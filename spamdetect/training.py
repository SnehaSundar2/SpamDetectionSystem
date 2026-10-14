"""The training workflow, step by step.

    raw data → clean → split (train/test, same spam share) → compare models by cross-validation (train only)
             → tune the best → refit → out-of-fold probabilities on the training set → choose the decision
             thresholds → evaluate ONCE on the untouched test set → explain, find mistakes → save

The test set is used exactly once, for the final report; every choice (which model, which settings, which
threshold) is made with cross-validation on the training messages.
"""

import logging
import platform
import sys
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate, train_test_split

from . import __version__
from . import evaluation as E
from . import models as M
from . import preprocessing as P
from . import schema as S
from .exceptions import DataError

log = logging.getLogger("spamdetect")


@dataclass
class TrainingResult:
    pipeline: object
    metadata: dict
    comparison: list
    cleaning: P.CleaningReport
    test_frame: pd.DataFrame          # text / actual / probability / predicted for every test message
    reference: pd.DataFrame           # cleaned TRAINING messages (used to find similar known messages)
    errors: dict = field(default_factory=dict)


def train(df_raw, seed=42, fast=False, tune=True, test_size=0.2, cv=5, progress=None, source="dataset"):
    """Run the whole workflow. `progress(message)` is called as each stage starts."""
    say = progress or (lambda message: None)
    started = datetime.now()

    say("Cleaning the data ...")
    df, report = P.clean(df_raw)
    X, y = P.split_xy(df)

    say("Splitting into training and test sets ...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
    if y_train.sum() < cv or (len(y_train) - y_train.sum()) < cv:
        raise DataError(f"Too few messages of one kind to cross-validate ({int(y_train.sum())} spam in training).")

    comparison = M.compare(X_train, y_train, cv=cv, seed=seed, fast=fast, progress=say)
    best = M.best_candidate(comparison)
    chosen, params = best["model"], {}

    tuned_note = ""
    if tune and chosen in M.TUNING_GRIDS:
        _, params = M.tune(chosen, X_train, y_train, seed=seed, fast=fast, progress=say)
        if params:
            say("Checking the tuned settings ...")
            tuned_pipe = M.pipeline_with_params(chosen, params, seed, fast)
            scores = cross_validate(tuned_pipe, X_train, y_train, cv=StratifiedKFold(cv, shuffle=True, random_state=seed),
                                    scoring=M.SCORING)
            tuned = M._summarise(f"{chosen} (tuned)", scores, 0.0)
            if tuned["average_precision"] > best["average_precision"]:
                comparison = sorted(comparison + [tuned], key=lambda r: (r["average_precision"], r["f1"]), reverse=True)
                best, tuned_note = tuned, "tuned"
            else:
                params = {}
                tuned_note = "tuning did not improve cross-validation, kept defaults"

    say("Training the final model ...")
    pipeline = (M.pipeline_with_params(chosen, params, seed, fast) if params
                else M.build_pipeline(M.candidates(seed, fast)[chosen]))
    pipeline.fit(X_train, y_train)

    say("Choosing the decision thresholds (from training data only) ...")
    oof = cross_val_predict(clone(pipeline), X_train, y_train, cv=StratifiedKFold(cv, shuffle=True, random_state=seed),
                            method="predict_proba")[:, 1]
    thresholds = E.choose_thresholds(y_train, oof)

    say("Evaluating on the untouched test set ...")
    test_p = pipeline.predict_proba(X_test)[:, 1]
    train_p = pipeline.predict_proba(X_train)[:, 1]
    t = thresholds["spam"]
    test_metrics = {**E.classification_metrics(y_test, test_p, t), **E.ranking_metrics(y_test, test_p)}
    test_metrics["default_threshold"] = E.classification_metrics(y_test, test_p, 0.5)
    test_metrics["majority_accuracy"] = float(100 * max(y_test.mean(), 1 - y_test.mean()))
    train_metrics = {**E.classification_metrics(y_train, train_p, t), **E.ranking_metrics(y_train, train_p)}
    review = (test_p >= thresholds["review"]) & (test_p < t)
    test_metrics["review_band"] = {"n": int(review.sum()), "spam": int((review & (y_test.to_numpy() == S.SPAM)).sum())}
    baseline_row = next(r for r in comparison if r["model"] == M.BASELINE)

    test_frame = X_test.copy()
    test_frame["actual"] = y_test.to_numpy()
    test_frame["probability"] = test_p
    test_frame["predicted"] = (test_p >= t).astype(int)

    say("Finding the most telling words and the mistakes ...")
    words = E.top_words(X_train["text"], y_train)
    errors = E.error_examples(test_frame)

    reference = pd.concat([X_train, y_train], axis=1).reset_index(drop=True)
    metadata = {
        "project_version": __version__,
        "trained_at": started.isoformat(timespec="seconds"),
        "training_seconds": round((datetime.now() - started).total_seconds(), 1),
        "source": source,
        "seed": seed,
        "model": chosen,
        "model_note": tuned_note,
        "model_params": {k: (v if v is None or isinstance(v, (int, float, str)) else str(v)) for k, v in params.items()},
        "rows": {"raw": report.rows_in, "clean": report.rows_out, "train": int(len(X_train)), "test": int(len(X_test))},
        "class_counts": report.class_counts,
        "metrics": {"test": test_metrics, "train": train_metrics, "cv": best, "baseline_cv": baseline_row},
        "thresholds": thresholds,
        "environment": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                        "numpy": np.__version__, "pandas": pd.__version__, "platform": sys.platform},
        "comparison": comparison,
        "cleaning": report.to_dict(),
        "top_words": words,
        "errors": errors,
        "segments": E.summarise_frame(test_frame),
        "pr_curve": E.pr_curve_points(y_test, test_p),
        "histograms": {"not_spam": E.probability_histogram(test_p[y_test.to_numpy() == S.HAM]),
                       "spam": E.probability_histogram(test_p[y_test.to_numpy() == S.SPAM])},
    }
    log.info("Trained %s: test precision=%.1f recall=%.1f", chosen, test_metrics["precision"], test_metrics["recall"])
    return TrainingResult(pipeline, metadata, comparison, report, test_frame, reference, errors)
