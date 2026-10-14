"""Candidate models, cross-validated comparison and light hyper-parameter tuning.

Every candidate is wrapped in the SAME pipeline (text features → classifier), so the comparison is
fair and nothing learned in one validation fold leaks into another. Folds keep the spam share equal
(stratified), because spam is the minority class.

Models are ranked by average precision (area under the precision-recall curve): it does not depend
on any threshold and, unlike accuracy, it cannot be gamed by calling everything "not spam".
"""

import time
import warnings

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, f1_score, precision_score, recall_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from . import preprocessing as P

BASELINE = "Baseline (always 'not spam')"
SCORING = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
    "roc_auc": "roc_auc",
    "average_precision": "average_precision",
}

TUNING_GRIDS = {
    "Naive Bayes": {"model__alpha": [0.01, 0.03, 0.1, 0.3, 1.0]},
    "Logistic Regression": {"model__C": [1, 3, 10, 30, 100]},
    "Linear SVM": {"model__estimator__C": [0.1, 0.3, 1, 3]},
}


def build_pipeline(estimator):
    return Pipeline([("features", P.build_features()), ("model", estimator)])


def candidates(seed=42, fast=False):
    """name -> unfitted estimator (the pipeline is built around it)."""
    return {
        BASELINE: DummyClassifier(strategy="prior"),
        "Naive Bayes": MultinomialNB(alpha=0.1),
        "Logistic Regression": LogisticRegression(C=10, max_iter=2000, solver="liblinear", random_state=seed),
        "Linear SVM": CalibratedClassifierCV(LinearSVC(C=1.0, random_state=seed), cv=3, method="sigmoid"),
        "Random Forest": RandomForestClassifier(n_estimators=60 if fast else 200, n_jobs=1 if fast else -1,
                                                random_state=seed),
    }


def _summarise(name, scores, seconds):
    out = {"model": name, "seconds": round(seconds, 1)}
    for key in SCORING:
        out[key] = float(scores[f"test_{key}"].mean())
    out["average_precision_std"] = float(scores["test_average_precision"].std())
    return out


def compare(X, y, cv=5, seed=42, fast=False, progress=None):
    """Cross-validate every candidate on the TRAINING data only. Best (by average precision) first."""
    folds = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    results = []
    for name, estimator in candidates(seed, fast).items():
        if progress:
            progress(f"Cross-validating {name} ...")
        started = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = cross_validate(build_pipeline(estimator), X, y, cv=folds, scoring=SCORING)
        results.append(_summarise(name, scores, time.perf_counter() - started))
    return sorted(results, key=lambda r: (r["average_precision"], r["f1"]), reverse=True)


def best_candidate(results):
    """The best-scoring model that is not the baseline."""
    return next(r for r in results if r["model"] != BASELINE)


def tune(name, X, y, seed=42, cv=3, iterations=6, fast=False, progress=None):
    """Randomised search over a small grid. Returns (search, tuned_params) or (None, {}) if there is no grid."""
    grid = TUNING_GRIDS.get(name)
    if not grid:
        return None, {}
    if progress:
        progress(f"Tuning {name} ({iterations} settings × {cv} folds) ...")
    search = RandomizedSearchCV(build_pipeline(candidates(seed, fast)[name]), grid,
                                n_iter=int(min(iterations, np.prod([len(v) for v in grid.values()]))),
                                cv=StratifiedKFold(cv, shuffle=True, random_state=seed),
                                scoring="average_precision", random_state=seed, n_jobs=1, refit=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        search.fit(X, y)
    return search, {k.replace("model__", "", 1): v for k, v in search.best_params_.items()}


def pipeline_with_params(name, params, seed=42, fast=False):
    """A fresh pipeline for `name` with tuned parameters applied."""
    estimator = candidates(seed, fast)[name]
    estimator.set_params(**params)
    return build_pipeline(estimator)
