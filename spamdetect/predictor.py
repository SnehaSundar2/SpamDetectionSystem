"""Checking messages: spam probability, verdict, the words behind it, similar known messages and batch files."""

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import persistence
from . import schema as S
from . import textproc
from .exceptions import DataError, NotTrainedError, ValidationError
from .preprocessing import normalise_headers
from .validators import validate_message

MAX_BATCH_ROWS = 5000
EFFECT_SHOWN = 0.25       # a piece must change the log-odds of spam by at least this much (odds x1.3) to be mentioned


@dataclass
class Prediction:
    text: str
    probability: float
    verdict: str
    spam_threshold: float
    review_threshold: float
    warnings: list = field(default_factory=list)
    signals: list = field(default_factory=list)
    toward_spam: list = field(default_factory=list)       # [{"word", "effect", "odds"}] pieces that raise the odds of spam
    toward_genuine: list = field(default_factory=list)    # words that lower it
    effects: dict = field(default_factory=dict)           # lower-case piece -> change in log-odds (used to highlight text)
    similar: list = field(default_factory=list)

    def to_dict(self):
        return {"verdict": self.verdict, "spam_probability": round(self.probability, 4),
                "spam_threshold": round(self.spam_threshold, 4), "review_threshold": round(self.review_threshold, 4),
                "warnings": self.warnings, "signals": self.signals,
                "words_toward_spam": self.toward_spam, "words_toward_not_spam": self.toward_genuine,
                "similar_messages": self.similar}


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class Predictor:
    def __init__(self, loaded):
        self.model = loaded.pipeline
        self.meta = loaded.metadata
        self.reference = loaded.reference
        self.load_warnings = list(loaded.warnings)
        self.spam_threshold = self.meta["thresholds"]["spam"]
        self.review_threshold = self.meta["thresholds"]["review"]
        self._reference_matrix = None

    @classmethod
    def from_directory(cls, directory):
        return cls(persistence.load(directory))

    # ------------------------------------------------------------------ core
    def _proba(self, texts):
        return np.asarray(self.model.predict_proba(pd.DataFrame({"text": list(texts)}))[:, 1], float)

    def verdict(self, probability):
        return S.verdict_for(probability, self.review_threshold, self.spam_threshold)

    def check(self, raw, explain=True, similar=True):
        """Check one message. Raises ValidationError for an unusable message."""
        text, warnings = validate_message(raw)
        probability = float(self._proba([text])[0])
        result = Prediction(text, probability, self.verdict(probability), self.spam_threshold, self.review_threshold,
                            warnings, textproc.signals(text))
        if explain:
            self._explain(result)
        if similar:
            result.similar = self._similar(text)
        return result

    # ------------------------------------------------------------------ explanations
    def _explain(self, result, top=6):
        """Which parts of the message push the score up or down.

        Each piece (a word, a link, a phone number) is taken out in turn and the message is re-scored. The
        effect is the change in the LOG-ODDS of spam, so it still shows up when the message is already
        99% spam (a probability would be stuck near 100% either way). exp(effect) is the factor by which that
        piece multiplies the odds of spam.
        """
        pieces = textproc.units(result.text)
        if not pieces:
            return
        variants = [textproc.remove_unit(result.text, kind, piece) for kind, piece in pieces]
        effects = _logit(result.probability) - _logit(self._proba(variants))
        rows = []
        for (kind, piece), effect in zip(pieces, effects):
            name = {"link": "[link] " + piece, "phone": "[phone number] " + piece}.get(kind, piece)
            rows.append({"word": name, "effect": round(float(effect), 3), "odds": round(float(np.exp(effect)), 2)})
            result.effects[piece.lower()] = float(effect)
        result.toward_spam = sorted((r for r in rows if r["effect"] >= EFFECT_SHOWN), key=lambda r: -r["effect"])[:top]
        result.toward_genuine = sorted((r for r in rows if r["effect"] <= -EFFECT_SHOWN), key=lambda r: r["effect"])[:top]

    def _reference_vectors(self):
        if self._reference_matrix is None:
            vectorizer = self.model.named_steps["features"].named_transformers_["word"]
            self._reference_matrix = vectorizer.transform(self.reference["text"])
        return self._reference_matrix

    def _similar(self, text, n=3):
        """The known training messages that use the most similar words."""
        vectorizer = self.model.named_steps["features"].named_transformers_["word"]
        query = vectorizer.transform([text])
        sims = (self._reference_vectors() @ query.T).toarray().ravel()
        rows = []
        for idx in np.argsort(-sims)[:n]:
            if sims[idx] < 0.05:
                break
            known = self.reference.iloc[int(idx)]
            shown = known["text"] if len(known["text"]) <= 140 else known["text"][:139] + "…"
            rows.append({"text": shown, "label": S.CLASS_NAMES[int(known["label"])], "similarity": round(float(sims[idx]), 2)})
        return rows

    # ------------------------------------------------------------------ batch
    def predict_csv(self, in_path, out_path):
        """Check every row of a CSV. Bad rows are reported (with line numbers), good rows still get a verdict.

        Returns (ok_count, errors) where errors is a list of 'Line N: message'. The output CSV has the original
        columns plus spam_probability, verdict and a status column.
        """
        in_path, out_path = Path(str(in_path).strip().strip('"')), Path(str(out_path).strip().strip('"'))
        if not in_path.exists():
            raise DataError(f"File not found: {in_path}")
        if out_path.exists():
            raise DataError(f"{out_path} already exists; choose a new file name.")
        try:
            df = pd.read_csv(in_path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
        except pd.errors.EmptyDataError:
            raise DataError(f"{in_path.name} is empty.") from None
        except (pd.errors.ParserError, UnicodeDecodeError, OSError) as exc:
            raise DataError(f"Could not read {in_path.name}: {exc}") from None
        df.columns = normalise_headers(df.columns)
        if len(df) > MAX_BATCH_ROWS:
            raise DataError(f"{in_path.name} has {len(df)} rows; the limit is {MAX_BATCH_ROWS}.")
        column = next((c for c in S.TEXT_COLUMNS if c in df.columns), None)
        if column is None:
            raise DataError(f"{in_path.name} needs a text column (any of: {', '.join(S.TEXT_COLUMNS)}).")

        good, errors, status = [], [], []
        for position, value in enumerate(df[column]):
            line = position + 2   # header is line 1
            try:
                text, warnings = validate_message(value)
                good.append((position, text))
                status.append("ok" + (f" ({'; '.join(w.split('.')[0].lower() for w in warnings)})" if warnings else ""))
            except ValidationError as exc:
                errors.append(f"Line {line}: {exc}")
                status.append(f"error: {exc}")
        out = df.copy()
        out["spam_probability"], out["verdict"] = "", ""
        if good:
            probabilities = self._proba([text for _, text in good])
            for (i, _), p in zip(good, probabilities):
                out.loc[i, "spam_probability"] = f"{p:.4f}"
                out.loc[i, "verdict"] = self.verdict(float(p))
        out["status"] = status
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out.to_csv(out_path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
        except OSError as exc:
            raise DataError(f"Could not write {out_path}: {exc.strerror or exc}") from exc
        return len(good), errors


def highlight_pieces(text, effects, minimum=EFFECT_SHOWN):
    """Split a message into [(chunk, effect_or_None)] so a page can mark the words that raise the spam score.

    Links and phone numbers (keys that are not a plain word) are matched whole; everything else word by word.
    """
    long_keys = sorted((k for k in effects if not re.fullmatch(r"[a-z0-9']+", k)), key=len, reverse=True)
    pattern = "|".join([re.escape(k) for k in long_keys] + [r"[A-Za-z0-9']+", r"[^A-Za-z0-9']"])
    chunks = []
    for match in re.finditer(pattern, text, flags=re.I):
        chunk = match.group(0)
        effect = effects.get(chunk.lower().strip("'"))
        chunks.append((chunk, effect if effect is not None and effect >= minimum else None))
    return chunks


def require_predictor(directory):
    """Load the saved model or raise NotTrainedError with instructions."""
    if not persistence.is_trained(directory):
        raise NotTrainedError(f"No trained model found in '{directory}'. Train one first "
                              "(menu option 'Train a model', or: python main.py --train).")
    return Predictor.from_directory(directory)
