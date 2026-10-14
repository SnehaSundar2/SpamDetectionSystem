"""Model evaluation for a two-class problem where mistakes cost different amounts.

Blocking a genuine message (a false positive) is worse than letting one spam through (a false negative),
so besides accuracy this module measures precision, recall, the false-alarm rate, picks the decision
threshold on TRAINING data, and lists the actual mistakes.
"""

import math
import re
from collections import Counter

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from . import schema as S
from .textproc import URL, normalise_text, pretty_token


def classification_metrics(actual, probability, threshold):
    """Counts and percentages when every message scoring at or above `threshold` is called spam."""
    actual = np.asarray(actual, int)
    flagged = np.asarray(probability, float) >= threshold
    spam = actual == S.SPAM
    tp, fp = int((flagged & spam).sum()), int((flagged & ~spam).sum())
    fn, tn = int((~flagged & spam).sum()), int((~flagged & ~spam).sum())
    precision = 100 * tp / (tp + fp) if tp + fp else 0.0
    recall = 100 * tp / (tp + fn) if tp + fn else 0.0
    return {
        "n": int(len(actual)), "threshold": float(threshold),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "accuracy": 100 * (tp + tn) / len(actual),
        "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "false_alarm_rate": 100 * fp / (fp + tn) if fp + tn else 0.0,
    }


def ranking_metrics(actual, probability):
    """Threshold-free scores: ROC AUC and average precision (needs both classes present)."""
    actual = np.asarray(actual, int)
    if len(set(actual.tolist())) < 2:
        return {"roc_auc": 0.0, "average_precision": 0.0}
    return {"roc_auc": float(roc_auc_score(actual, probability)),
            "average_precision": float(average_precision_score(actual, probability))}


def choose_thresholds(actual, probability, precision_target=S.PRECISION_TARGET, review_precision=S.REVIEW_PRECISION):
    """Pick the two decision thresholds from (out-of-fold) training predictions.

    spam threshold   — the one catching the most spam while at least `precision_target` of flagged messages
                       really are spam (genuine messages are protected). If no threshold reaches the target, the
                       one with the best F1 is used and `precision_target_met` is False.
    review threshold — the lowest threshold for which at least `review_precision` of the messages between it and
                       the spam threshold really are spam. Messages in that band are called "Suspicious" and left for
                       a person to look at.
    """
    actual, probability = np.asarray(actual, int), np.asarray(probability, float)
    precision, recall, thresholds = precision_recall_curve(actual, probability)
    precision, recall = precision[:-1], recall[:-1]
    if len(thresholds) == 0:
        return {"spam": 0.5, "review": 0.5, "precision_target": precision_target, "precision_target_met": False,
                "review_precision": review_precision}
    reaches = precision >= precision_target
    if reaches.any():
        best_recall = recall[reaches].max()
        spam_t = float(thresholds[reaches & (recall == best_recall)].max())
    else:
        f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
        spam_t = float(thresholds[int(f1.argmax())])
    below = probability < spam_t
    order = np.argsort(-probability[below])
    band_p, band_spam = probability[below][order], (actual[below][order] == S.SPAM)
    review_t = spam_t
    if len(band_p):
        # Look only at whole score levels (a threshold always lets every tied score through together).
        levels = np.unique(band_p)[::-1]
        count = np.searchsorted(-band_p, -levels, side="right")
        share = np.cumsum(band_spam)[count - 1] / count
        good = np.flatnonzero(share >= review_precision)
        if len(good):
            review_t = float(levels[good[-1]])
    return {"spam": spam_t, "review": min(review_t, spam_t), "precision_target": precision_target,
            "precision_target_met": bool(reaches.any()), "review_precision": review_precision}


def pr_curve_points(actual, probability, max_points=60):
    """[(recall, precision)] for a precision-recall chart, thinned to at most `max_points`."""
    precision, recall, _ = precision_recall_curve(np.asarray(actual, int), np.asarray(probability, float))
    points = list(zip(recall.tolist(), precision.tolist()))
    if len(points) > max_points:
        keep = np.linspace(0, len(points) - 1, max_points).round().astype(int)
        points = [points[i] for i in sorted(set(keep.tolist()))]
    return [(round(r, 4), round(p, 4)) for r, p in points]


def probability_histogram(probability, bins=10):
    """Counts of messages per spam-probability band (0-10%, 10-20%, ... 90-100%)."""
    counts, _ = np.histogram(np.clip(np.asarray(probability, float), 0, 1), bins=bins, range=(0, 1))
    return [int(c) for c in counts]


def length_band(text):
    n = len(text)
    return "Short (under 40 characters)" if n < 40 else "Medium (40-120)" if n <= 120 else "Long (over 120)"


def segment_errors(frame, by):
    """Mistakes per group of messages (e.g. per length band), most error-prone first."""
    rows = []
    for key, group in frame.groupby(by, observed=True):
        wrong = int((group["actual"] != group["predicted"]).sum())
        rows.append({"group": str(key), "n": int(len(group)), "spam": int((group["actual"] == S.SPAM).sum()),
                     "errors": wrong, "error_rate": 100 * wrong / len(group)})
    return sorted(rows, key=lambda r: (-r["error_rate"], r["group"]))


def top_words(texts, labels, k=12, min_df=5):
    """The most telling words: smoothed log-odds of appearing in spam rather than genuine messages.

    Counted over the TRAINING messages only, once per message. Independent of the chosen model.
    """
    labels = np.asarray(labels, int)
    spam_docs, ham_docs = Counter(), Counter()
    for text, label in zip(texts, labels):
        words = set(re.findall(r"[a-z]+", normalise_text(text)))
        (spam_docs if label == S.SPAM else ham_docs).update(words)
    n_spam, n_ham = max(int((labels == S.SPAM).sum()), 1), max(int((labels != S.SPAM).sum()), 1)
    rows = []
    for word in set(spam_docs) | set(ham_docs):
        s, h = spam_docs.get(word, 0), ham_docs.get(word, 0)
        if s + h < min_df:
            continue
        odds = math.log((s + 1) / (n_spam + 2)) - math.log((h + 1) / (n_ham + 2))
        rows.append({"word": pretty_token(word), "spam_messages": s, "genuine_messages": h, "log_odds": round(odds, 3)})
    rows.sort(key=lambda r: r["log_odds"], reverse=True)
    return {"spam": rows[:k], "ham": sorted(rows[-k:], key=lambda r: r["log_odds"])}


def error_examples(frame, k=8, width=170):
    """The messages the model got wrong on the test set, most confident mistakes first."""
    def rows(mask, most_confident):
        picked = frame[mask].sort_values("probability", ascending=not most_confident).head(k)
        return [{"text": t if len(t) <= width else t[:width - 1] + "…", "probability": round(float(p), 3)}
                for t, p in zip(picked["text"], picked["probability"])]
    return {"false_positives": rows((frame["predicted"] == S.SPAM) & (frame["actual"] == S.HAM), True),
            "false_negatives": rows((frame["predicted"] == S.HAM) & (frame["actual"] == S.SPAM), False)}


def has_link(text):
    return bool(URL.search(text))


def noise_ceiling(label_noise):
    """For SYNTHETIC data only: with `label_noise` of labels flipped at random, no model can beat this accuracy."""
    return 100 * (1 - label_noise)


def summarise_frame(frame):
    """A DataFrame of test messages (text, actual, probability, predicted) → the segment tables for the report."""
    frame = frame.copy()
    frame["length"] = frame["text"].map(length_band)
    frame["has_link"] = frame["text"].map(lambda t: "Contains a link" if has_link(t) else "No link")
    return {"length": segment_errors(frame, "length"), "link": segment_errors(frame, "has_link")}
