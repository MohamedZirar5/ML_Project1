"""Classification metrics for Project 1 (NumPy only).

All functions expect binary labels in {0, 1}. Use `to_binary` first if your
labels are in {-1, 1}.
"""

import numpy as np


def to_binary(y):
    """Map labels in {-1, 1} (or {0, 1}) to {0, 1}."""

    return (np.asarray(y) == 1).astype(int)


def confusion_counts(y_true, y_pred):
    """Return (tp, fp, fn, tn) for binary labels in {0, 1}."""

    y_true = np.asarray(y_true).astype(bool)
    y_pred = np.asarray(y_pred).astype(bool)

    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    tn = int(np.sum(~y_true & ~y_pred))

    return tp, fp, fn, tn


def accuracy(y_true, y_pred):
    tp, fp, fn, tn = confusion_counts(y_true, y_pred)
    return (tp + tn) / (tp + fp + fn + tn)


def precision(y_true, y_pred):
    tp, fp, _, _ = confusion_counts(y_true, y_pred)
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def recall(y_true, y_pred):
    tp, _, fn, _ = confusion_counts(y_true, y_pred)
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def f1_score(y_true, y_pred):
    # F1 = 2PR / (P + R) = 2TP / (2TP + FP + FN)
    tp, fp, fn, _ = confusion_counts(y_true, y_pred)
    denom = 2 * tp + fp + fn
    return 2 * tp / denom if denom > 0 else 0.0


def compute_metrics(y_true, y_pred):
    """All metrics at once, as a dict."""

    tp, fp, fn, tn = confusion_counts(y_true, y_pred)
    n = tp + fp + fn + tn

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_denom = 2 * tp + fp + fn

    return {
        "accuracy": (tp + tn) / n,
        "precision": prec,
        "recall": rec,
        "f1": 2 * tp / f1_denom if f1_denom > 0 else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def predict_labels(scores, threshold):
    """Predict 1 where score >= threshold, else 0."""

    return (np.asarray(scores) >= threshold).astype(int)


def best_f1_threshold(y_true, scores):
    """Threshold on `scores` that maximizes F1, found in O(N log N).

    Works with any score where higher means "more likely positive": the raw
    output tx @ w of least squares / ridge, or probabilities from logistic
    regression.

    Inputs :

    - y_true ndarray (N, ) : labels in {0, 1}

    - scores ndarray (N, ) : model scores

    Returns :

    - threshold float : predict 1 when score >= threshold

    - f1 float : F1-score obtained with that threshold"""

    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)

    n_pos = y_true.sum()
    if n_pos == 0:
        return float(scores.max()) + 1.0, 0.0

    # Sort by decreasing score: predicting the first i samples as positive
    # gives tp = cumsum(y) and fp = i - tp.
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    tp = np.cumsum(y_true[order])
    fp = np.arange(1, len(y_true) + 1) - tp

    # Only cut at the last position of each run of equal scores, otherwise
    # we would count a threshold that splits tied samples.
    is_cut = np.r_[sorted_scores[1:] != sorted_scores[:-1], True]
    tp, fp, cut_scores = tp[is_cut], fp[is_cut], sorted_scores[is_cut]

    # fn = n_pos - tp, so 2TP + FP + FN = tp + fp + n_pos
    f1 = 2 * tp / (tp + fp + n_pos)
    best = np.argmax(f1)

    return float(cut_scores[best]), float(f1[best])
