"""k-fold cross-validation for the six models in implementations.py.

Typical use:

    k_indices = build_k_indices(y, k_fold=5, seed=1)
    result = cross_validation(y, tx, "ridge_regression", {"lambda_": 1e-4}, k_indices)
    print(result["f1_mean"], result["threshold"])

`tx` must already contain the bias column, and `y` must be in {0, 1}.
"""

import numpy as np

import implementations as impl
from metrics import best_f1_threshold, compute_metrics, predict_labels

# ----------------------------------------------------------------------- models

# Models whose score is a probability (sigmoid(tx @ w)). For the others the
# score is the raw linear output tx @ w. In both cases we pick a threshold.
LOGISTIC_MODELS = {"logistic_regression", "reg_logistic_regression"}

# Full-batch gradient descent is deterministic, so running n1 iterations and
# then n2 more from the result gives exactly the (n1 + n2)-iteration weights.
# This lets us evaluate several max_iters for the price of the largest one.
# (SGD is excluded: its random sample sequence would differ.)
WARM_START_MODELS = {
    "mean_squared_error_gd",
    "logistic_regression",
    "reg_logistic_regression",
}


def fit_model(model, y, tx, params, initial_w=None):
    """Train `model` (a name from implementations.py) and return the weights.

    `params` holds the hyperparameters the model needs:
    - least_squares: {}
    - ridge_regression: {"lambda_"}
    - mean_squared_error_gd / _sgd, logistic_regression: {"max_iters", "gamma"}
    - reg_logistic_regression: {"lambda_", "max_iters", "gamma"}

    `initial_w` (iterative models only) defaults to zeros.

    `model` can also be a function f(y, tx, params) -> w, for models outside
    implementations.py; its score is then the raw linear output tx @ w.
    """

    if callable(model):
        return model(y, tx, params)

    if initial_w is None:
        initial_w = np.zeros(tx.shape[1])

    if model == "least_squares":
        w, _ = impl.least_squares(y, tx)
    elif model == "ridge_regression":
        w, _ = impl.ridge_regression(y, tx, params["lambda_"])
    elif model == "mean_squared_error_gd":
        w, _ = impl.mean_squared_error_gd(
            y, tx, initial_w, params["max_iters"], params["gamma"]
        )
    elif model == "mean_squared_error_sgd":
        w, _ = impl.mean_squared_error_sgd(
            y, tx, initial_w, params["max_iters"], params["gamma"]
        )
    elif model == "logistic_regression":
        w, _ = impl.logistic_regression(
            y, tx, initial_w, params["max_iters"], params["gamma"]
        )
    elif model == "reg_logistic_regression":
        w, _ = impl.reg_logistic_regression(
            y, tx, params["lambda_"], initial_w, params["max_iters"], params["gamma"]
        )
    else:
        raise ValueError(f"Unknown model: {model}")

    return w


def predict_scores(model, tx, w):
    """Score of each sample: higher means more likely to be positive."""

    if model in LOGISTIC_MODELS:
        return impl.sigmoid(tx @ w)
    return tx @ w


# ---------------------------------------------------------------- fold building


def build_k_indices(y, k_fold, seed=1, stratified=True):
    """Split the sample indices into `k_fold` disjoint folds.

    Every sample ends up in exactly one fold (no rows are dropped). With
    `stratified=True` each fold keeps the class ratio of `y`, which matters a
    lot when positives are rare.

    Returns :

    - k_indices list of k_fold ndarrays of indices"""

    rng = np.random.default_rng(seed)
    y = np.asarray(y)

    if stratified:
        folds = [[] for _ in range(k_fold)]
        for label in np.unique(y):
            label_idx = rng.permutation(np.where(y == label)[0])
            for k, chunk in enumerate(np.array_split(label_idx, k_fold)):
                folds[k].append(chunk)
        return [rng.permutation(np.concatenate(fold)) for fold in folds]

    return np.array_split(rng.permutation(len(y)), k_fold)


def train_test_indices(k_indices, k):
    """Indices for the k-th fold: (train = all other folds, test = fold k)."""

    te_idx = k_indices[k]
    tr_idx = np.concatenate([k_indices[i] for i in range(len(k_indices)) if i != k])
    return tr_idx, te_idx


# ------------------------------------------------------------ class rebalancing
# Apply these to the TRAINING folds only (pass them as `resample_fn`). Never
# rebalance before splitting: duplicated positives would land in both the
# training and the validation fold and inflate the score.


def random_undersample(x, y, seed=1, ratio=1.0):
    """Keep all positives and `ratio * n_pos` random negatives."""

    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]

    n_neg = min(len(neg), int(ratio * len(pos)))
    idx = rng.permutation(np.r_[pos, rng.choice(neg, n_neg, replace=False)])
    return x[idx], y[idx]


def random_oversample(x, y, seed=1, ratio=1.0):
    """Duplicate random positives until n_pos = `ratio * n_neg`."""

    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]

    n_extra = max(0, int(ratio * len(neg)) - len(pos))
    idx = rng.permutation(np.r_[neg, pos, rng.choice(pos, n_extra, replace=True)])
    return x[idx], y[idx]


# ------------------------------------------------------------- cross-validation


def cross_validation(
    y, tx, model, params, k_indices, resample_fn=None, threshold=None, seed=1
):
    """k-fold cross-validation of one model with one set of hyperparameters.

    For each fold, the model is trained on the other folds (after
    `resample_fn`, if given) and scores the held-out fold. The decision
    threshold is then chosen to maximize F1 on all out-of-fold scores pooled
    together, unless a fixed `threshold` is passed. This treats the threshold
    as one more hyperparameter; it is slightly optimistic, so use a fixed
    threshold or a separate hold-out set for the final reported number.

    Inputs :

    - y ndarray (N, ) : labels in {0, 1}

    - tx ndarray (N, D) : data matrix, bias column included

    - model str : name of a function in implementations.py

    - params dict : hyperparameters, see `fit_model`

    - k_indices list : output of `build_k_indices`

    - resample_fn callable or None : f(x, y, seed) -> (x, y), training folds only

    - threshold float or None : fixed threshold, or None to tune it for F1

    Returns :

    - result dict : threshold, mean/std of each metric over folds,
      per-fold metrics and the out-of-fold scores"""

    y = np.asarray(y)
    k_fold = len(k_indices)
    oof_scores = np.full(len(y), np.nan)

    for k in range(k_fold):
        tr_idx, te_idx = train_test_indices(k_indices, k)
        x_tr, y_tr = tx[tr_idx], y[tr_idx]

        if resample_fn is not None:
            x_tr, y_tr = resample_fn(x_tr, y_tr, seed=seed + k)

        w = fit_model(model, y_tr, x_tr, params)
        oof_scores[te_idx] = predict_scores(model, tx[te_idx], w)

    return summarize_oof(y, oof_scores, k_indices, threshold)


def cross_validation_path(
    y, tx, model, params, iters_list, k_indices, resample_fn=None, seed=1
):
    """`cross_validation` for every value in `iters_list` at once.

    Only for models in WARM_START_MODELS. In each fold the model is trained up
    to the smallest max_iters, scored, then trained further from those
    weights up to the next one, and so on. The result for each max_iters is
    identical to a separate run from zero, but the total cost is that of
    max(iters_list) instead of sum(iters_list).

    Inputs : as `cross_validation`, plus

    - params dict : hyperparameters without "max_iters"

    - iters_list list of int : values of max_iters to evaluate

    Returns :

    - results list of dict : one `cross_validation` result per value of
      sorted(iters_list)"""

    if model not in WARM_START_MODELS:
        raise ValueError(f"{model} cannot be warm-started")

    y = np.asarray(y)
    iters_list = sorted(iters_list)
    oof_scores = {n: np.full(len(y), np.nan) for n in iters_list}

    for k in range(len(k_indices)):
        tr_idx, te_idx = train_test_indices(k_indices, k)
        x_tr, y_tr = tx[tr_idx], y[tr_idx]

        if resample_fn is not None:
            x_tr, y_tr = resample_fn(x_tr, y_tr, seed=seed + k)

        w, done = None, 0
        for n in iters_list:
            w = fit_model(model, y_tr, x_tr, {**params, "max_iters": n - done}, w)
            done = n
            oof_scores[n][te_idx] = predict_scores(model, tx[te_idx], w)

    return [summarize_oof(y, oof_scores[n], k_indices) for n in iters_list]


def summarize_oof(y, oof_scores, k_indices, threshold=None):
    """Pick the F1-optimal threshold (unless given) and compute fold metrics."""

    k_fold = len(k_indices)

    if threshold is None:
        covered = np.concatenate(k_indices)
        threshold, _ = best_f1_threshold(y[covered], oof_scores[covered])

    fold_metrics = []
    for k in range(k_fold):
        te_idx = k_indices[k]
        y_pred = predict_labels(oof_scores[te_idx], threshold)
        fold_metrics.append(compute_metrics(y[te_idx], y_pred))

    result = {"threshold": threshold, "folds": fold_metrics, "oof_scores": oof_scores}
    for name in ("f1", "accuracy", "precision", "recall"):
        values = [m[name] for m in fold_metrics]
        result[f"{name}_mean"] = float(np.mean(values))
        result[f"{name}_std"] = float(np.std(values))

    return result
