"""Grid search over hyperparameters, scored with k-fold cross-validation.

From Python:

    best, results = grid_search(
        y, tx, "ridge_regression", {"lambda_": np.logspace(-8, 0, 17)}, k_fold=5
    )

From the command line (uses dataset/prepared/clean_full.npz by default):

    python grid_search.py --model ridge_regression
    python grid_search.py --model reg_logistic_regression --undersample
"""

import argparse
import itertools
import time
from pathlib import Path

import numpy as np

from cross_validation import (
    WARM_START_MODELS,
    build_k_indices,
    cross_validation,
    cross_validation_path,
    random_oversample,
    random_undersample,
)

# The MSE GD and regularized logistic grids are the ones of notebooks 03 and 04
# (shared across the team so the report compares like with like), except for
# MSE GD: we drop max_iters = 10000 (no gain over 5000 in notebook 03, twice
# the cost) and gamma = 0.2: on clean_full.npz the largest eigenvalue of
# tx.T @ tx / N is L ~ 20, so MSE GD diverges for gamma > 2/L ~ 0.1. For the
# logistic loss the limit is 8/L (~ 0.4 on linear features, ~ 0.18 with the
# squared features of notebook 04). See notebook 05.
DEFAULT_GRIDS = {
    "least_squares": {},
    "ridge_regression": {"lambda_": np.logspace(-8, 0, 17)},
    "mean_squared_error_gd": {
        "gamma": [0.01, 0.05, 0.1],
        "max_iters": [500, 1000, 5000],
    },
    "mean_squared_error_sgd": {"gamma": [1e-4, 1e-3], "max_iters": [20000]},
    "logistic_regression": {"gamma": [0.05, 0.1, 0.3], "max_iters": [500]},
    "reg_logistic_regression": {
        "lambda_": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        "gamma": [0.005, 0.01, 0.02, 0.05, 0.1],
        "max_iters": [300, 600, 1000],
    },
}


def iterate_grid(param_grid):
    """Yield every combination of `param_grid` as a dict.

    {"a": [1, 2], "b": [3]} -> {"a": 1, "b": 3}, {"a": 2, "b": 3}"""

    names = list(param_grid)
    for values in itertools.product(*(param_grid[name] for name in names)):
        yield dict(zip(names, values))


def grid_search(
    y,
    tx,
    model,
    param_grid,
    k_fold=5,
    seed=1,
    resample_fn=None,
    metric="f1_mean",
    verbose=True,
):
    """Run cross-validation for every hyperparameter combination.

    The same folds are used for every combination so the scores are
    comparable.

    Inputs :

    - y ndarray (N, ) : labels in {0, 1}

    - tx ndarray (N, D) : data matrix, bias column included

    - model str : name of a function in implementations.py

    - param_grid dict : {param name: list of values}

    - k_fold int : number of folds

    - resample_fn callable or None : rebalancing applied to training folds only

    - metric str : key of the cross_validation result to maximize

    Returns :

    - best dict : the best entry of `results`

    - results list of dict : one entry per combination, with "params" and the
      cross-validation summary (out-of-fold scores are dropped to save memory)"""

    k_indices = build_k_indices(y, k_fold, seed=seed)

    # For GD models, all values of max_iters are evaluated in one warm-started
    # run (see cross_validation_path), so we loop over the other parameters.
    warm_start = model in WARM_START_MODELS and len(param_grid.get("max_iters", [])) > 1
    if warm_start:
        iters_list = sorted(param_grid["max_iters"])
        base_grid = {k: v for k, v in param_grid.items() if k != "max_iters"}
    else:
        base_grid = param_grid
    combos = list(iterate_grid(base_grid))
    results = []

    for i, base in enumerate(combos, start=1):
        start = time.time()
        if warm_start:
            cvs = cross_validation_path(
                y, tx, model, base, iters_list, k_indices,
                resample_fn=resample_fn, seed=seed,
            )
            params_list = [{**base, "max_iters": n} for n in iters_list]
        else:
            cvs = [cross_validation(
                y, tx, model, base, k_indices, resample_fn=resample_fn, seed=seed
            )]
            params_list = [base]

        for params, cv in zip(params_list, cvs):
            cv.pop("oof_scores")
            cv["params"] = params
            results.append(cv)

            if verbose:
                print(
                    f"[{i}/{len(combos)}] {format_params(params)}  "
                    f"F1 = {cv['f1_mean']:.4f} ± {cv['f1_std']:.4f}  "
                    f"acc = {cv['accuracy_mean']:.4f}  "
                    f"thr = {cv['threshold']:.4g}  ({time.time() - start:.1f}s)"
                )

    best = max(results, key=lambda r: r[metric])
    return best, results


def format_params(params):
    if not params:
        return "(no hyperparameters)"
    return ", ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}"
                     for k, v in params.items())


def print_results(results, metric="f1_mean", top=10):
    """Print the `top` combinations sorted by `metric`."""

    ranked = sorted(results, key=lambda r: r[metric], reverse=True)[:top]
    print(f"\n{'rank':>4}  {'F1':>15}  {'accuracy':>8}  {'precision':>9}  "
          f"{'recall':>6}  {'threshold':>9}  params")
    for rank, r in enumerate(ranked, start=1):
        print(
            f"{rank:>4}  {r['f1_mean']:.4f} ± {r['f1_std']:.4f}  "
            f"{r['accuracy_mean']:>8.4f}  {r['precision_mean']:>9.4f}  "
            f"{r['recall_mean']:>6.4f}  {r['threshold']:>9.4g}  "
            f"{format_params(r['params'])}"
        )


# --------------------------------------------------------------------- CLI


def load_prepared(path):
    """Load a prepared .npz (x_train, y_train in {0, 1}) and add the bias column."""

    data = np.load(path, allow_pickle=True)
    x = data["x_train"].astype(np.float64)
    y = (data["y_train"] == 1).astype(int)
    tx = np.c_[np.ones(len(x)), x]
    return y, tx


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="ridge_regression", choices=DEFAULT_GRIDS)
    parser.add_argument("--data", default=root / "dataset/prepared/clean_full.npz")
    parser.add_argument("--k-fold", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    balance = parser.add_mutually_exclusive_group()
    balance.add_argument("--undersample", action="store_true",
                         help="random undersampling of the training folds")
    balance.add_argument("--oversample", action="store_true",
                         help="random oversampling of the training folds")
    args = parser.parse_args()

    y, tx = load_prepared(args.data)
    print(f"{args.model}: N = {len(y)}, D = {tx.shape[1]}, "
          f"positives = {y.mean():.2%}, {args.k_fold} folds")

    resample_fn = None
    if args.undersample:
        resample_fn = random_undersample
    elif args.oversample:
        resample_fn = random_oversample

    best, results = grid_search(
        y, tx, args.model, DEFAULT_GRIDS[args.model],
        k_fold=args.k_fold, seed=args.seed, resample_fn=resample_fn,
    )
    print_results(results)
    print(f"\nBest: {format_params(best['params'])}, threshold = "
          f"{best['threshold']:.4g}, F1 = {best['f1_mean']:.4f}")


if __name__ == "__main__":
    main()
