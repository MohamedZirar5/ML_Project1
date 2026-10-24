"""Reproduce our best AIcrowd submission (F1 = 0.446).

    python run.py

Pipeline (details and the comparison with the other models in
notebooks/05_feature_engineering_and_final_models.ipynb):

1. Features: the cleaning of notebook 01, plus three feature groups that
   notebook 01 drops ("redundant", "risk_management", "optional_modules"),
   standardized squares of the numeric columns, and interactions of every
   original column with age and sex (1236 columns with the bias).
2. Model: L2-regularized logistic regression (lambda = 1e-3), each positive
   row weighted 2x, solved with Newton's method. Same objective as
   reg_logistic_regression in implementations.py, run to convergence.
3. Threshold: the one maximizing F1 on the out-of-fold scores of a
   stratified 5-fold cross-validation.
4. The model is retrained on all training rows and the test predictions are
   written to submission.csv.

The first run parses the CSVs in dataset/ (about 2 minutes) and caches them
in dataset/prepared/raw_arrays.npz. A full run takes about 3 minutes.
"""

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code"))

import feature_engineering as fe  # noqa: E402
from cross_validation import build_k_indices, cross_validation  # noqa: E402
from helpers import create_csv_submission  # noqa: E402

GROUPS = ("redundant", "risk_management", "optional_modules")
INTERACT = ("_AGE80", "SEX=2")
LAMBDA = 1e-3
POS_WEIGHT = 2.0
K_FOLD = 5
SEED = 1


def fit(y, tx, params):
    """Weighted, regularized logistic regression (Newton's method)."""

    w, _ = fe.reg_logistic_newton(y, tx, params["lambda_"], pos_weight=params["pos_weight"])
    return w


def main():
    start = time.time()
    raw = fe.load_raw(ROOT / "dataset")
    y = raw["y_train"]
    x_train, x_test, _ = fe.build_features(
        raw, keep_groups=GROUPS, square_numeric=True, interact_with=INTERACT
    )
    tx_train = np.c_[np.ones(len(x_train)), x_train]
    tx_test = np.c_[np.ones(len(x_test)), x_test]
    del x_train, x_test
    print(f"features: {tx_train.shape[1]} columns, {len(y)} train rows, "
          f"{len(tx_test)} test rows ({time.time() - start:.0f}s)")

    params = {"lambda_": LAMBDA, "pos_weight": POS_WEIGHT}
    k_indices = build_k_indices(y, K_FOLD, seed=SEED)
    cv = cross_validation(y, tx_train, fit, params, k_indices)
    threshold = cv["threshold"]
    print(f"{K_FOLD}-fold CV: F1 = {cv['f1_mean']:.4f} ± {cv['f1_std']:.4f}, "
          f"threshold on tx @ w = {threshold:.4f} ({time.time() - start:.0f}s)")

    w = fit(y, tx_train, params)
    predictions = np.where(tx_test @ w >= threshold, 1, -1)
    print(f"predicted positive rate on test: {np.mean(predictions == 1):.4f}")

    path = ROOT / "submission.csv"
    create_csv_submission(raw["test_ids"], predictions, path)
    print(f"submission written to {path} ({time.time() - start:.0f}s)")


if __name__ == "__main__":
    main()
