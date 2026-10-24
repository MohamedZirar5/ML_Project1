"""Tests for metrics.py, cross_validation.py and grid_search.py.

Run from the repository root with:
    pytest code/scripts/test_model_selection.py -v
"""

import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))  # implementations.py
sys.path.insert(0, str(ROOT / "code"))  # helper modules
import cross_validation as cv  # noqa: E402
import grid_search as gs  # noqa: E402
import metrics  # noqa: E402


@pytest.fixture
def imbalanced_data():
    """Linearly separable-ish data with ~10% positives."""
    rng = np.random.default_rng(0)
    N, D = 2000, 6
    x = rng.normal(size=(N, D - 1))
    logits = x @ rng.normal(size=D - 1) - 2.5
    y = (rng.random(N) < 1 / (1 + np.exp(-logits))).astype(int)
    tx = np.c_[np.ones(N), x]
    return y, tx


# --------------------------------------------------------------------- metrics


def test_metrics_known_values():
    y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    y_pred = np.array([1, 1, 0, 1, 0, 0, 0, 0])
    m = metrics.compute_metrics(y_true, y_pred)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 1, 1, 4)
    assert m["accuracy"] == pytest.approx(6 / 8)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(2 / 3)
    assert metrics.f1_score(y_true, y_pred) == pytest.approx(m["f1"])


def test_metrics_no_positive_predictions():
    m = metrics.compute_metrics(np.array([1, 0]), np.array([0, 0]))
    assert m["precision"] == 0.0
    assert m["f1"] == 0.0


def test_to_binary():
    np.testing.assert_array_equal(metrics.to_binary([-1, 1, 1, -1]), [0, 1, 1, 0])


def test_best_f1_threshold_matches_brute_force():
    rng = np.random.default_rng(1)
    y = (rng.random(500) < 0.15).astype(int)
    # Rounded scores create ties, which the fast version must handle.
    scores = np.round(y * 0.5 + rng.normal(size=500), 1)

    thr, f1 = metrics.best_f1_threshold(y, scores)

    brute = max(
        metrics.f1_score(y, metrics.predict_labels(scores, t)) for t in np.unique(scores)
    )
    assert f1 == pytest.approx(brute)
    assert metrics.f1_score(y, metrics.predict_labels(scores, thr)) == pytest.approx(f1)


# ------------------------------------------------------------ cross-validation


@pytest.mark.parametrize("stratified", [True, False])
def test_k_indices_partition_all_rows(imbalanced_data, stratified):
    y, _ = imbalanced_data
    k_indices = cv.build_k_indices(y, 5, seed=3, stratified=stratified)
    all_idx = np.concatenate(k_indices)
    assert len(all_idx) == len(y)
    assert len(np.unique(all_idx)) == len(y)


def test_k_indices_stratified_keeps_ratio(imbalanced_data):
    y, _ = imbalanced_data
    for fold in cv.build_k_indices(y, 5, seed=3):
        assert abs(y[fold].mean() - y.mean()) < 0.01


def test_resamplers_balance_classes(imbalanced_data):
    y, tx = imbalanced_data
    _, y_under = cv.random_undersample(tx, y)
    _, y_over = cv.random_oversample(tx, y)
    assert y_under.mean() == pytest.approx(0.5)
    assert y_over.mean() == pytest.approx(0.5, abs=1e-3)
    assert y_under.sum() == y.sum()
    assert (y_over == 0).sum() == (y == 0).sum()


@pytest.mark.parametrize(
    "model, params",
    [
        ("least_squares", {}),
        ("ridge_regression", {"lambda_": 1e-4}),
        ("mean_squared_error_gd", {"gamma": 0.1, "max_iters": 100}),
        ("mean_squared_error_sgd", {"gamma": 1e-3, "max_iters": 500}),
        ("logistic_regression", {"gamma": 0.5, "max_iters": 100}),
        ("reg_logistic_regression", {"lambda_": 1e-4, "gamma": 0.5, "max_iters": 100}),
    ],
)
def test_cross_validation_runs_for_every_model(imbalanced_data, model, params):
    y, tx = imbalanced_data
    k_indices = cv.build_k_indices(y, 4)
    result = cv.cross_validation(y, tx, model, params, k_indices)
    assert len(result["folds"]) == 4
    assert not np.isnan(result["oof_scores"]).any()
    # Much better than predicting everything positive (F1 ~ 2p / (1 + p)).
    assert result["f1_mean"] > 2 * y.mean() / (1 + y.mean())


def test_cross_validation_fixed_threshold(imbalanced_data):
    y, tx = imbalanced_data
    k_indices = cv.build_k_indices(y, 3)
    result = cv.cross_validation(
        y, tx, "ridge_regression", {"lambda_": 1e-3}, k_indices, threshold=0.5
    )
    assert result["threshold"] == 0.5


@pytest.mark.parametrize(
    "model, params",
    [
        ("mean_squared_error_gd", {"gamma": 0.1}),
        ("reg_logistic_regression", {"lambda_": 1e-4, "gamma": 0.5}),
    ],
)
def test_cross_validation_path_matches_separate_runs(imbalanced_data, model, params):
    y, tx = imbalanced_data
    k_indices = cv.build_k_indices(y, 3)
    path = cv.cross_validation_path(y, tx, model, params, [50, 20, 120], k_indices)
    for n, warm in zip([20, 50, 120], path):
        cold = cv.cross_validation(y, tx, model, {**params, "max_iters": n}, k_indices)
        np.testing.assert_allclose(warm["oof_scores"], cold["oof_scores"], rtol=1e-10)
        assert warm["f1_mean"] == pytest.approx(cold["f1_mean"])


def test_cross_validation_with_undersampling(imbalanced_data):
    y, tx = imbalanced_data
    k_indices = cv.build_k_indices(y, 3)
    result = cv.cross_validation(
        y, tx, "least_squares", {}, k_indices, resample_fn=cv.random_undersample
    )
    assert 0 < result["f1_mean"] <= 1


# ---------------------------------------------------------------- grid search


def test_iterate_grid():
    combos = list(gs.iterate_grid({"a": [1, 2], "b": [3, 4, 5]}))
    assert len(combos) == 6
    assert {"a": 2, "b": 5} in combos
    assert list(gs.iterate_grid({})) == [{}]


def test_grid_search_prefers_small_lambda(imbalanced_data):
    y, tx = imbalanced_data
    best, results = gs.grid_search(
        y, tx, "ridge_regression", {"lambda_": [1e-6, 10.0]}, k_fold=3, verbose=False
    )
    assert len(results) == 2
    assert best["params"]["lambda_"] == 1e-6
    assert best["f1_mean"] == max(r["f1_mean"] for r in results)


def test_grid_search_warm_start_covers_every_combination(imbalanced_data):
    y, tx = imbalanced_data
    grid = {"gamma": [0.05, 0.1], "max_iters": [30, 10]}
    _, results = gs.grid_search(
        y, tx, "mean_squared_error_gd", grid, k_fold=3, verbose=False
    )
    found = sorted((r["params"]["gamma"], r["params"]["max_iters"]) for r in results)
    assert found == [(0.05, 10), (0.05, 30), (0.1, 10), (0.1, 30)]
