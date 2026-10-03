"""Tests for the functions implemented in code/implementations.py.

Run from the repository root with:
    pytest code/scripts/test_implementations.py -v
"""

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import implementations as impl  # noqa: E402

RTOL = 1e-4
ATOL = 1e-8


# Same toy data as the public grading tests.
@pytest.fixture
def y():
    return np.array([0.1, 0.3, 0.5])


@pytest.fixture
def tx():
    return np.array([[2.3, 3.2], [1.0, 0.1], [1.4, 2.3]])


@pytest.fixture
def random_data():
    rng = np.random.default_rng(0)
    N, D = 200, 5
    tx = np.c_[np.ones(N), rng.normal(size=(N, D - 1))]
    w_true = rng.normal(size=D)
    y = tx @ w_true + 0.1 * rng.normal(size=N)
    return y, tx, w_true


def mse(y, tx, w):
    e = y - tx @ w
    return e @ e / (2 * len(y))


# ---------------------------------------------------------------- least_squares


def test_least_squares_public_values(y, tx):
    w, loss = impl.least_squares(y, tx)

    np.testing.assert_allclose(w, [0.218786, -0.053837], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(loss, 0.026942, rtol=RTOL, atol=ATOL)


def test_least_squares_output_shapes(y, tx):
    w, loss = impl.least_squares(y, tx)

    assert w.shape == (tx.shape[1],)
    assert np.ndim(loss) == 0


def test_least_squares_matches_numpy_lstsq(random_data):
    y, tx, _ = random_data
    w, _ = impl.least_squares(y, tx)

    w_ref = np.linalg.lstsq(tx, y, rcond=None)[0]
    np.testing.assert_allclose(w, w_ref, rtol=1e-8, atol=1e-10)


def test_least_squares_loss_is_half_mse(random_data):
    y, tx, _ = random_data
    w, loss = impl.least_squares(y, tx)

    np.testing.assert_allclose(loss, mse(y, tx, w), rtol=1e-10)


def test_least_squares_recovers_exact_weights_without_noise(random_data):
    _, tx, w_true = random_data
    y = tx @ w_true
    w, loss = impl.least_squares(y, tx)

    np.testing.assert_allclose(w, w_true, rtol=1e-8, atol=1e-10)
    assert loss == pytest.approx(0.0, abs=1e-20)


def test_least_squares_residual_orthogonal_to_features(random_data):
    y, tx, _ = random_data
    w, _ = impl.least_squares(y, tx)

    # Normal equations: X^T (y - Xw) = 0
    np.testing.assert_allclose(tx.T @ (y - tx @ w), 0.0, atol=1e-9)


def test_least_squares_is_optimal(random_data):
    y, tx, _ = random_data
    w, loss = impl.least_squares(y, tx)

    rng = np.random.default_rng(1)
    for _ in range(20):
        w_perturbed = w + 0.01 * rng.normal(size=w.shape)
        assert mse(y, tx, w_perturbed) >= loss


# ------------------------------------------------------------- ridge_regression


def test_ridge_regression_returns_w_and_loss(y, tx):
    result = impl.ridge_regression(y, tx, 1.0)

    assert isinstance(result, tuple), "ridge_regression must return (w, loss)"
    assert len(result) == 2, "ridge_regression must return (w, loss)"


def test_ridge_regression_lambda0_public_values(y, tx):
    w, loss = impl.ridge_regression(y, tx, 0.0)

    np.testing.assert_allclose(w, [0.218786, -0.053837], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(loss, 0.026942, rtol=RTOL, atol=ATOL)
    assert np.ndim(loss) == 0


def test_ridge_regression_lambda1_public_values(y, tx):
    w, loss = impl.ridge_regression(y, tx, 1.0)

    np.testing.assert_allclose(w, [0.054303, 0.042713], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(loss, 0.03175, rtol=RTOL, atol=ATOL)
    assert np.ndim(loss) == 0
    assert w.shape == (tx.shape[1],)


def test_ridge_regression_lambda0_equals_least_squares(random_data):
    y, tx, _ = random_data
    w_ridge, loss_ridge = impl.ridge_regression(y, tx, 0.0)
    w_ls, loss_ls = impl.least_squares(y, tx)

    np.testing.assert_allclose(w_ridge, w_ls, rtol=1e-10)
    np.testing.assert_allclose(loss_ridge, loss_ls, rtol=1e-10)


def test_ridge_regression_loss_excludes_penalty(random_data):
    y, tx, _ = random_data
    w, loss = impl.ridge_regression(y, tx, 0.5)

    np.testing.assert_allclose(loss, mse(y, tx, w), rtol=1e-10)


def test_ridge_regression_satisfies_normal_equations(random_data):
    y, tx, _ = random_data
    N = tx.shape[0]
    lambda_ = 0.1
    w, _ = impl.ridge_regression(y, tx, lambda_)

    # Gradient of 1/(2N)||y - Xw||^2 + lambda ||w||^2 must vanish.
    grad = -tx.T @ (y - tx @ w) / N + 2 * lambda_ * w
    np.testing.assert_allclose(grad, 0.0, atol=1e-10)


def test_ridge_regression_shrinks_weights(random_data):
    y, tx, _ = random_data
    lambdas = [0.0, 0.01, 0.1, 1.0, 10.0]
    norms = [np.linalg.norm(impl.ridge_regression(y, tx, lam)[0]) for lam in lambdas]

    assert all(a > b for a, b in zip(norms, norms[1:]))


def test_ridge_regression_training_loss_increases_with_lambda(random_data):
    y, tx, _ = random_data
    lambdas = [0.0, 0.01, 0.1, 1.0, 10.0]
    losses = [impl.ridge_regression(y, tx, lam)[1] for lam in lambdas]

    assert all(a <= b for a, b in zip(losses, losses[1:]))


def test_ridge_regression_large_lambda_drives_weights_to_zero(random_data):
    y, tx, _ = random_data
    w, _ = impl.ridge_regression(y, tx, 1e8)

    np.testing.assert_allclose(w, 0.0, atol=1e-6)


def test_ridge_regression_handles_singular_gram_matrix():
    # Duplicate column -> X^T X is singular, but ridge with lambda > 0 is solvable.
    rng = np.random.default_rng(2)
    col = rng.normal(size=50)
    tx = np.c_[col, col]
    y = 3 * col
    w, loss = impl.ridge_regression(y, tx, 1e-3)

    assert np.all(np.isfinite(w))
    np.testing.assert_allclose(w[0], w[1], rtol=1e-10)
    assert np.isfinite(loss)
