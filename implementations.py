"""Core machine-learning implementations for Project 1."""

import numpy as np


def compute_loss(y, tx, w):
    """Return the mean squared error loss for predictions ``tx @ w``."""
    residual = y - tx @ w
    return 0.5 * np.mean(residual**2)


def compute_gradient(y, tx, w):
    """Return the gradient of the mean squared error loss."""
    residual = y - tx @ w
    return -(tx.T @ residual) / y.shape[0]


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Optimize mean squared error with full-batch gradient descent."""
    w = initial_w.copy()
    for _ in range(max_iters):
        w -= gamma * compute_gradient(y, tx, w)
    return w, compute_loss(y, tx, w)


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Optimize mean squared error with stochastic gradient descent."""
    w = initial_w.copy()
    for _ in range(max_iters):
        index = np.random.randint(y.shape[0])
        sample_y = y[index : index + 1]
        sample_tx = tx[index : index + 1]
        w -= gamma * compute_gradient(sample_y, sample_tx, w)
    return w, compute_loss(y, tx, w)


def least_squares(y, tx):
    """Compute the least-squares solution and its mean squared error loss."""
    w = np.linalg.lstsq(tx, y, rcond=None)[0]
    return w, compute_loss(y, tx, w)


def ridge_regression(y, tx, lambda_):
    """Compute the ridge-regression solution and its mean squared error loss."""
    n_samples, n_features = tx.shape
    regularized_matrix = tx.T @ tx + 2 * n_samples * lambda_ * np.eye(n_features)
    w = np.linalg.solve(regularized_matrix, tx.T @ y)
    return w, compute_loss(y, tx, w)


def sigmoid(t):
    """Return the elementwise logistic sigmoid of ``t``."""
    return np.where(t >= 0, 1 / (1 + np.exp(-t)), np.exp(t) / (1 + np.exp(t)))


def compute_logistic_loss(y, tx, w):
    """Return the average logistic loss for binary labels in ``y``."""
    scores = tx @ w
    return np.mean(np.logaddexp(0, scores) - y * scores)


def compute_logistic_gradient(y, tx, w):
    """Return the gradient of the average logistic loss."""
    return tx.T @ (sigmoid(tx @ w) - y) / y.shape[0]


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Optimize logistic loss with gradient descent."""
    w = initial_w.copy()
    for _ in range(max_iters):
        w -= gamma * compute_logistic_gradient(y, tx, w)
    return w, compute_logistic_loss(y, tx, w)


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Optimize L2-regularized logistic loss with gradient descent."""
    w = initial_w.copy()
    for _ in range(max_iters):
        gradient = compute_logistic_gradient(y, tx, w) + 2 * lambda_ * w
        w -= gamma * gradient
    return w, compute_logistic_loss(y, tx, w)
