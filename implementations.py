"""Core machine-learning implementations for Project 1."""

import numpy as np


def compute_loss(y, tx, w):
    """MSE implementation

    Inputs :

    - y ndarray (N, ) : labels

    - tx ndarray (N, D) : data matrix

    - w ndarray (D, ) : weights

    Returns :

    - loss float : MSE loss"""

    N = y.shape[0]

    # tx @ w is the prediction vector of shape (N,)
    loss = np.sum((y - tx @ w) ** 2) / (2 * N)

    return loss


def compute_gradient(y, tx, w):
    """Gradient of the MSE.

    Inputs :

    - y ndarray (N, ) : labels

    - tx ndarray (N, D) : data matrix

    - w ndarray (D, ) : weights

    Returns :

    - grad ndarray (D, ) : gradient of the loss function"""

    N = y.shape[0]

    # y - tx @ w is the error "e", and its derivative was computed during the course.
    # We obtain the following gradient:
    grad = -(tx.T @ (y - tx @ w)) / N

    return grad


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent

    Inputs :

    - y ndarray (N, ) : labels

    - tx ndarray (N, D) : data matrix

    - initial_w ndarray (D, ) : initial weights

    - max_iters int : number of iterations

    - gamma float : learning rate

    Returns :

    - w ndarray (D, ) : final weights

    - loss float : final MSE loss"""

    w = initial_w.copy()

    for _ in range(max_iters):
        grad = compute_gradient(y, tx, w)
        w = w - gamma * grad

    loss = compute_loss(y, tx, w)

    return w, loss

def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    raise NotImplementedError


def least_squares(y, tx):
    raise NotImplementedError


def ridge_regression(y, tx, lambda_):
    raise NotImplementedError


def sigmoid(t):
    return 1 / (1 + np.exp(-t))


def compute_logistic_loss(y, tx, w):
    raise NotImplementedError


def compute_logistic_gradient(y, tx, w):
    raise NotImplementedError


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    raise NotImplementedError

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    raise NotImplementedError