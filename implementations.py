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
    """Linear regression using stochastic gradient descent

    Inputs :

    - y ndarray (N,) : expected output

    - tx ndarray (N, D) : features matrix

    - initial_w  ndarray (D,) : initial weights

    - max_iters int : maximum of iterations

    - gamma float : stepsize

    Returns :

    - w ndarray (D,) : learned weights

    - loss float : MSE loss with stochastic gradient descent

    """

    w = initial_w.copy()

    for _ in range(max_iters):
        idx = np.random.randint(y.shape[0])
        y_sample = y[[idx]]
        tx_sample = tx[[idx]]

        sto_grad = compute_gradient(y_sample, tx_sample, w)
        w = w - gamma * sto_grad

    loss = compute_loss(y, tx, w)

    return w, loss


def least_squares(y, tx):
    """Least-squares regression using the normal equations."""

    A = tx.T @ tx
    b = tx.T @ y
    w = np.linalg.solve(A, b)
    return w, compute_loss(y, tx, w)


def ridge_regression(y, tx, lambda_):
    """Ridge regression using the regularized normal equations."""

    N, D = tx.shape
    A = tx.T @ tx + 2 * N * lambda_ * np.eye(D)
    b = tx.T @ y
    w = np.linalg.solve(A, b)
    return w, compute_loss(y, tx, w)


def sigmoid(t):
    # Avoid overflow in exp for large negative values
    t = np.clip(t, -500, 500)

    return 1 / (1 + np.exp(-t))


def compute_logistic_loss(y, tx, w):
    """Logistic loss implementation.

    Inputs :

    - y ndarray (N, ) : labels

    - tx ndarray (N, D) : data matrix

    - w ndarray (D, ) : weights

    Returns :

    - loss float : cross-entropy loss for logistic regression"""

    N = y.shape[0]

    # pred = tx @ w is the linear prediction.
    pred = tx @ w

    # Starting from the cross-entropy loss and replacing the sigmoid then rearranging,
    # we obtain the equivalent expression:
    # log(1 + exp(pred)) - y * pred.
    # np.logaddexp(0, pred) computes log(exp(0) + exp(pred)) = log(1 + exp(pred)).
    loss = np.sum(np.logaddexp(0, pred) - y * pred) / N

    return loss


def compute_logistic_gradient(y, tx, w):
    """Gradient of the logistic loss implementation.

    Inputs :
    - y ndarray (N, ) : labels
    - tx ndarray (N, D) : data matrix
    - w ndarray (D, ) : weights

    Returns :
    - gradient ndarray (D, ) : gradient of the cross-entropy loss
    """

    N = y.shape[0]

    # pred = tx @ w is the linear prediction.
    pred = tx @ w

    # sigmoid(pred) converts the linear prediction into probabilities
    # between 0 and 1.
    prob = sigmoid(pred)

    # The gradient of the cross-entropy loss is
    # X^T (sigmoid(pred) - y), averaged over all samples, using the chain rule.
    gradient = tx.T @ (prob - y) / N

    return gradient


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent.

    Inputs :

        - y ndarray (N,) : expected output

        - tx ndarray (N, D) : features matrix

        - initial_w  ndarray (D,) : initial weights

        - max_iters int : maximum of iterations

        - gamma float : stepsize

        Returns :

        - w ndarray (D,) : weights

        - loss float : loss of logistic regression

    """

    w = initial_w.copy()

    for _ in range(max_iters):
        grad = compute_logistic_gradient(y, tx, w)
        w = w - gamma * grad

    loss = compute_logistic_loss(y, tx, w)

    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent.

    Inputs:
    - y ndarray (N,): labels
    - tx ndarray (N, D): data matrix
    - lambda_ float: regularization strength
    - initial_w ndarray (D,): initial weights
    - max_iters int: number of iterations
    - gamma float: learning rate

    Returns:
    - w ndarray (D,): final weights
    - loss float: logistic loss without the regularization penalty
    """

    w = initial_w.copy()

    for _ in range(max_iters):
        grad = compute_logistic_gradient(y, tx, w)
        # The penalty lambda * ||w||^2 has gradient 2 * lambda * w.
        grad += 2 * lambda_ * w
        w -= gamma * grad

    # The returned loss must NOT include the regularization penalty.
    loss = compute_logistic_loss(y, tx, w)

    return w, loss
