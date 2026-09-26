"""Core machine-learning implementations for Project 1."""

import numpy as np


def compute_loss(y, tx, w):
    raise NotImplementedError


def compute_gradient(y, tx, w):
    raise NotImplementedError


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    raise NotImplementedError

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