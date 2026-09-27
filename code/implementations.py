"""Core machine-learning implementations for Project 1."""

import numpy as np


def least_squares(y, tx):
    "Implementation of Least squares regression using normal equations."
    A = tx.T @ tx 
    b = tx.T @ y
    w = np.linalg.solve(A , b)
    e = y - tx @ w
    mse = (1 / (2 * len(y))) * (e @ e) #### reutiliser apres une fonction deja implemente??????
    return w, mse

    # ou juste utiliser une solution un peu plus clean ? computatielement plus rapide....

def ridge_regression(y, tx, lambda_):
    "Implementation of Ridge regression using normal equations."
    N, D = tx.shape
    A = tx.T @ tx + 2 * N * lambda_ * np.eye(D)
    b = tx.T @ y
    w = np.linalg.solve(A, b)
    e = y - tx @ w
    mse = (1 / (2 * N)) * (e @ e)
    return w, mse
