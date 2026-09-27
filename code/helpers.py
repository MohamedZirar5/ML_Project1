# -*- coding: utf-8 -*-
"""some helper functions."""

import os
import numpy as np

# dataset/ lives next to code/, so resolve it relative to this file
# (works no matter which directory you run python from)
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dataset")


def load_data(data_path=DATA_DIR, sub_sample=False):
    """Load the BRFSS training and test data.

    Args:
        data_path: folder containing x_train.csv, y_train.csv and x_test.csv.
        sub_sample: if True, keep only every 50th training row (fast for debugging).

    Returns:
        x_train:   (N_train, D) feature matrix, NaN where the value is missing.
        x_test:    (N_test, D) feature matrix.
        y_train:   (N_train,) labels in {-1, 1}.
        train_ids: (N_train,) Id column of x_train.
        test_ids:  (N_test,) Id column of x_test.
    """
    y_train = np.genfromtxt(os.path.join(data_path, "y_train.csv"),delimiter=",",skip_header=1,dtype=int,usecols=1,)
    x_train = np.genfromtxt(os.path.join(data_path, "x_train.csv"), delimiter=",", skip_header=1)
    x_test = np.genfromtxt(os.path.join(data_path, "x_test.csv"), delimiter=",", skip_header=1)

    # first column is the Id, the rest are the features
    train_ids = x_train[:, 0].astype(int)
    test_ids = x_test[:, 0].astype(int)
    x_train = x_train[:, 1:]
    x_test = x_test[:, 1:]

    if sub_sample:
        y_train = y_train[::50]
        x_train = x_train[::50]
        train_ids = train_ids[::50]

    return x_train, x_test, y_train, train_ids, test_ids


def load_feature_names(data_path=DATA_DIR):
    """Return the list of feature names (header of x_train.csv, without Id)."""
    with open(os.path.join(data_path, "x_train.csv")) as f:
        header = f.readline().strip().split(",")
    return header[1:]


def create_csv_submission(ids, y_pred, name):
    """Write predictions in the AIcrowd submission format.

    Args:
        ids: (N,) test sample ids.
        y_pred: (N,) predictions in {-1, 1}.
        name: output file name, e.g. "submission.csv".
    """
    if not all(i in [-1, 1] for i in np.unique(y_pred)):
        raise ValueError("y_pred can only contain values -1, 1")
    with open(name, "w") as f:
        f.write("Id,Prediction\n")
        for i, p in zip(ids, y_pred):
            f.write(f"{int(i)},{int(p)}\n")
