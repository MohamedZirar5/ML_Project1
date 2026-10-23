"""Tests for feature_engineering.py (no dataset needed).

Run from the repository root with:
    pytest code/scripts/test_feature_engineering.py -v
"""

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import feature_engineering as fe  # noqa: E402
import implementations as impl  # noqa: E402


def test_newton_reaches_the_gd_optimum():
    rng = np.random.default_rng(0)
    N, D = 1000, 5
    tx = np.c_[np.ones(N), rng.normal(size=(N, D - 1))]
    y = (rng.random(N) < impl.sigmoid(tx @ rng.normal(size=D))).astype(float)
    lambda_ = 1e-2

    w_newton, n_iters = fe.reg_logistic_newton(y, tx, lambda_, chunk_rows=128)
    w_gd, _ = impl.reg_logistic_regression(y, tx, lambda_, np.zeros(D), 5000, 0.5)

    np.testing.assert_allclose(w_newton, w_gd, atol=1e-5)
    assert n_iters < 15


def test_selected_columns_groups():
    names = np.array(["GENHLTH", "SEQNO", "BPMEDS", "HAREHAB1", "VIDFCLT2", "ASTHMA3"])
    assert list(fe.selected_columns(names)) == [True, False, False, False, False, True]
    assert list(fe.selected_columns(names, ("cvd_care", "optional_modules"))) == [
        True, False, False, True, True, True,
    ]
    with pytest.raises(ValueError):
        fe.selected_columns(names, ("administrative",))


def test_build_features_small():
    names = np.array(["GENHLTH", "_AGE80", "BPMEDS"])
    x_train = np.array([[1, 30, 1], [2, 50, 2], [9, 70, 1], [2, np.nan, 2]] * 10, dtype=float)
    x_train[:, 1] += np.arange(40)  # > 15 distinct ages, so _AGE80 is numeric
    raw = {"x_train": x_train, "x_test": x_train[:3], "names": names}

    x_tr, x_te, out = fe.build_features(raw, min_level_count=1)
    # GENHLTH: 9 -> missing level (-1), one-hot drops the first level (-1).
    assert list(out) == ["_AGE80", "_AGE80_missing", "GENHLTH=1", "GENHLTH=2"]
    assert x_tr.shape == (40, 4) and x_te.shape == (3, 4)
    assert not np.isnan(x_tr).any()

    _, _, out = fe.build_features(raw, keep_groups=("risk_management",),
                                  interact_with=("_AGE80",), min_level_count=1)
    # Interactions only with columns from the base (notebook 01) set.
    assert "_AGE80*GENHLTH=2" in out
    assert not any(name.startswith("_AGE80*BPMEDS") for name in out)


def test_independent_columns_drops_exact_combinations():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(2, 200))
    onehot = (rng.random(200) < 0.3).astype(float)
    # columns: bias, a, b, a + 2b (dependent), onehot, 1 - onehot (dependent with bias)
    tx = np.c_[np.ones(200), a, b, a + 2 * b, onehot, 1 - onehot]
    keep = fe.independent_columns(tx)
    assert list(keep) == [0, 1, 2, 4]
    assert np.linalg.matrix_rank(tx[:, keep]) == len(keep) == np.linalg.matrix_rank(tx)
