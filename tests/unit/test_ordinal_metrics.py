from __future__ import annotations

import numpy as np
from src.models.metrics import (
    adjacent_accuracy,
    kids_as_mature_count,
    ordinal_mae,
    quadratic_weighted_kappa,
    severe_error_count,
)


def test_perfect():
    y = np.array([0, 1, 2, 3])
    assert quadratic_weighted_kappa(y, y) == 1.0
    assert ordinal_mae(y, y) == 0.0
    assert adjacent_accuracy(y, y) == 1.0
    assert severe_error_count(y, y) == 0
    assert kids_as_mature_count(y, y) == 0


def test_all_wrong_kids_as_mature():
    y = np.array([0, 0, 0, 0])
    p = np.array([3, 3, 3, 3])
    assert ordinal_mae(y, p) == 3.0
    assert adjacent_accuracy(y, p) == 0.0
    assert severe_error_count(y, p) == 4
    assert kids_as_mature_count(y, p) == 4
    # QWK is defined; for constant-true vs constant-pred it can be 0 or nan-like — just be finite or 0
    qwk = quadratic_weighted_kappa(y, p)
    assert qwk <= 0.0 or np.isnan(qwk)


def test_hand_ordinal_mae():
    y = np.array([0, 1, 2, 3])
    p = np.array([1, 1, 1, 1])
    assert ordinal_mae(y, p) == 1.0
    assert adjacent_accuracy(y, p) == 0.75
