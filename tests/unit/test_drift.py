from __future__ import annotations

import numpy as np
from src.monitoring.drift import psi


def test_psi_identical_near_zero():
    rng = np.random.default_rng(42)
    x = rng.normal(size=400)
    assert psi(x, x.copy()) < 0.05


def test_psi_shifted_exceeds_threshold():
    rng = np.random.default_rng(42)
    a = rng.normal(0, 1, 400)
    b = rng.normal(3, 1, 400)
    assert psi(a, b) > 0.2


def test_psi_hand_two_bins():
    expected = np.array([0.0] * 50 + [1.0] * 50)
    actual = np.array([0.0] * 50 + [1.0] * 50)
    assert psi(expected, actual, bins=2) < 0.05
