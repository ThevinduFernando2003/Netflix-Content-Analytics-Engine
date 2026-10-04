from __future__ import annotations

import numpy as np
from sklearn.metrics import adjusted_rand_score
from src.models.clustering import bootstrap_ari


def test_identical_labelings_ari_one():
    labels = np.array([0, 0, 1, 1, 2, 2])
    assert adjusted_rand_score(labels, labels) == 1.0


def test_bootstrap_ari_reproducible():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(80, 6))
    a = bootstrap_ari(X, k=3, n_resamples=8, frac=0.8, seed=42)
    b = bootstrap_ari(X, k=3, n_resamples=8, frac=0.8, seed=42)
    assert a == b
