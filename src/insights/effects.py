"""Effect sizes. A significant p-value on n=8790 is not a finding."""

from __future__ import annotations

import numpy as np


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return 0.0
    va = float(np.var(a, ddof=1))
    vb = float(np.var(b, ddof=1))
    pooled = np.sqrt(((len(a) - 1) * va + (len(b) - 1) * vb) / max(len(a) + len(b) - 2, 1))
    if pooled == 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled)


def cliffs_delta(a: np.ndarray, b: np.ndarray, max_n: int = 500) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return 0.0
    rng = np.random.default_rng(42)
    if len(a) > max_n:
        a = rng.choice(a, size=max_n, replace=False)
    if len(b) > max_n:
        b = rng.choice(b, size=max_n, replace=False)
    gt = int(np.sum(a[:, None] > b[None, :]))
    lt = int(np.sum(a[:, None] < b[None, :]))
    return float(gt - lt) / float(len(a) * len(b))


def rank_biserial(a: np.ndarray, b: np.ndarray) -> float:
    """Rank-biserial correlation; equals cliffs_delta for two independent samples."""
    return cliffs_delta(a, b)


def cramers_v(chi2: float, n: int, r: int, c: int) -> float:
    if n <= 0 or min(r - 1, c - 1) <= 0:
        return 0.0
    return float(np.sqrt(chi2 / (n * min(r - 1, c - 1))))


def hhi(shares: np.ndarray) -> float:
    s = np.asarray(shares, dtype=float)
    s = s[np.isfinite(s)]
    total = s.sum()
    if total <= 0:
        return 0.0
    p = s / total
    return float(np.sum(p**2))


def gini(values: np.ndarray) -> float:
    x = np.sort(np.asarray(values, dtype=float))
    x = x[np.isfinite(x)]
    if len(x) == 0 or x.sum() == 0:
        return 0.0
    n = len(x)
    idx = np.arange(1, n + 1)
    return float((2 * np.sum(idx * x) / (n * x.sum())) - (n + 1) / n)


def top_n_share(values: np.ndarray, n: int = 5) -> float:
    x = np.sort(np.asarray(values, dtype=float))[::-1]
    total = x.sum()
    if total <= 0:
        return 0.0
    return float(x[:n].sum() / total)
