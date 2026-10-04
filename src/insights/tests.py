"""The right test per comparison. Assumptions are checked and named in comments."""

from __future__ import annotations

import numpy as np
from scipy import stats


def _as_float(values: np.ndarray) -> np.ndarray:
    """Cast to float64. Boolean and pandas-nullable inputs are not valid scipy dtypes."""
    return np.asarray(values, dtype=float)


def welch_t(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Welch's t-test. Used only when both groups pass a normality screen; default path is Mann-Whitney."""
    a = _as_float(a)
    b = _as_float(b)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan"), 1.0
    res = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
    return float(res.statistic), float(res.pvalue)


def mann_whitney(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Mann-Whitney U. Default two-group test: catalogue metrics are skewed (checked via sample skewness)."""
    a = _as_float(a)
    b = _as_float(b)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return float("nan"), 1.0
    # Identical groups → p = 1. Tied-heavy groups are handled by scipy's asymptotic method.
    if np.array_equal(np.sort(a), np.sort(b)) and len(a) == len(b):
        return 0.0, 1.0
    res = stats.mannwhitneyu(a, b, alternative="two-sided")
    return float(res.statistic), float(res.pvalue)


def kruskal_wallis(*groups: np.ndarray) -> tuple[float, float]:
    clean = [g[~np.isnan(g)] for g in groups if len(g)]
    if len(clean) < 2:
        return float("nan"), 1.0
    res = stats.kruskal(*clean)
    return float(res.statistic), float(res.pvalue)


def chi_square(table: np.ndarray) -> tuple[float, float]:
    """χ² test of independence. Fisher's exact is used when any expected cell is < 5 and the table is 2x2.

    A zero row or column sum makes the expected table singular; that comparison is not a finding.
    """
    table = np.asarray(table, dtype=float)
    if table.ndim != 2 or table.size == 0 or table.sum() == 0:
        return 0.0, 1.0
    if (table.sum(axis=0) == 0).any() or (table.sum(axis=1) == 0).any():
        return 0.0, 1.0
    try:
        chi2, p, _dof, expected = stats.chi2_contingency(table, correction=False)
    except ValueError:
        return 0.0, 1.0
    if table.shape == (2, 2) and expected.min() < 5:
        _odds, p = stats.fisher_exact(table)
        return float(chi2), float(p)
    return float(chi2), float(p)


def trend_regression(years: np.ndarray, values: np.ndarray) -> dict[str, float]:
    """OLS of metric on a time index. Reports slope, 95% CI, R², and the F-test p-value."""
    x = np.asarray(years, dtype=float)
    y = np.asarray(values, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return {
            "slope": 0.0,
            "intercept": 0.0,
            "r2": 0.0,
            "p": 1.0,
            "ci_lo": 0.0,
            "ci_hi": 0.0,
            "n": float(len(x)),
        }
    res = stats.linregress(x, y)
    # 95% CI on slope from the standard error.
    tcrit = stats.t.ppf(0.975, df=max(len(x) - 2, 1))
    ci_lo = float(res.slope - tcrit * res.stderr)
    ci_hi = float(res.slope + tcrit * res.stderr)
    return {
        "slope": float(res.slope),
        "intercept": float(res.intercept),
        "r2": float(res.rvalue**2),
        "p": float(res.pvalue),
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "n": float(len(x)),
    }


def looks_normal(sample: np.ndarray, max_n: int = 200) -> bool:
    """Shapiro-Wilk on a cap of 200 observations. Skewed catalogue metrics typically fail this.

    Binary flags are not continuous; Welch's t-test is the wrong default for them.
    """
    x = np.asarray(sample, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 8 or len(np.unique(x)) <= 2:
        return False
    rng = np.random.default_rng(42)
    if len(x) > max_n:
        x = rng.choice(x, size=max_n, replace=False)
    stat = float(np.abs(stats.skew(x)))
    if stat > 1.0:
        return False
    p = stats.shapiro(x).pvalue
    return p > 0.05


def two_group_numeric(a: np.ndarray, b: np.ndarray) -> dict[str, float | str]:
    """Pick Mann-Whitney by default; Welch only if both groups look normal."""
    if looks_normal(a) and looks_normal(b):
        stat, p = welch_t(a, b)
        return {"test": "welch_t", "stat": stat, "p": p}
    stat, p = mann_whitney(a, b)
    return {"test": "mannwhitney", "stat": stat, "p": p}
