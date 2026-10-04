from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from src.insights.effects import cliffs_delta, cohens_d, cramers_v
from src.insights.engine import bh_fdr, generate_candidates, run_insight_engine
from src.insights.narrative import assert_no_causal_language, causal_hits
from src.insights.templates import render
from src.insights.tests import mann_whitney, trend_regression, welch_t
from statsmodels.stats.multitest import multipletests


def test_mann_whitney_matches_scipy():
    rng = np.random.default_rng(42)
    a = rng.normal(0, 1, 40)
    b = rng.normal(0.2, 1, 40)
    stat, p = mann_whitney(a, b)
    ref = stats.mannwhitneyu(a, b, alternative="two-sided")
    assert abs(stat - ref.statistic) < 1e-8
    assert abs(p - ref.pvalue) < 1e-8


def test_equal_means_and_ties():
    a = np.ones(20)
    stat, p = mann_whitney(a, a.copy())
    assert p == 1.0
    assert stat == 0.0


def test_welch_matches_scipy():
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    b = np.array([2.0, 3.0, 4.0, 5.0, 7.0])
    stat, p = welch_t(a, b)
    ref = stats.ttest_ind(a, b, equal_var=False)
    assert abs(stat - ref.statistic) < 1e-8
    assert abs(p - ref.pvalue) < 1e-8


def test_cohens_d_identical_is_zero():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert cohens_d(x, x.copy()) == 0.0


def test_cliffs_delta_hand():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    # every pair a < b → delta = -1
    assert cliffs_delta(a, b) == -1.0


def test_cramers_v_hand():
    # Perfect association 2x2, n=100, chi2 = 100, min(r-1,c-1)=1 → V=1
    assert abs(cramers_v(100, 100, 2, 2) - 1.0) < 1e-9


def test_bh_fdr_matches_statsmodels():
    p = [0.001, 0.01, 0.04, 0.2, 0.8]
    rej, adj = bh_fdr(p, 0.05)
    ref_rej, ref_adj, *_ = multipletests(p, alpha=0.05, method="fdr_bh")
    assert np.allclose(adj, ref_adj)
    assert np.array_equal(rej, ref_rej)
    assert np.all(adj >= np.array(p) - 1e-12)


def test_effect_floor_discards_tiny_significant(cfg):
    # A 0.15 SD shift is significant at this n, but Cliff's delta stays under the 0.20 floor.
    rng = np.random.default_rng(0)
    n = 4000
    years = np.concatenate([rng.normal(0.0, 1.0, n), rng.normal(0.15, 1.0, n)])
    frame = pd.DataFrame(
        {
            "type": ["Movie"] * n + ["TV Show"] * n,
            "years_to_platform": years,
            "catalogue_prominence": np.ones(2 * n),
            "n_genres": np.ones(2 * n),
            "n_countries": np.ones(2 * n),
            "is_likely_original": np.zeros(2 * n),
            "is_movie": [1] * n + [0] * n,
            "year_added": [2018] * n + [2019] * n,
            "maturity_level": ["Teens"] * (2 * n),
            "primary_country": ["X"] * (2 * n),
            "primary_genre": ["Y"] * (2 * n),
            "acquisition_bucket": ["Recent"] * (2 * n),
            "title_length": np.ones(2 * n),
            "description_length": np.zeros(2 * n),
        }
    )
    payload = run_insight_engine(cfg, frame)
    segs = [
        f
        for f in payload["findings"]
        if f["kind"] == "segment_difference" and f["metric"] == "years_to_platform"
    ]
    assert segs == []


def test_min_support_discards(cfg):
    frame = pd.DataFrame(
        {
            "type": ["Movie"] * 5 + ["TV Show"] * 5,
            "years_to_platform": np.arange(10, dtype=float),
            "catalogue_prominence": np.arange(10, dtype=float),
            "n_genres": np.ones(10),
            "n_countries": np.ones(10),
            "is_likely_original": np.zeros(10),
            "is_movie": [1] * 5 + [0] * 5,
            "year_added": list(range(2010, 2020)),
            "maturity_level": ["Teens"] * 10,
            "primary_country": ["X"] * 10,
            "primary_genre": ["Y"] * 10,
            "acquisition_bucket": ["Recent"] * 10,
            "title_length": np.ones(10),
            "description_length": np.zeros(10),
        }
    )
    cands = generate_candidates(frame, cfg)
    segs = [c for c in cands if c["kind"] == "segment_difference"]
    assert segs == []


def test_dedup_nested(cfg, featured):
    payload = run_insight_engine(cfg, featured)
    keys = [(f["kind"], f["metric"], f["segment"]) for f in payload["findings"]]
    assert len(keys) == len(set(keys))


def test_noise_yields_near_zero_findings(cfg):
    rng = np.random.default_rng(42)
    n = 400
    frame = pd.DataFrame(
        {
            "type": rng.choice(["Movie", "TV Show"], n),
            "years_to_platform": rng.normal(4, 1, n),
            "catalogue_prominence": rng.normal(0.4, 0.05, n),
            "n_genres": rng.integers(1, 4, n),
            "n_countries": rng.integers(1, 3, n),
            "is_likely_original": rng.integers(0, 2, n),
            "is_movie": rng.integers(0, 2, n),
            "year_added": rng.integers(2015, 2021, n),
            "maturity_level": rng.choice(["Kids", "Older Kids", "Teens", "Mature"], n),
            "primary_country": rng.choice(["A", "B", "C", "D"], n),
            "primary_genre": rng.choice(["G1", "G2", "G3", "G4"], n),
            "acquisition_bucket": rng.choice(["Day-one", "Fast", "Recent", "Back-catalogue"], n),
            "title_length": rng.integers(5, 20, n),
            "description_length": rng.integers(0, 5, n),
        }
    )
    payload = run_insight_engine(cfg, frame)
    assert payload["n_survivors"] <= 3


def test_causal_language_fails():
    with pytest.raises(ValueError):
        assert_no_causal_language(["X causes Y (q=0.01, n=40)."], ["causes"])
    assert causal_hits("X is associated with Y", ["causes"]) == []


def test_template_numbers_match_record():
    values = {
        "metric": "years_to_platform",
        "segment": "the full catalogue",
        "direction": "decreased",
        "start_val": 6.0,
        "end_val": 3.0,
        "start_year": 2015,
        "end_year": 2021,
        "pct_change": -0.5,
        "r2": 0.42,
        "q_value": 0.012,
        "n": 100,
    }
    text = render("trend", values)
    assert "6.0" in text and "3.0" in text and "0.012" in text and "100" in text


def test_trend_regression_known():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = 2 * x + 1
    tr = trend_regression(x, y)
    assert abs(tr["slope"] - 2) < 1e-9
    assert abs(tr["r2"] - 1) < 1e-9
    assert tr["p"] < 1e-6
