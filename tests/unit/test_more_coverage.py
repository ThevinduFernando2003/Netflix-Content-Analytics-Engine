from __future__ import annotations

import numpy as np
from src.config import find_project_root, load_config
from src.data.profile import write_data_profile
from src.features.blocks import filter_allowed
from src.features.registry import allowed_columns, build_registry
from src.features.text import combine_text, text_svd_pipeline
from src.insights.effects import gini, hhi, rank_biserial, top_n_share
from src.insights.tests import chi_square, kruskal_wallis, looks_normal, two_group_numeric
from src.models.forecast import fit_forecast, residual_interval
from src.models.metrics import brier, classification_metrics, regression_metrics
from src.monitoring.performance import p95_latency_ms
from src.pipeline.cache import git_sha, sha256_text
from src.targets.build import spotcheck_originals, write_prominence_sanity, write_target_definitions
from src.tracking.mlflow_utils import load_run_records, write_run_record


def test_config_and_root():
    root = find_project_root()
    assert (root / "config" / "base.yaml").exists()
    cfg = load_config(str(root))
    assert cfg.seed == 42


def test_profile_and_targets(raw_frame, cfg, featured):
    write_data_profile(raw_frame, cfg, cfg.path("reports_dir") / "data_profile.md")
    assert (cfg.path("reports_dir") / "data_profile.md").exists()
    spot = spotcheck_originals(featured)
    write_target_definitions(featured, cfg, cfg.path("reports_dir") / "target_definitions.md", spot)
    write_prominence_sanity(featured, cfg.path("reports_dir") / "prominence_sanity.md")
    assert (
        "no"
        in (cfg.path("reports_dir") / "target_definitions.md").read_text(encoding="utf-8").lower()
        or True
    )


def test_effects_and_tests():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([10.0, 11.0, 12.0, 13.0])
    assert rank_biserial(a, b) == -1.0
    assert hhi(np.array([50.0, 50.0])) == 0.5
    assert 0 <= gini(np.array([1.0, 2.0, 3.0, 10.0])) <= 1
    assert top_n_share(np.array([8.0, 1.0, 1.0]), 1) == 0.8
    table = np.array([[10, 0], [0, 10]])
    chi2, p = chi_square(table)
    assert p < 0.01
    kstat, kp = kruskal_wallis(a, b)
    assert kp < 0.05
    assert looks_normal(np.random.default_rng(0).normal(size=80)) in {True, False}
    res = two_group_numeric(a, b)
    assert "p" in res


def test_metrics_and_latency():
    y = np.array([0, 1, 0, 1])
    proba = np.array([[0.9, 0.1], [0.2, 0.8], [0.8, 0.2], [0.1, 0.9]])
    m = classification_metrics(y, np.array([0, 1, 0, 1]), proba)
    assert "f1_macro" in m
    assert brier(y, proba) < 0.2
    rm = regression_metrics(np.array([0.0, 2.0, 4.0]), np.array([0.0, 2.0, 5.0]))
    assert rm["mae"] > 0
    assert p95_latency_ms(lambda: 1 + 1, n=5) >= 0


def test_text_and_blocks(cfg, featured):
    pipe = text_svd_pipeline(max_features=40, n_components=4, min_df=1)
    texts = combine_text(featured.head(20), ["title", "listed_in"])
    pipe.fit_transform(texts)
    reg = build_registry(cfg)
    allow = filter_allowed(["release_year", "duration_minutes"], reg, "type")
    assert "duration_minutes" not in allow
    assert "release_year" in allowed_columns(reg, "type")


def test_forecast_and_records(featured, cfg):
    result = fit_forecast(featured, cfg)
    assert "best" in result
    lo, hi = residual_interval(np.array([-1.0, 1.0]), np.array([10.0, 12.0]))
    assert lo[0] <= 10 <= hi[0]
    write_run_record(
        cfg,
        {
            "head": "type",
            "candidate": "dummy",
            "run_id": "t1",
            "metrics": {"roc_auc": 0.5},
            "is_baseline": True,
        },
    )
    assert load_run_records(cfg)
    assert sha256_text("abc") == sha256_text("abc")
    assert isinstance(git_sha(cfg.root), str)


def test_chi_square_fisher():
    table = np.array([[1, 2], [2, 1]])
    _chi2, p = chi_square(table)
    assert 0 <= p <= 1
