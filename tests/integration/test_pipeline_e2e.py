from __future__ import annotations

from src.pipeline.cache import read_manifest
from src.pipeline.stages import (
    _stage_dir,
    stage_features,
    stage_ingest,
    stage_insights,
)


def test_ingest_features_insights_artifacts(cfg):
    stage_ingest(cfg, force=True)
    stage_features(cfg, force=True)
    stage_insights(cfg, force=True)
    assert (cfg.path("reports_dir") / "data_profile.md").exists()
    assert (cfg.path("reports_dir") / "validation_report.html").exists()
    assert (cfg.path("reports_dir") / "target_definitions.md").exists()
    assert (cfg.path("reports_dir") / "insights.md").exists()
    assert (cfg.path("features_dir") / "feature_registry.json").exists()
    for name in ("ingest", "features", "insights"):
        man = read_manifest(_stage_dir(cfg, name))
        assert man is not None
        assert "cache_key" in man
        assert "git_sha" in man


def test_cache_then_from_stage(cfg):
    stage_ingest(cfg, force=True)
    stage_features(cfg, force=True)
    skipped = stage_ingest(cfg, force=False)
    assert skipped["skipped"] is True
    rerun = stage_features(cfg, force=True)
    assert rerun["skipped"] is False
