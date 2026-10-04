from __future__ import annotations

from src.pipeline.cache import cache_key
from src.pipeline.stages import stage_features, stage_ingest


def test_second_run_skips(cfg):
    a = stage_ingest(cfg, force=True)
    assert a.get("skipped") is False
    b = stage_ingest(cfg, force=False)
    assert b.get("skipped") is True


def test_from_stage_features_reruns(cfg):
    stage_ingest(cfg, force=True)
    first = stage_features(cfg, force=True)
    assert first.get("skipped") is False
    cached = stage_features(cfg, force=False)
    assert cached.get("skipped") is True
    rerun = stage_features(cfg, force=True)
    assert rerun.get("skipped") is False


def test_failed_stage_leaves_no_success(cfg, tmp_path):
    from src.pipeline.cache import success_path
    from src.pipeline.stages import _stage_dir, run_stage

    def boom() -> dict:
        raise RuntimeError("nope")

    try:
        run_stage("boom", cfg, inputs={}, outputs={"x": tmp_path / "x"}, fn=boom, force=True)
    except RuntimeError:
        pass
    assert not success_path(_stage_dir(cfg, "boom")).exists()


def test_parquet_hash_ignores_cluster_columns(tmp_path):
    import pandas as pd
    from src.pipeline.cache import sha256_parquet_excluding

    path = tmp_path / "features.parquet"
    frame = pd.DataFrame({"show_id": ["a", "b"], "n_genres": [1, 2], "cluster_id": [None, None]})
    frame.to_parquet(path, index=False)
    before = sha256_parquet_excluding(path, ("cluster_id", "portfolio_fit"))
    frame["cluster_id"] = [0, 1]
    frame["portfolio_fit"] = [0.1, 0.2]
    frame.to_parquet(path, index=False)
    after = sha256_parquet_excluding(path, ("cluster_id", "portfolio_fit"))
    assert before == after


def test_cache_key_stable():
    k1 = cache_key({"a": "1"}, "c", "d")
    k2 = cache_key({"a": "1"}, "c", "d")
    assert k1 == k2
    assert k1 != cache_key({"a": "2"}, "c", "d")
