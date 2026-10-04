"""Independently runnable pipeline stages. Each writes a _SUCCESS manifest."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.config import AppConfig, load_config
from src.data.load import load_raw
from src.data.profile import write_data_profile
from src.data.validate import run_validation_gate
from src.features.store import write_feature_store
from src.logging_conf import get_logger
from src.pipeline.cache import (
    cache_key,
    code_hash,
    config_hash,
    git_sha,
    is_cache_hit,
    remove_success,
    sha256_parquet_excluding,
    sha256_path,
    utc_now,
    write_manifest,
)
from src.targets.build import spotcheck_originals, write_prominence_sanity, write_target_definitions

log = get_logger(__name__)


def _stage_dir(cfg: AppConfig, name: str) -> Path:
    return cfg.path("artifacts_dir") / "stages" / name


def _code_files(cfg: AppConfig) -> list[Path]:
    src = cfg.root / "src"
    return sorted(p for p in src.rglob("*.py") if p.is_file())


def run_stage(
    name: str,
    cfg: AppConfig,
    inputs: dict[str, Path],
    outputs: dict[str, Path],
    fn: Callable[[], dict[str, Any]],
    force: bool = False,
    input_hashes: dict[str, str] | None = None,
) -> dict[str, Any]:
    stage_dir = _stage_dir(cfg, name)
    stage_dir.mkdir(parents=True, exist_ok=True)
    hashed = {k: sha256_path(v) for k, v in inputs.items()}
    if input_hashes:
        hashed.update(input_hashes)
    input_hashes = hashed
    key = cache_key(input_hashes, config_hash(cfg), code_hash(_code_files(cfg)))
    if not force and is_cache_hit(stage_dir, key):
        log.info("stage_skipped_cache_hit", stage=name)
        return {"stage": name, "skipped": True, "cache_key": key}

    remove_success(stage_dir)
    log.info("stage_start", stage=name)
    started = time.perf_counter()
    try:
        stats = fn()
    except Exception:
        remove_success(stage_dir)
        log.exception("stage_failed", stage=name)
        raise
    duration = time.perf_counter() - started
    manifest = {
        "stage": name,
        "cache_key": key,
        "input_hashes": input_hashes,
        "outputs": {k: str(v) for k, v in outputs.items()},
        "git_sha": git_sha(cfg.root),
        "config_hash": config_hash(cfg),
        "finished_at": utc_now(),
        "duration_seconds": round(duration, 3),
        "stats": stats,
        "skipped": False,
    }
    write_manifest(stage_dir, manifest)
    log.info("stage_end", stage=name, duration=round(duration, 3), stats=stats)
    return manifest


def stage_ingest(cfg: AppConfig | None = None, force: bool = False) -> dict[str, Any]:
    cfg = cfg or load_config()
    raw_path = cfg.path("raw_csv")
    interim = cfg.path("interim_dir")
    reports = cfg.path("reports_dir")

    def _fn() -> dict[str, Any]:
        raw = load_raw(raw_path, cfg)
        profile = write_data_profile(raw, cfg, reports / "data_profile.md")
        run_validation_gate(raw, cfg, reports / "validation_report.html")
        interim.mkdir(parents=True, exist_ok=True)
        raw.to_csv(interim / "raw_loaded.csv", index=False)
        n_rows = int(str(profile["n_rows"]))
        return {"rows_in": n_rows, "rows_out": n_rows, **profile}

    return run_stage(
        "ingest",
        cfg,
        inputs={"raw": raw_path, "base": cfg.root / "config" / "base.yaml"},
        outputs={"raw_loaded": interim / "raw_loaded.csv", "profile": reports / "data_profile.md"},
        fn=_fn,
        force=force,
    )


def stage_features(cfg: AppConfig | None = None, force: bool = False) -> dict[str, Any]:
    cfg = cfg or load_config()
    raw_path = cfg.path("raw_csv")
    feat_dir = cfg.path("features_dir")
    reports = cfg.path("reports_dir")

    def _fn() -> dict[str, Any]:
        raw = load_raw(raw_path, cfg)
        store, registry = write_feature_store(raw, cfg, feat_dir)
        spot = spotcheck_originals(store)
        write_target_definitions(store, cfg, reports / "target_definitions.md", spot)
        write_prominence_sanity(store, reports / "prominence_sanity.md")
        (reports / "originals_spotcheck.md").write_text(
            "# is_likely_original spot-check\n\n"
            f"Found {spot['n_found']} of {spot['n_looked_up']} hand-listed titles. "
            f"Agreed with the hand label on {spot['n_correct']}. "
            f"Hit rate: {spot['hit_rate']}.\n\n"
            "This is a heuristic quality number, not ground-truth accuracy.\n\n"
            + json.dumps(spot["rows"], indent=2, default=str)
            + "\n",
            encoding="utf-8",
        )
        return {"rows_in": len(raw), "rows_out": len(store), "schema_hash": registry["schema_hash"]}

    return run_stage(
        "features",
        cfg,
        inputs={"raw": raw_path, "features_cfg": cfg.root / "config" / "features.yaml"},
        outputs={
            "store": feat_dir / "features.parquet",
            "registry": feat_dir / "feature_registry.json",
        },
        fn=_fn,
        force=force,
    )


def _cluster_labels_missing(feat_dir: Path) -> bool:
    parquet = feat_dir / "features.parquet"
    if not parquet.exists():
        return True
    try:
        import pyarrow.parquet as pq

        table = pq.read_table(parquet, columns=["cluster_id"])
    except Exception:
        return True
    if table.num_rows == 0 or "cluster_id" not in table.column_names:
        return True
    return int(table.column("cluster_id").null_count) == table.num_rows


def stage_train(cfg: AppConfig | None = None, force: bool = False) -> dict[str, Any]:
    cfg = cfg or load_config()
    from src.models.train_all import train_all_heads

    feat_dir = cfg.path("features_dir")
    # A features rewrite restores the pre-cluster store. Training must run again
    # or the catalogue loses its segment labels while the train stage looks cached.
    if _cluster_labels_missing(feat_dir):
        force = True

    def _fn() -> dict[str, Any]:
        return train_all_heads(cfg)

    store = feat_dir / "features.parquet"
    # cluster_id and portfolio_fit are written by this stage into the feature file.
    # Hashing them would miss the cache on every later run.
    store_hash = sha256_parquet_excluding(store, ("cluster_id", "portfolio_fit"))
    return run_stage(
        "train",
        cfg,
        inputs={
            "store": store,
            "models": cfg.root / "config" / "models.yaml",
        },
        outputs={"mlruns": cfg.path("mlruns_dir")},
        fn=_fn,
        force=force,
        input_hashes={"store": store_hash},
    )


def stage_evaluate(cfg: AppConfig | None = None, force: bool = False) -> dict[str, Any]:
    cfg = cfg or load_config()
    from src.models.evaluate import write_leaderboard

    def _fn() -> dict[str, Any]:
        return write_leaderboard(cfg)

    return run_stage(
        "evaluate",
        cfg,
        inputs={"mlruns": cfg.path("mlruns_dir")},
        outputs={"leaderboard": cfg.path("reports_dir") / "model_leaderboard.md"},
        fn=_fn,
        force=force,
    )


def stage_insights(cfg: AppConfig | None = None, force: bool = False) -> dict[str, Any]:
    cfg = cfg or load_config()
    from src.insights.engine import run_insight_engine

    feat_dir = cfg.path("features_dir")

    def _fn() -> dict[str, Any]:
        return run_insight_engine(cfg)

    return run_stage(
        "insights",
        cfg,
        inputs={
            "store": feat_dir / "features.parquet",
            "insights": cfg.root / "config" / "insights.yaml",
        },
        outputs={"insights": cfg.path("reports_dir") / "insights.md"},
        fn=_fn,
        force=force,
    )


def stage_promote(cfg: AppConfig | None = None, force: bool = False) -> dict[str, Any]:
    cfg = cfg or load_config()
    from src.tracking.promote import promote_all

    def _fn() -> dict[str, Any]:
        return promote_all(cfg)

    return run_stage(
        "promote",
        cfg,
        inputs={"mlruns": cfg.path("mlruns_dir")},
        outputs={"registry": cfg.path("registry_dir")},
        fn=_fn,
        force=force,
    )


def stage_drift(cfg: AppConfig | None = None, force: bool = False) -> dict[str, Any]:
    cfg = cfg or load_config()
    from src.monitoring.drift import run_drift

    feat_dir = cfg.path("features_dir")

    def _fn() -> dict[str, Any]:
        return run_drift(cfg)

    return run_stage(
        "drift",
        cfg,
        inputs={"store": feat_dir / "features.parquet"},
        outputs={"drift": cfg.path("reports_dir") / "drift_report.html"},
        fn=_fn,
        force=force,
    )


STAGE_ORDER = ["ingest", "features", "train", "evaluate", "insights", "promote", "drift"]
STAGE_FNS: dict[str, Callable[..., dict[str, Any]]] = {
    "ingest": stage_ingest,
    "features": stage_features,
    "train": stage_train,
    "evaluate": stage_evaluate,
    "insights": stage_insights,
    "promote": stage_promote,
    "drift": stage_drift,
}
