"""Write and read the parquet feature store plus feature_registry.json."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import AppConfig
from src.features.engineer import engineer_features
from src.features.registry import build_registry, dump_registry, load_registry, schema_hash
from src.logging_conf import get_logger
from src.pipeline.cache import sha256_file, utc_now

log = get_logger(__name__)

STORE_COLUMNS = [
    "show_id",
    "title",
    "type",
    "is_movie",
    "director",
    "country",
    "listed_in",
    "description",
    "rating",
    "maturity_level",
    "release_year",
    "duration_minutes",
    "duration_seasons",
    "n_genres",
    "n_countries",
    "n_directors",
    "n_cast",
    "description_length",
    "title_length",
    "title_word_count",
    "has_director",
    "has_country",
    "has_cast",
    "has_description",
    "date_added_parsed",
    "year_added",
    "month_added",
    "day_added",
    "weekday_added",
    "quarter_added",
    "years_to_platform",
    "years_to_platform_raw",
    "is_likely_original",
    "acquisition_bucket",
    "catalogue_prominence",
    "primary_country",
    "primary_genre",
    "portfolio_fit",
    "cluster_id",
]


class SchemaMismatchError(ValueError):
    """Raised when a model was trained on a different feature schema."""


def write_feature_store(
    raw: pd.DataFrame, cfg: AppConfig, out_dir: Path
) -> tuple[pd.DataFrame, dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    featured = engineer_features(raw, cfg)
    keep = [c for c in STORE_COLUMNS if c in featured.columns]
    store = featured[keep].copy()
    parquet_path = out_dir / "features.parquet"
    try:
        store.to_parquet(parquet_path, index=False)
    except ImportError:
        csv_path = out_dir / "features.csv"
        store.to_csv(csv_path, index=False)
        log.warning("pyarrow_missing_wrote_csv", path=str(csv_path))
    extra = [c for c in store.columns if c]
    registry = build_registry(cfg, extra_columns=extra)
    dump_registry(registry, out_dir / "feature_registry.json")
    manifest = {
        "n_rows": int(len(store)),
        "n_cols": int(store.shape[1]),
        "schema_hash": registry["schema_hash"],
        "written_at": utc_now(),
        "parquet": str(parquet_path),
    }
    (out_dir / "store_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.info("feature_store_written", n_rows=len(store), schema_hash=registry["schema_hash"])
    return store, registry


def read_feature_store(out_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    parquet_path = out_dir / "features.parquet"
    csv_path = out_dir / "features.csv"
    if parquet_path.exists():
        frame = pd.read_parquet(parquet_path)
    elif csv_path.exists():
        frame = pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(f"No feature store in {out_dir}")
    registry = load_registry(out_dir / "feature_registry.json")
    return frame, registry


def assert_schema_hash(registry: dict[str, Any], expected: str) -> None:
    current = registry.get("schema_hash") or schema_hash(registry)
    if current != expected:
        raise SchemaMismatchError(
            f"Feature schema hash mismatch: model={expected[:12]}… store={current[:12]}…. "
            "Retrain against the current feature store."
        )


def data_sha(path: Path) -> str:
    return sha256_file(path)
