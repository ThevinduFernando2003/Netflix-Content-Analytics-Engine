"""Feature registry: name, dtype, description, source, transform, null policy, per-task allow."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from src.config import AppConfig

FEATURE_SPECS: list[dict[str, Any]] = [
    {
        "name": "show_id",
        "dtype": "string",
        "description": "Stable catalogue identifier.",
        "source_columns": ["show_id"],
        "transformation": "identity",
        "null_policy": "required",
        "role": "id",
    },
    {
        "name": "title",
        "dtype": "string",
        "description": "Title text.",
        "source_columns": ["title"],
        "transformation": "strip; missing → empty string",
        "null_policy": "empty_string",
        "role": "text",
    },
    {
        "name": "type",
        "dtype": "string",
        "description": "Movie or TV Show. Observed label for Head A; feature for other heads.",
        "source_columns": ["type"],
        "transformation": "identity",
        "null_policy": "required",
        "role": "categorical",
    },
    {
        "name": "is_movie",
        "dtype": "bool",
        "description": "1 if type == Movie.",
        "source_columns": ["type"],
        "transformation": "type == Movie",
        "null_policy": "false",
        "role": "target",
    },
    {
        "name": "release_year",
        "dtype": "int",
        "description": "Billed first-release year.",
        "source_columns": ["release_year"],
        "transformation": "identity",
        "null_policy": "required",
        "role": "numeric",
    },
    {
        "name": "rating",
        "dtype": "string",
        "description": "Maturity rating as listed.",
        "source_columns": ["rating"],
        "transformation": "duration-in-rating repaired; missing → empty",
        "null_policy": "empty_string",
        "role": "categorical",
    },
    {
        "name": "maturity_level",
        "dtype": "string",
        "description": "Ordinal collapse of rating (Kids / Older Kids / Teens / Mature).",
        "source_columns": ["rating"],
        "transformation": "config/base.yaml maturity_ladder; NR/UR dropped",
        "null_policy": "nullable",
        "role": "target",
    },
    {
        "name": "duration_minutes",
        "dtype": "float",
        "description": "Parsed movie runtime. Banned for Head A.",
        "source_columns": ["duration"],
        "transformation": "regex minutes",
        "null_policy": "nullable",
        "role": "numeric",
    },
    {
        "name": "duration_seasons",
        "dtype": "float",
        "description": "Parsed TV season count. Banned for Head A.",
        "source_columns": ["duration"],
        "transformation": "regex seasons",
        "null_policy": "nullable",
        "role": "numeric",
    },
    {
        "name": "n_genres",
        "dtype": "int",
        "description": "Count of listed_in tokens.",
        "source_columns": ["listed_in"],
        "transformation": "split/strip",
        "null_policy": "zero",
        "role": "numeric",
    },
    {
        "name": "n_countries",
        "dtype": "int",
        "description": "Count of country tokens.",
        "source_columns": ["country"],
        "transformation": "split/strip; Not Given → empty",
        "null_policy": "zero",
        "role": "numeric",
    },
    {
        "name": "n_directors",
        "dtype": "int",
        "description": "Count of credited directors.",
        "source_columns": ["director"],
        "transformation": "split/strip; Not Given → empty",
        "null_policy": "zero",
        "role": "numeric",
    },
    {
        "name": "n_cast",
        "dtype": "int",
        "description": "Cast size. Zero on the shipped file (column absent).",
        "source_columns": ["cast"],
        "transformation": "split/strip",
        "null_policy": "zero",
        "role": "numeric",
    },
    {
        "name": "description_length",
        "dtype": "int",
        "description": "Synopsis character length. Zero on the shipped file (column absent).",
        "source_columns": ["description"],
        "transformation": "len(strip)",
        "null_policy": "zero",
        "role": "numeric",
    },
    {
        "name": "title_length",
        "dtype": "int",
        "description": "Title character length.",
        "source_columns": ["title"],
        "transformation": "len",
        "null_policy": "zero",
        "role": "numeric",
    },
    {
        "name": "title_word_count",
        "dtype": "int",
        "description": "Title word count.",
        "source_columns": ["title"],
        "transformation": "split len",
        "null_policy": "zero",
        "role": "numeric",
    },
    {
        "name": "has_director",
        "dtype": "bool",
        "description": "Director credited after missing-token cleanup.",
        "source_columns": ["director"],
        "transformation": "indicator",
        "null_policy": "false",
        "role": "binary",
    },
    {
        "name": "has_country",
        "dtype": "bool",
        "description": "At least one country listed.",
        "source_columns": ["country"],
        "transformation": "indicator",
        "null_policy": "false",
        "role": "binary",
    },
    {
        "name": "has_cast",
        "dtype": "bool",
        "description": "At least one cast member listed.",
        "source_columns": ["cast"],
        "transformation": "indicator",
        "null_policy": "false",
        "role": "binary",
    },
    {
        "name": "has_description",
        "dtype": "bool",
        "description": "Non-empty synopsis.",
        "source_columns": ["description"],
        "transformation": "indicator",
        "null_policy": "false",
        "role": "binary",
    },
    {
        "name": "date_added_parsed",
        "dtype": "datetime",
        "description": "Parsed platform-add date. Banned for Heads C/D.",
        "source_columns": ["date_added"],
        "transformation": "explicit formats then coerce",
        "null_policy": "nullable",
        "role": "temporal",
    },
    {
        "name": "year_added",
        "dtype": "int",
        "description": "Year of date_added. Banned for Heads C/D — leaks years_to_platform.",
        "source_columns": ["date_added"],
        "transformation": "dt.year",
        "null_policy": "nullable",
        "role": "temporal",
    },
    {
        "name": "month_added",
        "dtype": "int",
        "description": "Month of date_added. Banned for Heads C/D.",
        "source_columns": ["date_added"],
        "transformation": "dt.month",
        "null_policy": "nullable",
        "role": "temporal",
    },
    {
        "name": "day_added",
        "dtype": "int",
        "description": "Day of date_added. Banned for Heads C/D.",
        "source_columns": ["date_added"],
        "transformation": "dt.day",
        "null_policy": "nullable",
        "role": "temporal",
    },
    {
        "name": "weekday_added",
        "dtype": "int",
        "description": "Weekday of date_added. Banned for Heads C/D.",
        "source_columns": ["date_added"],
        "transformation": "dt.weekday",
        "null_policy": "nullable",
    },
    {
        "name": "quarter_added",
        "dtype": "int",
        "description": "Quarter of date_added. Banned for Heads C/D.",
        "source_columns": ["date_added"],
        "transformation": "dt.quarter",
        "null_policy": "nullable",
        "role": "temporal",
    },
    {
        "name": "years_to_platform",
        "dtype": "float",
        "description": "year_added - release_year, floored. PROXY acquisition lag.",
        "source_columns": ["date_added", "release_year"],
        "transformation": "difference; floor at configured value",
        "null_policy": "nullable",
        "role": "target",
        "is_proxy": True,
    },
    {
        "name": "years_to_platform_raw",
        "dtype": "float",
        "description": "Uncapped year_added - release_year.",
        "source_columns": ["date_added", "release_year"],
        "transformation": "difference",
        "null_policy": "nullable",
        "role": "audit",
    },
    {
        "name": "is_likely_original",
        "dtype": "bool",
        "description": "Heuristic originals flag: years_to_platform <= 0. PROXY.",
        "source_columns": ["years_to_platform"],
        "transformation": "<= 0",
        "null_policy": "false",
        "role": "target",
        "is_proxy": True,
    },
    {
        "name": "acquisition_bucket",
        "dtype": "string",
        "description": "Ordinal bucket of years_to_platform. PROXY.",
        "source_columns": ["years_to_platform"],
        "transformation": "config/base.yaml acquisition_buckets",
        "null_policy": "nullable",
        "role": "target",
        "is_proxy": True,
    },
    {
        "name": "catalogue_prominence",
        "dtype": "float",
        "description": "Constructed metadata-richness index. PROXY.",
        "source_columns": ["listed_in", "country", "director", "cast", "description"],
        "transformation": "weighted min-max of configured components",
        "null_policy": "zero",
        "role": "numeric",
        "is_proxy": True,
    },
    {
        "name": "primary_country",
        "dtype": "string",
        "description": "First listed country or empty.",
        "source_columns": ["country"],
        "transformation": "first token",
        "null_policy": "empty_string",
        "role": "categorical",
    },
    {
        "name": "primary_genre",
        "dtype": "string",
        "description": "First listed genre.",
        "source_columns": ["listed_in"],
        "transformation": "first token",
        "null_policy": "empty_string",
        "role": "categorical",
    },
    {
        "name": "listed_in",
        "dtype": "string",
        "description": "Raw genre list. TV-prefixed tokens banned for Head A at encoding time.",
        "source_columns": ["listed_in"],
        "transformation": "identity",
        "null_policy": "empty_string",
        "role": "text",
    },
    {
        "name": "description",
        "dtype": "string",
        "description": "Synopsis. Empty on the shipped file.",
        "source_columns": ["description"],
        "transformation": "missing → empty string (never 'Unknown')",
        "null_policy": "empty_string",
        "role": "text",
    },
    {
        "name": "director",
        "dtype": "string",
        "description": "Raw director string after missing-token cleanup.",
        "source_columns": ["director"],
        "transformation": "Not Given → empty",
        "null_policy": "empty_string",
        "role": "text",
    },
    {
        "name": "country",
        "dtype": "string",
        "description": "Raw country string after missing-token cleanup.",
        "source_columns": ["country"],
        "transformation": "Not Given → empty",
        "null_policy": "empty_string",
        "role": "text",
    },
    {
        "name": "portfolio_fit",
        "dtype": "float",
        "description": "1 - normalised distance to cluster centroid. PROXY. Filled after Head E.",
        "source_columns": ["cluster_id"],
        "transformation": "clustering head",
        "null_policy": "nullable",
        "role": "target",
        "is_proxy": True,
    },
    {
        "name": "cluster_id",
        "dtype": "int",
        "description": "Segment assignment from Head E.",
        "source_columns": [],
        "transformation": "spherical k-means",
        "null_policy": "nullable",
        "role": "derived",
    },
]


def _banned_for(cfg: AppConfig, leakage_key: str, name: str) -> bool:
    rules = cfg.features["leakage"].get(leakage_key, {})
    prefixes = list(rules.get("banned_prefixes", []))
    return any(name == p or name.startswith(p) for p in prefixes)


def build_registry(cfg: AppConfig, extra_columns: list[str] | None = None) -> dict[str, Any]:
    heads = list(cfg.models["heads"].keys())
    leakage_keys = {h: cfg.models["heads"][h].get("leakage_key", h) for h in heads}
    features: list[dict[str, Any]] = []
    known = {spec["name"] for spec in FEATURE_SPECS}
    for spec in FEATURE_SPECS:
        item = dict(spec)
        allowed: dict[str, bool] = {}
        for head in heads:
            allowed[head] = not _banned_for(cfg, leakage_keys[head], spec["name"])
        item["allowed_for"] = allowed
        features.append(item)
    if extra_columns:
        for name in extra_columns:
            if name in known:
                continue
            allowed = {h: not _banned_for(cfg, leakage_keys[h], name) for h in heads}
            features.append(
                {
                    "name": name,
                    "dtype": "unknown",
                    "description": "Generated column",
                    "source_columns": [],
                    "transformation": "generated",
                    "null_policy": "nullable",
                    "role": "generated",
                    "allowed_for": allowed,
                }
            )
    payload: dict[str, Any] = {"features": features}
    payload["schema_hash"] = schema_hash(payload)
    return payload


def schema_hash(registry: dict[str, Any]) -> str:
    names = sorted(f["name"] for f in registry["features"])
    return hashlib.sha256("|".join(names).encode("utf-8")).hexdigest()


def dump_registry(registry: dict[str, Any], path: Any) -> None:
    path.write_text(json.dumps(registry, indent=2, sort_keys=True), encoding="utf-8")


def load_registry(path: Any) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def allowed_columns(registry: dict[str, Any], head: str) -> list[str]:
    return [f["name"] for f in registry["features"] if f.get("allowed_for", {}).get(head, True)]


def banned_columns(registry: dict[str, Any], head: str) -> list[str]:
    return [f["name"] for f in registry["features"] if not f.get("allowed_for", {}).get(head, True)]


def assert_no_banned(feature_names: list[str], registry: dict[str, Any], head: str) -> None:
    banned = set(banned_columns(registry, head))
    leaked = sorted(set(feature_names) & banned)
    if leaked:
        raise ValueError(f"Banned features reached head '{head}': {leaked}")
