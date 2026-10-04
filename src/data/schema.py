"""Pandera schema and rating / type vocabularies loaded from config."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

import pandas as pd

from src.config import AppConfig

if TYPE_CHECKING:
    from pandera import DataFrameSchema


def allowed_types(cfg: AppConfig) -> list[str]:
    return list(cfg.base["dataset"]["allowed_types"])


def rating_vocabulary(cfg: AppConfig) -> list[str]:
    return list(cfg.base["rating_vocabulary"])


def release_year_bounds(cfg: AppConfig) -> tuple[int, int]:
    year_min = int(cfg.base["dataset"]["release_year_min"])
    year_max = datetime.now().year + int(cfg.base["dataset"]["release_year_max_offset"])
    return year_min, year_max


def raw_schema(cfg: AppConfig) -> DataFrameSchema:
    """Schema applied to the raw listing after optional columns are filled."""
    from pandera import Check, Column, DataFrameSchema

    types = allowed_types(cfg)
    year_min, year_max = release_year_bounds(cfg)
    return DataFrameSchema(
        {
            "show_id": Column(str, checks=[Check.str_length(min_value=1)], nullable=False, unique=True),
            "type": Column(str, checks=[Check.isin(types)], nullable=False),
            "title": Column(str, nullable=False),
            "director": Column(str, nullable=True),
            "country": Column(str, nullable=True),
            "date_added": Column(str, nullable=True),
            "release_year": Column(int, checks=[Check.in_range(year_min, year_max)], nullable=False),
            "rating": Column(str, nullable=True),
            "duration": Column(str, nullable=True),
            "listed_in": Column(str, nullable=False),
            "cast": Column(str, nullable=True, required=False),
            "description": Column(str, nullable=True, required=False),
        },
        coerce=True,
        strict=False,
        name="netflix_raw",
    )


def maturity_map(cfg: AppConfig) -> dict[str, str]:
    mapping: dict[str, str] = {}
    ladder: dict[str, Any] = cfg.base["maturity_ladder"]
    for level, ratings in ladder.items():
        if level == "drop":
            continue
        for rating in ratings:
            mapping[str(rating)] = str(level)
    return mapping


def dropped_ratings(cfg: AppConfig) -> set[str]:
    return {str(x) for x in cfg.base["maturity_ladder"].get("drop", [])}


def maturity_order(cfg: AppConfig) -> list[str]:
    return [k for k in cfg.base["maturity_ladder"] if k != "drop"]


def bucket_order(cfg: AppConfig) -> list[str]:
    return [str(item["name"]) for item in cfg.base["acquisition_buckets"]]


def assign_acquisition_bucket(years: float, cfg: AppConfig) -> str:
    for item in cfg.base["acquisition_buckets"]:
        if float(item["min"]) <= years <= float(item["max"]):
            return str(item["name"])
    return "Back-catalogue"


def null_rate_tolerances(cfg: AppConfig) -> dict[str, float]:
    return {str(k): float(v) for k, v in cfg.base["validation"]["max_null_rate"].items()}


def is_missing_token(value: object, cfg: AppConfig) -> bool:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return True
    text = str(value).strip()
    if text == "":
        return True
    tokens = {str(t).strip().lower() for t in cfg.base["dataset"]["missing_tokens"]}
    return text.lower() in tokens
