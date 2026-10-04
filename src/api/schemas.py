"""Pydantic v2 request/response models. Proxy fields carry is_proxy + proxy_note."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from src.targets.definitions import PROXY_NOTES


class TitleIn(BaseModel):
    title: str = "Untitled"
    type: str | None = None
    director: str = ""
    country: str = ""
    listed_in: str = ""
    description: str = ""
    rating: str | None = None
    release_year: int = 2018
    duration: str | None = None
    date_added: str | None = None
    year_added: int | None = None
    cast: str = ""


class ProxyBlock(BaseModel):
    is_proxy: bool = True
    proxy_note: str


class AcquisitionOut(BaseModel):
    years_to_platform: float | None = None
    interval_lo: float | None = None
    interval_hi: float | None = None
    bucket: str | None = None
    is_likely_original: bool | None = None
    is_proxy: bool = True
    proxy_note: str = PROXY_NOTES["acquisition_priority"]


class TypeOut(BaseModel):
    label: str
    probability_movie: float
    is_proxy: bool = False
    proxy_note: str = "Observed catalogue field, not a success metric."


class RatingOut(BaseModel):
    label: str
    probabilities: dict[str, float]
    is_proxy: bool = False
    proxy_note: str = (
        "Observed catalogue field collapsed onto an ordinal ladder, not a success metric."
    )


class SegmentOut(BaseModel):
    cluster_id: int | None = None
    name: str | None = None
    distance: float | None = None
    outlier: bool = False
    portfolio_fit: float | None = None
    is_proxy: bool = True
    proxy_note: str = PROXY_NOTES["portfolio_fit"]


class AnalyzeOut(BaseModel):
    title: str
    type: TypeOut | None = None
    rating: RatingOut | None = None
    acquisition_priority: AcquisitionOut
    segment: SegmentOut
    catalogue_prominence: dict[str, Any] = Field(default_factory=dict)
    insights: list[dict[str, Any]] = Field(default_factory=list)
    summary: str = ""


class HealthOut(BaseModel):
    status: str


class MetaOut(BaseModel):
    version: str
    git_sha: str
    feature_schema_hash: str | None
    promoted: dict[str, Any]
    pipeline_run_id: str | None = None
