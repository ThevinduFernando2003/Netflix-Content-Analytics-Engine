"""Explicit handling of every known listing defect."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

from src.config import AppConfig
from src.data.schema import is_missing_token
from src.logging_conf import get_logger

log = get_logger(__name__)

DURATION_IN_RATING = re.compile(r"^\d+\s*min$", flags=re.IGNORECASE)
MINUTES = re.compile(r"^(\d+)\s*min$", flags=re.IGNORECASE)
SEASONS = re.compile(r"^(\d+)\s*seasons?$", flags=re.IGNORECASE)


def split_list(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value).strip()
    if text == "":
        return []
    return [part.strip() for part in text.split(",") if part.strip()]


def _empty_if_missing(series: pd.Series, cfg: AppConfig) -> pd.Series:
    return series.map(lambda v: "" if is_missing_token(v, cfg) else str(v).strip())


def parse_date_added(series: pd.Series, cfg: AppConfig) -> tuple[pd.Series, int]:
    """Parse with explicit formats then a generic fallback. Returns (dates, n_failed)."""
    parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    remaining = series.copy()
    formats = list(cfg.base["dataset"]["date_added_formats"])
    # Windows strftime does not accept %-m; keep only portable formats.
    portable = [fmt for fmt in formats if "%-" not in fmt]
    for fmt in portable:
        mask = remaining.notna() & parsed.isna()
        if not mask.any():
            break
        attempt = pd.to_datetime(remaining[mask], format=fmt, errors="coerce")
        parsed.loc[mask] = attempt
        remaining = remaining.where(parsed.isna(), other=pd.NA)
    still = parsed.isna() & series.notna() & (series.astype(str).str.strip() != "")
    if still.any():
        fallback = pd.to_datetime(series[still], errors="coerce")
        parsed.loc[still] = fallback
    failed = int((parsed.isna() & series.notna() & (series.astype(str).str.strip() != "")).sum())
    empty = int((series.isna() | (series.astype(str).str.strip() == "")).sum())
    return parsed, failed + empty


def repair_rating_duration(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Move duration strings that leaked into `rating` back to `duration`."""
    out = frame.copy()
    rating = out["rating"].astype("string")
    mask = rating.fillna("").str.match(DURATION_IN_RATING)
    affected = out.loc[mask, "show_id"].astype(str).tolist()
    if affected:
        duration_missing = out.loc[mask, "duration"].isna() | (
            out.loc[mask, "duration"].astype(str).str.strip() == ""
        )
        out.loc[mask & duration_missing, "duration"] = out.loc[mask & duration_missing, "rating"]
        out.loc[mask, "rating"] = pd.NA
        log.info("rating_duration_contamination", show_ids=affected, n=len(affected))
    return out, affected


def parse_duration(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series, int]:
    raw = frame["duration"].astype("string")
    minutes = pd.Series(pd.NA, index=frame.index, dtype="Float64")
    seasons = pd.Series(pd.NA, index=frame.index, dtype="Float64")
    failed = 0
    for idx, value in raw.items():
        if value is None or pd.isna(value) or str(value).strip() == "":
            continue
        text = str(value).strip()
        m_min = MINUTES.match(text)
        m_sea = SEASONS.match(text)
        if m_min:
            minutes.at[idx] = float(m_min.group(1))
        elif m_sea:
            seasons.at[idx] = float(m_sea.group(1))
        else:
            failed += 1
    return minutes, seasons, failed


def clean_catalogue(frame: pd.DataFrame, cfg: AppConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply every defect handler. Never drop rows for missing director/cast/country."""
    notes: dict[str, Any] = {}
    out = frame.copy()
    for col in ("director", "cast", "country", "title", "listed_in", "description", "rating", "duration"):
        if col not in out.columns:
            out[col] = ""
    before = len(out)
    out = out.drop_duplicates(subset=["show_id"], keep="first")
    notes["duplicate_show_ids_dropped"] = before - len(out)

    out, contaminated = repair_rating_duration(out)
    notes["rating_duration_show_ids"] = contaminated

    for col in ("director", "cast", "country", "title", "listed_in", "description", "rating", "duration"):
        if col in out.columns:
            out[col] = _empty_if_missing(out[col], cfg)

    parsed_dates, date_failures = parse_date_added(out["date_added"].astype("string"), cfg)
    out["date_added_parsed"] = parsed_dates
    notes["date_added_parse_failures"] = date_failures
    notes["date_added_missing_or_unparsed"] = int(parsed_dates.isna().sum())

    minutes, seasons, duration_failures = parse_duration(out)
    out["duration_minutes"] = minutes
    out["duration_seasons"] = seasons
    notes["duration_parse_failures"] = duration_failures

    out["directors"] = out["director"].map(split_list)
    out["cast_list"] = out["cast"].map(split_list)
    out["countries"] = out["country"].map(split_list)
    out["genres"] = out["listed_in"].map(split_list)

    out["has_director"] = out["directors"].map(lambda xs: len(xs) > 0)
    out["has_cast"] = out["cast_list"].map(lambda xs: len(xs) > 0)
    out["has_country"] = out["countries"].map(lambda xs: len(xs) > 0)
    out["has_description"] = out["description"].astype(str).str.strip().ne("")

    out["n_directors"] = out["directors"].map(len)
    out["n_cast"] = out["cast_list"].map(len)
    out["n_countries"] = out["countries"].map(len)
    out["n_genres"] = out["genres"].map(len)
    out["description_length"] = out["description"].astype(str).str.len().where(out["has_description"], 0)
    out["title_length"] = out["title"].astype(str).str.len()
    out["title_word_count"] = out["title"].astype(str).str.split().map(len)

    out["year_added"] = out["date_added_parsed"].dt.year.astype("Int64")
    out["month_added"] = out["date_added_parsed"].dt.month.astype("Int64")
    out["day_added"] = out["date_added_parsed"].dt.day.astype("Int64")
    out["weekday_added"] = out["date_added_parsed"].dt.weekday.astype("Int64")
    out["quarter_added"] = out["date_added_parsed"].dt.quarter.astype("Int64")

    floor = int(cfg.base["dataset"]["years_to_platform_floor"])
    raw_ytp = out["year_added"].astype("Float64") - out["release_year"].astype("Float64")
    notes["years_to_platform_negative"] = int((raw_ytp < 0).fillna(False).sum())
    notes["years_to_platform_below_floor"] = int((raw_ytp < floor).fillna(False).sum())
    notes["years_to_platform_floor"] = floor
    out["years_to_platform_raw"] = raw_ytp
    out["years_to_platform"] = raw_ytp.clip(lower=float(floor))

    out["is_movie"] = out["type"].astype(str).eq("Movie")
    notes["rows_in"] = before
    notes["rows_out"] = len(out)
    return out, notes
