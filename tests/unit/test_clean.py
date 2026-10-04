from __future__ import annotations

import pandas as pd
from src.data.clean import clean_catalogue, split_list


def test_rating_duration_moved(raw_frame, cfg):
    cleaned, notes = clean_catalogue(raw_frame, cfg)
    row = cleaned.loc[cleaned["show_id"] == "s200"].iloc[0]
    assert "s200" in notes["rating_duration_show_ids"]
    assert str(row["duration_minutes"]) == "74.0" or float(row["duration_minutes"]) == 74
    assert row["rating"] == "" or row["rating"] != "74 min"


def test_duration_units_split(raw_frame, cfg):
    cleaned, _ = clean_catalogue(raw_frame, cfg)
    movie = cleaned.loc[cleaned["show_id"] == "s1"].iloc[0]
    tv = cleaned.loc[cleaned["show_id"] == "s41"].iloc[0]
    assert float(movie["duration_minutes"]) > 0
    assert pd_isna(movie["duration_seasons"])
    assert float(tv["duration_seasons"]) == 1
    assert pd_isna(tv["duration_minutes"])


def test_multivalue_split():
    assert split_list("France, Germany") == ["France", "Germany"]
    assert split_list("") == []
    assert split_list(None) == []


def test_missing_text_is_empty_not_unknown(raw_frame, cfg):
    cleaned, _ = clean_catalogue(raw_frame, cfg)
    row = cleaned.loc[cleaned["show_id"] == "s201"].iloc[0]
    assert row["director"] == ""
    assert row["country"] == ""
    assert not row["has_director"]
    assert "Unknown" not in str(row["director"])


def test_not_given_becomes_indicator(raw_frame, cfg):
    cleaned, _ = clean_catalogue(raw_frame, cfg)
    kids = cleaned.loc[cleaned["show_id"] == "s41"].iloc[0]
    assert kids["director"] == ""
    assert bool(kids["has_director"]) is False


def test_negative_years_not_silently_zero(raw_frame, cfg):
    cleaned, notes = clean_catalogue(raw_frame, cfg)
    row = cleaned.loc[cleaned["show_id"] == "s203"].iloc[0]
    assert float(row["years_to_platform"]) < 0
    assert notes["years_to_platform_negative"] >= 1


def test_dedup_on_show_id(raw_frame, cfg):
    doubled = pd.concat([raw_frame.iloc[[0]], raw_frame], ignore_index=True)
    cleaned, notes = clean_catalogue(doubled, cfg)
    assert notes["duplicate_show_ids_dropped"] >= 1
    assert cleaned["show_id"].is_unique


def test_date_added_parse(raw_frame, cfg):
    cleaned, notes = clean_catalogue(raw_frame, cfg)
    assert cleaned["date_added_parsed"].notna().mean() > 0.9
    assert "date_added_parse_failures" in notes


def pd_isna(value: object) -> bool:
    import pandas as pd

    return bool(pd.isna(value))
