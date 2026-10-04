from __future__ import annotations

from src.features.blocks import genre_allowed_for_type, safe_genres_for_type
from src.features.registry import assert_no_banned, banned_columns, build_registry
from src.models.train_all import _head_xy


def test_type_bans_duration_and_tv_genres(cfg, featured):
    registry = build_registry(cfg, extra_columns=list(featured.columns))
    banned = banned_columns(registry, "type")
    assert any(name.startswith("duration") for name in banned)
    X, _y, numeric, categorical, _text = _head_xy(featured, cfg, "type", registry)
    used = numeric + categorical
    assert not any(c.startswith("duration") for c in used)
    assert "primary_genre" not in used
    for col in used:
        assert "TV " not in col


def test_safe_genres_strip_tv(cfg):
    text = safe_genres_for_type("Crime TV Shows, International TV Shows, Documentaries", cfg)
    assert "TV " not in text
    assert "Documentaries" in text
    assert genre_allowed_for_type("Kids' TV", cfg) is False


def test_acquisition_bans_date_added_derived(cfg, featured):
    registry = build_registry(cfg, extra_columns=list(featured.columns))
    for head in ("acquisition_reg", "acquisition_clf"):
        _X, _y, numeric, categorical, _text = _head_xy(featured, cfg, head, registry)
        used = set(numeric + categorical)
        for leak in (
            "date_added",
            "year_added",
            "month_added",
            "day_added",
            "weekday_added",
            "quarter_added",
            "years_to_platform",
            "is_likely_original",
            "acquisition_bucket",
        ):
            assert leak not in used, leak


def test_assert_no_banned_raises(cfg):
    registry = build_registry(cfg)
    try:
        assert_no_banned(["duration_minutes"], registry, "type")
        raised = False
    except ValueError:
        raised = True
    assert raised
