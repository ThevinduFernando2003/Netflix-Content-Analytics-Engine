from __future__ import annotations

from src.data.schema import assign_acquisition_bucket
from src.targets.build import add_catalogue_prominence, add_targets
from src.targets.definitions import proxy_payload


def test_years_to_platform_negative(featured):
    row = featured.loc[featured["show_id"] == "s203"].iloc[0]
    assert float(row["years_to_platform"]) == -1


def test_bucket_boundaries(cfg):
    assert assign_acquisition_bucket(0, cfg) == "Day-one"
    assert assign_acquisition_bucket(-2, cfg) == "Day-one"
    assert assign_acquisition_bucket(1, cfg) == "Fast"
    assert assign_acquisition_bucket(2, cfg) == "Recent"
    assert assign_acquisition_bucket(5, cfg) == "Recent"
    assert assign_acquisition_bucket(6, cfg) == "Back-catalogue"


def test_is_likely_original(featured):
    row = featured.loc[featured["show_id"] == "s203"].iloc[0]
    assert bool(row["is_likely_original"]) is True
    late = featured.loc[featured["show_id"] == "s204"].iloc[0]
    assert bool(late["is_likely_original"]) is False


def test_catalogue_prominence_hand_computed(featured, cfg):
    # Five-genre, two-country, credited director → high prominence.
    row = featured.loc[featured["show_id"] == "s204"].iloc[0]
    assert int(row["n_genres"]) == 5
    assert int(row["n_countries"]) == 2
    assert bool(row["has_director"]) is True
    assert float(row["catalogue_prominence"]) > 0


def test_prominence_matches_formula_on_row(featured, cfg):
    from src.data.load import load_raw
    from src.features.engineer import engineer_features

    raw = load_raw(cfg.path("raw_csv"), cfg)
    once = add_catalogue_prominence(add_targets(featured.copy(), cfg), cfg)
    again = engineer_features(raw, cfg)
    a = once.loc[once["show_id"] == "s1", "catalogue_prominence"].iloc[0]
    b = again.loc[again["show_id"] == "s1", "catalogue_prominence"].iloc[0]
    assert abs(float(a) - float(b)) < 1e-9


def test_proxy_payload():
    payload = proxy_payload("years_to_platform")
    assert payload["is_proxy"] is True
    assert "not audience success" in payload["proxy_note"]
