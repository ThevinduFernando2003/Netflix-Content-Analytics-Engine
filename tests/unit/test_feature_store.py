from __future__ import annotations

import pytest
from src.data.load import load_raw
from src.features.registry import build_registry, schema_hash
from src.features.store import SchemaMismatchError, assert_schema_hash, write_feature_store


def test_registry_covers_store_columns(cfg):
    raw = load_raw(cfg.path("raw_csv"), cfg)
    store, registry = write_feature_store(raw, cfg, cfg.path("features_dir"))
    names = {f["name"] for f in registry["features"]}
    for col in store.columns:
        assert col in names


def test_schema_hash_changes_when_feature_added(cfg):
    a = build_registry(cfg)
    b = build_registry(cfg, extra_columns=["brand_new_feature"])
    assert schema_hash(a) != schema_hash(b)


def test_mismatched_schema_raises(cfg):
    reg = build_registry(cfg)
    with pytest.raises(SchemaMismatchError):
        assert_schema_hash(reg, "not-the-hash")


def test_allow_lists_respected(cfg):
    reg = build_registry(cfg)
    type_feat = next(f for f in reg["features"] if f["name"] == "duration_minutes")
    assert type_feat["allowed_for"]["type"] is False
    year = next(f for f in reg["features"] if f["name"] == "year_added")
    assert year["allowed_for"]["acquisition_reg"] is False
    assert year["allowed_for"]["acquisition_clf"] is False
