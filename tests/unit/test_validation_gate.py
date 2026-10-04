from __future__ import annotations

import pandas as pd
import pytest
from src.data.validate import ValidationGateError, collect_issues, gate_fails, run_validation_gate


def test_gate_passes_on_fixture(raw_frame, cfg, tmp_path):
    issues = collect_issues(raw_frame, cfg)
    assert not gate_fails(issues)
    run_validation_gate(raw_frame, cfg, tmp_path / "val.html")
    assert (tmp_path / "val.html").exists()


def test_missing_column_fails(raw_frame, cfg):
    bad = raw_frame.drop(columns=["title"])
    issues = collect_issues(bad, cfg)
    assert any(i.get("check") == "missing_column" for i in issues if not i.get("ok", True))


def test_duplicate_show_id_fails(raw_frame, cfg):
    bad = pd.concat([raw_frame, raw_frame.iloc[[0]]], ignore_index=True)
    issues = collect_issues(bad, cfg)
    assert any(i.get("check") == "duplicate_show_id" for i in issues if not i.get("ok", True))


def test_out_of_range_release_year_fails(raw_frame, cfg):
    bad = raw_frame.copy()
    bad.loc[0, "release_year"] = 1800
    issues = collect_issues(bad, cfg)
    assert any(i.get("check") == "release_year_range" for i in issues if not i.get("ok", True))


def test_unknown_rating_quarantined(raw_frame, cfg):
    bad = raw_frame.copy()
    bad.loc[0, "rating"] = "XYZ-99"
    issues = collect_issues(bad, cfg)
    # quarantine is on, so this is recorded but not a hard fail
    assert any(i.get("check") == "unknown_rating_quarantined" for i in issues)


def test_null_rate_over_tolerance_fails(raw_frame, cfg):
    bad = raw_frame.copy()
    bad["type"] = ""
    issues = collect_issues(bad, cfg)
    assert any(i.get("check") == "null_rate" and i.get("column") == "type" for i in issues)


def test_run_validation_raises(raw_frame, cfg, tmp_path):
    bad = raw_frame.drop(columns=["show_id"])
    with pytest.raises(ValidationGateError):
        run_validation_gate(bad, cfg, tmp_path / "val.html")
