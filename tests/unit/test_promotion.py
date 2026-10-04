from __future__ import annotations

import pytest
from src.tracking.promote import PromotionRefused, evaluate_gates
from src.tracking.registry import rollback, write_promotion


def _rec(metric: float, banned=None) -> dict:
    return {
        "head": "type",
        "run_id": "abc",
        "candidate": "logreg",
        "metrics": {"roc_auc": metric, "brier_after": 0.1, "latency_ms_p95": 10},
        "banned_features": banned or [],
    }


def test_refused_when_loses_to_baseline(cfg):
    with pytest.raises(PromotionRefused) as exc:
        evaluate_gates(cfg, "type", _rec(0.50), _rec(0.80))
    assert exc.value.gate == "beat_baseline"
    assert "beat_baseline" in str(exc.value)


def test_refused_when_loses_to_incumbent(cfg):
    write_promotion(cfg, "type", _rec(0.95), None)
    with pytest.raises(PromotionRefused) as exc:
        evaluate_gates(cfg, "type", _rec(0.90), _rec(0.50))
    assert exc.value.gate == "beat_incumbent"


def test_refused_when_banned_feature(cfg):
    with pytest.raises(PromotionRefused) as exc:
        evaluate_gates(cfg, "type", _rec(0.99, banned=["duration_minutes"]), _rec(0.50))
    assert exc.value.gate == "no_banned_features"


def test_rollback_restores_previous(cfg):
    write_promotion(cfg, "type", {**_rec(0.80), "run_id": "first"}, None)
    write_promotion(cfg, "type", {**_rec(0.90), "run_id": "second"}, None)
    prev = rollback(cfg, "type")
    assert prev["run_id"] == "first"
