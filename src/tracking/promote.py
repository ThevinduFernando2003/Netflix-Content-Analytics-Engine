"""Promotion is a coded rule, not a judgement call. A refusal names the failing gate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.config import AppConfig
from src.logging_conf import get_logger
from src.tracking.cards import write_model_card
from src.tracking.mlflow_utils import load_run_records
from src.tracking.registry import read_incumbent, write_promotion

log = get_logger(__name__)


class PromotionRefused(Exception):
    def __init__(self, gate: str, message: str) -> None:
        self.gate = gate
        super().__init__(f"Promotion refused: gate '{gate}' failed — {message}")


def _primary(record: dict[str, Any], spec: dict[str, Any]) -> float | None:
    metric = spec["primary_metric"]
    value = (record.get("metrics") or {}).get(metric)
    if value is None:
        return None
    return float(value)


def _is_better(candidate: float, reference: float, greater: bool, margin: float) -> bool:
    if greater:
        return candidate >= reference + margin
    return candidate <= reference - margin


def evaluate_gates(
    cfg: AppConfig, head: str, record: dict[str, Any], baseline: dict[str, Any] | None
) -> None:
    spec = cfg.models["heads"][head]
    gates = cfg.models["quality_gates"]
    greater = bool(spec.get("primary_greater_is_better", True))
    margin = float(cfg.models["defaults"].get("promotion_margin", 0.01))
    cand = _primary(record, spec)
    if cand is None:
        raise PromotionRefused(
            "primary_metric_missing", f"{spec['primary_metric']} not on the candidate record"
        )

    if gates.get("require_beat_baseline", True):
        if baseline is None:
            raise PromotionRefused(
                "baseline_missing", "no dummy/baseline run recorded for this head"
            )
        base = _primary(baseline, spec)
        if base is None:
            raise PromotionRefused("baseline_metric_missing", spec["primary_metric"])
        if not _is_better(cand, base, greater, margin):
            raise PromotionRefused(
                "beat_baseline",
                f"{spec['primary_metric']}={cand:.4f} did not beat baseline {base:.4f} by margin {margin}",
            )

    if gates.get("require_beat_incumbent", True):
        incumbent = read_incumbent(cfg, head)
        if incumbent and incumbent.get("metrics"):
            inc = (incumbent.get("metrics") or {}).get(spec["primary_metric"])
            if inc is not None and not _is_better(cand, float(inc), greater, 0.0):
                raise PromotionRefused(
                    "beat_incumbent",
                    f"{spec['primary_metric']}={cand:.4f} did not beat incumbent {float(inc):.4f}",
                )

    if gates.get("require_no_banned_features", True) and record.get("banned_features"):
        raise PromotionRefused("no_banned_features", str(record["banned_features"]))

    if gates.get("require_calibration", True) and spec["task"] in {
        "binary_classification",
        "ordinal_classification",
        "multiclass",
    }:
        brier = (record.get("metrics") or {}).get("brier_after")
        tol = float(cfg.models["defaults"].get("calibration_brier_tolerance", 0.25))
        if brier is not None and float(brier) > 1.0 + tol:
            raise PromotionRefused("calibration", f"brier_after={brier} exceeds tolerance")

    if gates.get("require_latency", True):
        latency = (record.get("metrics") or {}).get("latency_ms_p95")
        budget = float(cfg.models["defaults"].get("latency_budget_ms", 250))
        if latency is not None and float(latency) > budget:
            raise PromotionRefused("latency", f"p95 {latency}ms > {budget}ms")


def _pick_best(records: list[dict[str, Any]], spec: dict[str, Any]) -> dict[str, Any] | None:
    metric = spec["primary_metric"]
    greater = bool(spec.get("primary_greater_is_better", True))
    scored = [r for r in records if (r.get("metrics") or {}).get(metric) is not None]
    if not scored:
        return None
    return sorted(scored, key=lambda r: float(r["metrics"][metric]), reverse=greater)[0]


def promote_head(cfg: AppConfig, head: str) -> dict[str, Any]:
    records = [r for r in load_run_records(cfg) if r.get("head") == head]
    spec = cfg.models["heads"][head]
    baseline = next((r for r in records if r.get("is_baseline")), None)
    challengers = [r for r in records if not r.get("is_baseline")]
    best = _pick_best(challengers, spec)
    if best is None:
        raise PromotionRefused("no_candidate", f"no challenger runs for head '{head}'")
    evaluate_gates(cfg, head, best, baseline)
    model_path = Path(best["model_path"]) if best.get("model_path") else None
    dest = write_promotion(cfg, head, best, model_path)
    decision = {"head": head, "promoted": True, "run_id": best.get("run_id"), "dir": str(dest)}
    log.info("promotion_accepted", **decision)
    return decision


def promote_all(cfg: AppConfig) -> dict[str, Any]:
    results: dict[str, Any] = {}
    refusals: list[str] = []
    for head in cfg.models["heads"]:
        try:
            results[head] = promote_head(cfg, head)
        except PromotionRefused as exc:
            results[head] = {"promoted": False, "gate": exc.gate, "reason": str(exc)}
            refusals.append(str(exc))
            log.info("promotion_refused", head=head, gate=exc.gate, reason=str(exc))
        record = next(
            (
                r
                for r in load_run_records(cfg)
                if r.get("head") == head and r.get("run_id") == results[head].get("run_id")
            ),
            None,
        )
        write_model_card(cfg, {"head": head, **results[head]}, record)
    path = cfg.path("artifacts_dir") / "promotion_decisions.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    if refusals:
        print("\n".join(refusals))
    return {"decisions": results, "n_refused": len(refusals)}
