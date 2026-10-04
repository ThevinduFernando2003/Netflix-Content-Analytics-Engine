"""One model card per promoted head, filled from the promotion record."""

from __future__ import annotations

from typing import Any

from src.config import AppConfig


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_model_card(
    cfg: AppConfig, decision: dict[str, Any], record: dict[str, Any] | None
) -> str:
    head = str(decision.get("head", "unknown"))
    spec = cfg.models["heads"][head]
    lines = [
        f"# Model card — {spec['display_name']}",
        "",
        "This platform analyses catalogue composition and acquisition behaviour, not audience success.",
        "The listing has no views, watch time, completion, revenue, subscriber impact, ratings scores,",
        "awards, budget, or renewal status. The score from this head is a proxy.",
        "",
        f"- Head: `{head}`",
        f"- Task: `{spec['task']}`",
        f"- Primary metric: `{spec['primary_metric']}`",
        f"- Promoted: `{decision.get('promoted')}`",
        f"- Run id: `{decision.get('run_id', '')}`",
    ]
    if not decision.get("promoted"):
        lines += [
            f"- Refused gate: `{decision.get('gate', '')}`",
            f"- Reason: {decision.get('reason', '')}",
            "",
        ]
    elif record:
        metrics = record.get("metrics") or {}
        lines += [
            f"- Candidate: `{record.get('candidate')}`",
            "",
            "## Holdout metrics",
            "",
        ]
        for key, value in metrics.items():
            lines.append(f"- `{key}`: {_fmt(value)}")
        lines.append("")
    path = cfg.root / "docs" / "model_cards" / f"{head}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)
