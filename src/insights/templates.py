"""Templated English. Numbers come from the record; the sentence never invents a fact."""

from __future__ import annotations

from typing import Any

TREND = (
    "{metric} in {segment} has {direction} from {start_val:.1f} to {end_val:.1f} "
    "between {start_year} and {end_year} ({pct_change:+.0%}, "
    "R²={r2:.2f}, q={q_value:.3f}, n={n})."
)

SEGMENT = (
    "{metric} is {direction} in {segment_a} than in {segment_b} "
    "({mean_a:.2f} vs {mean_b:.2f}, {effect_name}={effect:.2f}, q={q_value:.3f}, n={n})."
)

COMPOSITION = (
    "The share of {segment} in the catalogue {direction} from {start_share:.1%} to {end_share:.1%} "
    "between {start_period} and {end_period} (Δ={delta_pp:.1f} pp, q={q_value:.3f}, n={n})."
)

ASSOCIATION = (
    "{var_a} is associated with {var_b} " "(Cramér's V={effect:.2f}, q={q_value:.3f}, n={n})."
)

CONCENTRATION = (
    "The catalogue is concentrated in {dimension}: top-{top_n} share={top_share:.1%}, "
    "HHI={hhi:.3f}, Gini={gini:.3f} (n={n})."
)

GAP = (
    "{genre} × {country} has fewer titles than its marginals predict "
    "(residual={residual:.1f}, q={q_value:.3f}, n_cell={n})."
)

OUTLIER = (
    "{segment} is an outlier on {metric} " "(z={z:.2f}, value={value:.2f}, q={q_value:.3f}, n={n})."
)


def render(kind: str, values: dict[str, Any]) -> str:
    templates = {
        "trend": TREND,
        "segment_difference": SEGMENT,
        "composition_shift": COMPOSITION,
        "association": ASSOCIATION,
        "concentration": CONCENTRATION,
        "gap": GAP,
        "outlier": OUTLIER,
    }
    return templates[kind].format(**values)
