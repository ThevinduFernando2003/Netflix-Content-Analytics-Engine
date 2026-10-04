"""Proxy contract. Every target that is not an observed catalogue field is a proxy."""

from __future__ import annotations

PROXY_NOTES = {
    "years_to_platform": "Measures platform acquisition lag, not audience success.",
    "acquisition_priority": "Measures platform acquisition lag, not audience success.",
    "acquisition_bucket": "Bucketed acquisition lag (Day-one / Fast / Recent / Back-catalogue), not success.",
    "is_likely_original": (
        "Heuristic: added in or before release year. Not a Netflix Originals ground-truth flag."
    ),
    "catalogue_prominence": (
        "Constructed metadata-richness index, not an observed audience or editorial outcome."
    ),
    "portfolio_fit": "Distance to cluster centroid in metadata space. Not a quality or demand score.",
}

OBSERVED_FIELDS = {"type", "is_movie", "rating", "maturity_level"}


def proxy_payload(name: str) -> dict[str, object]:
    return {
        "is_proxy": name not in OBSERVED_FIELDS,
        "proxy_note": PROXY_NOTES.get(name, "Observed catalogue field, not a success metric."),
    }
