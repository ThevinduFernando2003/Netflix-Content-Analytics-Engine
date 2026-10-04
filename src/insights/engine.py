"""Scan the catalogue, correct for multiplicity, render only surviving findings."""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

from src.config import AppConfig
from src.features.store import read_feature_store
from src.insights.effects import cliffs_delta, cohens_d, cramers_v, gini, hhi, top_n_share
from src.insights.narrative import assert_no_causal_language
from src.insights.templates import render
from src.insights.tests import chi_square, trend_regression, two_group_numeric
from src.logging_conf import get_logger

log = get_logger(__name__)


def bh_fdr(p_values: list[float], alpha: float) -> tuple[np.ndarray, np.ndarray]:
    if not p_values:
        return np.array([], dtype=bool), np.array([])
    rejected, adj, _alphacSidak, _alphacBonf = multipletests(p_values, alpha=alpha, method="fdr_bh")
    return rejected, adj


def _importance(effect: float, n: int, novelty: float) -> float:
    return float(abs(effect) * np.log10(max(n, 10)) * novelty)


def _min_effect_ok(kind: str, effect: float, cfg: AppConfig) -> bool:
    floors = cfg.insights["thresholds"]["min_effect"]
    mapping = {
        "segment_difference": floors["cliffs_delta"],
        "trend": floors["slope_r2"],
        "association": floors["cramers_v"],
        "composition_shift": floors["composition_pp"],
        "concentration": floors["hhi"],
        "gap": 0.0,
        "outlier": 0.0,
    }
    return abs(effect) >= float(mapping.get(kind, 0.0))


def generate_candidates(frame: pd.DataFrame, cfg: AppConfig) -> list[dict[str, Any]]:
    min_n = int(cfg.insights["thresholds"]["min_support"])
    min_trend = int(cfg.insights["thresholds"]["min_trend_points"])
    candidates: list[dict[str, Any]] = []
    work = frame.copy()
    if "year_added" in work.columns:
        work["_year"] = pd.to_numeric(work["year_added"], errors="coerce")
    else:
        work["_year"] = np.nan

    # Trends
    trend_metrics = ["years_to_platform", "catalogue_prominence", "is_likely_original", "is_movie"]
    for metric in trend_metrics:
        if metric not in work.columns:
            continue
        yearly = work.dropna(subset=["_year"]).groupby("_year")[metric].mean()
        if len(yearly) < min_trend:
            continue
        tr = trend_regression(yearly.index.to_numpy(dtype=float), yearly.to_numpy(dtype=float))
        if tr["n"] < min_trend:
            continue
        direction = "increased" if tr["slope"] > 0 else "decreased"
        start_val, end_val = float(yearly.iloc[0]), float(yearly.iloc[-1])
        pct = (end_val - start_val) / start_val if start_val else 0.0
        candidates.append(
            {
                "kind": "trend",
                "theme": (
                    "acquisition" if "year" in metric or "original" in metric else "composition"
                ),
                "segment": "catalogue",
                "metric": metric,
                "p": tr["p"],
                "effect": tr["r2"],
                "effect_name": "R2",
                "n": int(tr["n"]),
                "values": {
                    "metric": metric,
                    "segment": "the full catalogue",
                    "direction": direction,
                    "start_val": start_val,
                    "end_val": end_val,
                    "start_year": int(yearly.index.min()),
                    "end_year": int(yearly.index.max()),
                    "pct_change": pct,
                    "r2": tr["r2"],
                    "n": int(work["_year"].notna().sum()),
                },
            }
        )

    # Segment differences
    segment_cols = [c for c in cfg.insights["segments"]["by"] if c in work.columns]
    metrics = [m for m in cfg.insights["segments"]["metrics"] if m in work.columns]
    for seg_col in segment_cols:
        levels = [lv for lv in work[seg_col].dropna().astype(str).unique() if lv and lv != "nan"]
        if len(levels) > 12:
            top = work[seg_col].astype(str).value_counts().head(8).index.tolist()
            levels = [lv for lv in levels if lv in top]
            for metric in metrics:
                if metric == seg_col:
                    continue
                # Skip comparisons that are true by definition (the segment is the metric).
                if (seg_col, metric) in {
                    ("type", "is_movie"),
                    ("acquisition_bucket", "years_to_platform"),
                    ("acquisition_bucket", "is_likely_original"),
                }:
                    continue
            series = pd.to_numeric(work[metric], errors="coerce")
            for a, b in combinations(sorted(levels), 2):
                ga = series[work[seg_col].astype(str) == a].dropna().to_numpy()
                gb = series[work[seg_col].astype(str) == b].dropna().to_numpy()
                if len(ga) < min_n or len(gb) < min_n:
                    continue
                test = two_group_numeric(ga, gb)
                effect = cliffs_delta(ga, gb)
                d = cohens_d(ga, gb)
                direction = "higher" if np.mean(ga) > np.mean(gb) else "lower"
                candidates.append(
                    {
                        "kind": "segment_difference",
                        "theme": (
                            "geography"
                            if "country" in seg_col
                            else "genre" if "genre" in seg_col else "composition"
                        ),
                        "segment": f"{seg_col}:{a}|{b}",
                        "metric": metric,
                        "p": float(test["p"]),
                        "effect": abs(effect),
                        "effect_name": "cliffs_delta",
                        "n": int(len(ga) + len(gb)),
                        "values": {
                            "metric": metric,
                            "direction": direction,
                            "segment_a": a,
                            "segment_b": b,
                            "mean_a": float(np.mean(ga)),
                            "mean_b": float(np.mean(gb)),
                            "effect_name": "Cliff's delta",
                            "effect": abs(effect),
                            "cohens_d": d,
                            "n": int(len(ga) + len(gb)),
                        },
                    }
                )

    # Composition shifts (pre-2019 vs 2019+)
    if work["_year"].notna().any():
        early = work[work["_year"] < 2019]
        late = work[work["_year"] >= 2019]
        for col in ("type", "maturity_level", "primary_country", "primary_genre"):
            if col not in work.columns:
                continue
            for level in work[col].dropna().astype(str).value_counts().head(10).index:
                n_e = int((early[col].astype(str) == level).sum())
                n_l = int((late[col].astype(str) == level).sum())
                if (n_e + n_l) < min_n:
                    continue
                table = np.array(
                    [
                        [n_e, max(len(early) - n_e, 0)],
                        [n_l, max(len(late) - n_l, 0)],
                    ]
                )
                if table.min() < 0:
                    continue
                chi2, p = chi_square(table)
                share_e = n_e / max(len(early), 1)
                share_l = n_l / max(len(late), 1)
                delta = share_l - share_e
                candidates.append(
                    {
                        "kind": "composition_shift",
                        "theme": "composition",
                        "segment": f"{col}:{level}",
                        "metric": f"share_{col}",
                        "p": p,
                        "effect": abs(delta),
                        "effect_name": "share_delta",
                        "n": int(len(early) + len(late)),
                        "values": {
                            "segment": f"{level} ({col})",
                            "direction": "rose" if delta > 0 else "fell",
                            "start_share": share_e,
                            "end_share": share_l,
                            "start_period": "pre-2019",
                            "end_period": "2019+",
                            "delta_pp": delta * 100,
                            "n": int(len(early) + len(late)),
                        },
                    }
                )

    # Associations
    cat_pairs = [
        ("type", "maturity_level"),
        ("type", "acquisition_bucket"),
        ("maturity_level", "acquisition_bucket"),
        ("primary_country", "type"),
    ]
    for a, b in cat_pairs:
        if a not in work.columns or b not in work.columns:
            continue
        tab = pd.crosstab(work[a].astype(str), work[b].astype(str))
        if tab.size < 4 or tab.values.sum() < min_n:
            continue
        chi2, p = chi_square(tab.values)
        v = cramers_v(chi2, int(tab.values.sum()), tab.shape[0], tab.shape[1])
        candidates.append(
            {
                "kind": "association",
                "theme": "composition",
                "segment": f"{a}×{b}",
                "metric": f"{a}_vs_{b}",
                "p": p,
                "effect": v,
                "effect_name": "cramers_v",
                "n": int(tab.values.sum()),
                "values": {"var_a": a, "var_b": b, "effect": v, "n": int(tab.values.sum())},
            }
        )

    # Concentration
    for col, dim in (("primary_country", "country"), ("primary_genre", "genre")):
        if col not in work.columns:
            continue
        counts = (
            work[col]
            .astype(str)
            .replace({"": np.nan, "nan": np.nan})
            .dropna()
            .value_counts()
            .to_numpy()
        )
        if counts.sum() < min_n:
            continue
        candidates.append(
            {
                "kind": "concentration",
                "theme": "concentration",
                "segment": dim,
                "metric": f"hhi_{dim}",
                "p": 0.0,  # descriptive; still goes through FDR with p=0 so it can survive if effect floor passes
                "effect": hhi(counts),
                "effect_name": "hhi",
                "n": int(counts.sum()),
                "values": {
                    "dimension": dim,
                    "top_n": 5,
                    "top_share": top_n_share(counts, 5),
                    "hhi": hhi(counts),
                    "gini": gini(counts),
                    "n": int(counts.sum()),
                },
            }
        )

    # Gaps: genre × country cells vs independence
    if "primary_genre" in work.columns and "primary_country" in work.columns:
        tab = pd.crosstab(work["primary_genre"].astype(str), work["primary_country"].astype(str))
        tab = tab.loc[tab.sum(axis=1) >= min_n, tab.sum(axis=0) >= min_n]
        if tab.shape[0] >= 2 and tab.shape[1] >= 2:
            expected = np.outer(tab.sum(axis=1), tab.sum(axis=0)) / tab.values.sum()
            residual = tab.values - expected
            chi2, p = chi_square(tab.values)
            flat = residual.flatten()
            idx = int(np.argmin(flat))
            i, j = divmod(idx, residual.shape[1])
            genre = tab.index[i]
            country = tab.columns[j]
            if residual[i, j] <= -float(cfg.insights["thresholds"]["gap_residual"]):
                candidates.append(
                    {
                        "kind": "gap",
                        "theme": "gaps",
                        "segment": f"{genre}×{country}",
                        "metric": "independence_residual",
                        "p": p,
                        "effect": abs(float(residual[i, j])),
                        "effect_name": "residual",
                        "n": int(tab.values[i, j]),
                        "values": {
                            "genre": genre,
                            "country": country,
                            "residual": float(residual[i, j]),
                            "n": int(tab.values[i, j]),
                        },
                    }
                )

    # Outliers on years_to_platform by country
    if "primary_country" in work.columns and "years_to_platform" in work.columns:
        overall = pd.to_numeric(work["years_to_platform"], errors="coerce").dropna()
        mu, sd = float(overall.mean()), float(overall.std() or 1.0)
        for country, sub in work.groupby(work["primary_country"].astype(str)):
            vals = pd.to_numeric(sub["years_to_platform"], errors="coerce").dropna()
            if len(vals) < min_n or not country or country == "nan":
                continue
            z = (float(vals.mean()) - mu) / sd
            if abs(z) >= float(cfg.insights["thresholds"]["outlier_z"]):
                candidates.append(
                    {
                        "kind": "outlier",
                        "theme": "geography",
                        "segment": country,
                        "metric": "years_to_platform",
                        "p": float(2 * (1 - norm.cdf(abs(z)))),
                        "effect": abs(z),
                        "effect_name": "z",
                        "n": int(len(vals)),
                        "values": {
                            "segment": country,
                            "metric": "years_to_platform",
                            "z": z,
                            "value": float(vals.mean()),
                            "n": int(len(vals)),
                        },
                    }
                )
    return candidates


def deduplicate(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the strongest finding per (kind, metric, nested segment)."""
    kept: list[dict[str, Any]] = []
    for item in sorted(findings, key=lambda f: abs(f.get("effect", 0)), reverse=True):
        nest = False
        for other in kept:
            if item["kind"] == other["kind"] and item["metric"] == other["metric"]:
                if item["segment"] in other["segment"] or other["segment"] in item["segment"]:
                    nest = True
                    break
        if not nest:
            kept.append(item)
    return kept


def run_insight_engine(cfg: AppConfig, frame: pd.DataFrame | None = None) -> dict[str, Any]:
    if frame is None:
        frame, _reg = read_feature_store(cfg.path("features_dir"))
    raw_candidates = generate_candidates(frame, cfg)
    min_n = int(cfg.insights["thresholds"]["min_support"])
    filtered = [
        c for c in raw_candidates if c.get("n", 0) >= min_n and np.isfinite(c.get("p", np.nan))
    ]
    filtered = [c for c in filtered if _min_effect_ok(c["kind"], float(c.get("effect", 0.0)), cfg)]
    rejected, adj = bh_fdr(
        [float(c["p"]) for c in filtered], float(cfg.insights["thresholds"]["fdr_alpha"])
    )
    survivors = []
    for cand, keep, q in zip(filtered, rejected, adj, strict=False):
        cand = dict(cand)
        cand["q"] = float(q)
        cand["raw_p"] = float(cand["p"])
        if keep:
            values = dict(cand["values"])
            values["q_value"] = float(q)
            text = render(cand["kind"], values)
            cand["text"] = text
            cand["importance"] = _importance(float(cand["effect"]), int(cand["n"]), 1.0)
            survivors.append(cand)
    survivors = deduplicate(survivors)
    survivors.sort(key=lambda f: f["importance"], reverse=True)
    stoplist = list(cfg.insights["causal_stoplist"])
    assert_no_causal_language([s["text"] for s in survivors], stoplist)

    payload = {
        "n_candidates": len(raw_candidates),
        "n_after_floors": len(filtered),
        "n_survivors": len(survivors),
        "fdr_alpha": cfg.insights["thresholds"]["fdr_alpha"],
        "min_support": min_n,
        "min_effect": cfg.insights["thresholds"]["min_effect"],
        "findings": survivors,
        "methodology": (
            "Mann-Whitney U (default) or Welch t after a Shapiro/skew screen; "
            "χ² / Fisher for categorical associations; OLS + F-test for trends. "
            "Benjamini-Hochberg FDR across the whole candidate set. "
            "A finding must clear both q ≤ α and an effect-size floor."
        ),
    }
    art = cfg.path("artifacts_dir")
    art.mkdir(parents=True, exist_ok=True)
    (art / "insights.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    _write_insights_md(cfg.path("reports_dir") / "insights.md", payload)
    log.info(
        "insights_complete",
        n_candidates=payload["n_candidates"],
        n_survivors=payload["n_survivors"],
    )
    return payload


def _write_insights_md(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Automated insights",
        "",
        f"**{payload['n_candidates']} candidate findings tested, {payload['n_survivors']} survived "
        f"BH-FDR at q = {payload['fdr_alpha']} with the configured effect-size floor.**",
        "",
        "## Methodology",
        "",
        payload["methodology"],
        "",
        f"Minimum cell support: {payload['min_support']}. Effect-size floors: {payload['min_effect']}.",
        "",
        "Causal language is banned. Sentences use “is associated with”, never “causes”, “drives”, or “leads to”.",
        "",
    ]
    by_theme: dict[str, list] = {}
    for finding in payload["findings"]:
        by_theme.setdefault(finding.get("theme", "other"), []).append(finding)
    for theme, items in by_theme.items():
        lines += [f"## {theme}", ""]
        for item in items:
            lines.append(f"- {item['text']}")
            lines.append(
                f"  - kind=`{item['kind']}` effect={item['effect']:.3f} "
                f"({item['effect_name']}) raw p={item['raw_p']:.4g} q={item['q']:.4g} n={item['n']}"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
