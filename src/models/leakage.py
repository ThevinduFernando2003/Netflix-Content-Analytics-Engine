"""Mutual-information leakage audit, especially for Head A."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder

from src.features.blocks import genre_allowed_for_type
from src.features.registry import banned_columns
from src.logging_conf import get_logger

log = get_logger(__name__)


def audit_type_leakage(
    frame: pd.DataFrame,
    used_features: list[str],
    registry: dict,
    cfg,
    path: Path,
) -> dict[str, float]:
    banned = set(banned_columns(registry, "type"))
    leaked = sorted(set(used_features) & banned)
    y = frame["is_movie"].astype(int).to_numpy()
    scores: dict[str, float] = {}
    numericish = []
    for col in used_features:
        if col not in frame.columns:
            continue
        series = frame[col]
        if pd.api.types.is_numeric_dtype(series):
            numericish.append(col)
            x = pd.to_numeric(series, errors="coerce").fillna(0).to_numpy().reshape(-1, 1)
            scores[col] = float(mutual_info_classif(x, y, random_state=cfg.seed)[0])
        else:
            enc = LabelEncoder()
            x = enc.fit_transform(series.astype(str).fillna("")).reshape(-1, 1)
            scores[col] = float(mutual_info_classif(x, y, random_state=cfg.seed)[0])
    tv_tokens = []
    if "listed_in" in frame.columns:
        for genres in frame["listed_in"].astype(str):
            for g in [p.strip() for p in genres.split(",") if p.strip()]:
                if not genre_allowed_for_type(g, cfg) and g not in tv_tokens:
                    tv_tokens.append(g)
    path.parent.mkdir(parents=True, exist_ok=True)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    lines = [
        "# Leakage audit — Head A (content type)",
        "",
        "Duration columns and TV-prefixed genre tokens trivially determine Movie vs TV Show.",
        "If any of those appear in the fitted feature list, this audit is a failure.",
        "",
        f"- Banned features that reached the head: `{leaked or 'none'}`",
        f"- TV-prefixed / banned genre tokens present in raw `listed_in` (must be stripped before Head A): {len(tv_tokens)}",
        "",
        "## Mutual information with `is_movie`",
        "",
        "| feature | MI |",
        "|---|---|",
    ]
    for name, score in ranked[:40]:
        lines.append(f"| {name} | {score:.4f} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if leaked:
        raise ValueError(f"Head A leakage: banned features present: {leaked}")
    return scores
