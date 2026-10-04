"""Build proxy targets and write reports/target_definitions.md from computed statistics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.config import AppConfig
from src.data.schema import assign_acquisition_bucket, dropped_ratings, maturity_map, maturity_order
from src.logging_conf import get_logger
from src.targets.definitions import PROXY_NOTES

log = get_logger(__name__)

# Titles independently identifiable as Netflix Originals or clearly licensed.
# Used only for a heuristic spot-check of is_likely_original — not as training labels.
SPOTCHECK_ORIGINALS = [
    "Stranger Things",
    "The Crown",
    "The Witcher",
    "The Queen's Gambit",
    "Bird Box",
    "Roma",
    "The Irishman",
    "Extraction",
    "Don't Look Up",
    "Red Notice",
    "Squid Game",
    "Bridgerton",
    "Lupin",
    "Dark",
    "Sex Education",
    "Money Heist",
    "Orange Is the New Black",
    "Midnight Mass",
    "The Platform",
    "Our Planet",
]
SPOTCHECK_LICENSED = [
    "Friends",
    "The Office",
    "Breaking Bad",
    "Seinfeld",
    "Inception",
    "The Dark Knight",
    "Forrest Gump",
    "Goodfellas",
    "Jurassic Park",
    "Titanic",
    "The Matrix",
    "Pulp Fiction",
    "The Godfather",
    "Avatar",
    "The Shawshank Redemption",
    "Gladiator",
    "Fight Club",
    "The Departed",
    "Saving Private Ryan",
    "Schindler's List",
]


def add_targets(frame: pd.DataFrame, cfg: AppConfig) -> pd.DataFrame:
    out = frame.copy()
    mapping = maturity_map(cfg)
    drop = dropped_ratings(cfg)
    out["maturity_level"] = out["rating"].map(
        lambda r: mapping.get(str(r)) if str(r) not in drop else pd.NA
    )
    out["is_likely_original"] = (out["years_to_platform"] <= 0).fillna(False)
    out["acquisition_bucket"] = out["years_to_platform"].map(
        lambda y: assign_acquisition_bucket(float(y), cfg) if pd.notna(y) else pd.NA
    )
    return out


def _minmax(series: pd.Series, lo: float, hi: float) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").astype(float)
    clipped = numeric.clip(lower=lo, upper=hi)
    span = hi - lo
    if span <= 0:
        return pd.Series(0.0, index=series.index)
    return (clipped - lo) / span


def add_catalogue_prominence(frame: pd.DataFrame, cfg: AppConfig) -> pd.DataFrame:
    weights: dict[str, float] = dict(cfg.features["catalogue_prominence"]["weights"])
    clips: dict[str, list[float]] = dict(cfg.features["catalogue_prominence"]["clip_quantiles"])
    out = frame.copy()
    components = {
        "n_genres": out["n_genres"].astype(float),
        "n_countries": out["n_countries"].astype(float),
        "has_director": out["has_director"].astype(float),
        "n_directors": out["n_directors"].astype(float),
        "n_cast": out["n_cast"].astype(float),
        "description_length": out["description_length"].astype(float),
    }
    score = pd.Series(0.0, index=out.index)
    for name, series in components.items():
        weight = float(weights.get(name, 0.0))
        if name == "has_director":
            normed = series.clip(0, 1)
        else:
            qlo, qhi = clips.get(name, [0.0, 1.0])
            lo = float(series.quantile(qlo)) if series.notna().any() else 0.0
            hi = float(series.quantile(qhi)) if series.notna().any() else 1.0
            if hi <= lo:
                hi = lo + 1.0
            normed = _minmax(series, lo, hi)
        score = score + weight * normed.fillna(0.0)
    out["catalogue_prominence"] = score
    return out


def spotcheck_originals(frame: pd.DataFrame) -> dict[str, Any]:
    rows = []
    for title, expected in [(t, True) for t in SPOTCHECK_ORIGINALS] + [
        (t, False) for t in SPOTCHECK_LICENSED
    ]:
        hit = frame[frame["title"].astype(str).str.casefold() == title.casefold()]
        if hit.empty:
            rows.append({"title": title, "expected_original": expected, "found": False})
            continue
        row = hit.iloc[0]
        pred = bool(row["is_likely_original"])
        rows.append(
            {
                "title": title,
                "expected_original": expected,
                "found": True,
                "predicted": pred,
                "years_to_platform": row.get("years_to_platform"),
                "correct": pred == expected,
            }
        )
    found = [r for r in rows if r.get("found")]
    correct = [r for r in found if r.get("correct")]
    return {
        "n_looked_up": len(rows),
        "n_found": len(found),
        "n_correct": len(correct),
        "hit_rate": (len(correct) / len(found)) if found else None,
        "rows": rows,
    }


def write_target_definitions(
    frame: pd.DataFrame, cfg: AppConfig, path: Path, spotcheck: dict[str, Any]
) -> None:
    ytp = pd.to_numeric(frame["years_to_platform"], errors="coerce")
    buckets = frame["acquisition_bucket"].value_counts(dropna=False).to_dict()
    drop = dropped_ratings(cfg)
    n_nr = int(frame["rating"].isin(drop).sum())
    n_neg = int((ytp < 0).fillna(False).sum())
    n_zero = int((ytp == 0).fillna(False).sum())
    weights = cfg.features["catalogue_prominence"]["weights"]
    hit = spotcheck.get("hit_rate")
    hit_txt = f"{hit:.0%}" if hit is not None else "n/a (no spot-check titles found)"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""# Target definitions — what this platform can and cannot measure

This platform analyses **catalogue composition and platform acquisition behaviour**.
It does **not** measure audience success.

The shipped file is a Netflix **catalogue listing** of {len(frame):,} titles.
It contains `show_id`, `type`, `title`, `director`, `country`, `date_added`,
`release_year`, `rating`, `duration`, `listed_in`. The public Kaggle listing
also has `cast` and `description`; those columns are **absent** from the shipped
file and are treated as empty. The file contains **no** view count, watch time,
completion rate, revenue, subscriber impact, IMDb / Rotten Tomatoes score,
award record, budget, or renewal status.

**Success is not measurable from this file.** Every target below is an
explicitly labelled **proxy**. Required fields for a real success model are
listed in `docs/DATA_ROADMAP.md`.

---

## 1. `years_to_platform` — acquisition lag (strongest genuine signal)

`years_to_platform = year_added − release_year`.

| statistic | value |
|---|---|
| n with a parseable add-year | {int(ytp.notna().sum())} / {len(frame)} |
| mean | {float(ytp.mean()):.2f} |
| median | {float(ytp.median()):.2f} |
| std | {float(ytp.std()):.2f} |
| min | {float(ytp.min()):.0f} |
| max | {float(ytp.max()):.0f} |
| negative (not silently clipped to 0) | {n_neg} |
| zero (added in release year) | {n_zero} |
| floor applied | {cfg.base["dataset"]["years_to_platform_floor"]} |
| below-floor rows capped | {int((frame["years_to_platform_raw"] < cfg.base["dataset"]["years_to_platform_floor"]).fillna(False).sum()) if "years_to_platform_raw" in frame.columns else 0} |

**Label: acquisition priority. Never success.**
Proxy note: {PROXY_NOTES["years_to_platform"]}

## 2. Acquisition-priority buckets

{chr(10).join(f"- **{k}**: {v}" for k, v in buckets.items())}

## 3. `is_likely_original` (heuristic)

`years_to_platform <= 0`. Share flagged: {float(frame["is_likely_original"].mean()):.1%}.

Spot-check of independently identifiable titles: {spotcheck.get("n_found")} found of {spotcheck.get("n_looked_up")} looked up; {spotcheck.get("n_correct")} agreed with the hand label; hit rate **{hit_txt}**. This is a heuristic quality number, not ground-truth accuracy.

Proxy note: {PROXY_NOTES["is_likely_original"]}

## 4. `catalogue_prominence` (constructed index)

Weights from `config/features.yaml`: {weights}.
`n_cast` and `description_length` are zero on this file because those columns are absent.
Mean prominence: {float(frame["catalogue_prominence"].mean()):.3f}. Median: {float(frame["catalogue_prominence"].median()):.3f}.

Proxy note: {PROXY_NOTES["catalogue_prominence"]}

## 5. `portfolio_fit`

Filled after Head E. Distance to cluster centroid; not a quality score.

## 6. Observed (still not success) labels

- Head A: `type` / `is_movie` — leakage-banned duration and TV-prefixed genres.
- Head B: `maturity_level` ladder {maturity_order(cfg)}. NR/UR dropped: {n_nr} rows.
- Head F: monthly titles-added from `date_added`. Structural break in 2020.

## 7. Labelling contract

API responses carry `is_proxy: true` and a `proxy_note` on every proxy field.
""",
        encoding="utf-8",
    )


def write_prominence_sanity(frame: pd.DataFrame, path: Path) -> list[str]:
    top = frame.sort_values("catalogue_prominence", ascending=False).head(20)
    titles = top["title"].astype(str).tolist()
    lines = [
        "# Catalogue prominence sanity check",
        "",
        "Top 20 titles by the constructed `catalogue_prominence` index.",
        "This is a metadata-richness ranking, not an audience ranking.",
        "",
        "| rank | title | type | year | n_genres | n_countries | has_director | prominence |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for i, row in enumerate(top.itertuples(index=False), start=1):
        lines.append(
            f"| {i} | {row.title} | {row.type} | {row.release_year} | {row.n_genres} | "
            f"{row.n_countries} | {row.has_director} | {row.catalogue_prominence:.3f} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return titles
