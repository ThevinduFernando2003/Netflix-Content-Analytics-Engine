"""Write reports/data_profile.md from the raw listing. Every number is computed."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import AppConfig
from src.data.clean import DURATION_IN_RATING, MINUTES, SEASONS, parse_date_added, split_list
from src.data.schema import is_missing_token


def write_data_profile(raw: pd.DataFrame, cfg: AppConfig, path: Path) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(raw)
    dtypes = {c: str(t) for c, t in raw.dtypes.items()}
    nulls = {}
    for col in raw.columns:
        n_null = int(raw[col].map(lambda v: is_missing_token(v, cfg)).sum())
        nulls[col] = {"n": n_null, "pct": 100.0 * n_null / n if n else 0.0}
    dup_ids = int(raw["show_id"].duplicated().sum()) if "show_id" in raw.columns else 0
    type_counts = raw["type"].value_counts(dropna=False).to_dict() if "type" in raw.columns else {}
    rating_counts = raw["rating"].value_counts(dropna=False).to_dict() if "rating" in raw.columns else {}
    genres: list[str] = []
    if "listed_in" in raw.columns:
        for value in raw["listed_in"].tolist():
            genres.extend(split_list(value))
    genre_vocab = pd.Series(genres).value_counts().to_dict() if genres else {}
    year_vc = raw["release_year"].value_counts().sort_index().to_dict() if "release_year" in raw.columns else {}
    parsed, date_fail = parse_date_added(raw["date_added"].astype("string"), cfg)
    duration_fail = 0
    contaminated = []
    if "rating" in raw.columns:
        mask = raw["rating"].astype("string").fillna("").str.match(DURATION_IN_RATING)
        contaminated = raw.loc[mask, "show_id"].astype(str).tolist()
    if "duration" in raw.columns:
        for value in raw["duration"].astype("string").fillna(""):
            text = str(value).strip()
            if text and not MINUTES.match(text) and not SEASONS.match(text):
                duration_fail += 1
    lines = [
        "# Data profile",
        "",
        "Every number in this file is computed by `src/data/profile.py` from the raw listing.",
        "",
        f"- **rows**: {n}",
        f"- **columns**: {list(raw.columns)}",
        f"- **duplicate show_id**: {dup_ids}",
        f"- **date_added parse failures / missing**: {date_fail}",
        f"- **date_added min / max**: {parsed.min()} / {parsed.max()}",
        f"- **duration parse failures**: {duration_fail}",
        f"- **rating values that look like duration**: {contaminated}",
        "",
        "## dtypes",
        "",
        "| column | dtype |",
        "|---|---|",
    ]
    for col, dtype in dtypes.items():
        lines.append(f"| {col} | {dtype} |")
    lines += ["", "## nulls (including configured missing tokens such as `Not Given`)", "", "| column | n | pct |", "|---|---|---|"]
    for col, rec in nulls.items():
        lines.append(f"| {col} | {rec['n']} | {rec['pct']:.2f}% |")
    lines += ["", "## type value counts", ""]
    for k, v in type_counts.items():
        lines.append(f"- `{k}`: {v}")
    lines += ["", "## rating value counts", ""]
    for k, v in rating_counts.items():
        lines.append(f"- `{k}`: {v}")
    lines += ["", "## exploded genre vocabulary", ""]
    for k, v in list(genre_vocab.items())[:80]:
        lines.append(f"- `{k}`: {v}")
    lines += ["", "## release_year distribution (year: count)", ""]
    for k, v in year_vc.items():
        lines.append(f"- {k}: {v}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "n_rows": n,
        "n_cols": len(raw.columns),
        "duplicate_show_ids": dup_ids,
        "date_added_failures": date_fail,
        "duration_failures": duration_fail,
        "rating_duration_show_ids": contaminated,
    }
