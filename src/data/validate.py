"""Hard validation gate. A silent pass on bad data is the failure mode this layer exists to prevent."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from pandera.errors import SchemaError, SchemaErrors

from src.config import AppConfig
from src.data.schema import (
    is_missing_token,
    null_rate_tolerances,
    rating_vocabulary,
    raw_schema,
    release_year_bounds,
)
from src.logging_conf import get_logger

log = get_logger(__name__)


class ValidationGateError(Exception):
    """Raised when the catalogue fails a hard validation check."""


def _is_null(series: pd.Series, cfg: AppConfig) -> pd.Series:
    return series.map(lambda v: is_missing_token(v, cfg))


def collect_issues(frame: pd.DataFrame, cfg: AppConfig) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    expected = list(cfg.base["dataset"]["expected_columns"])
    for col in expected:
        if col not in frame.columns:
            issues.append({"check": "missing_column", "column": col, "ok": False, "detail": "column absent"})

    if "show_id" in frame.columns:
        null_ids = int(frame["show_id"].isna().sum() + (frame["show_id"].astype(str).str.strip() == "").sum())
        dups = int(frame["show_id"].duplicated().sum())
        if null_ids:
            issues.append({"check": "show_id_null", "ok": False, "n": null_ids})
        if dups and cfg.base["validation"]["fail_on_duplicate_show_id"]:
            issues.append({"check": "duplicate_show_id", "ok": False, "n": dups})

    if "type" in frame.columns:
        allowed = set(cfg.base["dataset"]["allowed_types"])
        bad = frame.loc[~frame["type"].isin(allowed), "type"].astype(str).unique().tolist()
        if bad:
            issues.append({"check": "type_vocabulary", "ok": False, "values": bad})

    if "release_year" in frame.columns:
        year_min, year_max = release_year_bounds(cfg)
        years = pd.to_numeric(frame["release_year"], errors="coerce")
        oob = int(((years < year_min) | (years > year_max) | years.isna()).sum())
        if oob:
            issues.append(
                {
                    "check": "release_year_range",
                    "ok": False,
                    "n": oob,
                    "min": year_min,
                    "max": year_max,
                }
            )

    vocab = set(rating_vocabulary(cfg))
    if "rating" in frame.columns:
        ratings = frame["rating"].astype("string")
        unknown_mask = ~ratings.isna() & ratings.str.strip().ne("") & ~ratings.isin(vocab)
        unknown = ratings[unknown_mask].dropna().unique().tolist()
        if unknown and not cfg.base["validation"]["quarantine_unknown_ratings"]:
            issues.append({"check": "unknown_rating", "ok": False, "values": unknown})
        elif unknown:
            issues.append({"check": "unknown_rating_quarantined", "ok": True, "values": unknown})

    tolerances = null_rate_tolerances(cfg)
    n = max(len(frame), 1)
    for col, max_rate in tolerances.items():
        if col not in frame.columns:
            continue
        rate = float(_is_null(frame[col], cfg).mean()) if n else 0.0
        if rate > max_rate:
            issues.append(
                {
                    "check": "null_rate",
                    "column": col,
                    "ok": False,
                    "rate": rate,
                    "max": max_rate,
                    "n": int(_is_null(frame[col], cfg).sum()),
                }
            )

    try:
        raw_schema(cfg).validate(frame, lazy=True)
    except (SchemaError, SchemaErrors) as exc:
        issues.append({"check": "pandera_schema", "ok": False, "detail": str(exc)})

    return issues


def gate_fails(issues: list[dict[str, Any]]) -> bool:
    return any(not item.get("ok", True) for item in issues)


def write_validation_html(path: Path, issues: list[dict[str, Any]], n_rows: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    failed = [i for i in issues if not i.get("ok", True)]
    status = "FAIL" if failed else "PASS"
    rows = []
    for item in issues:
        rows.append(
            "<tr>"
            f"<td>{item.get('check')}</td>"
            f"<td>{'PASS' if item.get('ok', True) else 'FAIL'}</td>"
            f"<td><code>{item}</code></td>"
            "</tr>"
        )
    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>Validation report — {status}</title>
<style>
body {{ font-family: Georgia, serif; margin: 2rem; color: #111; }}
.banner {{ padding: 1rem 1.25rem; border-radius: 6px; color: #fff;
  background: {"#8B0000" if failed else "#0B3D2E"}; }}
table {{ border-collapse: collapse; width: 100%; margin-top: 1.5rem; }}
th, td {{ border: 1px solid #ccc; padding: 0.5rem 0.75rem; text-align: left; }}
th {{ background: #f4f1ea; }}
code {{ font-size: 0.85rem; }}
</style></head><body>
<div class="banner"><h1>Validation gate: {status}</h1>
<p>Rows inspected: {n_rows}. Failed checks: {len(failed)}.</p></div>
<table><thead><tr><th>Check</th><th>Result</th><th>Detail</th></tr></thead>
<tbody>{''.join(rows) if rows else '<tr><td colspan="3">All configured checks passed.</td></tr>'}</tbody>
</table>
<p>A silent pass on bad data is the failure mode this layer exists to prevent.</p>
</body></html>"""
    path.write_text(html, encoding="utf-8")


def run_validation_gate(frame: pd.DataFrame, cfg: AppConfig, report_path: Path) -> list[dict[str, Any]]:
    issues = collect_issues(frame, cfg)
    write_validation_html(report_path, issues, len(frame))
    if gate_fails(issues):
        failed = [i for i in issues if not i.get("ok", True)]
        log.error("validation_gate_failed", n_failed=len(failed), issues=failed)
        raise ValidationGateError(f"Validation gate failed ({len(failed)} checks): {failed}")
    log.info("validation_gate_passed", n_checks=len(issues), n_rows=len(frame))
    return issues
