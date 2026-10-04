"""Feature, target and performance drift using a pre-2019 / 2019+ split."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import AppConfig
from src.features.store import read_feature_store
from src.logging_conf import get_logger

log = get_logger(__name__)


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index. Identical distributions → ~0."""
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]
    if len(expected) == 0 or len(actual) == 0:
        return 0.0
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(expected, quantiles))
    if len(edges) < 3:
        return 0.0
    e_hist, _ = np.histogram(expected, bins=edges)
    a_hist, _ = np.histogram(actual, bins=edges)
    e_prop = (e_hist + 1e-6) / (e_hist.sum() + 1e-6 * len(e_hist))
    a_prop = (a_hist + 1e-6) / (a_hist.sum() + 1e-6 * len(a_hist))
    return float(np.sum((a_prop - e_prop) * np.log(a_prop / e_prop)))


def ks_pvalue(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import ks_2samp

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 5 or len(b) < 5:
        return 1.0
    return float(ks_2samp(a, b).pvalue)


def _period_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    year = pd.to_numeric(frame.get("year_added"), errors="coerce")
    ref = frame[year < 2019]
    cur = frame[year >= 2019]
    return ref, cur


def _perf_degradation(ref: pd.DataFrame, cur: pd.DataFrame, seed: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    feats = [
        c
        for c in ("release_year", "n_genres", "n_countries", "has_director", "title_length")
        if c in ref.columns
    ]

    def _xy(df: pd.DataFrame, target: str) -> tuple[np.ndarray, pd.Series]:
        x = df[feats].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy()
        y = df[target]
        return x, y

    # Type head
    if "is_movie" in ref.columns and len(ref) > 30 and len(cur) > 30:
        x_r, y_r_s = _xy(ref, "is_movie")
        x_c, y_c_s = _xy(cur, "is_movie")
        y_r = y_r_s.astype(int).to_numpy()
        y_c = y_c_s.astype(int).to_numpy()
        x_tr, x_te, y_tr, y_te = train_test_split(
            x_r, y_r, test_size=0.2, random_state=seed, stratify=y_r
        )
        clf = Pipeline([("sc", StandardScaler()), ("m", LogisticRegression(max_iter=300))])
        clf.fit(x_tr, y_tr)
        random_split = float(roc_auc_score(y_te, clf.predict_proba(x_te)[:, 1]))
        temporal = float(roc_auc_score(y_c, clf.predict_proba(x_c)[:, 1]))
        out["type"] = {
            "metric": "roc_auc",
            "random_split": random_split,
            "temporal": temporal,
            "degradation": random_split - temporal,
        }

    # Acquisition regression
    if "years_to_platform" in ref.columns:
        x_r = ref[feats].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy()
        y_reg = pd.to_numeric(ref["years_to_platform"], errors="coerce")
        mask = y_reg.notna().to_numpy()
        x_r = x_r[mask]
        y_r = y_reg[mask].to_numpy()
        x_c = cur[feats].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy()
        y_c = pd.to_numeric(cur["years_to_platform"], errors="coerce").fillna(0).to_numpy()
        if len(y_r) > 30:
            x_tr, x_te, y_tr, y_te = train_test_split(x_r, y_r, test_size=0.2, random_state=seed)
            reg = Pipeline([("sc", StandardScaler()), ("m", Ridge())])
            reg.fit(x_tr, y_tr)
            random_split = float(mean_absolute_error(y_te, reg.predict(x_te)))
            temporal = float(mean_absolute_error(y_c, reg.predict(x_c)))
            out["acquisition_reg"] = {
                "metric": "mae",
                "random_split": random_split,
                "temporal": temporal,
                "degradation": temporal - random_split,
            }
    return out


def run_drift(cfg: AppConfig) -> dict[str, Any]:
    frame, _reg = read_feature_store(cfg.path("features_dir"))
    ref, cur = _period_split(frame)
    numeric_cols = [
        c
        for c in (
            "years_to_platform",
            "catalogue_prominence",
            "n_genres",
            "n_countries",
            "release_year",
            "title_length",
        )
        if c in frame.columns
    ]
    feature_drift = {}
    for col in numeric_cols:
        a = pd.to_numeric(ref[col], errors="coerce").dropna().to_numpy()
        b = pd.to_numeric(cur[col], errors="coerce").dropna().to_numpy()
        value = psi(a, b)
        feature_drift[col] = {
            "psi": value,
            "ks_p": ks_pvalue(a, b),
            "flag": "serious" if value > 0.25 else "warning" if value > 0.2 else "ok",
        }
    target_drift = {}
    for col in ("maturity_level", "acquisition_bucket"):
        if col not in frame.columns:
            continue
        ref_s = ref[col].astype(str).value_counts(normalize=True)
        cur_s = cur[col].astype(str).value_counts(normalize=True)
        keys = sorted(set(ref_s.index) | set(cur_s.index))
        e = np.array([float(ref_s.get(k, 0.0)) for k in keys])
        a = np.array([float(cur_s.get(k, 0.0)) for k in keys])
        # PSI on already-normalised category shares
        e = (e + 1e-6) / (e.sum() + 1e-6 * len(e))
        a = (a + 1e-6) / (a.sum() + 1e-6 * len(a))
        value = float(np.sum((a - e) * np.log(a / e)))
        target_drift[col] = {
            "psi": value,
            "flag": "serious" if value > 0.25 else "warning" if value > 0.2 else "ok",
        }

    perf = _perf_degradation(ref, cur, cfg.seed)
    summary = {
        "reference": "date_added year < 2019",
        "current": "date_added year >= 2019",
        "n_reference": int(len(ref)),
        "n_current": int(len(cur)),
        "feature_drift": feature_drift,
        "target_drift": target_drift,
        "performance_drift": perf,
    }
    html = _render_html(summary)
    path = cfg.path("reports_dir") / "drift_report.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    (cfg.path("artifacts_dir") / "drift.json").write_text(
        __import__("json").dumps(summary, indent=2), encoding="utf-8"
    )
    log.info("drift_complete", n_ref=len(ref), n_cur=len(cur))
    return summary


def _render_html(summary: dict[str, Any]) -> str:
    def rows(block: dict) -> str:
        out = []
        for k, v in block.items():
            if isinstance(v, dict):
                out.append(f"<tr><td>{k}</td><td><code>{v}</code></td></tr>")
        return "".join(out)

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>Drift report</title>
<style>
body {{ font-family: Georgia, serif; margin: 2rem; color: #111; background: #faf8f5; }}
h1 {{ color: #8B0000; }}
table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
th, td {{ border: 1px solid #ccc; padding: 0.45rem 0.7rem; text-align: left; }}
th {{ background: #f4f1ea; }}
</style></head><body>
<h1>Temporal drift — real date_added split</h1>
<p>Reference: {summary['reference']} (n={summary['n_reference']}).
Current: {summary['current']} (n={summary['n_current']}).</p>
<p>This is genuine temporal distribution shift, not a synthetic perturbation.
PSI &gt; 0.2 is a warning; &gt; 0.25 is serious. Performance degradation
(random-split score vs train-on-reference / evaluate-on-current) is the number
that tells you how often the model would need retraining.</p>
<h2>Feature drift</h2>
<table><thead><tr><th>feature</th><th>detail</th></tr></thead><tbody>{rows(summary['feature_drift'])}</tbody></table>
<h2>Target drift</h2>
<table><thead><tr><th>target</th><th>detail</th></tr></thead><tbody>{rows(summary['target_drift'])}</tbody></table>
<h2>Performance drift</h2>
<table><thead><tr><th>head</th><th>detail</th></tr></thead><tbody>{rows(summary['performance_drift'])}</tbody></table>
</body></html>"""
