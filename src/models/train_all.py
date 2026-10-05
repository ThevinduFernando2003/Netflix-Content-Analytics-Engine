"""Train every configured head against the shared feature store."""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance
from sklearn.model_selection import RandomizedSearchCV, train_test_split

from src.config import AppConfig
from src.data.schema import bucket_order, maturity_order
from src.features.blocks import BINARY_CANDIDATES, CATEGORICAL_CANDIDATES, NUMERIC_CANDIDATES
from src.features.registry import assert_no_banned
from src.features.store import data_sha, read_feature_store
from src.logging_conf import get_logger
from src.models.clustering import class_tfidf_terms, fit_clusters
from src.models.forecast import fit_forecast
from src.models.leakage import audit_type_leakage
from src.models.metrics import brier, classification_metrics, ordinal_bundle, regression_metrics
from src.models.pipelines import make_estimator, make_preprocessor, wrap
from src.pipeline.cache import git_sha
from src.tracking.mlflow_utils import log_mlflow_run, setup_mlflow, write_run_record

log = get_logger(__name__)

ID_COLS = {
    "show_id",
    "title",
    "directors",
    "cast_list",
    "countries",
    "genres",
    "date_added_parsed",
    "listed_in",
    "description",
    "director",
    "country",
    "portfolio_fit",
    "cluster_id",
    "years_to_platform_raw",
}


def _n_iter(cfg: AppConfig) -> int:
    defaults = cfg.models["defaults"]
    return int(defaults["n_iter_fast"] if cfg.settings.fast_mode else defaults["n_iter"])


def _param_grid(cfg: AppConfig, candidate: str) -> dict[str, list[Any]]:
    raw = cfg.models["search_spaces"].get(candidate, {})
    return {f"model__{k}": v for k, v in raw.items() if v}


def _head_xy(
    frame: pd.DataFrame,
    cfg: AppConfig,
    head: str,
    registry: dict[str, Any],
) -> tuple[pd.DataFrame, pd.Series, list[str], list[str], str | None]:
    spec = cfg.models["heads"][head]
    target = spec.get("target")
    exclude = set(ID_COLS)
    if target:
        exclude.add(target)
    if head == "type":
        exclude.update({"is_movie", "type", "primary_genre"})
    if head == "rating":
        exclude.update({"rating", "maturity_level"})
    if head in {"acquisition_reg", "acquisition_clf"}:
        exclude.update({"years_to_platform", "is_likely_original", "acquisition_bucket"})

    numeric = [c for c in NUMERIC_CANDIDATES if c in frame.columns and c not in exclude]
    binary = [c for c in BINARY_CANDIDATES if c in frame.columns and c not in exclude]
    categorical = [c for c in CATEGORICAL_CANDIDATES if c in frame.columns and c not in exclude]
    # Honour registry bans (duration* for type; date_added-derived for acquisition).
    from src.features.registry import allowed_columns

    allow = set(allowed_columns(registry, head))
    numeric = [c for c in numeric if c in allow]
    binary = [c for c in binary + numeric if c in allow]
    # de-dup preserving order
    seen: set[str] = set()
    numeric_bin: list[str] = []
    for c in numeric + binary:
        if c not in seen:
            seen.add(c)
            numeric_bin.append(c)
    categorical = [c for c in categorical if c in allow]

    work = frame.copy()
    if head == "type":
        # listed_in is withheld entirely. Stripping TV-prefixed tokens leaves
        # "Movies" / "Documentaries" / empty strings, which still separate type.
        work["text_blob"] = work["title"].astype(str)
        text_col = "text_blob"
    else:
        work["text_blob"] = (
            work["title"].astype(str)
            + " "
            + work["listed_in"].astype(str)
            + " "
            + work.get("description", pd.Series("", index=work.index)).astype(str)
        )
        text_col = "text_blob"

    feature_cols = numeric_bin + categorical + [text_col]
    assert_no_banned(
        [c for c in feature_cols if c != "text_blob" and c != "listed_in_safe"], registry, head
    )
    X = _coerce_model_frame(work[feature_cols].copy(), numeric_bin, categorical, text_col)
    y = work[target] if target else pd.Series(index=work.index, dtype=object)
    if spec.get("drop_null_target") and target:
        mask = y.notna()
        X, y = X.loc[mask], y.loc[mask]
    return X, y, numeric_bin, categorical, text_col


def _encode_ordinal(y: pd.Series, order: list[str]) -> tuple[np.ndarray, list[str]]:
    mapping = {name: i for i, name in enumerate(order)}
    return y.map(mapping).to_numpy(), order


def _split(X: pd.DataFrame, y: pd.Series, cfg: AppConfig, stratify: bool) -> tuple:
    strat = y if stratify and y.nunique() > 1 and y.notna().all() else None
    try:
        return train_test_split(
            X,
            y,
            test_size=float(cfg.base["split"]["test_size"]),
            random_state=cfg.seed,
            stratify=strat,
        )
    except ValueError:
        return train_test_split(
            X, y, test_size=float(cfg.base["split"]["test_size"]), random_state=cfg.seed
        )


def _coerce_model_frame(
    frame: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
    text_col: str | None,
) -> pd.DataFrame:
    """Float numerics and string categoricals. pandas NA is not a valid sklearn missing value."""
    out = frame.copy()
    for col in numeric:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype(float)
    for col in categorical:
        if col in out.columns:
            out[col] = out[col].astype("string").fillna("missing").astype(str)
    if text_col and text_col in out.columns:
        out[text_col] = out[text_col].astype("string").fillna("").astype(str)
    return out


def _search_or_fit(pipe, X, y, cfg, candidate, scoring: str):
    grid = _param_grid(cfg, candidate)
    n_iter = _n_iter(cfg)
    if not grid or candidate.startswith("dummy"):
        pipe.fit(X, y)
        return pipe, {}
    n_iter = min(n_iter, int(np.prod([len(v) for v in grid.values()])))
    search = RandomizedSearchCV(
        pipe,
        param_distributions=grid,
        n_iter=max(1, n_iter),
        scoring=scoring,
        cv=int(cfg.models["defaults"]["cv_folds"]),
        random_state=cfg.seed,
        n_jobs=1,
        refit=True,
    )
    search.fit(X, y)
    return search.best_estimator_, search.best_params_


def train_supervised_head(
    frame: pd.DataFrame,
    cfg: AppConfig,
    head: str,
    registry: dict[str, Any],
    schema_hash: str,
) -> list[dict[str, Any]]:
    spec = cfg.models["heads"][head]
    X, y, numeric, categorical, text_col = _head_xy(frame, cfg, head, registry)
    order: list[str] | None = None
    if spec["task"] == "ordinal_classification":
        order = maturity_order(cfg) if head == "rating" else bucket_order(cfg)
        y_enc, order = _encode_ordinal(pd.Series(y, index=X.index), order)
        mask = ~pd.isna(y_enc)
        X, y_arr = X.loc[mask], np.asarray(y_enc[mask], dtype=int)
        y_split = pd.Series(y_arr, index=X.index)
    elif spec["task"] == "binary_classification":
        y_split = y.astype(int)
        y_arr = y_split.to_numpy()
        order = ["TV Show", "Movie"] if head == "type" else None
    else:
        y_split = pd.to_numeric(y, errors="coerce")
        mask = y_split.notna()
        X, y_split = X.loc[mask], y_split.loc[mask]
        y_arr = y_split.to_numpy(dtype=float)

    X_tr, X_te, y_tr, y_te = _split(X, y_split, cfg, stratify=spec["task"] != "regression")
    scoring = {
        "binary_classification": "roc_auc",
        "ordinal_classification": "f1_macro",
        "regression": "neg_mean_absolute_error",
    }[spec["task"]]

    records = []
    models_dir = cfg.path("artifacts_dir") / "models" / head
    models_dir.mkdir(parents=True, exist_ok=True)

    for candidate in spec["candidates"]:
        started = time.perf_counter()
        prep = make_preprocessor(
            numeric,
            categorical,
            text_col,
            cfg.seed,
            max_features=int(cfg.features["text"]["tfidf_max_features"]),
            svd_components=int(cfg.features["text"]["svd_components"]),
        )
        est = make_estimator(candidate, spec["task"], cfg.seed)
        pipe = wrap(prep, est)
        y_fit = y_tr.to_numpy()
        floor = float(cfg.base["dataset"]["years_to_platform_floor"])
        if spec["task"] == "regression":
            y_fit = np.log1p(np.clip(y_fit - floor, 0, None))
        fitted, params = _search_or_fit(pipe, X_tr, y_fit, cfg, candidate, scoring)

        latency_samples = []
        for _ in range(min(30, len(X_te))):
            t0 = time.perf_counter()
            fitted.predict(X_te.iloc[:1])
            latency_samples.append((time.perf_counter() - t0) * 1000)

        if spec["task"] == "regression":
            pred_log = np.asarray(fitted.predict(X_te), dtype=float)
            pred = np.expm1(pred_log) + floor
            metrics = regression_metrics(y_te.to_numpy(dtype=float), pred)
            residuals = y_te.to_numpy(dtype=float) - pred
            q_lo, q_hi = np.quantile(residuals, [0.1, 0.9])
            metrics["interval_coverage_nominal"] = 0.8
            extra = {"residual_q10": float(q_lo), "residual_q90": float(q_hi)}
            proba = None
        else:
            pred_idx = np.asarray(fitted.predict(X_te))
            proba = fitted.predict_proba(X_te) if hasattr(fitted, "predict_proba") else None
            metrics = classification_metrics(y_te.to_numpy(), pred_idx, proba)
            extra = {}
            if spec["task"] == "ordinal_classification":
                metrics.update(ordinal_bundle(y_te.to_numpy(), pred_idx))
            if proba is not None:
                metrics["brier_before"] = brier(y_te.to_numpy(), proba)
                if not candidate.startswith("dummy") and not cfg.settings.fast_mode:
                    try:
                        calibrated = CalibratedClassifierCV(
                            fitted,
                            method=str(cfg.models["defaults"].get("calibration", "isotonic")),
                            cv=int(cfg.models["defaults"].get("calibration_cv", 3)),
                        )
                        calibrated.fit(X_tr, y_tr.to_numpy())
                        proba_c = calibrated.predict_proba(X_te)
                        metrics["brier_after"] = brier(y_te.to_numpy(), proba_c)
                        fitted = calibrated
                    except Exception as exc:
                        log.warning(
                            "calibration_failed", head=head, candidate=candidate, error=str(exc)
                        )
                        metrics["brier_after"] = metrics["brier_before"]
                else:
                    metrics["brier_after"] = metrics["brier_before"]

        metrics["latency_ms_p95"] = (
            float(np.quantile(latency_samples, 0.95)) if latency_samples else 0.0
        )
        metrics["n_train"] = float(len(X_tr))
        metrics["n_test"] = float(len(X_te))

        run_id = uuid.uuid4().hex[:12]
        model_path = models_dir / f"{candidate}_{run_id}.joblib"
        joblib.dump(
            {
                "pipeline": fitted,
                "head": head,
                "candidate": candidate,
                "feature_schema_hash": schema_hash,
                "numeric": numeric,
                "categorical": categorical,
                "text_col": text_col,
                "order": order,
                "extra": extra,
            },
            model_path,
        )
        record = {
            "head": head,
            "candidate": candidate,
            "run_id": run_id,
            "run_type": "train",
            "is_baseline": candidate.startswith("dummy") or candidate == "seasonal_naive",
            "params": {**params, "candidate": candidate, "seed": cfg.seed},
            "metrics": metrics,
            "feature_schema_hash": schema_hash,
            "git_sha": git_sha(cfg.root),
            "data_sha": data_sha(cfg.path("raw_csv")),
            "model_path": str(model_path),
            "features_used": numeric + categorical,
            "banned_features": [],
            "duration_seconds": round(time.perf_counter() - started, 3),
            "display_name": spec["display_name"],
            "primary_metric": spec["primary_metric"],
        }
        write_run_record(cfg, record)
        log_mlflow_run(record, models_dir)
        records.append(record)
        log.info("trained_candidate", head=head, candidate=candidate, metrics=metrics)

        if head == "type" and candidate == spec["candidates"][-1]:
            audit_type_leakage(
                frame,
                numeric + categorical,
                registry,
                cfg,
                cfg.path("reports_dir") / "leakage_audit.md",
            )

        _maybe_importance(fitted, X_te, y_te, cfg, head, candidate)

    _write_eval_md(cfg, head, records, spec)
    return records


def _maybe_importance(model, X_te, y_te, cfg, head, candidate) -> None:
    if cfg.settings.fast_mode or candidate.startswith("dummy"):
        return
    try:
        sample = X_te.sample(n=min(80, len(X_te)), random_state=cfg.seed)
        y_s = y_te.loc[sample.index] if hasattr(y_te, "loc") else y_te[: len(sample)]
        res = permutation_importance(
            model, sample, y_s, n_repeats=5, random_state=cfg.seed, n_jobs=1
        )
        names = list(sample.columns)
        ranked = sorted(
            zip(names, res.importances_mean, strict=False), key=lambda t: t[1], reverse=True
        )[:15]
        path = cfg.path("reports_dir") / "evaluation" / f"{head}_{candidate}_importance.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "# Permutation importance (test sample)\n\n"
            + "\n".join(f"- `{n}`: {v:.4f}" for n, v in ranked)
            + "\n",
            encoding="utf-8",
        )
    except Exception as exc:
        log.warning("permutation_importance_failed", error=str(exc))


def _write_eval_md(
    cfg: AppConfig, head: str, records: list[dict[str, Any]], spec: dict[str, Any]
) -> None:
    path = cfg.path("reports_dir") / "evaluation" / f"{head}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Evaluation — {spec['display_name']}",
        "",
        f"Primary metric: `{spec['primary_metric']}`.",
        "",
    ]
    for rec in records:
        lines.append(f"## {rec['candidate']}")
        for k, v in rec["metrics"].items():
            lines.append(f"- **{k}**: {v}")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def train_cluster_head(
    frame: pd.DataFrame, cfg: AppConfig, schema_hash: str
) -> list[dict[str, Any]]:
    result = fit_clusters(frame, cfg)
    out = frame.copy()
    out["cluster_id"] = result["labels"]
    out["portfolio_fit"] = result["portfolio_fit"]
    feat_dir = cfg.path("features_dir")
    keep_path = feat_dir / "features.parquet"
    try:
        out.to_parquet(keep_path, index=False)
    except ImportError:
        out.to_csv(feat_dir / "features.csv", index=False)
    terms = class_tfidf_terms(frame, result["labels"])
    payload = {
        "k": result["k"],
        "silhouette": result["silhouette"],
        "choice": result["choice"],
        "profiles": result["profiles"],
        "c_tfidf": terms,
        "labels": result["labels"].tolist(),
        "portfolio_fit": result["portfolio_fit"].tolist(),
    }
    art = cfg.path("artifacts_dir") / "models" / "cluster"
    art.mkdir(parents=True, exist_ok=True)
    (art / "cluster.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    joblib.dump(
        {
            "model": result["model"],
            "bundle": result["bundle"],
            "distance_min": float(result["distances"].min()),
            "distance_max": float(result["distances"].max()),
        },
        art / "kmeans.joblib",
    )
    record = {
        "head": "cluster",
        "candidate": "kmeans",
        "run_id": uuid.uuid4().hex[:12],
        "run_type": "train",
        "is_baseline": False,
        "params": {"k": result["k"], "seed": cfg.seed},
        "metrics": {
            "silhouette": result["silhouette"],
            "chosen_k": float(result["k"]),
            "bootstrap_ari": float(
                next(r["bootstrap_ari"] for r in result["choice"]["grid"] if r["k"] == result["k"])
            ),
        },
        "feature_schema_hash": schema_hash,
        "git_sha": git_sha(cfg.root),
        "data_sha": data_sha(cfg.path("raw_csv")),
        "model_path": str(art / "kmeans.joblib"),
        "features_used": ["title", "listed_in", "primary_country"],
        "banned_features": [],
        "display_name": cfg.models["heads"]["cluster"]["display_name"],
        "primary_metric": "silhouette",
    }
    write_run_record(cfg, record)
    log_mlflow_run(record, art)
    (cfg.path("reports_dir") / "evaluation").mkdir(parents=True, exist_ok=True)
    (cfg.path("reports_dir") / "evaluation" / "cluster.md").write_text(
        f"# Clustering\n\nChosen k={result['k']} by silhouette + bootstrap ARI "
        f"(≥ {cfg.models['heads']['cluster']['bootstrap_resamples']} resamples of "
        f"{cfg.models['heads']['cluster']['bootstrap_frac']:.0%}).\n\n"
        f"Silhouette={result['silhouette']:.3f}.\n\n"
        + json.dumps(result["profiles"], indent=2)
        + "\n",
        encoding="utf-8",
    )
    return [record]


def train_forecast_head(
    frame: pd.DataFrame, cfg: AppConfig, schema_hash: str
) -> list[dict[str, Any]]:
    result = fit_forecast(frame, cfg)
    art = cfg.path("artifacts_dir") / "models" / "forecast"
    art.mkdir(parents=True, exist_ok=True)
    (art / "forecast.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    records = []
    for cand in result["candidates"]:
        record = {
            "head": "forecast",
            "candidate": cand["candidate"],
            "run_id": uuid.uuid4().hex[:12],
            "run_type": "train",
            "is_baseline": cand.get("is_baseline", False),
            "params": {"horizon": result["horizon"]},
            "metrics": {"mae": cand["mae"]},
            "feature_schema_hash": schema_hash,
            "git_sha": git_sha(cfg.root),
            "data_sha": data_sha(cfg.path("raw_csv")),
            "model_path": str(art / "forecast.json"),
            "features_used": ["lag_1", "lag_12", "month"],
            "banned_features": [],
            "display_name": cfg.models["heads"]["forecast"]["display_name"],
            "primary_metric": "mae",
        }
        write_run_record(cfg, record)
        log_mlflow_run(record, art)
        records.append(record)
    note = "baseline wins" if result["baseline_wins"] else f"best={result['best']}"
    (cfg.path("reports_dir") / "evaluation").mkdir(parents=True, exist_ok=True)
    (cfg.path("reports_dir") / "evaluation" / "forecast.md").write_text(
        f"# Forecast\n\n{result['structural_break_note']}\n\n"
        f"Best candidate: {result['best']} ({note}).\n\n"
        + json.dumps({k: v for k, v in result.items() if k != "series_values"}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return records


def train_all_heads(cfg: AppConfig) -> dict[str, Any]:
    setup_mlflow(cfg)
    frame, registry = read_feature_store(cfg.path("features_dir"))
    schema_hash = registry["schema_hash"]
    summary: dict[str, Any] = {"heads": {}}
    for head, spec in cfg.models["heads"].items():
        log.info("train_head_start", head=head, task=spec["task"])
        if spec["task"] == "clustering":
            recs = train_cluster_head(frame, cfg, schema_hash)
            # reload store with cluster columns for later stages
            frame, registry = read_feature_store(cfg.path("features_dir"))
        elif spec["task"] == "forecasting":
            recs = train_forecast_head(frame, cfg, schema_hash)
        else:
            recs = train_supervised_head(frame, cfg, head, registry, schema_hash)
        summary["heads"][head] = {"n_runs": len(recs)}
    return summary
