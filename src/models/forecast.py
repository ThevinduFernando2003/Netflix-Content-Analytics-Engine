"""Monthly titles-added forecast. Time-based split only. Seasonal-naive is the baseline."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

from src.config import AppConfig
from src.logging_conf import get_logger

log = get_logger(__name__)


def monthly_volume(frame: pd.DataFrame) -> pd.Series:
    dates = pd.to_datetime(frame["date_added_parsed"], errors="coerce")
    s = dates.dropna().dt.to_period("M").value_counts().sort_index()
    if s.empty:
        return pd.Series(dtype=float)
    full = pd.period_range(s.index.min(), s.index.max(), freq="M")
    s = s.reindex(full, fill_value=0).astype(float)
    s.index = s.index.to_timestamp()
    return s


def seasonal_naive(y: np.ndarray, season: int, horizon: int) -> np.ndarray:
    if len(y) < season:
        return np.repeat(y[-1] if len(y) else 0.0, horizon)
    pattern = y[-season:]
    reps = int(np.ceil(horizon / season))
    return np.tile(pattern, reps)[:horizon]


def lag_matrix(y: np.ndarray, lags: list[int], season: int) -> tuple[np.ndarray, np.ndarray]:
    max_lag = max(lags + [season])
    rows = []
    targets = []
    for t in range(max_lag, len(y)):
        feats = [y[t - lag] for lag in lags]
        feats.append(y[t - season] if t >= season else y[0])
        month = (t % 12) + 1
        feats.extend([month, t])
        rows.append(feats)
        targets.append(y[t])
    return np.asarray(rows, dtype=float), np.asarray(targets, dtype=float)


def residual_interval(
    errors: np.ndarray, point: np.ndarray, q: float = 0.1
) -> tuple[np.ndarray, np.ndarray]:
    lo = float(np.quantile(errors, q)) if len(errors) else -1.0
    hi = float(np.quantile(errors, 1 - q)) if len(errors) else 1.0
    return point + lo, point + hi


def fit_forecast(frame: pd.DataFrame, cfg: AppConfig) -> dict[str, Any]:
    spec = cfg.models["heads"]["forecast"]
    series = monthly_volume(frame)
    y = series.to_numpy(dtype=float)
    index = series.index
    season = int(spec.get("seasonal_period", 12))
    horizon = int(spec.get("horizon", 6))
    cutoff = pd.Timestamp(spec.get("cutoff", "2019-12-01"))
    train_mask = index <= cutoff
    if train_mask.sum() < int(spec.get("min_train_months", 24)):
        # Fall back to an 80/20 time split when the configured cutoff is too late/early.
        cut = max(int(0.8 * len(y)), season + 2)
        train_y, test_y = y[:cut], y[cut:]
        train_idx, test_idx = index[:cut], index[cut:]
    else:
        train_y, test_y = y[train_mask], y[~train_mask]
        train_idx, test_idx = index[train_mask], index[~train_mask]

    results: list[dict[str, Any]] = []
    # Seasonal naive — the baseline that must be reported even if it wins.
    naive_pred = seasonal_naive(train_y, season, len(test_y) if len(test_y) else horizon)
    naive_mae = (
        float(mean_absolute_error(test_y, naive_pred[: len(test_y)]))
        if len(test_y)
        else float("nan")
    )
    results.append(
        {"candidate": "seasonal_naive", "is_baseline": True, "mae": naive_mae, "pred": naive_pred}
    )

    # Holt-Winters
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        hw = ExponentialSmoothing(
            train_y,
            trend="add",
            seasonal="add" if len(train_y) >= 2 * season else None,
            seasonal_periods=season if len(train_y) >= 2 * season else None,
        ).fit(optimized=True)
        hw_pred = np.asarray(hw.forecast(len(test_y) if len(test_y) else horizon), dtype=float)
        hw_mae = (
            float(mean_absolute_error(test_y, hw_pred[: len(test_y)]))
            if len(test_y)
            else float("nan")
        )
        results.append(
            {"candidate": "holt_winters", "is_baseline": False, "mae": hw_mae, "pred": hw_pred}
        )
    except Exception as exc:  # pragma: no cover
        log.warning("holt_winters_failed", error=str(exc))

    lags = [1, 2, 3, 6, 12]
    X_all, y_all = lag_matrix(train_y, lags, season)
    if len(y_all) >= 8:
        tscv = TimeSeriesSplit(n_splits=min(4, max(2, len(y_all) // 8)))
        for name, model in (
            ("ridge_lags", Ridge(alpha=1.0)),
            (
                "hist_gb_lags",
                HistGradientBoostingRegressor(random_state=cfg.seed, max_depth=4, max_iter=80),
            ),
        ):
            cv_err = []
            for tr, te in tscv.split(X_all):
                model.fit(X_all[tr], y_all[tr])
                cv_err.append(mean_absolute_error(y_all[te], model.predict(X_all[te])))
            model.fit(X_all, y_all)
            # Walk forward on test using observed history only (point-in-time).
            history = list(train_y)
            preds = []
            for actual in test_y:
                x_t, _ = lag_matrix(np.asarray(history + [0.0]), lags, season)
                pred = float(model.predict(x_t[-1:])) if len(x_t) else float(history[-1])
                preds.append(pred)
                history.append(
                    actual
                )  # evaluate with observed lag updates, never future y as a feature at t
            pred_arr = np.asarray(preds, dtype=float)
            mae = float(mean_absolute_error(test_y, pred_arr)) if len(test_y) else float("nan")
            results.append(
                {
                    "candidate": name,
                    "is_baseline": False,
                    "mae": mae,
                    "pred": pred_arr,
                    "cv_mae": float(np.mean(cv_err)),
                }
            )

    best = min(results, key=lambda r: r["mae"] if np.isfinite(r["mae"]) else 1e18)
    errors = (test_y - best["pred"][: len(test_y)]) if len(test_y) else np.array([0.0])
    future = (
        seasonal_naive(y, season, horizon)
        if best["candidate"] == "seasonal_naive"
        else best["pred"][:horizon]
    )
    if len(future) < horizon:
        future = seasonal_naive(y, season, horizon)
    lo, hi = residual_interval(errors, np.asarray(future, dtype=float)[:horizon])
    log.info(
        "forecast_complete",
        best=best["candidate"],
        best_mae=best["mae"],
        baseline_mae=naive_mae,
        baseline_wins=best["candidate"] == "seasonal_naive",
        n_months=int(len(y)),
        train_months=int(len(train_y)),
        test_months=int(len(test_y)),
    )
    return {
        "series_index": [str(t.date()) for t in index],
        "series_values": y.tolist(),
        "train_end": str(train_idx.max().date()) if len(train_idx) else None,
        "test_index": [str(t.date()) for t in test_idx],
        "candidates": [
            {k: v for k, v in r.items() if k != "pred"} | {"pred": np.asarray(r["pred"]).tolist()}
            for r in results
        ],
        "best": best["candidate"],
        "baseline_wins": best["candidate"] == "seasonal_naive",
        "horizon_point": np.asarray(future[:horizon], dtype=float).tolist(),
        "horizon_lo": lo.tolist(),
        "horizon_hi": hi.tolist(),
        "horizon": horizon,
        "structural_break_note": (
            "The catalogue's monthly add rate changes sharply around 2020. "
            "Forecasts beyond a few months are weakly supported."
        ),
    }
