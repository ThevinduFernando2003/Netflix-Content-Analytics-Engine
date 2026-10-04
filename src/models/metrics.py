"""Task metrics, including ordinal measures that accuracy cannot see."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


def quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))


def ordinal_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))))


def adjacent_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(
        np.mean(np.abs(np.asarray(y_true, dtype=int) - np.asarray(y_pred, dtype=int)) <= 1)
    )


def severe_error_count(y_true: np.ndarray, y_pred: np.ndarray, min_distance: int = 3) -> int:
    return int(
        np.sum(
            np.abs(np.asarray(y_true, dtype=int) - np.asarray(y_pred, dtype=int)) >= min_distance
        )
    )


def kids_as_mature_count(
    y_true: np.ndarray, y_pred: np.ndarray, kids: int = 0, mature: int = 3
) -> int:
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    return int(np.sum((yt == kids) & (yp == mature)))


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    average: str = "macro",
) -> dict[str, float]:
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
    }
    if y_proba is not None and y_proba.ndim == 2 and y_proba.shape[1] == 2:
        out["roc_auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
    elif y_proba is not None and y_proba.ndim == 2 and y_proba.shape[1] > 2:
        try:
            out["roc_auc"] = float(
                roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
            )
        except ValueError:
            out["roc_auc"] = float("nan")
    return out


def ordinal_bundle(y_true_idx: np.ndarray, y_pred_idx: np.ndarray) -> dict[str, float]:
    return {
        "quadratic_weighted_kappa": quadratic_weighted_kappa(y_true_idx, y_pred_idx),
        "ordinal_mae": ordinal_mae(y_true_idx, y_pred_idx),
        "adjacent_accuracy": adjacent_accuracy(y_true_idx, y_pred_idx),
        "severe_error_count": float(severe_error_count(y_true_idx, y_pred_idx)),
        "kids_predicted_as_mature": float(kids_as_mature_count(y_true_idx, y_pred_idx)),
    }


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mae = float(mean_absolute_error(yt, yp))
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    r2 = float(r2_score(yt, yp))
    # MAPE is undefined / explosive near zero — computed only to demonstrate the failure.
    denom = np.where(np.abs(yt) < 1e-8, np.nan, np.abs(yt))
    mape = float(np.nanmean(np.abs(yt - yp) / denom)) if np.isfinite(denom).any() else float("inf")
    return {"mae": mae, "rmse": rmse, "r2": r2, "mape_unstable_near_zero": mape}


def brier(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    yt = np.asarray(y_true)
    yp = np.asarray(y_proba)
    if yp.ndim == 1:
        return float(np.mean((yp - yt) ** 2))
    n_classes = yp.shape[1]
    onehot = np.eye(n_classes)[yt.astype(int)]
    return float(np.mean(np.sum((yp - onehot) ** 2, axis=1)))
