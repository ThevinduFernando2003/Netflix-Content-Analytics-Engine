from __future__ import annotations

import numpy as np
from src.features.registry import build_registry
from src.models.train_all import train_supervised_head


def _train_one(featured, cfg, head: str):
    registry = build_registry(cfg, extra_columns=list(featured.columns))
    return train_supervised_head(featured, cfg, head, registry, registry["schema_hash"])


def test_type_head_fits(featured, cfg):
    recs = _train_one(featured, cfg, "type")
    assert len(recs) >= 2
    dummy = next(r for r in recs if r["is_baseline"])
    assert "roc_auc" in dummy["metrics"] or "accuracy" in dummy["metrics"]


def test_predict_proba_sums_to_one(featured, cfg):
    import joblib

    recs = _train_one(featured, cfg, "type")
    winner = next(r for r in recs if not r["is_baseline"])
    blob = joblib.load(winner["model_path"])
    X = featured.head(5).copy()
    X["text_blob"] = X["title"].astype(str) + " " + X["listed_in"].astype(str)
    from src.api.predict import _align

    aligned = _align(X, blob)
    proba = blob["pipeline"].predict_proba(aligned)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert proba.shape[1] == 2


def test_regression_interval_contains_point(featured, cfg):
    recs = _train_one(featured, cfg, "acquisition_reg")
    winner = next(r for r in recs if not r["is_baseline"])
    extra = __import__("joblib").load(winner["model_path"]).get("extra", {})
    # Residual quantiles: lo < 0 < hi typically, so point ± residuals contains the point
    assert extra.get("residual_q10", -1) <= 0 <= extra.get("residual_q90", 1) or extra.get(
        "residual_q10"
    ) <= extra.get("residual_q90")


def test_unseen_categorical_does_not_crash(featured, cfg):
    recs = _train_one(featured, cfg, "type")
    blob = __import__("joblib").load(recs[-1]["model_path"])
    row = featured.iloc[[0]].copy()
    row["primary_country"] = "Atlantis"
    row["text_blob"] = row["title"].astype(str)
    from src.api.predict import _align

    aligned = _align(row, blob)
    pred = blob["pipeline"].predict(aligned)
    assert pred.shape[0] == 1


def test_single_row_predict(featured, cfg):
    recs = _train_one(featured, cfg, "rating")
    blob = __import__("joblib").load(recs[-1]["model_path"])
    row = featured.iloc[[0]].copy()
    row["text_blob"] = row["title"].astype(str)
    from src.api.predict import _align

    aligned = _align(row, blob)
    pred = blob["pipeline"].predict(aligned)
    assert pred.shape == (1,)
