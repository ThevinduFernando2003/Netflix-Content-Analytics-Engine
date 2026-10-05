"""Shared prediction helpers used by the routers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from fastapi import HTTPException

from src.api.deps import AppState
from src.api.schemas import TitleIn
from src.features.engineer import engineer_features
from src.features.store import assert_schema_hash
from src.models.clustering import transform_with_bundle
from src.targets.definitions import PROXY_NOTES


def reject_leaks(payload: TitleIn, head: str) -> None:
    if head == "type" and payload.duration:
        raise HTTPException(
            status_code=422,
            detail=(
                "duration is banned for /predict/type — it trivially determines Movie vs TV Show. "
                "Banned list: duration*, TV-prefixed genre tokens. See GET /meta."
            ),
        )
    if head in {"acquisition", "acquisition_reg", "acquisition_clf"}:
        if payload.date_added or payload.year_added is not None:
            raise HTTPException(
                status_code=422,
                detail=(
                    "date_added / year_added are banned for /predict/acquisition — "
                    "years_to_platform is computed from year_added. See GET /meta."
                ),
            )


def title_to_frame(payload: TitleIn, cfg) -> pd.DataFrame:
    raw = pd.DataFrame(
        [
            {
                "show_id": "api-1",
                "type": payload.type or "Movie",
                "title": payload.title,
                "director": payload.director,
                "country": payload.country,
                "date_added": payload.date_added or "",
                "release_year": payload.release_year,
                "rating": payload.rating or "",
                "duration": payload.duration or "",
                "listed_in": payload.listed_in,
                "cast": payload.cast,
                "description": payload.description,
            }
        ]
    )
    return engineer_features(raw, cfg)


def predict_type(state: AppState, payload: TitleIn) -> dict[str, Any]:
    reject_leaks(payload, "type")
    model = state.models.get("type")
    if not model:
        raise HTTPException(503, "type model is not loaded")
    if isinstance(model, dict) and model.get("feature_schema_hash") and state.registry:
        assert_schema_hash(state.registry, model["feature_schema_hash"])
    frame = title_to_frame(payload, state.cfg)
    pipe = model["pipeline"] if isinstance(model, dict) else model
    X = _align(frame, model)
    proba = pipe.predict_proba(X)[0]
    label_idx = int(pipe.predict(X)[0])
    return {
        "label": "Movie" if label_idx == 1 else "TV Show",
        "probability_movie": float(proba[1] if len(proba) > 1 else proba[0]),
        "is_proxy": False,
        "proxy_note": "Observed catalogue field, not a success metric.",
    }


def predict_rating(state: AppState, payload: TitleIn) -> dict[str, Any]:
    model = state.models.get("rating")
    if not model:
        raise HTTPException(503, "rating model is not loaded")
    frame = title_to_frame(payload, state.cfg)
    pipe = model["pipeline"] if isinstance(model, dict) else model
    X = _align(frame, model)
    idx = int(pipe.predict(X)[0])
    order = model.get("order") if isinstance(model, dict) else None
    order = order or ["Kids", "Older Kids", "Teens", "Mature"]
    proba = pipe.predict_proba(X)[0]
    probs = {order[i]: float(proba[i]) for i in range(min(len(order), len(proba)))}
    return {
        "label": order[idx] if 0 <= idx < len(order) else str(idx),
        "probabilities": probs,
        "is_proxy": False,
        "proxy_note": "Observed catalogue field collapsed onto an ordinal ladder, not a success metric.",
    }


def predict_acquisition(state: AppState, payload: TitleIn) -> dict[str, Any]:
    reject_leaks(payload, "acquisition")
    model = state.models.get("acquisition_reg")
    clf = state.models.get("acquisition_clf")
    if not model:
        raise HTTPException(503, "acquisition model is not loaded")
    frame = title_to_frame(payload, state.cfg)
    pipe = model["pipeline"] if isinstance(model, dict) else model
    X = _align(frame, model)
    extra = (model.get("extra") if isinstance(model, dict) else {}) or {}
    floor = float(state.cfg.base["dataset"]["years_to_platform_floor"])
    pred_log = float(pipe.predict(X)[0])
    point = float(__import__("numpy").expm1(pred_log) + floor)
    lo = point + float(extra.get("residual_q10", -2))
    hi = point + float(extra.get("residual_q90", 2))
    bucket = None
    if clf:
        cpipe = clf["pipeline"] if isinstance(clf, dict) else clf
        Xc = _align(frame, clf)
        bidx = int(cpipe.predict(Xc)[0])
        order = clf.get("order") if isinstance(clf, dict) else None
        order = order or ["Day-one", "Fast", "Recent", "Back-catalogue"]
        bucket = order[bidx] if 0 <= bidx < len(order) else None
    return {
        "years_to_platform": point,
        "interval_lo": min(lo, point),
        "interval_hi": max(hi, point),
        "bucket": bucket,
        "is_likely_original": point <= 0,
        "is_proxy": True,
        "proxy_note": PROXY_NOTES["acquisition_priority"],
    }


def predict_segment(state: AppState, payload: TitleIn) -> dict[str, Any]:
    profiles = (state.cluster or {}).get("profiles") or []
    frame = title_to_frame(payload, state.cfg)
    artifact = state.models.get("cluster")
    scored = _score_cluster(frame, artifact)
    cluster_id: int | None
    distance: float | None
    fit: float | None
    name: str
    outlier: bool
    if scored is not None:
        cluster_id = int(scored["cluster_id"])
        distance = float(scored["distance"])
        fit = float(scored["portfolio_fit"])
        prof = next((p for p in profiles if int(p["cluster_id"]) == cluster_id), None)
        name = prof["name"] if prof else f"cluster_{cluster_id}"
        outlier = fit < 0.05
    else:
        cluster_id, name = _genre_mode_cluster(state, frame, profiles)
        distance = None
        fit = None
        outlier = False
    return {
        "cluster_id": cluster_id,
        "name": name,
        "distance": distance,
        "outlier": outlier,
        "portfolio_fit": fit,
        "is_proxy": True,
        "proxy_note": PROXY_NOTES["portfolio_fit"],
    }


def prominence_value(frame: pd.DataFrame) -> float | None:
    if "catalogue_prominence" not in frame.columns:
        return None
    value = frame.iloc[0]["catalogue_prominence"]
    if value is None or (isinstance(value, float) and np.isnan(value)) or pd.isna(value):
        return None
    return float(value)


def _score_cluster(frame: pd.DataFrame, artifact: Any) -> dict[str, float | int] | None:
    if not isinstance(artifact, dict) or "bundle" not in artifact or "model" not in artifact:
        return None
    embedded = transform_with_bundle(frame, artifact["bundle"])
    model = artifact["model"]
    label = int(model.predict(embedded)[0])
    center = model.cluster_centers_[label]
    distance = float(np.linalg.norm(embedded[0] - center))
    dmin = float(artifact.get("distance_min", 0.0))
    dmax = float(artifact.get("distance_max", 1.0))
    span = (dmax - dmin) or 1.0
    fit = float(np.clip(1.0 - (distance - dmin) / span, 0.0, 1.0))
    return {"cluster_id": label, "distance": distance, "portfolio_fit": fit}


def _genre_mode_cluster(
    state: AppState, frame: pd.DataFrame, profiles: list[dict[str, Any]]
) -> tuple[int | None, str]:
    """Fallback when the fitted vectorizer was not saved with the cluster model."""
    name = "unassigned"
    if state.store is None or "cluster_id" not in state.store.columns:
        return None, name
    genre = str(frame.iloc[0].get("primary_genre") or "")
    if not genre:
        return None, name
    subset = state.store[state.store["primary_genre"].astype(str) == genre]
    if not len(subset) or not subset["cluster_id"].notna().any():
        return None, name
    cluster_id = int(subset["cluster_id"].mode().iloc[0])
    prof = next((p for p in profiles if int(p["cluster_id"]) == cluster_id), None)
    return cluster_id, prof["name"] if prof else name


def _align(frame: pd.DataFrame, model: Any) -> pd.DataFrame:
    if not isinstance(model, dict):
        return frame
    numeric = model.get("numeric") or []
    categorical = model.get("categorical") or []
    text_col = model.get("text_col")
    cols = [c for c in numeric + categorical if c in frame.columns]
    out = frame[cols].copy()
    if text_col:
        if text_col not in out.columns:
            title = (
                frame["title"].astype(str)
                if "title" in frame.columns
                else pd.Series("", index=frame.index)
            )
            # Head A is trained on title only. Genre strings still separate type.
            if model.get("head") == "type":
                out[text_col] = title
            else:
                listed = (
                    frame["listed_in"].astype(str)
                    if "listed_in" in frame.columns
                    else pd.Series("", index=frame.index)
                )
                out[text_col] = (title + " " + listed).astype(str)
    for c in numeric:
        if c not in out.columns:
            out[c] = 0
    for c in categorical:
        if c not in out.columns:
            out[c] = ""
    ordered = [c for c in numeric + categorical if c in out.columns]
    if text_col and text_col in out.columns:
        ordered.append(text_col)
    out = out[ordered]
    for col in numeric:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype(float)
    for col in categorical:
        if col in out.columns:
            out[col] = out[col].astype("string").fillna("missing").astype(str)
    if text_col and text_col in out.columns:
        out[text_col] = out[text_col].astype("string").fillna("").astype(str)
    return out
