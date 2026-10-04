"""Shared prediction helpers used by the routers."""

from __future__ import annotations

from typing import Any

import pandas as pd
from fastapi import HTTPException

from src.api.deps import AppState
from src.api.schemas import TitleIn
from src.features.engineer import engineer_features
from src.features.store import assert_schema_hash
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
    # Nearest-centroid in a cheap metadata space when the kmeans artifact is present.
    cluster_id = 0
    distance = 0.0
    name = profiles[0]["name"] if profiles else "unassigned"
    if state.store is not None and "cluster_id" in state.store.columns:
        # Match on primary genre when possible.
        genre = str(frame.iloc[0].get("primary_genre") or "")
        subset = state.store[state.store["primary_genre"].astype(str) == genre]
        if len(subset) and subset["cluster_id"].notna().any():
            cluster_id = int(subset["cluster_id"].mode().iloc[0])
            prof = next((p for p in profiles if int(p["cluster_id"]) == cluster_id), None)
            name = prof["name"] if prof else name
    fit = float(frame.iloc[0].get("catalogue_prominence") or 0.5)
    return {
        "cluster_id": cluster_id,
        "name": name,
        "distance": distance,
        "outlier": False,
        "portfolio_fit": fit,
        "is_proxy": True,
        "proxy_note": PROXY_NOTES["portfolio_fit"],
    }


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
