"""Spherical k-means on TruncatedSVD text+metadata. k by silhouette AND bootstrap ARI."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import normalize

from src.config import AppConfig
from src.features.text import text_svd_pipeline
from src.logging_conf import get_logger

log = get_logger(__name__)


_EXTRA_COLUMNS = ("n_genres", "n_countries", "has_director", "release_year", "is_movie")


def _texts(frame: pd.DataFrame) -> list[str]:
    text_cols = [
        c for c in ("title", "listed_in", "primary_country", "primary_genre") if c in frame.columns
    ]
    return frame[text_cols].astype(str).agg(" ".join, axis=1).tolist()


def _design_matrix(frame: pd.DataFrame, cfg: AppConfig) -> tuple[np.ndarray, dict[str, Any]]:
    texts = _texts(frame)
    n_comp = int(cfg.models["heads"]["cluster"].get("svd_components", 24))
    pipe = text_svd_pipeline(
        max_features=int(cfg.features["text"]["tfidf_max_features"]),
        n_components=n_comp,
        ngram_range=tuple(cfg.features["text"]["ngram_range"]),
        min_df=int(cfg.features["text"]["min_df"]),
        random_state=cfg.seed,
    )
    reduced = pipe.fit_transform(texts)
    extra_names = [col for col in _EXTRA_COLUMNS if col in frame.columns]
    mean = np.zeros(len(extra_names))
    scale = np.ones(len(extra_names))
    if extra_names:
        extra = _extra_block(frame, extra_names)
        mean = extra.mean(axis=0)
        scale = extra.std(axis=0)
        scale[scale == 0] = 1.0
        reduced = np.hstack([reduced, (extra - mean) / scale])
    bundle = {
        "text_pipe": pipe,
        "extra_names": extra_names,
        "extra_mean": mean,
        "extra_scale": scale,
    }
    return normalize(reduced), bundle


def _extra_block(frame: pd.DataFrame, names: list[str]) -> np.ndarray:
    cols = []
    for name in names:
        if name in frame.columns:
            values = pd.to_numeric(frame[name], errors="coerce").fillna(0).to_numpy()
        else:
            values = np.zeros(len(frame))
        cols.append(values.reshape(-1, 1))
    return np.hstack(cols)


def transform_with_bundle(frame: pd.DataFrame, bundle: dict[str, Any]) -> np.ndarray:
    """Project new rows with the vectorizer fitted at training time."""
    reduced = bundle["text_pipe"].transform(_texts(frame))
    names = list(bundle.get("extra_names") or [])
    if names:
        extra = _extra_block(frame, names)
        mean = np.asarray(bundle["extra_mean"])
        scale = np.asarray(bundle["extra_scale"])
        reduced = np.hstack([reduced, (extra - mean) / scale])
    return normalize(reduced)


def bootstrap_ari(
    X: np.ndarray,
    k: int,
    n_resamples: int,
    frac: float,
    seed: int,
    full_labels: np.ndarray | None = None,
) -> float:
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    take = min(n, max(k + 1, int(frac * n)))
    scores: list[float] = []
    if full_labels is None:
        full = KMeans(n_clusters=k, n_init=10, random_state=seed)
        full_labels = full.fit_predict(X)
    for i in range(n_resamples):
        idx = rng.choice(n, size=take, replace=False)
        km = KMeans(n_clusters=k, n_init=10, random_state=seed + i)
        part = km.fit_predict(X[idx])
        scores.append(adjusted_rand_score(full_labels[idx], part))
    return float(np.mean(scores))


def choose_k(X: np.ndarray, cfg: AppConfig) -> dict[str, Any]:
    spec = cfg.models["heads"]["cluster"]
    k_min = int(spec.get("k_min", 3))
    k_max = int(spec.get("k_max", 8))
    n_boot = int(spec.get("bootstrap_resamples", 20))
    frac = float(spec.get("bootstrap_frac", 0.8))
    rows = []
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=cfg.seed)
        labels = km.fit_predict(X)
        sil = float(silhouette_score(X, labels)) if len(set(labels)) > 1 else -1.0
        ari = bootstrap_ari(X, k, n_boot, frac, cfg.seed, full_labels=labels)
        rows.append({"k": k, "silhouette": sil, "bootstrap_ari": ari, "score": sil + ari})
        log.info("cluster_k_eval", k=k, silhouette=sil, bootstrap_ari=ari)
    best = max(rows, key=lambda r: r["score"])
    return {"chosen_k": int(best["k"]), "grid": rows}


def fit_clusters(frame: pd.DataFrame, cfg: AppConfig) -> dict[str, Any]:
    X, bundle = _design_matrix(frame, cfg)
    choice = choose_k(X, cfg)
    k = int(choice["chosen_k"])
    model = KMeans(n_clusters=k, n_init=10, random_state=cfg.seed)
    labels = model.fit_predict(X)
    dist = np.linalg.norm(X - model.cluster_centers_[labels], axis=1)
    dmin, dmax = float(dist.min()), float(dist.max())
    span = (dmax - dmin) or 1.0
    fit = 1.0 - (dist - dmin) / span
    profiles = _profile_clusters(frame, labels, cfg)
    return {
        "k": k,
        "labels": labels,
        "portfolio_fit": fit,
        "distances": dist,
        "centers": model.cluster_centers_,
        "choice": choice,
        "profiles": profiles,
        "X": X,
        "bundle": bundle,
        "model": model,
        "silhouette": float(silhouette_score(X, labels)) if len(set(labels)) > 1 else -1.0,
    }


def _profile_clusters(
    frame: pd.DataFrame, labels: np.ndarray, cfg: AppConfig
) -> list[dict[str, Any]]:
    out = []
    genres = frame["listed_in"].astype(str)
    overall: dict[str, int] = {}
    for text in genres:
        for g in [p.strip() for p in text.split(",") if p.strip()]:
            overall[g] = overall.get(g, 0) + 1
    n = max(len(frame), 1)
    for k in sorted(set(labels.tolist())):
        mask = labels == k
        sub = genres[mask]
        counts: dict[str, int] = {}
        for text in sub:
            for g in [p.strip() for p in text.split(",") if p.strip()]:
                counts[g] = counts.get(g, 0) + 1
        nk = max(int(mask.sum()), 1)
        lift = []
        for g, c in counts.items():
            base = overall.get(g, 0) / n
            share = c / nk
            lift.append((g, share / base if base else 0.0, share, c))
        lift.sort(key=lambda t: t[1], reverse=True)
        name = lift[0][0] if lift else f"cluster_{k}"
        out.append(
            {
                "cluster_id": int(k),
                "name": f"{name} / n={nk}",
                "n": nk,
                "top_genre_lift": [
                    {"genre": g, "lift": round(lv, 2), "share": round(sh, 3)}
                    for g, lv, sh, _ in lift[:8]
                ],
            }
        )
    return out


def class_tfidf_terms(
    frame: pd.DataFrame, labels: np.ndarray, top: int = 8
) -> dict[int, list[str]]:
    from sklearn.feature_extraction.text import TfidfVectorizer

    texts = frame["listed_in"].astype(str) + " " + frame["title"].astype(str)
    vec = TfidfVectorizer(max_features=400, stop_words="english")
    x = vec.fit_transform(texts)
    vocab = np.array(vec.get_feature_names_out())
    out: dict[int, list[str]] = {}
    for k in sorted(set(labels.tolist())):
        mean = np.asarray(x[labels == k].mean(axis=0)).ravel()
        idx = mean.argsort()[::-1][:top]
        out[int(k)] = vocab[idx].tolist()
    return out
