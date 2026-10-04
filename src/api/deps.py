"""Shared application state, loaded once in the lifespan handler."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.config import AppConfig, load_config
from src.features.store import read_feature_store
from src.pipeline.cache import git_sha
from src.tracking.registry import read_incumbent


@dataclass
class AppState:
    cfg: AppConfig
    models: dict[str, Any] = field(default_factory=dict)
    insights: dict[str, Any] = field(default_factory=dict)
    store: pd.DataFrame | None = None
    registry: dict[str, Any] | None = None
    forecast: dict[str, Any] = field(default_factory=dict)
    drift: dict[str, Any] = field(default_factory=dict)
    cluster: dict[str, Any] = field(default_factory=dict)
    git_sha: str = "unknown"
    ready: bool = False


STATE = AppState(cfg=load_config())


def _load_json(path: Path) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def load_state(cfg: AppConfig | None = None) -> AppState:
    cfg = cfg or load_config()
    state = AppState(cfg=cfg, git_sha=git_sha(cfg.root))
    feat_dir = cfg.path("features_dir")
    try:
        state.store, state.registry = read_feature_store(feat_dir)
    except FileNotFoundError:
        state.store, state.registry = None, None
    art = cfg.path("artifacts_dir")
    state.insights = _load_json(art / "insights.json")
    state.forecast = _load_json(art / "models" / "forecast" / "forecast.json")
    state.drift = _load_json(art / "drift.json")
    state.cluster = _load_json(art / "models" / "cluster" / "cluster.json")
    models_root = art / "models"
    if models_root.exists():
        for head_dir in models_root.iterdir():
            if not head_dir.is_dir():
                continue
            joblibs = sorted(head_dir.glob("*.joblib"))
            if joblibs:
                try:
                    state.models[head_dir.name] = joblib.load(joblibs[-1])
                except Exception:
                    continue
    for head in cfg.models["heads"]:
        inc = read_incumbent(cfg, head)
        if inc and inc.get("dir"):
            rec_path = Path(inc["dir"]) / "record.json"
            if rec_path.exists():
                rec = json.loads(rec_path.read_text(encoding="utf-8"))
                mp = rec.get("model_path")
                if mp and Path(mp).exists():
                    try:
                        state.models[head] = joblib.load(mp)
                    except Exception:
                        pass
    state.ready = state.store is not None
    global STATE
    STATE = state
    return state


def get_state() -> AppState:
    return STATE
