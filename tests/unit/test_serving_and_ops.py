from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sklearn.dummy import DummyClassifier, DummyRegressor
from src.api.deps import load_state
from src.api.main import create_app
from src.features.registry import build_registry
from src.features.store import read_feature_store, write_feature_store
from src.logging_conf import configure_logging
from src.models.base import BaseModelTask
from src.models.evaluate import write_leaderboard
from src.models.pipelines import make_preprocessor, wrap
from src.monitoring.drift import run_drift
from src.pipeline.orchestrator import run_pipeline
from src.pipeline.stages import stage_features, stage_ingest, stage_insights
from src.tracking.mlflow_utils import log_mlflow_run, setup_mlflow, write_run_record
from src.tracking.promote import promote_all


class _Task(BaseModelTask):
    name = "toy"
    target = "type"

    def build_pipeline(self, candidate: str):
        return make_preprocessor(["release_year"], [], None, 42)

    def fit(self, pipeline, X, y):
        return pipeline

    def predict(self, pipeline, X):
        return X

    def evaluate(self, pipeline, X, y):
        return {"mae": 1.0}

    def explain(self, pipeline, X):
        return super().explain(pipeline, X)


def _fit_head(frame, numeric, categorical, target, task, order=None):
    X = frame[numeric + categorical].copy()
    for col in numeric:
        X[col] = X[col].astype(float)
    for col in categorical:
        X[col] = X[col].astype(str).fillna("missing")
    if task == "regression":
        y = frame[target].astype(float)
        est = DummyRegressor(strategy="median")
    else:
        mapping = {name: i for i, name in enumerate(order or [])}
        y = frame[target].astype(str).map(mapping)
        est = DummyClassifier(strategy="most_frequent")
    mask = y.notna()
    pipe = wrap(make_preprocessor(numeric, categorical, None, 42), est)
    y_fit = y.loc[mask].astype(float if task == "regression" else int)
    pipe.fit(X.loc[mask], y_fit)
    return {
        "pipeline": pipe,
        "numeric": numeric,
        "categorical": categorical,
        "text_col": None,
        "order": order,
        "extra": {"residual_q10": -1.5, "residual_q90": 2.5},
        "feature_schema_hash": "unused",
    }


def test_logging_and_base_and_store(cfg, featured):
    configure_logging("INFO", json_logs=True)
    configure_logging("INFO", json_logs=False)
    task = _Task()
    assert task.explain(None, featured) == {}
    from src.data.load import load_raw

    store, registry = write_feature_store(
        load_raw(cfg.path("raw_csv"), cfg), cfg, cfg.path("features_dir")
    )
    loaded, reg = read_feature_store(cfg.path("features_dir"))
    assert len(loaded) == len(store)
    assert reg["schema_hash"] == registry["schema_hash"]
    with pytest.raises(FileNotFoundError):
        read_feature_store(cfg.path("artifacts_dir") / "missing-store")


def test_drift_leaderboard_promote_mlflow(cfg):
    from src.data.load import load_raw

    write_feature_store(load_raw(cfg.path("raw_csv"), cfg), cfg, cfg.path("features_dir"))
    summary = run_drift(cfg)
    assert summary["n_reference"] >= 0
    assert (cfg.path("reports_dir") / "drift_report.html").exists()
    write_run_record(
        cfg,
        {
            "head": "type",
            "candidate": "logreg",
            "run_id": "r1",
            "is_baseline": False,
            "metrics": {"roc_auc": 0.9, "brier_after": 0.1, "latency_ms_p95": 5},
            "params": {"C": 1},
            "feature_schema_hash": "abc",
            "banned_features": [],
        },
    )
    write_run_record(
        cfg,
        {
            "head": "type",
            "candidate": "dummy",
            "run_id": "r0",
            "is_baseline": True,
            "metrics": {"roc_auc": 0.5, "brier_after": 0.2, "latency_ms_p95": 1},
            "params": {},
            "feature_schema_hash": "abc",
            "banned_features": [],
        },
    )
    board = write_leaderboard(cfg)
    assert board["n_runs"] >= 2
    setup_mlflow(cfg)
    log_mlflow_run(
        {"head": "type", "candidate": "logreg", "metrics": {"roc_auc": 0.9}, "params": {}}
    )
    decisions = promote_all(cfg)
    assert "type" in decisions["decisions"]


def test_orchestrator_controls(cfg, monkeypatch):
    monkeypatch.setattr("src.pipeline.orchestrator.load_config", lambda: cfg)
    with pytest.raises(ValueError):
        run_pipeline(only="not-a-stage")
    with pytest.raises(ValueError):
        run_pipeline(from_stage="not-a-stage")
    result = run_pipeline(only="ingest", force=True)
    assert result[0]["stage"] == "ingest"


def test_api_routes_and_predictions(cfg, featured):
    stage_ingest(cfg, force=True)
    stage_features(cfg, force=True)
    stage_insights(cfg, force=True)
    state = load_state(cfg)
    registry = build_registry(cfg)
    numeric = ["release_year", "n_genres", "title_length"]
    categorical = ["primary_country"]
    work = featured.copy()
    work["maturity_level"] = work["maturity_level"].astype("string").fillna("Teens")
    state.models["type"] = _fit_head(
        work, numeric, categorical, "type", "classification", ["TV Show", "Movie"]
    )
    state.models["type"]["feature_schema_hash"] = registry["schema_hash"]
    state.registry = registry
    state.models["rating"] = _fit_head(
        work,
        numeric,
        categorical,
        "maturity_level",
        "classification",
        ["Kids", "Older Kids", "Teens", "Mature"],
    )
    state.models["acquisition_reg"] = _fit_head(
        work, numeric, categorical, "years_to_platform", "regression"
    )
    state.models["acquisition_clf"] = _fit_head(
        work,
        numeric,
        categorical,
        "acquisition_bucket",
        "classification",
        ["Day-one", "Fast", "Recent", "Back-catalogue"],
    )
    state.forecast = {"best": "seasonal_naive", "horizon_point": [1, 2], "baseline_wins": True}
    state.drift = {"performance_drift": {"type": {"degradation": 0.01}}}
    state.ready = True

    app = create_app()
    with TestClient(app) as client:
        load_state(cfg)
        # reload models wiped by lifespan; put them back
        state2 = load_state(cfg)
        state2.models = state.models
        state2.forecast = state.forecast
        state2.drift = state.drift
        state2.ready = True
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 200
        assert client.get("/meta").status_code == 200
        assert client.get("/models").status_code == 200
        assert client.get("/models/type/card").status_code == 200
        payload = {
            "title": "US Doc 1",
            "listed_in": "Documentaries",
            "release_year": 2015,
            "country": "United States",
            "rating": "PG-13",
        }
        assert client.post("/predict/type", json=payload).status_code == 200
        assert client.post("/predict/rating", json=payload).status_code == 200
        acq = client.post("/predict/acquisition", json=payload)
        assert acq.status_code == 200
        body = acq.json()
        assert body["is_proxy"] is True
        assert body["interval_lo"] <= body["years_to_platform"] <= body["interval_hi"]
        assert client.post("/segment", json=payload).status_code == 200
        analyzed = client.post("/analyze", json=payload)
        assert analyzed.status_code == 200
        assert analyzed.json()["acquisition_priority"]["is_proxy"] is True
        assert (
            client.post("/predict/type", json={**payload, "duration": "90 min"}).status_code == 422
        )
        assert (
            client.post("/predict/acquisition", json={**payload, "year_added": 2020}).status_code
            == 422
        )
        assert client.get("/insights").status_code == 200
        assert client.get("/stats/overview").status_code == 200
        assert client.get("/stats/timeseries", params={"metric": "titles_added"}).status_code == 200
        assert client.get("/stats/segments", params={"by": "type"}).status_code == 200
        assert client.get("/forecast").status_code == 200
        assert client.get("/drift").status_code == 200
        assert "ncae_titles" in client.get("/metrics").text
