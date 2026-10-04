from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from src.api.deps import load_state
from src.api.main import create_app
from src.pipeline.stages import stage_features, stage_ingest, stage_insights


@pytest.fixture
def client(cfg):
    stage_ingest(cfg, force=True)
    stage_features(cfg, force=True)
    stage_insights(cfg, force=True)
    app = create_app()
    with TestClient(app) as test_client:
        load_state(cfg)
        yield test_client


def test_health_ready(client):
    assert client.get("/health").json()["status"] == "ok"
    ready = client.get("/ready")
    assert ready.status_code in {200, 503}


def test_meta_and_insights(client):
    meta = client.get("/meta")
    assert meta.status_code == 200
    ins = client.get("/insights", params={"limit": 5, "min_effect": 0.0})
    assert ins.status_code == 200
    body = ins.json()
    assert "items" in body


def test_leaky_type_rejected(client):
    r = client.post(
        "/predict/type", json={"title": "x", "duration": "90 min", "release_year": 2018}
    )
    assert r.status_code == 422
    assert "duration" in str(r.json()["detail"]).lower()


def test_leaky_acquisition_rejected(client):
    r = client.post(
        "/predict/acquisition",
        json={"title": "x", "date_added": "1/1/2020", "release_year": 2018},
    )
    assert r.status_code == 422
    assert "date_added" in str(r.json()["detail"]).lower()


def test_invalid_field_422(client):
    r = client.get("/stats/timeseries", params={"metric": "not_a_metric"})
    assert r.status_code == 422


def test_analyze_proxy_flags(client):
    r = client.post(
        "/analyze",
        json={
            "title": "US Doc 1",
            "listed_in": "Documentaries",
            "release_year": 2015,
            "country": "United States",
        },
    )
    # 503 if models missing is acceptable in this fixture; 200 must carry proxy flags
    assert r.status_code in {200, 503}
    if r.status_code == 200:
        body = r.json()
        assert body["acquisition_priority"]["is_proxy"] is True
        assert "success" in body["acquisition_priority"]["proxy_note"].lower()


def test_adversarial_inputs_no_500(client):
    payloads = [
        {"title": "", "description": ""},
        {"title": "x" * 10000, "description": "🔥" * 50},
        {"title": "'; DROP TABLE titles;--", "listed_in": "../etc/passwd"},
        {"title": "old", "release_year": 1900, "listed_in": ",".join(["g"] * 50)},
    ]
    for payload in payloads:
        r = client.post("/analyze", json=payload)
        assert r.status_code < 500


def test_overview_and_metrics(client):
    ov = client.get("/stats/overview")
    assert ov.status_code == 200
    m = client.get("/metrics")
    assert m.status_code == 200
    assert "ncae_titles" in m.text


def test_insight_pagination(client):
    r = client.get("/insights", params={"limit": 1})
    assert r.status_code == 200
    assert len(r.json()["items"]) <= 1
