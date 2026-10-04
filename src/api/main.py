"""FastAPI application. Artifacts load once in the lifespan handler."""

from __future__ import annotations

import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import structlog
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse

from src import __version__
from src.api.deps import get_state, load_state
from src.api.predict import predict_acquisition, predict_rating, predict_segment, predict_type
from src.api.schemas import AnalyzeOut, HealthOut, MetaOut, TitleIn
from src.config import load_config
from src.logging_conf import configure_logging, get_logger
from src.targets.definitions import PROXY_NOTES
from src.tracking.mlflow_utils import load_run_records
from src.tracking.registry import read_incumbent

log = get_logger(__name__)
REQUESTS: dict[str, list[float]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging(os.environ.get("LOG_LEVEL", "INFO"))
    load_state()
    yield


def create_app() -> FastAPI:
    cfg = load_config()
    origins = [o.strip() for o in cfg.settings.cors_origins.split(",") if o.strip()]
    app = FastAPI(
        title="Netflix Catalogue Analytics API",
        description=(
            "Analyses catalogue composition and acquisition behaviour — "
            "not audience success. Proxy fields carry is_proxy=true."
        ),
        version=__version__,
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins or ["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def request_id_and_limit(request: Request, call_next):
        rid = request.headers.get("X-Request-ID", uuid.uuid4().hex)
        structlog.contextvars.bind_contextvars(request_id=rid, path=request.url.path)
        limit = int(os.environ.get("RATE_LIMIT_PER_MINUTE", "60"))
        bucket = REQUESTS.setdefault(request.client.host if request.client else "anon", [])
        now = time.time()
        REQUESTS[request.client.host if request.client else "anon"] = [
            t for t in bucket if now - t < 60
        ]
        if len(REQUESTS[request.client.host if request.client else "anon"]) >= limit:
            return JSONResponse({"detail": "rate limit exceeded"}, status_code=429)
        REQUESTS[request.client.host if request.client else "anon"].append(now)
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            log.exception("unhandled_error", request_id=rid)
            return JSONResponse({"detail": "internal error", "request_id": rid}, status_code=500)
        response.headers["X-Request-ID"] = rid
        log.info(
            "http_access",
            status=response.status_code,
            duration_ms=round((time.perf_counter() - started) * 1000, 2),
        )
        return response

    @app.get("/health", response_model=HealthOut, tags=["ops"])
    def health() -> HealthOut:
        return HealthOut(status="ok")

    @app.get("/ready", tags=["ops"])
    def ready() -> dict[str, Any]:
        state = get_state()
        if not state.ready:
            raise HTTPException(503, "feature store not loaded")
        return {"status": "ready", "models": list(state.models.keys())}

    @app.get("/meta", response_model=MetaOut, tags=["ops"])
    def meta() -> MetaOut:
        state = get_state()
        promoted = {h: read_incumbent(state.cfg, h) for h in state.cfg.models["heads"]}
        return MetaOut(
            version=__version__,
            git_sha=state.git_sha,
            feature_schema_hash=(state.registry or {}).get("schema_hash"),
            promoted=promoted,
        )

    @app.get("/models", tags=["models"])
    def models() -> list[dict[str, Any]]:
        state = get_state()
        records = load_run_records(state.cfg)
        out = []
        for rec in records:
            inc = read_incumbent(state.cfg, rec["head"])
            out.append(
                {
                    **{
                        k: rec.get(k)
                        for k in ("head", "candidate", "run_id", "metrics", "is_baseline")
                    },
                    "promoted": bool(inc and inc.get("run_id") == rec.get("run_id")),
                }
            )
        return out

    @app.get("/models/{head}/card", tags=["models"])
    def model_card(head: str) -> dict[str, Any]:
        state = get_state()
        path = state.cfg.root / "docs" / "MODEL_CARD.md"
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        inc = read_incumbent(state.cfg, head)
        return {"head": head, "incumbent": inc, "card_markdown": text}

    @app.post("/predict/type", tags=["predict"])
    def pred_type(payload: TitleIn) -> dict[str, Any]:
        return predict_type(get_state(), payload)

    @app.post("/predict/rating", tags=["predict"])
    def pred_rating(payload: TitleIn) -> dict[str, Any]:
        return predict_rating(get_state(), payload)

    @app.post("/predict/acquisition", tags=["predict"])
    def pred_acq(payload: TitleIn) -> dict[str, Any]:
        return predict_acquisition(get_state(), payload)

    @app.post("/segment", tags=["predict"])
    def pred_seg(payload: TitleIn) -> dict[str, Any]:
        return predict_segment(get_state(), payload)

    @app.post("/analyze", response_model=AnalyzeOut, tags=["predict"])
    def analyze(payload: TitleIn) -> AnalyzeOut:
        state = get_state()
        typ = None
        rat = None
        try:
            typ_raw = predict_type(state, payload)
            typ = typ_raw  # type: ignore[assignment]
        except HTTPException:
            typ = None
        try:
            rat = predict_rating(state, payload)
        except HTTPException:
            rat = None
        try:
            acq = predict_acquisition(state, payload)
        except HTTPException:
            acq = {
                "years_to_platform": None,
                "interval_lo": None,
                "interval_hi": None,
                "bucket": None,
                "is_likely_original": None,
                "is_proxy": True,
                "proxy_note": PROXY_NOTES["acquisition_priority"],
            }
        seg = predict_segment(state, payload)
        insights = (state.insights.get("findings") or [])[:3]
        summary = (
            f"{payload.title} is analysed as catalogue metadata only. "
            f"Acquisition-priority (proxy): {acq.get('years_to_platform')} years "
            f"[{acq.get('interval_lo')}, {acq.get('interval_hi')}], bucket={acq.get('bucket')}."
        )
        return AnalyzeOut(
            title=payload.title,
            type=typ,  # type: ignore[arg-type]
            rating=rat,  # type: ignore[arg-type]
            acquisition_priority=acq,  # type: ignore[arg-type]
            segment=seg,  # type: ignore[arg-type]
            catalogue_prominence={
                "value": None,
                "is_proxy": True,
                "proxy_note": PROXY_NOTES["catalogue_prominence"],
            },
            insights=insights,
            summary=summary,
        )

    @app.get("/insights", tags=["insights"])
    def insights(
        theme: str | None = None,
        limit: int = Query(20, ge=1, le=200),
        min_effect: float = 0.0,
    ) -> dict[str, Any]:
        state = get_state()
        items = list(state.insights.get("findings") or [])
        if theme:
            items = [i for i in items if i.get("theme") == theme]
        items = [i for i in items if abs(float(i.get("effect") or 0)) >= min_effect]
        return {
            "n_candidates": state.insights.get("n_candidates"),
            "n_survivors": state.insights.get("n_survivors"),
            "items": items[:limit],
        }

    @app.get("/insights/{finding_id}", tags=["insights"])
    def insight_one(finding_id: int) -> dict[str, Any]:
        state = get_state()
        items = list(state.insights.get("findings") or [])
        if finding_id < 0 or finding_id >= len(items):
            raise HTTPException(404, "finding not found")
        return items[finding_id]

    @app.get("/stats/overview", tags=["stats"])
    def overview() -> dict[str, Any]:
        state = get_state()
        if state.store is None:
            raise HTTPException(503, "feature store not loaded")
        df = state.store
        ytp = df["years_to_platform"] if "years_to_platform" in df.columns else None
        recent = 0.0
        if "year_added" in df.columns:
            ymax = pd_to_num_max(df["year_added"])
            recent = float((df["year_added"] >= ymax - 2).mean()) if ymax else 0.0
        return {
            "n_titles": int(len(df)),
            "n_movies": int((df["type"] == "Movie").sum()) if "type" in df.columns else 0,
            "n_tv": int((df["type"] == "TV Show").sum()) if "type" in df.columns else 0,
            "n_countries": (
                int(df["primary_country"].nunique()) if "primary_country" in df.columns else 0
            ),
            "n_genres": int(df["primary_genre"].nunique()) if "primary_genre" in df.columns else 0,
            "median_acquisition_lag": float(ytp.median()) if ytp is not None else None,
            "share_added_last_3y": recent,
            "is_proxy": True,
            "proxy_note": PROXY_NOTES["years_to_platform"],
        }

    @app.get("/stats/timeseries", tags=["stats"])
    def timeseries(metric: str = "titles_added", freq: str = "Y") -> dict[str, Any]:
        state = get_state()
        if state.store is None:
            raise HTTPException(503, "feature store not loaded")
        df = state.store
        year = df["year_added"] if "year_added" in df.columns else None
        if year is None:
            return {"metric": metric, "points": []}
        if metric == "titles_added":
            vc = year.dropna().astype(int).value_counts().sort_index()
            return {"metric": metric, "points": [{"t": int(k), "v": int(v)} for k, v in vc.items()]}
        if metric in df.columns:
            g = df.groupby(year)[metric].mean()
            return {
                "metric": metric,
                "points": [{"t": int(k), "v": float(v)} for k, v in g.items() if pd_ok(k)],
            }
        raise HTTPException(422, f"unknown metric {metric}")

    @app.get("/stats/segments", tags=["stats"])
    def segments(by: str = "primary_country") -> dict[str, Any]:
        state = get_state()
        if state.store is None or by not in state.store.columns:
            raise HTTPException(422, f"unknown segment {by}")
        vc = state.store[by].astype(str).value_counts().head(25)
        return {"by": by, "counts": [{"key": k, "n": int(v)} for k, v in vc.items()]}

    @app.get("/forecast", tags=["stats"])
    def forecast(horizon: int = 6) -> dict[str, Any]:
        state = get_state()
        data = dict(state.forecast)
        data["requested_horizon"] = horizon
        return data

    @app.get("/drift", tags=["ops"])
    def drift() -> dict[str, Any]:
        return get_state().drift

    @app.get("/metrics", response_class=PlainTextResponse, tags=["ops"])
    def metrics() -> str:
        state = get_state()
        n = len(state.store) if state.store is not None else 0
        return (
            "# HELP ncae_titles Catalogue size\n"
            "# TYPE ncae_titles gauge\n"
            f"ncae_titles {n}\n"
            "# HELP ncae_models_loaded Promoted/loaded heads\n"
            "# TYPE ncae_models_loaded gauge\n"
            f"ncae_models_loaded {len(state.models)}\n"
        )

    return app


def pd_to_num_max(series) -> int:
    import pandas as pd

    s = pd.to_numeric(series, errors="coerce").dropna()
    return int(s.max()) if len(s) else 0


def pd_ok(value: object) -> bool:
    try:
        int(str(value))
        return True
    except (TypeError, ValueError):
        return False


app = create_app()
