"""Local-file MLflow helpers. A JSON sidecar is always written so reports do not depend on the server."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from src.config import AppConfig
from src.logging_conf import get_logger
from src.pipeline.cache import git_sha, utc_now

log = get_logger(__name__)


def setup_mlflow(cfg: AppConfig) -> str:
    raw_uri = cfg.settings.mlflow_tracking_uri or str(cfg.path("mlruns_dir"))
    if "://" in raw_uri:
        uri = raw_uri
    else:
        uri = Path(raw_uri).resolve().as_uri()
    Path(cfg.path("mlruns_dir")).mkdir(parents=True, exist_ok=True)
    os.environ["MLFLOW_TRACKING_URI"] = uri
    try:
        import mlflow

        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(cfg.base["project"]["name"])
    except Exception as exc:  # pragma: no cover - optional dependency
        log.warning("mlflow_unavailable", error=str(exc))
    return uri


def runs_dir(cfg: AppConfig) -> Path:
    path = cfg.path("artifacts_dir") / "runs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_run_record(cfg: AppConfig, record: dict[str, Any]) -> Path:
    record = {
        **record,
        "git_sha": record.get("git_sha") or git_sha(cfg.root),
        "written_at": utc_now(),
    }
    path = (
        runs_dir(cfg)
        / f"{record['head']}__{record['candidate']}__{record.get('run_id', 'local')}.json"
    )
    path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    return path


def load_run_records(cfg: AppConfig) -> list[dict[str, Any]]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(runs_dir(cfg).glob("*.json"))]


def log_mlflow_run(record: dict[str, Any], artifact_dir: Path | None = None) -> None:
    try:
        import mlflow

        with mlflow.start_run(run_name=f"{record['head']}-{record['candidate']}"):
            mlflow.set_tags(
                {
                    "head": record.get("head", ""),
                    "candidate": record.get("candidate", ""),
                    "run_type": record.get("run_type", "train"),
                }
            )
            params = record.get("params") or {}
            mlflow.log_params({k: str(v)[:250] for k, v in params.items()})
            metrics = {
                k: float(v) for k, v in (record.get("metrics") or {}).items() if _is_number(v)
            }
            mlflow.log_metrics(metrics)
            mlflow.log_param("feature_schema_hash", record.get("feature_schema_hash", ""))
            mlflow.log_param("git_sha", record.get("git_sha", ""))
            mlflow.log_param("data_sha", record.get("data_sha", ""))
            if artifact_dir and artifact_dir.exists():
                mlflow.log_artifacts(str(artifact_dir))
    except Exception as exc:  # pragma: no cover
        log.warning("mlflow_log_failed", error=str(exc))


def _is_number(value: object) -> bool:
    try:
        float(value)  # type: ignore[arg-type]
        return True
    except (TypeError, ValueError):
        return False
