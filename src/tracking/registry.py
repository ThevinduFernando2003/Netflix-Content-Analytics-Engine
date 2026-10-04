"""On-disk model registry under registry/."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from src.config import AppConfig
from src.pipeline.cache import utc_now


def registry_root(cfg: AppConfig) -> Path:
    path = cfg.path("registry_dir")
    path.mkdir(parents=True, exist_ok=True)
    return path


def head_dir(cfg: AppConfig, head: str) -> Path:
    path = registry_root(cfg) / head
    path.mkdir(parents=True, exist_ok=True)
    return path


def current_pointer(cfg: AppConfig, head: str) -> Path:
    return head_dir(cfg, head) / "CURRENT.json"


def read_incumbent(cfg: AppConfig, head: str) -> dict[str, Any] | None:
    path = current_pointer(cfg, head)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_promotion(
    cfg: AppConfig, head: str, record: dict[str, Any], model_src: Path | None
) -> Path:
    dest = head_dir(cfg, head) / record.get("run_id", "promoted")
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "record.json").write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    if model_src and model_src.exists():
        target = dest / model_src.name
        if model_src.resolve() != target.resolve():
            shutil.copy2(model_src, target)
        record = {**record, "model_path": str(target)}
        (dest / "record.json").write_text(
            json.dumps(record, indent=2, default=str), encoding="utf-8"
        )
    pointer = {
        "head": head,
        "run_id": record.get("run_id"),
        "candidate": record.get("candidate"),
        "metrics": record.get("metrics"),
        "promoted_at": utc_now(),
        "dir": str(dest),
        "previous": read_incumbent(cfg, head),
    }
    current_pointer(cfg, head).write_text(
        json.dumps(pointer, indent=2, default=str), encoding="utf-8"
    )
    return dest


def rollback(cfg: AppConfig, head: str) -> dict[str, Any]:
    current = read_incumbent(cfg, head)
    if not current or not current.get("previous"):
        raise ValueError(f"No previous model to roll back for head '{head}'")
    previous = current["previous"]
    current_pointer(cfg, head).write_text(
        json.dumps(previous, indent=2, default=str), encoding="utf-8"
    )
    return previous
