"""Load and validate the four YAML configs. No magic numbers live in code."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def find_project_root(start: Path | None = None) -> Path:
    """Walk up from *start* (or cwd) until pyproject.toml is found."""
    here = (start or Path.cwd()).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "pyproject.toml").exists() and (
            candidate / "config" / "base.yaml"
        ).exists():
            return candidate
    env_root = os.environ.get("PROJECT_ROOT")
    if env_root:
        return Path(env_root).resolve()
    return here


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: str = "development"
    log_level: str = "INFO"
    random_seed: int = 42
    port: int = 8000
    api_base_url: str = "http://127.0.0.1:8000"
    cors_origins: str = "http://localhost:8501,http://127.0.0.1:8501"
    enrichment_enabled: bool = False
    tmdb_api_key: str = ""
    omdb_api_key: str = ""
    insight_llm_polish: bool = False
    openai_api_key: str = ""
    rate_limit_per_minute: int = 60
    mlflow_tracking_uri: str = ""
    fast_mode: bool = False


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    raw: dict[str, Any]
    base: dict[str, Any]
    features: dict[str, Any]
    models: dict[str, Any]
    insights: dict[str, Any]
    root: Path
    settings: AppSettings = Field(default_factory=AppSettings)

    @property
    def seed(self) -> int:
        return int(self.base.get("project", {}).get("random_seed", 42))

    def path(self, key: str) -> Path:
        rel = self.base["paths"][key]
        return (self.root / rel).resolve()


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML at {path} must be a mapping")
    return data


@lru_cache(maxsize=4)
def load_config(root: str | None = None) -> AppConfig:
    project_root = Path(root).resolve() if root else find_project_root()
    cfg_dir = project_root / "config"
    settings = AppSettings()
    if settings.fast_mode or os.environ.get("NETFLIX_FAST") == "1":
        settings.fast_mode = True
    return AppConfig(
        raw={
            "base": _read_yaml(cfg_dir / "base.yaml"),
            "features": _read_yaml(cfg_dir / "features.yaml"),
            "models": _read_yaml(cfg_dir / "models.yaml"),
            "insights": _read_yaml(cfg_dir / "insights.yaml"),
        },
        base=_read_yaml(cfg_dir / "base.yaml"),
        features=_read_yaml(cfg_dir / "features.yaml"),
        models=_read_yaml(cfg_dir / "models.yaml"),
        insights=_read_yaml(cfg_dir / "insights.yaml"),
        root=project_root,
        settings=settings,
    )


def reset_config_cache() -> None:
    load_config.cache_clear()
