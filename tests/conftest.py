from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest
from src.config import load_config, reset_config_cache

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "mini_netflix.csv"


@pytest.fixture
def cfg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("PROJECT_ROOT", str(ROOT))
    monkeypatch.setenv("NETFLIX_FAST", "1")
    reset_config_cache()
    config = load_config(str(ROOT))
    config.settings.fast_mode = True
    # Redirect writable outputs into the temp dir.
    config.base["paths"] = {
        **config.base["paths"],
        "interim_dir": str(tmp_path / "interim"),
        "processed_dir": str(tmp_path / "processed"),
        "features_dir": str(tmp_path / "features"),
        "artifacts_dir": str(tmp_path / "artifacts"),
        "reports_dir": str(tmp_path / "reports"),
        "registry_dir": str(tmp_path / "registry"),
        "mlruns_dir": str(tmp_path / "mlruns"),
        "raw_csv": str(tmp_path / "raw" / "netflix_titles.csv"),
    }
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(FIXTURE, raw_dir / "netflix_titles.csv")
    for key in (
        "interim_dir",
        "processed_dir",
        "features_dir",
        "artifacts_dir",
        "reports_dir",
        "registry_dir",
        "mlruns_dir",
    ):
        Path(config.base["paths"][key]).mkdir(parents=True, exist_ok=True)
    return config


@pytest.fixture
def raw_frame() -> pd.DataFrame:
    return pd.read_csv(FIXTURE, dtype=str)


@pytest.fixture
def featured(cfg):
    from src.data.load import load_raw
    from src.features.engineer import engineer_features

    raw = load_raw(cfg.path("raw_csv"), cfg)
    return engineer_features(raw, cfg)
