"""Load the raw catalogue listing and normalise column presence."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import AppConfig
from src.logging_conf import get_logger

log = get_logger(__name__)

OPTIONAL = ("cast", "description")


def load_raw(path: Path, cfg: AppConfig) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Raw catalogue not found at {path}")
    frame = pd.read_csv(path, dtype=str, keep_default_na=True)
    expected = list(cfg.base["dataset"]["expected_columns"])
    missing = [col for col in expected if col not in frame.columns]
    if missing:
        raise ValueError(f"Raw file is missing required columns: {missing}")
    for col in OPTIONAL:
        if col not in frame.columns:
            frame[col] = ""
            log.info("optional_column_absent", column=col, filled_with="empty_string")
    if "release_year" in frame.columns:
        frame["release_year"] = pd.to_numeric(frame["release_year"], errors="coerce")
    return frame
