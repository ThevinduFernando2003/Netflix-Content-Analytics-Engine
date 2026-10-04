"""Deterministic, non-learned feature engineering. Learned transforms stay in model pipelines."""

from __future__ import annotations

import pandas as pd

from src.config import AppConfig
from src.data.clean import clean_catalogue
from src.targets.build import add_catalogue_prominence, add_targets


def _primary(items: object) -> str:
    if isinstance(items, list) and items:
        return str(items[0])
    return ""


def engineer_features(raw: pd.DataFrame, cfg: AppConfig) -> pd.DataFrame:
    cleaned, _notes = clean_catalogue(raw, cfg)
    featured = add_targets(cleaned, cfg)
    featured = add_catalogue_prominence(featured, cfg)
    featured["primary_country"] = featured["countries"].map(_primary)
    featured["primary_genre"] = featured["genres"].map(_primary)
    if "portfolio_fit" not in featured.columns:
        featured["portfolio_fit"] = pd.NA
    if "cluster_id" not in featured.columns:
        featured["cluster_id"] = pd.NA
    return featured
