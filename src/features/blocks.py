"""Column groups consumed by model pipelines, honouring per-head bans."""

from __future__ import annotations

from src.config import AppConfig
from src.data.clean import split_list
from src.features.registry import allowed_columns

NUMERIC_CANDIDATES = [
    "release_year",
    "duration_minutes",
    "duration_seasons",
    "n_genres",
    "n_countries",
    "n_directors",
    "n_cast",
    "description_length",
    "title_length",
    "title_word_count",
    "catalogue_prominence",
    "year_added",
    "month_added",
    "weekday_added",
    "years_to_platform",
]
BINARY_CANDIDATES = [
    "has_director",
    "has_country",
    "has_cast",
    "has_description",
    "is_movie",
    "is_likely_original",
]
CATEGORICAL_CANDIDATES = [
    "type",
    "rating",
    "maturity_level",
    "acquisition_bucket",
    "primary_country",
    "primary_genre",
]
TEXT_CANDIDATES = ["title", "listed_in", "description"]


def filter_allowed(names: list[str], registry: dict, head: str) -> list[str]:
    allow = set(allowed_columns(registry, head))
    return [n for n in names if n in allow]


def genre_allowed_for_type(genre: str, cfg: AppConfig) -> bool:
    rules = cfg.features["leakage"]["type"]
    if genre in rules.get("banned_genre_exact", []):
        return False
    for token in rules.get("banned_genre_contains", []):
        if token in genre:
            return False
    return True


def safe_genres_for_type(listed_in: object, cfg: AppConfig) -> str:
    kept = [g for g in split_list(listed_in) if genre_allowed_for_type(g, cfg)]
    return " ".join(kept)
