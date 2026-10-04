"""Shared sklearn preprocessors. Every learned transform is fitted per fold."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features.text import text_svd_pipeline


def make_preprocessor(
    numeric: list[str],
    categorical: list[str],
    text_col: str | None,
    seed: int,
    max_features: int = 400,
    svd_components: int = 24,
) -> ColumnTransformer:
    transformers: list[tuple] = []
    if numeric:
        transformers.append(
            (
                "num",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
                        (
                            "ohe",
                            OneHotEncoder(
                                handle_unknown="ignore", sparse_output=True, min_frequency=5
                            ),
                        ),
                    ]
                ),
                categorical,
            )
        )
    if text_col:
        transformers.append(
            (
                "text",
                text_svd_pipeline(
                    max_features=max_features,
                    n_components=min(svd_components, max(2, max_features - 1)),
                    random_state=seed,
                ),
                text_col,
            )
        )
    return ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.0)


def make_estimator(kind: str, task: str, seed: int) -> object:
    if kind == "dummy":
        return DummyClassifier(strategy="most_frequent", random_state=seed)
    if kind == "dummy_median":
        return DummyRegressor(strategy="median")
    if task in {"binary_classification", "ordinal_classification", "multiclass"}:
        if kind == "logreg":
            return LogisticRegression(max_iter=400, random_state=seed)
        if kind == "random_forest":
            return RandomForestClassifier(random_state=seed, n_jobs=-1)
        if kind == "extra_trees":
            return ExtraTreesClassifier(random_state=seed, n_jobs=-1)
        if kind == "hist_gb":
            return HistGradientBoostingClassifier(random_state=seed)
    if task == "regression":
        if kind == "ridge":
            return Ridge()
        if kind == "random_forest":
            return RandomForestRegressor(random_state=seed, n_jobs=-1)
        if kind == "extra_trees":
            return ExtraTreesRegressor(random_state=seed, n_jobs=-1)
        if kind == "hist_gb":
            return HistGradientBoostingRegressor(random_state=seed)
    raise ValueError(f"Unknown estimator {kind} for task {task}")


def wrap(preprocessor: ColumnTransformer, estimator: object) -> Pipeline:
    return Pipeline(steps=[("prep", preprocessor), ("model", estimator)])
