"""Shared task interface so the leaderboard and registry stay generic."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


class BaseModelTask(ABC):
    name: str
    target: str | None

    @abstractmethod
    def build_pipeline(self, candidate: str) -> Pipeline: ...

    @abstractmethod
    def fit(self, pipeline: Pipeline, X: pd.DataFrame, y: np.ndarray) -> Pipeline: ...

    @abstractmethod
    def predict(self, pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray: ...

    @abstractmethod
    def evaluate(self, pipeline: Pipeline, X: pd.DataFrame, y: np.ndarray) -> dict[str, float]: ...

    def explain(self, pipeline: Pipeline, X: pd.DataFrame) -> dict[str, Any]:
        return {}
