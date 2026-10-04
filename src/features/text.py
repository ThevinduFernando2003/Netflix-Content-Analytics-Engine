"""Text helpers. TF-IDF + TruncatedSVD belong inside model pipelines, never PCA on sparse input."""

from __future__ import annotations

from typing import Any

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline


class _CappedTruncatedSVD(BaseEstimator, TransformerMixin):
    """Shrink n_components when a small catalogue has fewer tokens than the config asks for."""

    def __init__(self, n_components: int = 24, random_state: int = 42) -> None:
        self.n_components = n_components
        self.random_state = random_state

    def _n(self, X: Any) -> int:
        width = int(X.shape[1])
        return min(int(self.n_components), max(1, width - 1))

    def fit(self, X: Any, y: Any = None) -> _CappedTruncatedSVD:
        self.svd_ = TruncatedSVD(n_components=self._n(X), random_state=self.random_state)
        self.svd_.fit(X, y)
        return self

    def transform(self, X: Any) -> Any:
        return self.svd_.transform(X)


def text_svd_pipeline(
    max_features: int = 400,
    n_components: int = 24,
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 2,
    random_state: int = 42,
) -> Pipeline:
    n_components = max(2, min(n_components, max_features - 1))
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=max_features,
                    ngram_range=ngram_range,
                    min_df=min_df,
                    stop_words="english",
                ),
            ),
            (
                "svd",
                _CappedTruncatedSVD(n_components=n_components, random_state=random_state),
            ),
        ]
    )


def combine_text(frame: object, columns: list[str]) -> list[str]:
    texts: list[str] = []
    for _, row in frame.iterrows():  # type: ignore[attr-defined]
        parts = [str(row[c]) for c in columns if c in row.index and str(row[c]).strip()]
        texts.append(" ".join(parts))
    return texts
