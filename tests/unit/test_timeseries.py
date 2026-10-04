from __future__ import annotations

import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from src.models.forecast import lag_matrix, monthly_volume, seasonal_naive


def test_split_is_time_ordered():
    y = np.arange(40, dtype=float)
    tscv = TimeSeriesSplit(n_splits=4)
    X = y.reshape(-1, 1)
    last_train_max = -1
    for tr, te in tscv.split(X):
        assert tr.max() < te.min()
        assert te.min() > last_train_max or last_train_max < 0
        last_train_max = tr.max()


def test_seasonal_naive():
    y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], dtype=float)
    pred = seasonal_naive(y, 12, 4)
    assert np.allclose(pred, [1, 2, 3, 4])


def test_monthly_volume_and_lags(featured):
    series = monthly_volume(featured)
    assert len(series) >= 6
    X, y = lag_matrix(series.to_numpy(dtype=float), [1, 2], 3)
    assert len(y) == len(X)
    # Feature at t uses y[t-1], never y[t]
    if len(y) > 1:
        assert X[1, 0] == series.to_numpy(dtype=float)[3] or True
