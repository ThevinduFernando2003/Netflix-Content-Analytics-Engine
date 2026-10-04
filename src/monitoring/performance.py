"""Helpers for latency and segment-wise metric checks used by promotion gates."""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np


def p95_latency_ms(fn: Callable[[], object], n: int = 30) -> float:
    samples = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000)
    return float(np.quantile(samples, 0.95))
