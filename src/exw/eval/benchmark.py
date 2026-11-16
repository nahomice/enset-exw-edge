"""Latency benchmark with an honest protocol: warm-up, many repeats, median and spread."""

from __future__ import annotations

import time

import numpy as np


def time_per_tile(predictor, tiles: np.ndarray, warmup: int = 5, repeats: int = 5) -> dict:
    predictor.logits(tiles[:warmup])
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        predictor.logits(tiles)
        ts.append((time.perf_counter() - t0) / len(tiles))
    ts = np.array(ts) * 1e3
    return dict(backend=predictor.name, ms_per_tile_median=float(np.median(ts)),
                ms_per_tile_min=float(ts.min()), ms_per_tile_max=float(ts.max()), n_tiles=len(tiles))
