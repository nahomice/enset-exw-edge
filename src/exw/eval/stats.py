"""Uncertainty: bootstrap by group (farm), so views of one farm are never treated as independent."""

from __future__ import annotations

import numpy as np


def grouped_bootstrap(groups: np.ndarray, stat_fn, n_boot: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    """``stat_fn(idx)`` computes a statistic on the sample indices ``idx``. Returns (point, lo95, hi95)."""
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    members = {g: np.where(groups == g)[0] for g in ug}
    point = stat_fn(np.arange(len(groups)))
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(ug, size=len(ug), replace=True)
        v = stat_fn(np.concatenate([members[g] for g in pick]))
        if not np.isnan(v):
            vals.append(v)
    lo, hi = (np.percentile(vals, [2.5, 97.5]) if vals else (np.nan, np.nan))
    return float(point), float(lo), float(hi)
