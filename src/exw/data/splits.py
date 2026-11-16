"""Grouped, stratified splits. Splits are made on whole groups (site, farm or date), BEFORE tiling."""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_splits(manifest: pd.DataFrame, group_by: str = "site_id", fractions=(0.70, 0.15, 0.15),
                seed: int = 42, holdout_styles=()) -> pd.DataFrame:
    """Return a copy of ``manifest`` with a ``split`` column in {train, val, test, test_shift}.

    * all images of a group go to the same split (no leakage between neighbouring views);
    * groups are stratified by their most severe farm label;
    * groups whose style is in ``holdout_styles`` form the synthetic domain-shift set ``test_shift``.
    """
    df = manifest.copy()
    rng = np.random.default_rng(seed)
    rank = {"healthy": 0, "other_condition": 1, "exw_suspect": 2}
    g = df.groupby(group_by).agg(label=("farm_label", lambda s: max(s, key=rank.get)),
                                 style=("style", "first")).reset_index()
    assignment: dict = {}
    shift = g[g["style"].isin(list(holdout_styles))]
    for gid in shift[group_by]:
        assignment[gid] = "test_shift"
    rest = g[~g[group_by].isin(assignment)]
    for _, sub in rest.groupby("label"):
        ids = list(sub[group_by])
        rng.shuffle(ids)
        n = len(ids)
        n_val = max(1, round(n * fractions[1])) if n >= 3 else 0
        n_test = max(1, round(n * fractions[2])) if n >= 3 else 0
        for i, gid in enumerate(ids):
            assignment[gid] = "val" if i < n_val else "test" if i < n_val + n_test else "train"
    df["split"] = df[group_by].map(assignment)
    assert_no_leakage(df, group_by)
    return df


def assert_no_leakage(df: pd.DataFrame, group_by: str = "site_id") -> None:
    """Raise if any group appears in more than one split."""
    per_group = df.groupby(group_by)["split"].nunique()
    bad = per_group[per_group > 1]
    if len(bad):
        raise AssertionError(f"leakage: groups in several splits: {list(bad.index)[:5]}")
    # independent check on the weakest unit, the farm
    per_farm = df.groupby("farm_id")["split"].nunique()
    if (per_farm > 1).any():
        raise AssertionError("leakage: a farm appears in several splits")
