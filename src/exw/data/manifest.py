"""Manifest (one row per image) and annotation I/O. The manifest is the single source of truth for splits."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from exw.data.splits import make_splits
from exw.data.synthetic import generate_dataset

MANIFEST = "manifest.csv"
ANNOTATIONS = "annotations.json"


def build_synthetic_manifest(cfg: dict, seed: int | None = None) -> pd.DataFrame:
    """Generate scenes and write ``manifest.csv`` with grouped splits."""
    data_dir = Path(cfg["paths"]["data_dir"])
    seed = cfg["seed"] if seed is None else seed
    raw = generate_dataset(cfg, data_dir, seed=seed)
    df = make_splits(raw, group_by=cfg["split"]["group_by"],
                     fractions=(cfg["split"]["train"], cfg["split"]["val"], cfg["split"]["test"]),
                     seed=seed, holdout_styles=cfg["synthetic"].get("holdout_styles", []))
    df.to_csv(data_dir / MANIFEST, index=False)
    return df


def load_manifest(data_dir: str | Path) -> pd.DataFrame:
    return pd.read_csv(Path(data_dir) / MANIFEST)


def load_annotations(data_dir: str | Path) -> dict:
    with open(Path(data_dir) / ANNOTATIONS) as f:
        return json.load(f)
