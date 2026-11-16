"""Build tile datasets (npz) from the manifest. Tile labels are DERIVED from boxes by program."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from exw.constants import HEALTHY, IGNORE
from exw.data.labels import tile_labels
from exw.data.manifest import load_annotations, load_manifest
from exw.data.tiling import extract_tiles


def load_image(data_dir: str | Path, rel_path: str) -> np.ndarray:
    return np.array(Image.open(Path(data_dir) / rel_path).convert("RGB"))


def build_split_tiles(cfg: dict, split: str, seed: int = 0, boxed_only: bool = True) -> dict:
    """Tiles + derived labels for one split. Images without boxes are skipped when ``boxed_only``."""
    data_dir = Path(cfg["paths"]["data_dir"])
    tc = cfg["tiles"]
    df = load_manifest(data_dir)
    ann = load_annotations(data_dir)
    df = df[df["split"] == split]
    if boxed_only:
        df = df[df["has_boxes"]]
    X, y, ids, xy = [], [], [], []
    for _, row in df.iterrows():
        img = load_image(data_dir, row["path"])
        tiles, coords, _ = extract_tiles(img, tc["size"], tc["stride"])
        lab = tile_labels(coords, ann[row["image_id"]]["boxes"], tc["size"], tc["box_frac"], tc["ignore_frac"])
        keep = lab != IGNORE
        X.append(tiles[keep]); y.append(lab[keep]); xy.append(coords[keep])
        ids += [row["image_id"]] * int(keep.sum())
    if not X:
        return dict(X=np.zeros((0, tc["size"], tc["size"], 3), np.uint8), y=np.zeros(0, np.int8),
                    image_id=np.array([]), xy=np.zeros((0, 2), np.int32))
    out = dict(X=np.concatenate(X), y=np.concatenate(y), image_id=np.array(ids), xy=np.concatenate(xy))
    if split == "train" and tc.get("bg_keep_ratio"):
        out = _balance_background(out, tc["bg_keep_ratio"], seed)
    return out


def _balance_background(d: dict, ratio: float, seed: int) -> dict:
    """Training only: keep at most ``ratio`` background tiles per suspect tile (val/test stay natural)."""
    rng = np.random.default_rng(seed)
    bg = np.where(d["y"] == HEALTHY)[0]
    n_sus = int((d["y"] != HEALTHY).sum())
    max_bg = int(ratio * max(n_sus, 1))
    if len(bg) > max_bg:
        drop = rng.choice(bg, size=len(bg) - max_bg, replace=False)
        keep = np.ones(len(d["y"]), bool)
        keep[drop] = False
        d = {k: v[keep] for k, v in d.items()}
    return d


def save_tiles(d: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **d)


def load_tiles(path: str | Path) -> dict:
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def class_counts(y: np.ndarray, n_classes: int = 3) -> list[int]:
    return [int((y == c).sum()) for c in range(n_classes)]


def build_all(cfg: dict, seed: int = 0) -> pd.DataFrame:
    """Write tiles for train/val/test/test_shift and return a summary table."""
    rows = []
    tiles_dir = Path(cfg["paths"]["tiles_dir"])
    for split in ["train", "val", "test", "test_shift"]:
        d = build_split_tiles(cfg, split, seed=seed)
        save_tiles(d, tiles_dir / f"{split}.npz")
        c = class_counts(d["y"])
        rows.append(dict(split=split, n_tiles=len(d["y"]), exw=c[0], other=c[1], healthy_bg=c[2]))
    return pd.DataFrame(rows)
