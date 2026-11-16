"""Wide-view image -> tile probability grid -> heat map, farm risk score R and cluster count."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from exw.constants import EXW
from exw.data.tiling import extract_tiles
from exw.infer.predictors import probs


def score_image(predictor, img: np.ndarray, size: int = 96, stride: int = 48, tile_thr: float = 0.5) -> dict:
    """Return tile probabilities, EXW heat grid, flagged mask, risk score R = N_flagged / N and cluster count."""
    tiles, coords, (gh, gw) = extract_tiles(img, size, stride)
    p = probs(predictor, tiles)
    grid = p[:, EXW].reshape(gh, gw)
    flagged = grid >= tile_thr
    _, n_clusters = ndimage.label(flagged)
    return dict(p=p, grid=grid, flagged=flagged, coords=coords, risk=float(flagged.mean()),
                n_clusters=int(n_clusters), shape=(gh, gw))


def overlay(img: np.ndarray, res: dict, size: int = 96, stride: int = 48, alpha: float = 0.45) -> np.ndarray:
    """Red heat overlay of EXW probability on the original image (nearest tile centre)."""
    h, w = img.shape[:2]
    heat = np.zeros((h, w), np.float32)
    cnt = np.zeros((h, w), np.float32)
    for (x, y), p in zip(res["coords"], res["p"][:, EXW]):
        heat[y:y + size, x:x + size] += p
        cnt[y:y + size, x:x + size] += 1
    heat = np.where(cnt > 0, heat / np.maximum(cnt, 1), 0)
    out = img.astype(np.float32).copy()
    red = np.array([255, 0, 0], np.float32)
    a = (alpha * heat)[..., None]
    return np.clip(out * (1 - a) + red * a, 0, 255).astype(np.uint8)
