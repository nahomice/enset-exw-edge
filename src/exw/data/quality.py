"""Quality checks for REAL images received from providers (docs/DATA_PROTOCOL.md, step 2-3).

Not needed for synthetic scenes, but part of the pipeline so the same code runs on real data later.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import imagehash
import numpy as np
from PIL import Image


def blur_score(img_rgb: np.ndarray) -> float:
    """Variance of the Laplacian: low value = blurred."""
    g = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def exposure_ok(img_rgb: np.ndarray, low: float = 40, high: float = 215) -> bool:
    m = float(img_rgb.mean())
    return low <= m <= high


def phash(path: str | Path):
    with Image.open(path) as im:
        return imagehash.phash(im.convert("RGB"))


def find_near_duplicates(paths: list[str | Path], max_distance: int = 4) -> list[tuple[str, str]]:
    """Pairs of images whose perceptual hashes differ by <= ``max_distance`` bits (O(n^2); fine for ~1e4)."""
    hashes = [(str(p), phash(p)) for p in paths]
    out = []
    for i in range(len(hashes)):
        for j in range(i + 1, len(hashes)):
            if hashes[i][1] - hashes[j][1] <= max_distance:
                out.append((hashes[i][0], hashes[j][0]))
    return out


def check_image(path: str | Path, min_blur: float = 30.0) -> dict:
    img = np.array(Image.open(path).convert("RGB"))
    return dict(path=str(path), height=img.shape[0], width=img.shape[1],
                blur=blur_score(img), blurred=blur_score(img) < min_blur, exposure_ok=exposure_ok(img))
