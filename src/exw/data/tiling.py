"""Tiling of wide-view images and reassembly of tile results into a grid (heat map)."""

from __future__ import annotations

import numpy as np


def tile_grid(height: int, width: int, size: int, stride: int):
    """Top-left corners of overlapping tiles. Returns (ys, xs); the last row/col is clamped inside."""
    ys = list(range(0, max(height - size, 0) + 1, stride))
    xs = list(range(0, max(width - size, 0) + 1, stride))
    if ys[-1] + size < height:
        ys.append(height - size)
    if xs[-1] + size < width:
        xs.append(width - size)
    return ys, xs


def extract_tiles(img: np.ndarray, size: int, stride: int):
    """Return (tiles N x size x size x C, coords N x 2 as (x, y), grid shape (rows, cols))."""
    h, w = img.shape[:2]
    ys, xs = tile_grid(h, w, size, stride)
    tiles, coords = [], []
    for y in ys:
        for x in xs:
            tiles.append(img[y:y + size, x:x + size])
            coords.append((x, y))
    return np.stack(tiles), np.array(coords, dtype=np.int32), (len(ys), len(xs))
