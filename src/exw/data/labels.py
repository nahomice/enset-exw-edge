"""Derive tile labels from coarse boxes by a fixed rule (so tile size / rule can change without re-annotating).

Rule (configurable):
  * a tile takes the class of a box if >= ``box_frac`` of the TILE area lies inside that box
    (EXW-suspect has priority over other condition);
  * a tile that overlaps a box by ``ignore_frac`` .. ``box_frac`` is IGNORE (excluded from training);
  * every other tile of a boxed image is healthy_background.
Images WITHOUT boxes yield no tile labels (they are used for farm-level evaluation only).
"""

from __future__ import annotations

import numpy as np

from exw.constants import EXW, HEALTHY, IGNORE, OTHER


def _inter_frac(tx, ty, size, b) -> float:
    ix = max(0.0, min(tx + size, b["x1"]) - max(tx, b["x0"]))
    iy = max(0.0, min(ty + size, b["y1"]) - max(ty, b["y0"]))
    return ix * iy / float(size * size)


def tile_labels(coords: np.ndarray, boxes: list[dict], size: int, box_frac: float = 0.5,
                ignore_frac: float = 0.1) -> np.ndarray:
    out = np.full(len(coords), HEALTHY, dtype=np.int8)
    for i, (x, y) in enumerate(coords):
        best = {EXW: 0.0, OTHER: 0.0}
        for b in boxes:
            best[b["cls"]] = max(best[b["cls"]], _inter_frac(x, y, size, b))
        if best[EXW] >= box_frac:
            out[i] = EXW
        elif best[OTHER] >= box_frac:
            out[i] = OTHER
        elif max(best.values()) >= ignore_frac:
            out[i] = IGNORE
    return out
