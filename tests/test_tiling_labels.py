import numpy as np

from exw.constants import EXW, HEALTHY, IGNORE, OTHER
from exw.data.labels import tile_labels
from exw.data.tiling import extract_tiles


def test_tile_grid_covers_image_and_matches_coords():
    img = np.arange(200 * 300 * 3, dtype=np.uint32).reshape(200, 300, 3).astype(np.uint8)
    tiles, coords, (gh, gw) = extract_tiles(img, 96, 48)
    assert tiles.shape == (gh * gw, 96, 96, 3) and len(coords) == gh * gw
    x, y = coords[7]
    assert np.array_equal(tiles[7], img[y:y + 96, x:x + 96])


def test_label_rule():
    coords = np.array([[0, 0], [60, 0], [200, 200], [0, 80]])
    boxes = [dict(x0=0, y0=0, x1=96, y1=96, cls=EXW)]
    lab = tile_labels(coords, boxes, 96, 0.5, 0.1)
    assert lab[0] == EXW            # fully inside the box
    assert lab[1] == IGNORE         # 37 % overlap -> ignored
    assert lab[2] == HEALTHY        # far from any box
    assert lab[3] == IGNORE or lab[3] == HEALTHY


def test_exw_has_priority_over_other():
    boxes = [dict(x0=0, y0=0, x1=96, y1=96, cls=OTHER), dict(x0=0, y0=0, x1=96, y1=96, cls=EXW)]
    assert tile_labels(np.array([[0, 0]]), boxes, 96)[0] == EXW
