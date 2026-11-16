import numpy as np

from exw.infer.heatmap import score_image


class _BrightMeansExw:
    """Fake predictor (no TensorFlow): a tile is EXW-suspect when its mean brightness is high."""
    name = "fake"

    def logits(self, tiles_uint8):
        bright = tiles_uint8.reshape(len(tiles_uint8), -1).mean(1) > 150
        z = np.zeros((len(tiles_uint8), 3), np.float32)
        z[:, 0] = np.where(bright, 10.0, -10.0)
        return z


def test_risk_and_clusters_on_a_known_image():
    img = np.full((192, 192, 3), 20, np.uint8)
    img[:, :96] = 200                                   # left half bright
    res = score_image(_BrightMeansExw(), img, size=96, stride=48, tile_thr=0.5)
    assert res["shape"] == (3, 3)
    # tiles at x=0 are fully bright (flagged); x=48 is half bright (mean 110, not flagged); x=96 is dark
    assert res["flagged"][:, 0].all() and not res["flagged"][:, 1:].any()
    assert abs(res["risk"] - 3 / 9) < 1e-9 and res["n_clusters"] == 1


def test_all_dark_image_has_zero_risk():
    res = score_image(_BrightMeansExw(), np.full((192, 192, 3), 20, np.uint8))
    assert res["risk"] == 0.0 and res["n_clusters"] == 0
