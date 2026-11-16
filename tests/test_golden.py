"""The key hardware test: NumPy golden model == TFLite reference kernels, bit for bit (untrained model is enough)."""
import numpy as np
import pytest

pytestmark = pytest.mark.tf


@pytest.fixture(scope="module")
def blob_and_tiles(tmp_path_factory):
    from exw.models.tilecnn import build_tilecnn
    from exw.quantize import convert_int8
    rng = np.random.default_rng(0)
    m = build_tilecnn()
    m.set_weights([w + rng.normal(0, 0.05, w.shape).astype(w.dtype) for w in m.get_weights()])
    calib = rng.integers(0, 256, (32, 96, 96, 3), dtype=np.uint8)
    blob = convert_int8(m, calib, tmp_path_factory.mktemp("q") / "m.tflite")
    return blob, rng.integers(0, 256, (12, 96, 96, 3), dtype=np.uint8)


def test_param_budget():
    from exw.models.tilecnn import build_tilecnn, count_params
    assert count_params(build_tilecnn()) < 200_000


def test_only_supported_ops(blob_and_tiles):
    from exw.quantize import make_interpreter
    ops = {o["op_name"] for o in make_interpreter(blob_and_tiles[0], True)._get_ops_details()}
    compute = {"CONV_2D", "DEPTHWISE_CONV_2D", "AVERAGE_POOL_2D", "RESHAPE", "FULLY_CONNECTED"}
    plumbing = {"SHAPE", "STRIDED_SLICE", "PACK"}      # constant shape arithmetic around the flatten; no compute
    assert ops <= compute | plumbing, ops


def test_golden_matches_reference_kernels_bit_exact(blob_and_tiles):
    from exw.golden import GoldenModel
    from exw.quantize import make_interpreter, run_int8
    blob, tiles = blob_and_tiles
    ref = run_int8(make_interpreter(blob, True), tiles)
    assert np.array_equal(GoldenModel(blob).forward_tiles(tiles), ref)


def test_optimized_kernels_within_two_lsb(blob_and_tiles):
    from exw.golden import GoldenModel
    from exw.quantize import make_interpreter, run_int8
    blob, tiles = blob_and_tiles
    opt = run_int8(make_interpreter(blob, False), tiles).astype(int)
    assert np.abs(opt - GoldenModel(blob).forward_tiles(tiles).astype(int)).max() <= 2


def test_export_writes_vectors(blob_and_tiles, tmp_path):
    from exw.export import export_hw
    blob, tiles = blob_and_tiles
    spec = export_hw(blob, tiles, tmp_path, n_vectors=2)
    assert (tmp_path / "model_spec.json").exists() and (tmp_path / "vectors" / "tile_01_logits.mem").exists()
    assert len(spec["layers"]) == 12   # 9 conv/depthwise + pool + reshape + FC


def test_fc_single_rounding_tie_case():
    """Found on real TFLite output: acc*mult lands on a tie after the first rounding.
    Reference per-channel FC rounds ONCE (85); double rounding would give 86."""
    from exw.golden import requantize
    acc, m, sh = np.array([45918]), np.array([2047290786]), np.array([-9])
    assert requantize(acc, m, sh, single=True)[0] == 85
    assert requantize(acc, m, sh, single=False)[0] == 86
