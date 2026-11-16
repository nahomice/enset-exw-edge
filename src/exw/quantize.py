"""Post-training INT8 quantization to a fully-integer TensorFlow Lite model (Stage 4).

The resulting ``.tflite`` file is the SINGLE reference model: the phone runs it with TensorFlow Lite,
the FPGA memory files and the test vectors are generated from it, and ``exw.golden`` reproduces it
bit for bit in NumPy.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import tensorflow as tf

from exw.models.tilecnn import build_tilecnn, preprocess


def _with_batch1(model: tf.keras.Model) -> tf.keras.Model:
    m = build_tilecnn(model.input_shape[1], 1.0, model.output_shape[-1], batch_size=1)
    m.set_weights(model.get_weights())
    return m


def convert_int8(model: tf.keras.Model, calib_tiles_uint8: np.ndarray, out_path: str | Path) -> bytes:
    """Convert a Keras model to full-integer INT8 TFLite.

    ``calib_tiles_uint8`` must come from the TRAIN split only. An all-black and an all-white tile are
    appended so that the input range is exactly [-1, 127/128]: input scale 1/128, zero point 0, i.e.
    the integer input is ``pixel - 128``.
    """
    calib = np.concatenate([calib_tiles_uint8,
                            np.zeros((1,) + calib_tiles_uint8.shape[1:], np.uint8),
                            np.full((1,) + calib_tiles_uint8.shape[1:], 255, np.uint8)])
    x = preprocess(calib)

    def rep():
        for i in range(len(x)):
            yield [x[i:i + 1]]

    # fixed batch of 1 gives a clean graph (no dynamic-shape ops) for the phone and the FPGA flow
    model = _with_batch1(model)
    tmp = tempfile.mkdtemp()
    try:
        model.export(tmp, format="tf_saved_model", verbose=False)
        conv = tf.lite.TFLiteConverter.from_saved_model(tmp)
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.representative_dataset = rep
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        conv.inference_input_type = tf.int8
        conv.inference_output_type = tf.int8
        blob = conv.convert()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_bytes(blob)
    return blob


def make_interpreter(tflite_bytes: bytes, reference_kernels: bool = False) -> tf.lite.Interpreter:
    """``reference_kernels=True`` selects the plain reference kernels (the arithmetic the FPGA must match)."""
    kw = {}
    if reference_kernels:
        kw["experimental_op_resolver_type"] = tf.lite.experimental.OpResolverType.BUILTIN_REF
    it = tf.lite.Interpreter(model_content=tflite_bytes, **kw)
    it.allocate_tensors()
    return it


def run_int8(it: tf.lite.Interpreter, tiles_uint8: np.ndarray) -> np.ndarray:
    """Run INT8 inference tile by tile. Returns raw int8 outputs (N, n_classes)."""
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    scale, zp = inp["quantization"]
    outs = []
    for t in tiles_uint8:
        x = preprocess(t[None])
        q = np.clip(np.round(x / scale) + zp, -128, 127).astype(np.int8)
        it.set_tensor(inp["index"], q)
        it.invoke()
        outs.append(it.get_tensor(out["index"])[0].copy())
    return np.stack(outs)


def dequantize_output(it: tf.lite.Interpreter, q: np.ndarray) -> np.ndarray:
    scale, zp = it.get_output_details()[0]["quantization"]
    return (q.astype(np.float32) - zp) * scale
