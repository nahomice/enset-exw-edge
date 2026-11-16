"""One interface, several back-ends: Keras FP32, TFLite INT8 (optimized or reference kernels), NumPy golden."""

from __future__ import annotations

import numpy as np

from exw.models.tilecnn import preprocess


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


class KerasPredictor:
    name = "fp32_keras"

    def __init__(self, model, batch: int = 256):
        self.model, self.batch = model, batch

    def logits(self, tiles_uint8: np.ndarray) -> np.ndarray:
        if not len(tiles_uint8):
            return np.zeros((0, 3), np.float32)
        x = preprocess(tiles_uint8)
        return np.concatenate([self.model(x[i:i + self.batch], training=False).numpy()
                               for i in range(0, len(x), self.batch)])


class TflitePredictor:
    def __init__(self, tflite_bytes: bytes, reference_kernels: bool = False):
        from exw.quantize import make_interpreter
        self.it = make_interpreter(tflite_bytes, reference_kernels)
        self.name = "int8_tflite_ref" if reference_kernels else "int8_tflite"

    def logits(self, tiles_uint8: np.ndarray) -> np.ndarray:
        from exw.quantize import dequantize_output, run_int8
        return dequantize_output(self.it, run_int8(self.it, tiles_uint8))


class GoldenPredictor:
    name = "int8_golden_numpy"

    def __init__(self, tflite_bytes: bytes):
        from exw.golden import GoldenModel
        self.g = GoldenModel(tflite_bytes)

    def logits(self, tiles_uint8: np.ndarray) -> np.ndarray:
        q = self.g.forward_tiles(tiles_uint8)
        s, z = self.g.out_q
        return (q.astype(np.float32) - z) * s


def probs(predictor, tiles_uint8: np.ndarray) -> np.ndarray:
    return softmax(predictor.logits(tiles_uint8))
