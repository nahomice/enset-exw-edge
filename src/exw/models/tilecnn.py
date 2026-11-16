"""Tile classifier: small depthwise-separable CNN (< 200 000 parameters).

Layers are chosen so that every op has a simple integer implementation shared by the phone
(TensorFlow Lite INT8) and the FPGA accelerator:
    CONV_2D, DEPTHWISE_CONV_2D, AVERAGE_POOL_2D, RESHAPE, FULLY_CONNECTED   (ReLU6 fused)
The network input is the float tensor (pixel - 128) / 128 in [-1, 0.992]. With INT8 input quantization
(scale 1/128, zero point 0) the integer input is simply ``pixel - 128``; see ``preprocess``.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from exw.constants import N_CLASSES


def preprocess(tiles_uint8: np.ndarray) -> np.ndarray:
    """uint8 pixels -> float32 network input in [-1, 0.992] (same as int8 = pixel - 128 at scale 1/128)."""
    return (tiles_uint8.astype(np.float32) - 128.0) / 128.0


def _ch(c: int, width: float) -> int:
    return max(8, int(round(c * width / 8)) * 8)


def build_tilecnn(size: int = 96, width: float = 1.0, n_classes: int = N_CLASSES,
                  batch_size: int | None = None) -> tf.keras.Model:
    L = tf.keras.layers
    inp = L.Input((size, size, 3), batch_size=batch_size, name="tile")
    x = L.Conv2D(_ch(16, width), 3, strides=2, padding="same", use_bias=False, name="conv1")(inp)
    x = L.BatchNormalization(name="bn1")(x)
    x = L.ReLU(6.0, name="relu1")(x)
    fmap = size // 2

    for i, (ch, stride) in enumerate([(32, 1), (64, 2), (128, 2), (256, 2)], start=1):
        x = L.DepthwiseConv2D(3, strides=stride, padding="same", use_bias=False, name=f"dw{i}")(x)
        x = L.BatchNormalization(name=f"dw{i}_bn")(x)
        x = L.ReLU(6.0, name=f"dw{i}_relu")(x)
        x = L.Conv2D(_ch(ch, width), 1, use_bias=False, name=f"pw{i}")(x)
        x = L.BatchNormalization(name=f"pw{i}_bn")(x)
        x = L.ReLU(6.0, name=f"pw{i}_relu")(x)
        fmap = (fmap + stride - 1) // stride

    x = L.AveragePooling2D(pool_size=fmap, name="gap")(x)   # global average pooling as a plain pool op
    x = L.Reshape((x.shape[-1],), name="flatten")(x)
    out = L.Dense(n_classes, name="logits")(x)              # logits (softmax is applied outside the model)
    return tf.keras.Model(inp, out, name="tilecnn")


def count_params(model: tf.keras.Model) -> int:
    return int(model.count_params())
