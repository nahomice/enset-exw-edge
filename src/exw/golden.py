"""Bit-exact NumPy reference of the INT8 TFLite model (the "golden model" for the FPGA).

Reproduces TensorFlow Lite's reference integer kernels: int32 accumulators, per-channel requantization
with a Q31 multiplier (conv/depthwise: SaturatingRoundingDoublingHighMul + RoundingDivideByPOT; FC: single rounding), ReLU6 clamped in the
quantized domain, SAME padding with zero contribution. The FPGA RTL and the phone must agree with this.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def quantize_multiplier(real: float) -> tuple[int, int]:
    """real = m * 2^(shift-31), m in [2^30, 2^31). Same decomposition as TFLite's QuantizeMultiplier."""
    if real == 0.0:
        return 0, 0
    frac, shift = math.frexp(real)
    q = int(round(frac * (1 << 31)))
    if q == (1 << 31):
        q //= 2
        shift += 1
    return q, shift


def _srdhm(a: np.ndarray, m: int) -> np.ndarray:
    """SaturatingRoundingDoublingHighMul for int32 a and scalar/array int32 m (done in int64)."""
    a = a.astype(np.int64)
    ab = a * np.asarray(m, np.int64)
    nudge = np.where(ab >= 0, 1 << 30, 1 - (1 << 30))
    num = ab + nudge
    # C++ integer division truncates toward zero
    res = np.sign(num) * (np.abs(num) >> 31)
    return np.clip(res, -(1 << 31), (1 << 31) - 1)


def _rdbpot(x: np.ndarray, exp: np.ndarray) -> np.ndarray:
    """RoundingDivideByPOT (round half away from zero), exp >= 0."""
    exp = np.asarray(exp, np.int64)
    mask = (np.int64(1) << exp) - 1
    rem = x & mask
    thr = (mask >> 1) + (x < 0).astype(np.int64)
    return (x >> exp) + (rem > thr).astype(np.int64)


def requantize_single(acc: np.ndarray, mult: np.ndarray, shift: np.ndarray) -> np.ndarray:
    """Single-rounding variant: round-half-away of acc * mult / 2^(31 - shift) (exact, 64-bit)."""
    acc = acc.astype(np.int64)
    total = (31 - np.asarray(shift, np.int64))
    prod = acc * np.asarray(mult, np.int64)
    r = (np.abs(prod) + (np.int64(1) << (total - 1))) >> total
    return np.where(prod >= 0, r, -r)


def requantize(acc: np.ndarray, mult: np.ndarray, shift: np.ndarray, single: bool = False) -> np.ndarray:
    """MultiplyByQuantizedMultiplier, per-channel on the last axis. acc int64 (fits int32 in practice).

    Measured against the TFLite BUILTIN_REF kernels (see tests): CONV_2D / DEPTHWISE_CONV_2D use DOUBLE rounding
    (SaturatingRoundingDoublingHighMul then RoundingDivideByPOT); the per-channel FULLY_CONNECTED kernel uses
    SINGLE rounding. The RTL must reproduce both.
    """
    if single:
        return requantize_single(acc, mult, shift)
    left = np.maximum(shift, 0).astype(np.int64)
    right = np.maximum(-shift, 0).astype(np.int64)
    x = (acc.astype(np.int64) << left)
    x = np.clip(x, -(1 << 31), (1 << 31) - 1)
    return _rdbpot(_srdhm(x, mult), right)


def _same_pad(n: int, k: int, s: int) -> tuple[int, int, int]:
    out = -(-n // s)
    total = max((out - 1) * s + k - n, 0)
    return out, total // 2, total - total // 2


def _windows(x: np.ndarray, k: int, s: int):
    """x: (H, W, C) int64 already offset by -zp. SAME zero padding. Returns (Ho, Wo, k, k, C) view."""
    h, w, _ = x.shape
    ho, pt, pb = _same_pad(h, k, s)
    wo, pl, pr = _same_pad(w, k, s)
    xp = np.pad(x, ((pt, pb), (pl, pr), (0, 0)))
    win = sliding_window_view(xp, (k, k), axis=(0, 1))[::s, ::s]      # (Ho, Wo, C, k, k)
    return win[:ho, :wo]


def _layer_params(d_in, d_w, d_out):
    sw = np.asarray(d_w["quantization_parameters"]["scales"], np.float64)
    si, so = float(d_in["quantization"][0]), float(d_out["quantization"][0])
    mm = [quantize_multiplier(si * w / so) for w in sw]
    return np.array([m for m, _ in mm], np.int64), np.array([s for _, s in mm], np.int64)


class GoldenModel:
    """Parses an INT8 TFLite flatbuffer (through the interpreter's tensor tables) and runs it in NumPy."""

    def __init__(self, tflite_bytes: bytes):
        import tensorflow as tf
        it = tf.lite.Interpreter(model_content=tflite_bytes)
        it.allocate_tensors()
        td = {t["index"]: t for t in it.get_tensor_details()}
        self.in_q = td[it.get_input_details()[0]["index"]]["quantization"]
        self.out_q = td[it.get_output_details()[0]["index"]]["quantization"]
        ops = it._get_ops_details()
        self.layers = []
        for op in ops:
            name = op["op_name"]
            if name in ("CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED"):
                i_in, i_w, i_b = op["inputs"][:3]
                d_in, d_w, d_out = td[i_in], td[i_w], td[op["outputs"][0]]
                w = it.get_tensor(i_w).astype(np.int64)
                b = it.get_tensor(i_b).astype(np.int64) if i_b >= 0 else np.zeros(w.shape[0], np.int64)
                mult, shift = _layer_params(d_in, d_w, d_out)
                s_out, z_out = d_out["quantization"]
                last = name == "FULLY_CONNECTED"
                lo, hi = (-128, 127) if last else (max(-128, int(round(0 / s_out)) + z_out),
                                                   min(127, int(round(6 / s_out)) + z_out))
                self.layers.append(dict(op=name, w=w, b=b, mult=mult, shift=shift, z_in=int(d_in["quantization"][1]),
                                        z_out=int(z_out), lo=lo, hi=hi,
                                        stride=self._stride(op, d_in, d_out, w, name)))
            elif name == "AVERAGE_POOL_2D":
                d_in = td[op["inputs"][0]]
                self.layers.append(dict(op=name, hw=d_in["shape"][1:3]))
            elif name == "RESHAPE":
                self.layers.append(dict(op=name))
            # SHAPE / STRIDED_SLICE / PACK (shape plumbing) are ignored; RESHAPE flattens.
        self.n_layers = len(self.layers)

    @staticmethod
    def _stride(op, d_in, d_out, w, name):
        if name == "FULLY_CONNECTED":
            return 1
        return int(round(d_in["shape"][1] / d_out["shape"][1]))

    # -- layers ---------------------------------------------------------------------------------------
    def _conv(self, x, L):
        x = x.astype(np.int64) - L["z_in"]
        win = _windows(x, 3 if L["w"].shape[1] == 3 else 1, L["stride"]) if L["op"] != "CONV_2D" or L["w"].shape[1] > 1 \
            else x[::L["stride"], ::L["stride"], :, None, None].transpose(0, 1, 2, 3, 4)
        if L["op"] == "DEPTHWISE_CONV_2D":
            w = L["w"][0]                                       # (k, k, C)
            acc = np.einsum("hwcij,ijc->hwc", win, w)
        elif L["w"].shape[1] == 1:                              # 1x1 conv
            acc = np.einsum("hwc,oc->hwo", x[::L["stride"], ::L["stride"]], L["w"][:, 0, 0, :])
        else:                                                   # k x k conv, w: (O, k, k, I)
            acc = np.einsum("hwcij,oijc->hwo", win, L["w"])
        return self._finish(acc + L["b"], L)

    def _finish(self, acc, L):
        y = requantize(acc, L["mult"], L["shift"], single=L["op"] == "FULLY_CONNECTED") + L["z_out"]
        return np.clip(y, L["lo"], L["hi"]).astype(np.int8)

    def forward_int8(self, x_int8: np.ndarray, return_all: bool = False):
        """x_int8: (H, W, 3) int8 (= pixel - 128). Returns int8 logits (n_classes,)."""
        x, acts = x_int8, []
        for L in self.layers:
            if L["op"] in ("CONV_2D", "DEPTHWISE_CONV_2D"):
                x = self._conv(x, L)
            elif L["op"] == "AVERAGE_POOL_2D":
                h, w = L["hw"]
                acc = x.astype(np.int64).reshape(h * w, -1).sum(0)
                cnt = h * w
                x = np.where(acc > 0, (acc + cnt // 2) // cnt, -((-acc + cnt // 2) // cnt)).astype(np.int8)
                x = x.reshape(1, 1, -1)
            elif L["op"] == "RESHAPE":
                x = x.reshape(-1)
            else:  # FULLY_CONNECTED
                acc = (x.astype(np.int64) - L["z_in"]) @ L["w"].T + L["b"]
                x = self._finish(acc, L)
            acts.append(x)
        return (x, acts) if return_all else x

    def forward_tiles(self, tiles_uint8: np.ndarray) -> np.ndarray:
        out = []
        for t in tiles_uint8:
            q = np.clip(np.round((t.astype(np.float32) - 128.0) / 128.0 / self.in_q[0]) + self.in_q[1], -128, 127)
            out.append(self.forward_int8(q.astype(np.int8)))
        return np.stack(out)
