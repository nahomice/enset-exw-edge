"""Export the golden INT8 model for hardware: weight/bias/requant memory files + layer-by-layer test vectors.

Outputs (in ``out_dir``):
  model_spec.json         layer list: op, shapes, stride, zero points, requant multipliers/shifts, clamp range
  <layer>_w.mem           int8 weights, one signed byte per line as 2 hex digits (``$readmemh`` compatible)
  <layer>_b.mem           int32 biases, 8 hex digits per line
  vectors/tile_XX_input.mem, tile_XX_L<k>.mem   input and every layer output of the golden model
  vectors/tile_XX_logits.mem
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from exw.golden import GoldenModel


def _hex8(a: np.ndarray) -> list[str]:
    return [f"{int(v) & 0xFF:02x}" for v in a.reshape(-1)]


def _hex32(a: np.ndarray) -> list[str]:
    return [f"{int(v) & 0xFFFFFFFF:08x}" for v in a.reshape(-1)]


def _write(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n")


def export_hw(tflite_bytes: bytes, tiles_uint8: np.ndarray, out_dir: str | Path, n_vectors: int = 4) -> dict:
    out = Path(out_dir)
    (out / "vectors").mkdir(parents=True, exist_ok=True)
    g = GoldenModel(tflite_bytes)
    spec = dict(input_scale=float(g.in_q[0]), input_zero_point=int(g.in_q[1]),
                output_scale=float(g.out_q[0]), output_zero_point=int(g.out_q[1]), layers=[])
    for k, L in enumerate(g.layers):
        entry = dict(index=k, op=L["op"])
        if "w" in L:
            name = f"L{k}_{L['op'].lower()}"
            _write(out / f"{name}_w.mem", _hex8(L["w"]))
            _write(out / f"{name}_b.mem", _hex32(L["b"]))
            _write(out / f"{name}_mult.mem", _hex32(L["mult"]))
            _write(out / f"{name}_shift.mem", _hex32(L["shift"]))
            entry.update(weight_shape=list(L["w"].shape), stride=L["stride"], input_zero_point=L["z_in"],
                         output_zero_point=L["z_out"], clamp=[L["lo"], L["hi"]], files=name)
        else:
            entry.update({k2: (list(map(int, v)) if hasattr(v, "__len__") else v) for k2, v in L.items() if k2 != "op"})
        spec["layers"].append(entry)
    (out / "model_spec.json").write_text(json.dumps(spec, indent=1))
    for i, t in enumerate(tiles_uint8[:n_vectors]):
        q = np.clip(np.round((t.astype(np.float32) - 128.0) / 128.0 / g.in_q[0]) + g.in_q[1], -128, 127).astype(np.int8)
        logits, acts = g.forward_int8(q, return_all=True)
        _write(out / "vectors" / f"tile_{i:02d}_input.mem", _hex8(q))
        for k, a in enumerate(acts):
            _write(out / "vectors" / f"tile_{i:02d}_L{k}.mem", _hex8(a))
        _write(out / "vectors" / f"tile_{i:02d}_logits.mem", _hex8(logits))
    return spec
