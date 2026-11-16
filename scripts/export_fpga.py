#!/usr/bin/env python
"""Step 4: export weights, requantization constants and golden test vectors for the FPGA project."""
import argparse
from pathlib import Path

from exw.config import add_cli, load_config
from exw.data.dataset import load_tiles
from exw.export import export_hw

ap = add_cli(argparse.ArgumentParser())
ap.add_argument("--seed", type=int, default=None)
a = ap.parse_args()
cfg = load_config(a.config, a.overrides)
seed = cfg["train"]["seeds"][0] if a.seed is None else a.seed
res = Path(cfg["paths"]["results_dir"])
test = load_tiles(f"{cfg['paths']['tiles_dir']}/test.npz")
spec = export_hw((res / f"model_seed{seed}_int8.tflite").read_bytes(), test["X"], res / "fpga_export")
print(f"exported {len(spec['layers'])} layers to {res / 'fpga_export'}")
