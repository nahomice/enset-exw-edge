#!/usr/bin/env python
"""Compare RTL simulation dumps against the golden layer-by-layer vectors and report the FIRST mismatching layer.

Golden vectors:  <results>/fpga_export/vectors/tile_XX_L<k>.mem          (from scripts/export_fpga.py)
RTL dumps:       fpga/sim/out/tile_XX_L<k>.mem                           (written by the testbench, same format)
Both are one 2-digit hex byte per line (signed int8, two's complement).
"""
import argparse
from pathlib import Path

from exw.config import add_cli, load_config

ap = add_cli(argparse.ArgumentParser())
ap.add_argument("--rtl-dir", default="fpga/sim/out")
a = ap.parse_args()
cfg = load_config(a.config, a.overrides)
gold_dir = Path(cfg["paths"]["results_dir"]) / "fpga_export" / "vectors"
rtl_dir = Path(a.rtl_dir)


def read(p: Path) -> list[str]:
    return [ln.strip().lower() for ln in p.read_text().split() if ln.strip()]


bad = 0
for gfile in sorted(gold_dir.glob("tile_*_L*.mem")):
    rfile = rtl_dir / gfile.name
    if not rfile.exists():
        print(f"MISSING  {rfile}")
        bad += 1
        continue
    g, r = read(gfile), read(rfile)
    if len(g) != len(r):
        print(f"LENGTH   {gfile.name}: golden {len(g)} vs rtl {len(r)}")
        bad += 1
        continue
    diff = [i for i, (x, y) in enumerate(zip(g, r)) if x != y]
    if diff:
        i = diff[0]
        print(f"MISMATCH {gfile.name}: {len(diff)} of {len(g)} values differ; first at index {i}: golden {g[i]} rtl {r[i]}")
        bad += 1
print("ALL LAYERS MATCH" if bad == 0 else f"{bad} file(s) differ")
raise SystemExit(1 if bad else 0)
