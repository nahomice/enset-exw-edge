#!/usr/bin/env python
"""Generate unit-test vectors for the RTL requantize block from the golden NumPy model.

Each line of the output is:  acc  mult  shift  single  zp_out  lo  hi  expected
(all decimal integers; `single` = 1 for the final FC layer, 0 for conv/depthwise).
Includes random cases and the tie case found against TFLite (acc=45918, mult=2047290786, shift=-9 -> 85 single / 86 double).
"""
import argparse
from pathlib import Path

import numpy as np

from exw.golden import quantize_multiplier, requantize

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=5000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="fpga/sim/vectors/requant_cases.txt")
ap.add_argument("--config", default=None, help="accepted for Makefile compatibility; unused")
a = ap.parse_args()

rng = np.random.default_rng(a.seed)
rows = [(45918, 2047290786, -9, 1, -128, -128, 127), (45918, 2047290786, -9, 0, -128, -128, 127)]
for _ in range(a.n):
    real = float(10 ** rng.uniform(-4.5, -1.5))            # typical conv requantization ratios
    mult, shift = quantize_multiplier(real)
    acc = int(rng.integers(-(2 ** 20), 2 ** 20))
    single = int(rng.integers(0, 2))
    zp = int(rng.integers(-128, 128))
    relu6 = int(rng.integers(0, 2))
    lo, hi = (zp, min(127, zp + int(rng.integers(20, 128)))) if relu6 else (-128, 127)   # ReLU6 clamp or none
    rows.append((acc, mult, shift, single, zp, lo, hi))

out = Path(a.out)
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w") as f:
    for acc, mult, shift, single, zp, lo, hi in rows:
        y = int(np.clip(requantize(np.array([acc]), np.array([mult]), np.array([shift]), single=bool(single))[0] + zp,
                        lo, hi))
        f.write(f"{acc} {mult} {shift} {single} {zp} {lo} {hi} {y}\n")
print(f"wrote {len(rows)} cases to {out}")
