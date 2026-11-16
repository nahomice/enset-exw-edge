#!/usr/bin/env python
"""Write a parity case: one image plus the Python tile probabilities, risk and clusters, so the Android app can be
checked against the reference implementation. Open the image in the app and compare the numbers it shows.

Usage: python android/parity/make_parity_case.py --tflite results/smoke/model_seed0_int8.tflite --image some.png
Output: android/parity/case_<name>.json (git-ignored images stay local; the JSON is small and may be committed).
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from exw.infer.heatmap import score_image
from exw.infer.predictors import TflitePredictor

ap = argparse.ArgumentParser()
ap.add_argument("--tflite", required=True)
ap.add_argument("--image", required=True)
ap.add_argument("--tile-thr", type=float, default=0.5)
a = ap.parse_args()

img = np.array(Image.open(a.image).convert("RGB"))
pred = TflitePredictor(Path(a.tflite).read_bytes(), reference_kernels=False)   # optimized kernels, like the phone
res = score_image(pred, img, 96, 48, a.tile_thr)
case = dict(image=Path(a.image).name, height=img.shape[0], width=img.shape[1], rows=res["shape"][0],
            cols=res["shape"][1], risk=res["risk"], clusters=res["n_clusters"],
            p_exw=np.round(res["grid"], 4).tolist())
out = Path(__file__).parent / f"case_{Path(a.image).stem}.json"
out.write_text(json.dumps(case))
print(f"wrote {out}: risk={case['risk']:.3f} clusters={case['clusters']} grid={case['rows']}x{case['cols']}")
