#!/usr/bin/env python
"""Step 3: INT8 quantization, golden-model check, tile + farm evaluation, benchmark, heat-map figures."""
import argparse
from pathlib import Path

import matplotlib
import numpy as np
import tensorflow as tf

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from exw.config import add_cli, load_config
from exw.data.dataset import load_image, load_tiles
from exw.data.manifest import load_manifest
from exw.eval.benchmark import time_per_tile
from exw.infer.heatmap import overlay, score_image
from exw.infer.predictors import GoldenPredictor, KerasPredictor, TflitePredictor
from exw.pipeline import calibration_tiles, farm_eval, save_json, tile_eval
from exw.quantize import convert_int8, run_int8

ap = add_cli(argparse.ArgumentParser())
ap.add_argument("--seed", type=int, default=None, help="model seed to evaluate (default: first in config)")
a = ap.parse_args()
cfg = load_config(a.config, a.overrides)
seed = cfg["train"]["seeds"][0] if a.seed is None else a.seed
res = Path(cfg["paths"]["results_dir"])
tiles = cfg["paths"]["tiles_dir"]
tr = load_tiles(f"{tiles}/train.npz")
test, shift = load_tiles(f"{tiles}/test.npz"), load_tiles(f"{tiles}/test_shift.npz")

model = tf.keras.models.load_model(res / f"model_seed{seed}.keras")
blob = convert_int8(model, calibration_tiles(cfg, tr, seed), res / f"model_seed{seed}_int8.tflite")
fp32, i8, i8ref, gold = KerasPredictor(model), TflitePredictor(blob), TflitePredictor(blob, True), GoldenPredictor(blob)

# 1) golden model must match the TFLite reference kernels bit for bit
sub = test["X"][:: max(1, len(test["X"]) // 200)][:200]
g_q = gold.g.forward_tiles(sub)

r_q = run_int8(i8ref.it, sub)
golden_exact = bool((g_q == r_q).all())
opt_diff = int(np.abs(run_int8(i8.it, sub).astype(int) - r_q.astype(int)).max())
print(f"golden == TFLite reference kernels on {len(sub)} tiles: {golden_exact}; optimized-kernel max |diff| = {opt_diff} LSB")

# 2) tile-level metrics
preds = {"fp32": fp32, "int8": i8ref}
report = dict(seed=seed, golden_bit_exact=golden_exact, optimized_kernel_max_lsb_diff=opt_diff,
              tile={s: tile_eval(preds, t) for s, t in {"test": test, "test_shift": shift}.items()})
drop = report["tile"]["test"]["fp32"]["macro_f1"] - report["tile"]["test"]["int8"]["macro_f1"]
report["int8_macro_f1_drop_pp"] = 100 * drop
print(f"tile macro-F1 test: FP32={report['tile']['test']['fp32']['macro_f1']:.3f} INT8={report['tile']['test']['int8']['macro_f1']:.3f} (drop {100*drop:.2f} pp)")
if 100 * drop > cfg["quant"]["max_drop_pp"]:
    print("WARNING: INT8 drop exceeds the limit -> run quantization-aware training (see docs/ARCHITECTURE.md)")

# 3) farm-level: threshold from VAL only
fe, scores = farm_eval(i8ref, cfg)
report["farm_int8"] = fe
for s in ("test", "test_shift"):
    if s in fe:
        print(f"farm {s}: sens={fe[s]['sensitivity']:.2f} spec={fe[s]['specificity']:.2f} AUC={fe[s]['auc']:.2f} (n={fe[s]['n_images']} images, thr={fe['threshold']:.3f})")
for s, d in scores.items():
    d.to_csv(res / f"farm_scores_{s}.csv", index=False)

# 4) benchmark (this CPU only; phone and board numbers come from the real devices)
bt = test["X"][:64]
report["benchmark_lab_cpu"] = [time_per_tile(p, bt) for p in (fp32, i8, i8ref)]

# 5) heat-map figures

df = load_manifest(cfg["paths"]["data_dir"])
(res / "figures").mkdir(exist_ok=True)
for split in ("test", "test_shift"):
    sub_df = df[df["split"] == split]
    picks = [sub_df[sub_df["farm_label"] == lab].iloc[0] for lab in ("exw_suspect", "healthy") if (sub_df["farm_label"] == lab).any()]
    if not picks:
        continue
    fig, ax = plt.subplots(len(picks), 2, figsize=(9, 3.4 * len(picks)), squeeze=False)
    for i, r in enumerate(picks):
        img = load_image(cfg["paths"]["data_dir"], r["path"])
        out = score_image(i8ref, img, cfg["tiles"]["size"], cfg["tiles"]["stride"], cfg["farm"]["tile_prob_threshold"])
        ax[i, 0].imshow(img); ax[i, 0].set_title(f"{r['image_id']} (truth: {r['farm_label']})", fontsize=8)
        ax[i, 1].imshow(overlay(img, out, cfg["tiles"]["size"], cfg["tiles"]["stride"]))
        ax[i, 1].set_title(f"INT8 heat map  R={out['risk']:.2f}  clusters={out['n_clusters']}", fontsize=8)
        for x in ax[i]: x.axis("off")
    fig.suptitle("SYNTHETIC scene - pipeline check only", fontsize=9)
    fig.tight_layout(); fig.savefig(res / "figures" / f"heatmaps_{split}.png", dpi=130); plt.close(fig)

save_json(report, res / "report.json")
print("wrote", res / "report.json")
