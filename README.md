# enset-exw-edge

Offline detection and flagging of **Enset Xanthomonas Wilt (EXW)** in **wide-view farm images** using a
**quantized CNN** on a **smartphone** (primary, TensorFlow Lite INT8) and an **FPGA** (Terasic DE1-SoC, office batch
accelerator and hardware benchmark).

BSc thesis, Addis Ababa University, School of Electrical and Computer Engineering (CTBE).
Students: Nahom Wondale Habtamu, Yoseph Alene Abegaz. Advisor: Dr. Biniam T. Gebru.

> **Data status.** No real wide-view images yet. Everything under `data/synthetic` is **synthetic (Tier 0)** and is
> used **only** to test the pipeline and the hardware path. Synthetic results are never field-accuracy claims.

## What the system does

```
wide-view photo --> overlapping 96x96 tiles (stride 48) --> INT8 tile CNN --> P(EXW-suspect) per tile
                --> heat map + clusters + farm risk score R = N_flagged / N  --> "flag for expert inspection"
```
One INT8 `.tflite` file is the **golden model**: the phone runs it, the FPGA memory files are generated from it, and
`exw.golden` reproduces it **bit for bit** in NumPy (checked by `pytest`).

## Quick start (no GPU, no real data needed)

```bash
pip install -e ".[tf,dev]"
pytest -q                      # tiling, labels, leakage, config, heat map, golden == TFLite reference
make smoke                     # synthetic data -> train -> INT8 -> evaluate -> FPGA export  (slow: tens of minutes on CPU)
```
Outputs go to `results/smoke/` (`report.json`, `figures/heatmaps_*.png`, `model_seed0_int8.tflite`, `fpga_export/`).

## Repository map

| Path | Purpose |
|---|---|
| `configs/` | `default.yaml`, `smoke_test.yaml`, `real_data.yaml`, `experiments/` (planned runs X1-X8) |
| `data/` | `schema/` templates (metadata, boxes, capture log, permission), `raw/ interim/ external/` (git-ignored contents) |
| `notebooks/` | data exploration, tile/label inspection, results figures |
| `src/exw/data/` | synthetic scene generator, manifest, **grouped splits**, tiling, **box-to-tile label rule**, quality checks |
| `src/exw/models/tilecnn.py` | depthwise-separable CNN, ~50 k parameters, hardware-friendly ops only |
| `src/exw/train.py`, `quantize.py` | FP32 training; full-integer INT8 conversion (calibration from train tiles only) |
| `src/exw/golden.py` | bit-exact NumPy INT8 reference (the FPGA contract) |
| `src/exw/export.py` | `.mem` weights, requantization constants, per-layer test vectors for RTL |
| `src/exw/infer/` | predictors (FP32 / INT8 / golden), heat map, risk score, clusters |
| `src/exw/eval/` | tile and farm metrics, farm-grouped bootstrap CIs, latency benchmark |
| `scripts/` | `prepare_data`, `train`, `evaluate`, `export_fpga`, `ingest_real`, `run_ablation.sh`, `make_requant_vectors`, `compare_rtl_outputs`, `check_env` |
| `fpga/` | `rtl/` (SystemVerilog), `tb/`, `sim/`, `quartus/`, `hls/`, `hps/` (ARM side), `docs/` |
| `android/` | Kotlin app (`app/`), `benchmark/` protocol and log, `parity/` check against Python |
| `docs/` | plan, data protocol, architecture, experiment log, defense checklist, results template, risks, glossary, thesis outline, `user_needs/`, `meetings/`, `slides/`, `figures/` |
| `results/` | run outputs (git-ignored; see its README) |

## Rules that protect the results
1. Split by **site/farm/date before tiling**; `assert_no_leakage` runs on every split.
2. Calibration tiles come from **train only**; the farm risk threshold is chosen on **validation only**.
3. Report **farm-level** results with **farm-grouped** confidence intervals, not just tile accuracy.
4. A held-out style/site set (`test_shift`) measures domain shift. Synthetic numbers stay labelled synthetic.
5. Speed-ups are measured, never assumed: lab CPU FP32, DE1-SoC ARM INT8, phone INT8, FPGA INT8.

## Real data
`python scripts/ingest_real.py --src <folder> --out data/real`, then add `--set paths.data_dir=data/real` to the scripts.
Format and labeling rules: `docs/DATA_PROTOCOL.md`.

License: MIT.
