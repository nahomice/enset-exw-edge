#!/usr/bin/env python
"""Train and evaluate the tile CNN directly from the .npy dataset.

    python scripts/train_npy.py --data-dir data/data_npy --out results/npy_run
    python scripts/train_npy.py --data-dir data/data_npy --set train.epochs=30 --set train.batch_size=64

Steps (each prints what it did):
  1 config + seed          5 FP32 evaluation (val / test / test_shift)
  2 load .npy tiles        6 INT8 conversion (calibration tiles from TRAIN only)
  3 sanity checks          7 golden model == TFLite reference kernels (bit-exact)
  4 train FP32             8 INT8 tile evaluation, then farm-level evaluation on scenes/*.npy
                           9 save report.json

Data are SYNTHETIC (Tier 0): this checks the pipeline, it is not a field-accuracy result.
"""

import argparse
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from exw.config import add_cli, load_config
from exw.constants import N_CLASSES

SPLITS = ("train", "val", "test", "test_shift")


# ---------------------------------------------------------------- 2. loading
def load_tiles_npy(root: Path, split: str, limit: int | None = None, seed: int = 0) -> dict:
    d = root / "tiles"
    X, y = np.load(d / f"{split}_X.npy"), np.load(d / f"{split}_y.npy")
    ids = np.load(d / f"{split}_image_id.npy")
    if limit and len(y) > limit:  # debug: random subset
        idx = np.sort(np.random.default_rng(seed).choice(len(y), limit, replace=False))
        X, y, ids = X[idx], y[idx], ids[idx]
    return dict(X=X, y=y, image_id=ids)


# ---------------------------------------------------------------- 3. sanity checks
def check_data(data: dict, size: int) -> None:
    print(f"{'split':11s}{'tiles':>7s}{'EXW':>7s}{'other':>7s}{'healthy':>9s}")
    sites = {}
    for sp, d in data.items():
        assert d["X"].dtype == np.uint8 and d["X"].shape[1:] == (size, size, 3), (
            f"{sp}: bad tile array {d['X'].shape}"
        )
        assert set(np.unique(d["y"])) <= set(range(N_CLASSES)), (
            f"{sp}: labels outside 0..{N_CLASSES - 1}"
        )
        c = np.bincount(d["y"], minlength=N_CLASSES)
        print(f"{sp:11s}{len(d['y']):7d}{c[0]:7d}{c[1]:7d}{c[2]:9d}")
        sites[sp] = {str(i).split("_")[1] for i in d["image_id"] if str(i).startswith("syn_")}
    assert (np.bincount(data["train"]["y"], minlength=N_CLASSES) > 0).all(), (
        "a class is missing from train"
    )
    names = list(sites)
    for i, a in enumerate(names):  # no site may appear in two splits
        for b in names[i + 1 :]:
            both = sites[a] & sites[b]
            assert not both, f"LEAKAGE: sites {sorted(both)[:5]} are in both {a} and {b}"
    print(
        "checks passed: shapes, dtype, labels, all classes in train, no site shared between splits"
    )


def pp(x: float) -> str:
    return f"{100 * x:5.1f}"


def print_tile_table(name: str, res: dict) -> None:
    for model, m in res.items():
        pc = " ".join(
            f"{k}: R={pp(v['recall'])} P={pp(v['precision'])}" for k, v in m["per_class"].items()
        )
        print(
            f"  {name:11s}{model:6s} acc={pp(m['accuracy'])} macro-F1={pp(m['macro_f1'])} EXW-AUC={m['exw_auc']:.3f} | {pc}"
        )


# ---------------------------------------------------------------- 8. farm level
def farm_level(predictor, root: Path, cfg: dict) -> dict:
    from exw.eval.metrics import choose_threshold, farm_metrics
    from exw.infer.heatmap import score_image

    sc = root / "scenes"
    risks = {}
    for sp in ("val", "test", "test_shift"):
        if not (sc / f"{sp}_X.npy").exists():
            continue
        X, y = np.load(sc / f"{sp}_X.npy", mmap_mode="r"), np.load(sc / f"{sp}_y_farm.npy")
        r = [
            score_image(
                predictor,
                np.asarray(img),
                cfg["tiles"]["size"],
                cfg["tiles"]["stride"],
                cfg["farm"]["tile_prob_threshold"],
            )["risk"]
            for img in X
        ]
        risks[sp] = (np.array(r), (y == 2).astype(int))  # farm label 2 = EXW-suspect
    out = {}
    thr = None
    if "val" in risks:  # threshold on validation ONLY
        thr = choose_threshold(*risks["val"])
        out["threshold"] = thr
    else:
        print(
            "  (no val scenes in the dataset: reporting AUC only; a threshold must come from validation, not test)"
        )
    for sp, (r, y) in risks.items():
        if sp == "val":
            continue
        auc = float(roc_auc_score(y, r)) if 0 < y.sum() < len(y) else float("nan")
        out[sp] = dict(
            n_scenes=len(y),
            auc=auc,
            risk_exw=r[y == 1].round(3).tolist(),
            risk_other=r[y == 0].round(3).tolist(),
        )
        if thr is not None:
            out[sp].update(farm_metrics(r, y, thr))
        print(
            f"  farm {sp:10s} scenes={len(y):3d} AUC={auc:.3f}  mean risk EXW farms={r[y == 1].mean():.3f}  other farms={r[y == 0].mean():.3f}"
        )
    return out


def main() -> None:
    ap = add_cli(
        argparse.ArgumentParser(
            description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
        )
    )
    ap.add_argument("--data-dir", default="data/data_npy")
    ap.add_argument("--out", default="results/npy_run")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--max-train", type=int, default=None, help="debug: use only N random training tiles"
    )
    a = ap.parse_args()
    t0 = time.time()

    # 1. config + seed ------------------------------------------------------------------------------
    cfg = load_config(a.config, a.overrides)
    root, out = Path(a.data_dir), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print(
        f"[1] config={a.config} seed={a.seed} epochs={cfg['train']['epochs']} data={root} out={out}"
    )

    # 2. load ---------------------------------------------------------------------------------------
    data = {
        sp: load_tiles_npy(root, sp, a.max_train if sp == "train" else None, a.seed)
        for sp in SPLITS
    }
    print(f"[2] loaded {sum(len(d['y']) for d in data.values())} tiles")

    # 3. checks -------------------------------------------------------------------------------------
    print("[3] sanity checks")
    check_data(data, cfg["tiles"]["size"])

    # heavy imports only after the cheap checks passed
    import tensorflow as tf

    from exw.infer.predictors import GoldenPredictor, KerasPredictor, TflitePredictor
    from exw.pipeline import calibration_tiles, save_json, tile_eval
    from exw.quantize import convert_int8, run_int8
    from exw.train import train_one

    # 4. train FP32 ---------------------------------------------------------------------------------
    print("[4] training FP32")
    info = train_one(cfg, data["train"], data["val"], a.seed, out)
    print(
        f"    {info['epochs_run']} epochs, params={info['n_params']}, final val_acc={info['val_acc'][-1]:.3f}"
    )
    model = tf.keras.models.load_model(out / f"model_seed{a.seed}.keras")

    # 5. FP32 evaluation ----------------------------------------------------------------------------
    print("[5] FP32 evaluation")
    fp32 = KerasPredictor(model)
    report = dict(seed=a.seed, data="synthetic (Tier 0)", train=info, tile={})
    for sp in ("val", "test", "test_shift"):
        report["tile"][sp] = tile_eval({"fp32": fp32}, data[sp])
        print_tile_table(sp, report["tile"][sp])

    # 6. INT8 ---------------------------------------------------------------------------------------
    print("[6] INT8 conversion (calibration from TRAIN tiles only)")
    blob = convert_int8(
        model,
        calibration_tiles(cfg, data["train"], a.seed),
        out / f"model_seed{a.seed}_int8.tflite",
    )
    int8_ref, int8_opt = TflitePredictor(blob, True), TflitePredictor(blob, False)
    print(f"    int8 model: {len(blob) / 1024:.1f} KB")

    # 7. golden model check ---------------------------------------------------------------------------
    print("[7] golden NumPy model vs TFLite reference kernels")
    sub = data["test"]["X"][:: max(1, len(data["test"]["X"]) // 200)][:200]
    gold = GoldenPredictor(blob).g.forward_tiles(sub)
    exact = bool((gold == run_int8(int8_ref.it, sub)).all())
    lsb = int(np.abs(run_int8(int8_opt.it, sub).astype(int) - gold.astype(int)).max())
    print(f"    bit-exact on {len(sub)} tiles: {exact}; optimized-kernel max difference: {lsb} LSB")
    report.update(golden_bit_exact=exact, optimized_kernel_max_lsb_diff=lsb)
    assert exact, "golden model does not match the TFLite reference kernels: stop and investigate"

    # 8. INT8 tile + farm evaluation ----------------------------------------------------------------
    print("[8] INT8 evaluation")
    for sp in ("val", "test", "test_shift"):
        r = tile_eval({"int8": int8_ref}, data[sp])
        report["tile"][sp].update(r)
        print_tile_table(sp, r)
    drop = 100 * (
        report["tile"]["test"]["fp32"]["macro_f1"] - report["tile"]["test"]["int8"]["macro_f1"]
    )
    report["int8_macro_f1_drop_pp"] = drop
    print(f"    INT8 macro-F1 drop on test: {drop:.2f} pp (limit {cfg['quant']['max_drop_pp']} pp)")
    if drop > cfg["quant"]["max_drop_pp"]:
        print(
            "    WARNING: above the limit -> more calibration tiles first, then quantization-aware training"
        )
    print("    farm level (INT8 reference kernels):")
    report["farm_int8"] = farm_level(int8_ref, root, cfg)

    # 9. save ---------------------------------------------------------------------------------------
    save_json(report, out / "report.json")
    print(f"[9] wrote {out / 'report.json'}  (total {time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
