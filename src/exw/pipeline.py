"""End-to-end orchestration used by the scripts and by the tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from exw.data.dataset import build_all, load_image
from exw.data.manifest import build_synthetic_manifest, load_manifest
from exw.eval.metrics import choose_threshold, exw_tile_auc, farm_metrics, tile_metrics
from exw.eval.stats import grouped_bootstrap
from exw.infer.heatmap import score_image
from exw.infer.predictors import probs


def prepare_data(cfg: dict) -> pd.DataFrame:
    """Synthetic scenes -> manifest with grouped splits -> tile npz files. Returns the tile-count table."""
    build_synthetic_manifest(cfg)
    return build_all(cfg, seed=cfg["seed"])


def tile_eval(predictors: dict, tiles: dict) -> dict:
    out = {}
    for name, pr in predictors.items():
        p = probs(pr, tiles["X"])
        m = tile_metrics(tiles["y"], p)
        m["exw_auc"] = exw_tile_auc(tiles["y"], p)
        out[name] = m
    return out


def farm_scores(predictor, cfg: dict, split: str) -> pd.DataFrame:
    """Risk score per wide-view image of one split (ALL images, boxed or not)."""
    data_dir = Path(cfg["paths"]["data_dir"])
    df = load_manifest(data_dir)
    df = df[df["split"] == split].reset_index(drop=True)
    rows = []
    for _, r in df.iterrows():
        res = score_image(predictor, load_image(data_dir, r["path"]), cfg["tiles"]["size"], cfg["tiles"]["stride"],
                          cfg["farm"]["tile_prob_threshold"])
        rows.append(dict(image_id=r["image_id"], farm_id=r["farm_id"], farm_label=r["farm_label"],
                         y_exw=int(r["farm_label"] == "exw_suspect"), risk=res["risk"], n_clusters=res["n_clusters"]))
    return pd.DataFrame(rows)


def farm_eval(predictor, cfg: dict, splits=("val", "test", "test_shift")) -> dict:
    """Choose the risk threshold on VAL only, then report on the other splits with farm-bootstrap CIs."""
    sc = {s: farm_scores(predictor, cfg, s) for s in splits}
    thr = choose_threshold(sc["val"]["risk"].values, sc["val"]["y_exw"].values)
    out = dict(threshold=thr)
    for s, d in sc.items():
        if not len(d):
            continue
        m = farm_metrics(d["risk"].values, d["y_exw"].values, thr)
        sens = lambda idx, d=d: farm_metrics(d["risk"].values[idx], d["y_exw"].values[idx], thr)["sensitivity"]
        spec = lambda idx, d=d: farm_metrics(d["risk"].values[idx], d["y_exw"].values[idx], thr)["specificity"]
        m["sensitivity_ci"] = grouped_bootstrap(d["farm_id"].values, sens, 500)
        m["specificity_ci"] = grouped_bootstrap(d["farm_id"].values, spec, 500)
        m["n_images"] = len(d)
        out[s] = m
    return out, sc


def calibration_tiles(cfg: dict, tr: dict, seed: int = 0) -> np.ndarray:
    """Calibration tiles from TRAIN only, class-stratified so rare classes are represented."""
    rng = np.random.default_rng(seed)
    n = cfg["quant"]["n_calib"]
    idx = []
    for c in range(3):
        ci = np.where(tr["y"] == c)[0]
        idx += list(rng.choice(ci, size=min(len(ci), n // 3), replace=False)) if len(ci) else []
    return tr["X"][np.array(idx, int)]


def save_json(obj, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
