#!/usr/bin/env python
"""Generate the SYNTHETIC wide-view dataset as plain .npy files (tiles = model input/output, scenes = farm level).

    python scripts/make_npy_dataset.py --out data_npy                 # full set (~260 MB, about 3 minutes on CPU)
    python scripts/make_npy_dataset.py --out data_npy --n-sites 80 --cap-train 8000   # bigger

Layout of <out>/ : tiles/<split>_{X,y,xy,image_id}.npy, scenes/<split>_{X,y_farm,farm_id,image_id}.npy + _boxes.json
Splits: train, val, test, test_shift (held-out appearance style). Tier-0 data: pipeline tests only, never field accuracy.
"""
import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

from exw.config import load_config, with_overrides
from exw.data.dataset import load_image, load_tiles
from exw.data.manifest import load_annotations, load_manifest
from exw.pipeline import prepare_data

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/default.yaml")
ap.add_argument("--out", default="data_npy")
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--n-sites", type=int, default=40)
ap.add_argument("--cap-train", type=int, default=3000)
ap.add_argument("--cap-eval", type=int, default=1500, help="cap for val / test / test_shift tiles")
ap.add_argument("--scenes-per-split", type=int, default=16)
a = ap.parse_args()

work = Path(tempfile.mkdtemp(prefix="exw_npy_"))
cfg = with_overrides(load_config(a.config), **{
    "paths.data_dir": str(work / "syn"), "paths.tiles_dir": str(work / "tiles"), "paths.results_dir": str(work / "res"),
    "synthetic.n_sites": a.n_sites, "seed": a.seed})
print(prepare_data(cfg).to_string(index=False))

out = Path(a.out)
(out / "tiles").mkdir(parents=True, exist_ok=True)
(out / "scenes").mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(a.seed)
caps = dict(train=a.cap_train, val=a.cap_eval, test=a.cap_eval, test_shift=a.cap_eval)
for sp, cap in caps.items():
    d = load_tiles(work / "tiles" / f"{sp}.npz")
    idx = np.sort(rng.choice(len(d["y"]), size=min(cap, len(d["y"])), replace=False))
    np.save(out / "tiles" / f"{sp}_X.npy", d["X"][idx])
    np.save(out / "tiles" / f"{sp}_y.npy", d["y"][idx].astype(np.int8))
    np.save(out / "tiles" / f"{sp}_xy.npy", d["xy"][idx].astype(np.int32))
    np.save(out / "tiles" / f"{sp}_image_id.npy", d["image_id"][idx].astype("U32"))
    print(sp, len(idx), "tiles; class counts (EXW/other/healthy):", np.bincount(d["y"][idx], minlength=3))

df, ann = load_manifest(work / "syn"), load_annotations(work / "syn")
farm_ids = {"healthy": 0, "other_condition": 1, "exw_suspect": 2}
share = {"exw_suspect": 0.375, "other_condition": 0.25, "healthy": 0.375}
for sp in ("test", "test_shift"):
    sub, pick = df[df.split == sp], []
    for lab, f in share.items():
        s = sub[sub.farm_label == lab]
        pick += list(s.sample(min(max(1, round(a.scenes_per_split * f)), len(s)), random_state=a.seed).index)
    sel = df.loc[pick]
    np.save(out / "scenes" / f"{sp}_X.npy", np.stack([load_image(work / "syn", p) for p in sel.path]))
    np.save(out / "scenes" / f"{sp}_y_farm.npy", np.array([farm_ids[v] for v in sel.farm_label], np.int8))
    np.save(out / "scenes" / f"{sp}_farm_id.npy", sel.farm_id.values.astype(np.int32))
    np.save(out / "scenes" / f"{sp}_image_id.npy", np.array([str(i) for i in sel.image_id], dtype="U32"))
    json.dump({i: ann[i]["boxes"] for i in sel.image_id}, open(out / "scenes" / f"{sp}_boxes.json", "w"))
    print(sp, "scenes:", len(sel), sel.farm_label.value_counts().to_dict())
print("done ->", out.resolve())
