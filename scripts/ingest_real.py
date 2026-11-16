#!/usr/bin/env python
"""Ingest REAL wide-view photos (from EAA / partners / your own field-edge shots) into the same pipeline.

Input folder layout:
    <src>/images/*.jpg|png
    <src>/metadata.csv      columns: filename, site_id, farm_id, date, farm_label, tier
                            farm_label in {healthy, other_condition, exw_suspect}   (confirmed by the expert)
    <src>/boxes.json        optional: {filename: [{"x0":..,"y0":..,"x1":..,"y1":..,"cls":0|1}, ...]}  (0=EXW, 1=other)

Writes <out>/manifest.csv (grouped splits), <out>/annotations.json and copies nothing: paths stay relative to <out>.
Run quality checks (blur, exposure, near-duplicates) and REPORTS problems instead of silently dropping images.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd

from exw.data.quality import check_image, find_near_duplicates
from exw.data.splits import make_splits

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--out", default="data/real")
ap.add_argument("--group-by", default="site_id", choices=["site_id", "farm_id", "date"])
ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()

src, out = Path(a.src), Path(a.out)
meta = pd.read_csv(src / "metadata.csv")
boxes = json.loads((src / "boxes.json").read_text()) if (src / "boxes.json").exists() else {}
(out / "images").mkdir(parents=True, exist_ok=True)

rows, ann, problems = [], {}, []
for _, r in meta.iterrows():
    f = src / "images" / r["filename"]
    q = check_image(f)
    if q["blurred"] or not q["exposure_ok"]:
        problems.append((r["filename"], q))
    shutil.copy2(f, out / "images" / f.name)
    bx = boxes.get(r["filename"], [])
    iid = f.stem
    ann[iid] = dict(boxes=bx, boxed=bool(bx))
    rows.append(dict(image_id=iid, path=f"images/{f.name}", site_id=r["site_id"], farm_id=r["farm_id"], date=r["date"],
                     source="real", tier=int(r.get("tier", 1)), style=-1, farm_label=r["farm_label"],
                     has_boxes=bool(bx), sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
df = pd.DataFrame(rows)
dups = find_near_duplicates([out / p for p in df["path"]])
dup_cross = [(x, y) for x, y in dups]
df = make_splits(df, group_by=a.group_by, seed=a.seed)
df.to_csv(out / "manifest.csv", index=False)
(out / "annotations.json").write_text(json.dumps(ann))
print(df.groupby(["split", "farm_label"]).size().unstack(fill_value=0))
print(f"{len(problems)} images failed quality checks; {len(dup_cross)} near-duplicate pairs (review them!)")
for p in problems[:20]:
    print("  QC:", p)
print("Next: python scripts/train.py etc. with --set paths.data_dir=" + str(out))
