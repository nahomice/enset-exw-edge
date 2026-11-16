"""Synthetic wide-view enset farm scenes (Tier 0 data).

PURPOSE: develop and test the pipeline (tiling, labels, splits, training, quantization, heat map,
hardware benchmarking). Results on these scenes are NOT claims about field accuracy.

Design rules (docs/DATA_PROTOCOL.md):
  * every label is exact and reproducible (seed + settings are stored with the scene);
  * the generator is deliberately imperfect: early EXW looks almost healthy, other conditions look
    similar to EXW, yellow weeds and trees act as distractors, and appearance "styles" differ so a
    held-out style can be used as a synthetic domain-shift test;
  * scenes of one farm share background layout, plant slots and style, so near-duplicates exist and
    random (non-grouped) splits would leak, exactly as in real data.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from exw.constants import EXW, FARM_LABELS, HEALTHY, OTHER

# appearance styles: soil colour, leaf-green, brightness, blur sigma
STYLES = [
    dict(soil=(125, 92, 62), leaf=(58, 150, 62), bright=1.00, blur=0.6),
    dict(soil=(150, 118, 84), leaf=(72, 160, 70), bright=1.12, blur=0.9),
    dict(soil=(98, 72, 52), leaf=(48, 132, 58), bright=0.88, blur=0.5),
    dict(soil=(170, 150, 120), leaf=(86, 168, 88), bright=1.20, blur=1.3),  # default hold-out style
]


@dataclass
class Plant:
    cx: float
    cy: float
    r: float
    cls: int           # EXW / OTHER / HEALTHY
    severity: float    # 0..1 (early EXW is hard to see)
    seed: int


def _lerp(a, b, t):
    return tuple(float(x + (y - x) * t) for x, y in zip(a, b))


def _draw_leaf(img, cx, cy, ang, length, width, color, vscale):
    ex = cx + np.cos(ang) * length / 2
    ey = cy + np.sin(ang) * length / 2 * vscale
    rot = np.degrees(np.arctan2(np.sin(ang) * vscale, np.cos(ang)))
    cv2.ellipse(img, (int(ex), int(ey)), (max(int(length / 2), 1), max(int(width / 2), 1)),
                rot, 0, 360, tuple(int(c) for c in color), -1, cv2.LINE_AA)
    return ex, ey


def draw_plant(img: np.ndarray, p: Plant, style: dict) -> None:
    rng = np.random.default_rng(p.seed)
    r, vscale = p.r, 0.62
    green = np.array(style["leaf"], float)
    # soft shadow
    sh = img.copy()
    cv2.ellipse(sh, (int(p.cx), int(p.cy + r * 0.45)), (int(r * 1.0), int(r * 0.45)), 0, 0, 360,
                (30, 25, 20), -1, cv2.LINE_AA)
    cv2.addWeighted(sh, 0.35, img, 0.65, 0, img)

    n = int(rng.integers(7, 11))
    if p.cls == EXW:
        n = max(4, int(n * (1 - 0.35 * p.severity)))
    for i in range(n):
        a = 2 * np.pi * i / n + rng.uniform(-0.25, 0.25)
        L = r * rng.uniform(0.85, 1.15)
        W = r * rng.uniform(0.26, 0.36)
        col = green * rng.uniform(0.9, 1.1, 3)
        cy_off = 0.0
        if p.cls == EXW:
            wilt = np.clip(p.severity * rng.uniform(0.6, 1.2), 0, 1)
            yellow, brown = (205, 190, 70), (125, 95, 45)
            col = _lerp(tuple(col), yellow, min(1.0, wilt * 1.2))
            if wilt > 0.7:
                col = _lerp(col, brown, (wilt - 0.7) / 0.3 * 0.8)
            cy_off = 0.25 * r * wilt           # leaves droop
            L *= 1 - 0.25 * wilt
        ex, ey = _draw_leaf(img, p.cx, p.cy + cy_off, a, L, W, col, vscale)
        if p.cls == OTHER:                     # spots / blotches (similar to EXW in colour)
            for _ in range(int(rng.integers(3, 9))):
                t = rng.uniform(0.2, 1.0)
                sx = p.cx + np.cos(a) * L * t / 1.0 * 0.95
                sy = p.cy + np.sin(a) * L * t * vscale
                cv2.circle(img, (int(sx), int(sy)), max(1, int(r * rng.uniform(0.05, 0.11))),
                           _lerp((150, 110, 50), (195, 170, 70), rng.uniform(0, 1)), -1, cv2.LINE_AA)
    if p.cls == EXW and p.severity > 0.6:     # collapsed centre
        cv2.circle(img, (int(p.cx), int(p.cy + r * 0.1)), max(2, int(r * 0.22)), (110, 85, 45), -1, cv2.LINE_AA)


def _background(h, w, style, farm_rng) -> np.ndarray:
    soil = np.array(style["soil"], float)
    yy = np.linspace(0.85, 1.1, h)[:, None, None]
    img = np.ones((h, w, 3)) * soil * yy
    noise = cv2.resize(farm_rng.normal(0, 1, (h // 8 + 1, w // 8 + 1)), (w, h))[:, :, None] * 9
    img = np.clip(img + noise, 0, 255).astype(np.uint8)
    # track
    y0 = int(farm_rng.uniform(0.15, 0.3) * h)
    pts = np.array([[0, y0], [w, y0 + int(farm_rng.uniform(-30, 30))],
                    [w, y0 + 26], [0, y0 + 30]], np.int32)
    cv2.fillPoly(img, [pts], tuple(int(c * 1.15) for c in soil))
    # roofs (non-enset distractors)
    for _ in range(int(farm_rng.integers(0, 3))):
        x, y = int(farm_rng.uniform(0, w - 70)), int(farm_rng.uniform(0, h * 0.25))
        col = [(165, 165, 170), (150, 90, 80)][int(farm_rng.integers(0, 2))]
        cv2.rectangle(img, (x, y), (x + int(farm_rng.uniform(40, 70)), y + int(farm_rng.uniform(25, 40))), col, -1)
    return img


def make_farm(farm_id: int, style_id: int, farm_type: str, seed: int, cfg: dict) -> dict:
    """Fixed per-farm layout: plant slots with classes, trees, weeds."""
    rng = np.random.default_rng(seed)
    h, w = cfg["height"], cfg["width"]
    n_rows, n_cols = int(rng.integers(4, 6)), int(rng.integers(6, 9))
    sc = cfg.get("plant_scale", w / 512.0)                 # pixel scale of plants/trees/weeds vs the 512-px design
    p_exw, p_oth = {"exw": (rng.uniform(0.15, 0.35), rng.uniform(0, 0.12)),
                    "other": (0.0, rng.uniform(0.12, 0.30)),
                    "healthy": (0.0, 0.0)}[farm_type]
    plants = []
    for ri in range(n_rows):
        t = (ri + 0.5) / n_rows                       # 0 far .. 1 near (front view from the field edge)
        y = h * (0.30 + 0.62 * t)
        r = (13 + 31 * t) * sc
        for ci in range(n_cols):
            x = (ci + 0.5) / n_cols * w + rng.uniform(-0.25, 0.25) * w / n_cols
            u = rng.uniform()
            cls = EXW if u < p_exw else OTHER if u < p_exw + p_oth else HEALTHY
            sev = float(rng.beta(2.2, 1.6)) if cls != HEALTHY else 0.0
            plants.append(Plant(x, y + rng.uniform(-4, 4) * sc, r * rng.uniform(0.9, 1.1), cls, sev,
                                int(rng.integers(0, 2**31 - 1))))
    trees = [(rng.uniform(0, w), rng.uniform(0.15, 0.35) * h, rng.uniform(18, 34) * sc) for _ in range(int(rng.integers(1, 4)))]
    weeds = [(rng.uniform(0, w), rng.uniform(0.35, 0.98) * h, rng.uniform(8, 16) * sc) for _ in range(int(rng.integers(5, 12)))]
    label = "exw_suspect" if any(p.cls == EXW and p.severity > 0.35 for p in plants) else (
        "other_condition" if any(p.cls == OTHER for p in plants) else "healthy")
    return dict(farm_id=farm_id, style=style_id, type=farm_type, plants=plants, trees=trees,
                weeds=weeds, bg_seed=int(rng.integers(0, 2**31 - 1)), label=label)


def render_scene(farm: dict, scene_idx: int, cfg: dict, seed: int):
    """Render one view of a farm. Returns (image uint8 HxWx3, plants-in-view list, scene metadata)."""
    rng = np.random.default_rng(seed)
    h, w = cfg["height"], cfg["width"]
    style = STYLES[farm["style"] % len(STYLES)]
    img = _background(h, w, style, np.random.default_rng(farm["bg_seed"]))
    sc = cfg.get("plant_scale", w / 512.0)
    dx, dy = rng.uniform(-12, 12) * sc, rng.uniform(-6, 6) * sc           # small camera shift between views
    for (x, y, rad) in farm["weeds"]:                           # yellow weeds: look like EXW colour
        col = _lerp((190, 175, 80), (95, 140, 60), rng.uniform())
        cv2.circle(img, (int(x + dx), int(y + dy)), int(rad), tuple(int(c) for c in col), -1, cv2.LINE_AA)
    for (x, y, rad) in farm["trees"]:
        cv2.circle(img, (int(x + dx), int(y + dy)), int(rad), (28, 80, 40), -1, cv2.LINE_AA)
        cv2.circle(img, (int(x + dx + rad * .3), int(y + dy - rad * .2)), int(rad * .6), (40, 105, 50), -1, cv2.LINE_AA)
    visible = []
    for p in sorted(farm["plants"], key=lambda q: q.cy):        # far to near
        if rng.uniform() < 0.1:
            continue                                            # plant not visible in this view
        q = Plant(p.cx + dx, p.cy + dy, p.r, p.cls, p.severity, p.seed)
        if -p.r < q.cx < w + p.r:
            draw_plant(img, q, style)
            visible.append(q)
    out = img.astype(np.float32) * style["bright"] * rng.uniform(0.95, 1.05)
    out = cv2.GaussianBlur(out, (0, 0), style["blur"] * sc)
    out += rng.normal(0, 3.0, out.shape)
    return np.clip(out, 0, 255).astype(np.uint8), visible


def boxes_from_plants(plants, cfg: dict, rng) -> list[dict]:
    """Coarse boxes around EXW / other plants (healthy plants need no box)."""
    h, w = cfg["height"], cfg["width"]
    j = cfg.get("box_jitter_px", 3)
    out = []
    for p in plants:
        if p.cls == HEALTHY or (p.cls == EXW and p.severity < 0.35):
            continue                                              # too early to be annotated as suspect
        x0, y0 = p.cx - p.r * 1.1 + rng.normal(0, j), p.cy - p.r * 0.95 + rng.normal(0, j)
        x1, y1 = p.cx + p.r * 1.1 + rng.normal(0, j), p.cy + p.r * 0.80 + rng.normal(0, j)
        out.append(dict(x0=float(np.clip(x0, 0, w)), y0=float(np.clip(y0, 0, h)),
                        x1=float(np.clip(x1, 0, w)), y1=float(np.clip(y1, 0, h)),
                        cls=int(p.cls)))
    return out


def generate_dataset(cfg: dict, out_dir: str | Path, seed: int = 42) -> pd.DataFrame:
    """Write PNG scenes, ``annotations.json`` and ``manifest_raw.csv`` (no split yet)."""
    sc = cfg["synthetic"]
    out_dir = Path(out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows, ann = [], {}
    farm_id = 0
    types = ["exw", "exw", "other", "healthy"]                    # ~50 % EXW farms
    for site in range(sc["n_sites"]):
        style = site % sc["n_styles"]
        for _ in range(sc["farms_per_site"]):
            ftype = types[int(rng.integers(0, len(types)))]
            farm = make_farm(farm_id, style, ftype, int(rng.integers(0, 2**31 - 1)), sc)
            for s in range(sc["scenes_per_farm"]):
                img, plants = render_scene(farm, s, sc, int(rng.integers(0, 2**31 - 1)))
                image_id = f"syn_s{site:02d}_f{farm_id:03d}_v{s}"
                path = out_dir / "images" / f"{image_id}.png"
                Image.fromarray(img).save(path)
                boxed = bool(rng.uniform() < sc["frac_boxed"])
                boxes = boxes_from_plants(plants, sc, rng) if boxed else []
                ann[image_id] = dict(boxes=boxes, boxed=boxed,
                                     plants=[dict(cx=p.cx, cy=p.cy, r=p.r, cls=p.cls, severity=p.severity) for p in plants])
                rows.append(dict(
                    image_id=image_id, path=str(path.relative_to(out_dir)), site_id=site, farm_id=farm_id,
                    date=f"2026-{1 + (site % 12):02d}-{1 + (farm_id % 27):02d}", source="synthetic", tier=0,
                    style=style, farm_label=farm["label"], has_boxes=boxed,
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            farm_id += 1
    with open(out_dir / "annotations.json", "w") as f:
        json.dump(ann, f)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "manifest_raw.csv", index=False)
    assert set(df.farm_label) <= set(FARM_LABELS)
    return df
