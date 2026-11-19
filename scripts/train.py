#!/usr/bin/env python
"""Step 2: train the FP32 tile CNN for every seed in the config."""

import argparse

from exw.config import add_cli, load_config
from exw.data.dataset import load_tiles
from exw.train import train_one

ap = add_cli(argparse.ArgumentParser())
a = ap.parse_args()
cfg = load_config(a.config, a.overrides)
tiles = cfg["paths"]["tiles_dir"]
tr, va = load_tiles(f"{tiles}/train.npz"), load_tiles(f"{tiles}/val.npz")
for seed in cfg["train"]["seeds"]:
    info = train_one(cfg, tr, va, seed, cfg["paths"]["results_dir"])
    print(
        f"seed {seed}: {info['epochs_run']} epochs, val_acc={info['val_acc'][-1]:.3f}, params={info['n_params']}"
    )
