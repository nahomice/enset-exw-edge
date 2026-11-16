#!/usr/bin/env python
"""Step 1: synthetic scenes -> manifest (grouped splits) -> tile arrays. For REAL data use scripts/ingest_real.py."""
import argparse

from exw.config import add_cli, load_config
from exw.pipeline import prepare_data, save_json

ap = add_cli(argparse.ArgumentParser())
a = ap.parse_args()
cfg = load_config(a.config, a.overrides)
table = prepare_data(cfg)
print(table.to_string(index=False))
save_json(table.to_dict("records"), f"{cfg['paths']['results_dir']}/tile_counts.json")
