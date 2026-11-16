"""Tiny YAML configuration loader with command-line overrides (``--set a.b=value``)."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml


def load_config(path: str | Path = "configs/default.yaml", overrides: list[str] | None = None) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for item in overrides or []:
        key, _, raw = item.partition("=")
        _set_dotted(cfg, key.strip(), yaml.safe_load(raw))
    return cfg


def _set_dotted(cfg: dict, key: str, value) -> None:
    parts = key.split(".")
    node = cfg
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = value


def with_overrides(cfg: dict, **flat) -> dict:
    """Return a copy of ``cfg`` with dotted keys replaced, e.g. ``with_overrides(cfg, **{"train.epochs": 2})``."""
    out = copy.deepcopy(cfg)
    for k, v in flat.items():
        _set_dotted(out, k, v)
    return out


def add_cli(parser):
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        help="override a config value, e.g. --set train.epochs=3")
    return parser
