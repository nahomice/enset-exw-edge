from pathlib import Path

import yaml

from exw.config import load_config, with_overrides


def _keys(d, prefix=""):
    out = set()
    for k, v in d.items():
        out.add(prefix + k)
        if isinstance(v, dict):
            out |= _keys(v, prefix + k + ".")
    return out


def test_overrides_parse_types_and_do_not_mutate():
    cfg = load_config("configs/default.yaml", ["train.epochs=3", "tiles.size=64", "train.seeds=[0]"])
    assert cfg["train"]["epochs"] == 3 and cfg["tiles"]["size"] == 64 and cfg["train"]["seeds"] == [0]
    base = load_config("configs/default.yaml")
    new = with_overrides(base, **{"tiles.size": 128})
    assert base["tiles"]["size"] == 96 and new["tiles"]["size"] == 128


def test_all_configs_have_the_same_keys_as_default():
    ref = _keys(yaml.safe_load(Path("configs/default.yaml").read_text()))
    for f in Path("configs").glob("*.yaml"):
        assert _keys(yaml.safe_load(f.read_text())) == ref, f"{f} has different keys than default.yaml"


def test_stride_is_half_tile_by_default_and_params_budget_is_set():
    cfg = load_config("configs/default.yaml")
    assert cfg["tiles"]["stride"] * 2 == cfg["tiles"]["size"]
    assert cfg["model"]["max_params"] == 200000
