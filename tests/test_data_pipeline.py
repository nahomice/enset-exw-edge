from exw.data.dataset import build_all, load_tiles
from exw.data.manifest import build_synthetic_manifest
from exw.data.splits import assert_no_leakage


def test_splits_have_no_leakage_and_all_splits_exist(tiny_cfg):
    df = build_synthetic_manifest(tiny_cfg)
    assert_no_leakage(df, "site_id")
    assert {"train", "val", "test"} <= set(df["split"])
    # held-out style forms the domain-shift set and is disjoint from the others
    shift_sites = set(df[df.split == "test_shift"].site_id)
    assert shift_sites.isdisjoint(set(df[df.split != "test_shift"].site_id))


def test_tiles_built_per_split_and_no_ignore_labels(tiny_cfg):
    build_synthetic_manifest(tiny_cfg)
    table = build_all(tiny_cfg, seed=0)
    assert table["n_tiles"].sum() > 0
    d = load_tiles(f"{tiny_cfg['paths']['tiles_dir']}/train.npz")
    assert (d["y"] >= 0).all() and d["X"].dtype.name == "uint8" and d["X"].shape[1:] == (96, 96, 3)


def test_synthetic_generation_is_deterministic(tiny_cfg, tmp_path):
    from exw.data.synthetic import generate_dataset
    a = generate_dataset(tiny_cfg["synthetic"] and {**tiny_cfg, "synthetic": tiny_cfg["synthetic"]}, tmp_path / "a", seed=1)
    b = generate_dataset(tiny_cfg, tmp_path / "b", seed=1)
    assert list(a["sha256"]) == list(b["sha256"])
