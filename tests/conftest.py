import pytest

from exw.config import load_config, with_overrides


@pytest.fixture(scope="session")
def tiny_cfg(tmp_path_factory):
    """A very small synthetic dataset (fast) written to a temp dir."""
    root = tmp_path_factory.mktemp("exw")
    cfg = load_config("configs/default.yaml")
    return with_overrides(cfg, **{
        "paths.data_dir": str(root / "data"), "paths.tiles_dir": str(root / "tiles"),
        "paths.results_dir": str(root / "res"),
        "synthetic.n_sites": 12, "synthetic.farms_per_site": 1, "synthetic.scenes_per_farm": 2,
        "synthetic.width": 480, "synthetic.height": 360, "synthetic.plant_scale": 1.875,
    })
