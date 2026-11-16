import numpy as np

from exw.eval.metrics import choose_threshold, farm_metrics
from exw.eval.stats import grouped_bootstrap


def test_threshold_keeps_sensitivity_on_val():
    risk = np.array([0.0, 0.01, 0.02, 0.2, 0.3, 0.4])
    y = np.array([0, 0, 0, 1, 1, 1])
    t = choose_threshold(risk, y, min_sens=1.0)
    m = farm_metrics(risk, y, t)
    assert m["sensitivity"] == 1.0 and m["specificity"] == 1.0


def test_bootstrap_resamples_groups():
    groups = np.repeat(np.arange(10), 3)
    vals = np.arange(30, dtype=float)
    pt, lo, hi = grouped_bootstrap(groups, lambda idx: vals[idx].mean(), 200)
    assert lo <= pt <= hi
