"""Statistics helpers and per-episode metrics."""

import math

import numpy as np
import pytest

from builders import make_scenario, make_sim, step_to_done
from uavta.evaluation.metrics import latency_stats, response_delays
from uavta.evaluation.stats import bootstrap_ci, holm_correction, iqm, paired_wilcoxon


def test_bootstrap_ci_contains_mean():
    x = np.random.default_rng(0).normal(5.0, 1.0, 200)
    lo, hi = bootstrap_ci(x)
    assert lo < x.mean() < hi


def test_iqm():
    assert iqm([1, 2, 3, 4, 100, -100, 2.5, 3.5]) == pytest.approx(np.mean([2, 2.5, 3, 3.5]))


def test_holm_known_values():
    adj = holm_correction([0.01, 0.04, 0.03])
    assert adj == pytest.approx([0.03, 0.06, 0.06])


def test_paired_wilcoxon_direction():
    rng = np.random.default_rng(1)
    b = rng.normal(0, 1, 50)
    a = b + 0.5 + rng.normal(0, 0.1, 50)
    res = paired_wilcoxon(a, b)
    assert res.p_value < 1e-6 and res.rank_biserial == pytest.approx(1.0)
    assert paired_wilcoxon(b, b).p_value == 1.0


def test_latency_stats():
    s = latency_stats([0.001, 0.002, 0.003])
    assert s["latency_median_ms"] == pytest.approx(2.0)
    assert math.isnan(latency_stats([])["latency_median_ms"])


def test_response_delay_after_crash():
    sc = make_scenario(
        uavs=[
            {"id": "U1", "type": "A", "position": [0.0, 0.0, 100.0]},
            {"id": "U2", "type": "A", "position": [5000.0, 0.0, 100.0]},
        ],
        events=[{"time": 105, "type": "uav_crash", "uav": "U1"}],
    )
    sim = make_sim(sc)
    step_to_done(sim)
    d = response_delays(sim)
    # T1 is re-assigned when U2 becomes free at 220 s.
    assert d["mean_crash_reassign_delay"] == pytest.approx(115.0)
    assert d["mean_response_delay"] == pytest.approx(115.0)
