"""Unit tests for the plan-level objectives (paper Eqs. 4, 8, 9–11)."""

import math

import numpy as np
import pytest

from uavta.dynamics.objectives import evaluate_plan

LAM_OBSE = 0.001
LAM_RISK = 0.0002


def _two_target_kwargs(**overrides):
    """One UAV at the origin, targets at x=1500 m and x=3000 m (hand-computable)."""
    kwargs = dict(
        t=0.0,
        uav_positions=[[0.0, 0.0, 0.0]],
        uav_speeds=[15.0],
        uav_rates=[1 / 1800],
        uav_energies=[1.0],
        uav_vulnerabilities=[0.9],
        uav_etas=[0.8],
        target_positions=[[1500.0, 0.0, 0.0], [3000.0, 0.0, 0.0]],
        target_values=[60.0, 90.0],
        target_observation_times=[10.0, 20.0],
        target_reference_times=[0.0, 0.0],
        unfinished=[True, True],
        lambda_obse=LAM_OBSE,
        lambda_risk=LAM_RISK,
    )
    kwargs.update(overrides)
    return kwargs


def test_hand_computed_two_targets():
    res = evaluate_plan([[0, 1]], **_two_target_kwargs())
    # Arrive T0 at 100 s, leave at 110 s, arrive T1 at 210 s, finish at 230 s.
    s1 = math.exp(-LAM_RISK * 0.9 * 1500)
    s2 = s1 * math.exp(-LAM_RISK * 0.9 * 1500)
    collected = 60 * 0.8 * s1 * math.exp(-LAM_OBSE * 100) + 90 * 0.8 * s2 * math.exp(
        -LAM_OBSE * 210
    )
    assert res.f2 == pytest.approx(230.0)
    assert res.collected_value[0] == pytest.approx(collected, rel=1e-12)
    assert res.f1 == pytest.approx(150.0 - collected, rel=1e-12)
    assert res.residual_energy[0] == pytest.approx(1 - 230 / 1800)
    assert res.feasible


def test_order_matters_for_makespan():
    near_first = evaluate_plan([[0, 1]], **_two_target_kwargs())
    far_first = evaluate_plan([[1, 0]], **_two_target_kwargs())
    assert far_first.f2 > near_first.f2


def test_plan_relative_decay_reference():
    # Planning at t=50 from the origin: absolute and plan-relative decay differ by exp(λ·50).
    kw = _two_target_kwargs(t=50.0, unfinished=[True, False])
    absolute = evaluate_plan([[0]], **kw)
    relative = evaluate_plan([[0]], decay_reference="plan", **kw)
    assert relative.collected_value[0] / absolute.collected_value[0] == pytest.approx(
        math.exp(LAM_OBSE * 50.0)
    )


def test_energy_infeasible():
    res = evaluate_plan([[0, 1]], **_two_target_kwargs(uav_energies=[0.1]))
    assert not res.feasible
    assert res.residual_energy[0] < 0


def test_duplicate_assignment_rejected():
    kw = _two_target_kwargs(
        uav_positions=[[0, 0, 0], [0, 0, 0]],
        uav_speeds=[15, 15],
        uav_rates=[1 / 1800] * 2,
        uav_energies=[1, 1],
        uav_vulnerabilities=[0.9, 0.9],
        uav_etas=[0.8, 0.8],
    )
    with pytest.raises(ValueError, match="Eq. 10"):
        evaluate_plan([[0, 1], [1]], **kw)


def test_incomplete_plan_rejected():
    with pytest.raises(ValueError, match="Eq. 9"):
        evaluate_plan([[0]], **_two_target_kwargs())
    res = evaluate_plan([[0]], require_complete=False, **_two_target_kwargs())
    assert res.f1 > 90.0  # target 1's whole value remains


def test_finished_target_rejected():
    with pytest.raises(ValueError, match="finished"):
        evaluate_plan([[0, 1]], **_two_target_kwargs(unfinished=[True, False]))


def test_idle_uav_completion_is_start_time():
    kw = _two_target_kwargs(
        t=40.0,
        uav_positions=[[0, 0, 0], [0, 0, 0]],
        uav_speeds=[15, 15],
        uav_rates=[1 / 1800] * 2,
        uav_energies=[1, 1],
        uav_vulnerabilities=[0.9, 0.9],
        uav_etas=[0.8, 0.8],
    )
    res = evaluate_plan([[0, 1], []], **kw)
    np.testing.assert_allclose(res.completion_times, [270.0, 40.0])
