"""Event-driven simulator: closed-form checks, events, lifecycle and invariants."""

import math

import numpy as np
import pytest

from builders import LAM_OBSE, LAM_RISK, make_scenario, make_sim, step_to_done
from uavta.entities.uav import UAVStatus
from uavta.environment.simulator import InvalidActionError, SimulationError


def v1_closed_form():
    return 50 * math.exp(-LAM_RISK * 1000) * math.exp(-LAM_OBSE * 100)


def v2_closed_form():
    return 80 * math.exp(-LAM_RISK * 3000) * math.exp(-LAM_OBSE * 310)


class TestClosedForm:
    def test_two_targets_in_order(self):
        sim = make_sim()
        assert sim.decision_uav == 0 and sim.t == 0.0
        sim.assign(0)
        sim.run_until_decision()
        # T1 reached at 100 s, observed until 110 s.
        assert sim.t == pytest.approx(110.0, abs=1e-9)
        assert sim.totals.collected_value == pytest.approx(v1_closed_form(), abs=1e-9)
        sim.assign(1)
        sim.run_until_decision()
        assert sim.terminated and sim.termination_reason == "all_targets_completed"
        s = sim.episode_summary()
        assert s["makespan"] == pytest.approx(330.0, abs=1e-9)
        assert s["collected_value"] == pytest.approx(v1_closed_form() + v2_closed_form(), abs=1e-9)
        assert s["energy_used_total"] == pytest.approx(330.0 / 10000.0, abs=1e-12)
        assert sim.uavs[0].survival == pytest.approx(math.exp(-LAM_RISK * 3000), abs=1e-12)
        assert s["completion_rate"] == 1.0

    def test_matches_plan_evaluation(self):
        from uavta.dynamics.objectives import evaluate_plan

        sim = make_sim()
        step_to_done(sim)
        plan = evaluate_plan(
            [[0, 1]],
            t=0.0,
            uav_positions=[[0, 0, 100]],
            uav_speeds=[10.0],
            uav_rates=[1e-4],
            uav_energies=[1.0],
            uav_vulnerabilities=[1.0],
            uav_etas=[1.0],
            target_positions=[[1000, 0, 100], [3000, 0, 100]],
            target_values=[50, 80],
            target_observation_times=[10, 20],
            target_reference_times=[0, 0],
            unfinished=[True, True],
            lambda_obse=LAM_OBSE,
            lambda_risk=LAM_RISK,
        )
        assert sim.totals.collected_value == pytest.approx(plan.collected_value[0], abs=1e-9)
        assert sim.totals.last_completion_time == pytest.approx(plan.f2, abs=1e-9)


class TestDecisions:
    def test_duplicate_assignment_prevented(self):
        sc = make_scenario(
            uavs=[
                {"id": "U1", "type": "A", "position": [0.0, 0.0, 100.0]},
                {"id": "U2", "type": "A", "position": [0.0, 10.0, 100.0]},
            ]
        )
        sim = make_sim(sc)
        assert sim.decision_uav == 0
        sim.assign(0)
        assert sim.decision_uav == 1 and sim.t == 0.0  # same instant
        assert not sim.valid_targets(1)[0] and sim.valid_targets(1)[1]
        with pytest.raises(InvalidActionError):
            sim.assign(0)

    def test_wait_rejected_when_targets_valid(self):
        sim = make_sim()
        with pytest.raises(InvalidActionError):
            sim.assign(None)

    def test_voluntary_wait_option(self):
        sim = make_sim(allow_voluntary_wait=True)
        sim.assign(None)
        assert sim.waiting[0]

    def test_energy_infeasible_target_masked(self):
        # Endurance 300 s: T1 needs 110 s (feasible), T2 needs 320 s (infeasible).
        types = make_scenario().uavs  # noqa: F841 - keep builder import used
        sc = make_scenario(
            uav_types={
                "A": {
                    "speed": 10.0,
                    "endurance": 300.0,
                    "sensors": {"VL": 1, "IR": 0, "RD": 0},
                    "vulnerability": 1.0,
                }
            }
        )
        sim = make_sim(sc)
        assert sim.valid_targets(0)[:2].tolist() == [True, False]

    def test_assign_without_pending_decision(self):
        sim = make_sim()
        step_to_done(sim)
        with pytest.raises(SimulationError):
            sim.assign(0)


class TestEvents:
    def test_crash_during_transit_releases_target(self):
        sc = make_scenario(events=[{"time": 50, "type": "uav_crash", "uav": "U1"}])
        sim = make_sim(sc)
        sim.assign(0)
        sim.run_until_decision()
        assert sim.terminated and sim.termination_reason == "no_operational_uav"
        assert sim.uavs[0].status == UAVStatus.FAILED
        assert sim.targets[0].locked_by is None and not sim.targets[0].completed
        assert sim.totals.collected_value == 0.0
        np.testing.assert_allclose(sim.uavs[0].position, [500.0, 0.0, 100.0])

    def test_crash_during_observation_loses_value_and_frees_target(self):
        sc = make_scenario(
            uavs=[
                {"id": "U1", "type": "A", "position": [0.0, 0.0, 100.0]},
                {"id": "U2", "type": "A", "position": [5000.0, 0.0, 100.0]},
            ],
            events=[{"time": 105, "type": "uav_crash", "uav": "U1"}],
        )
        sim = make_sim(sc)
        sim.assign(0)  # U1 -> T1 (arrive 100, observing until 110)
        sim.assign(1)  # U2 -> T2 (arrive 200)
        sim.run_until_decision()
        # Crash at 105 frees T1 (its interrupted observation yields nothing); U2 completes T2.
        assert sim.totals.interrupted_observations == 1
        t2_value = 80 * math.exp(-LAM_RISK * 2000) * math.exp(-LAM_OBSE * 200)
        assert sim.totals.collected_value == pytest.approx(t2_value, abs=1e-9)
        assert not sim.targets[0].completed and sim.targets[0].locked_by is None
        assert sim.t == pytest.approx(220.0)  # next decision: U2 finished T2
        assert sim.decision_uav == 1 and sim.valid_targets(1)[0]

    def test_sensor_damage_and_weather_lower_value(self):
        sc = make_scenario(
            uavs=[{"id": "U1", "type": "B", "position": [0.0, 0.0, 100.0]}],
            targets=[
                {"id": "T1", "value": 50, "observation_time": 10, "position": [1000.0, 0.0, 100.0]}
            ],
            events=[
                {"time": 20, "type": "weather", "omega": {"VL": 0.5, "IR": 0.5, "RD": 0.5}},
                {"time": 30, "type": "sensor_damage", "uav": "U1", "sensor": "VL"},
            ],
        )
        sim = make_sim(sc)
        sim.assign(0)
        sim.run_until_decision()
        eta = 1 - (1 - 0.5) * (1 - 0.5)  # IR + RD remain, both at 0.5
        expected = 50 * eta * math.exp(-LAM_RISK * 1000) * math.exp(-LAM_OBSE * 100)
        assert sim.totals.collected_value == pytest.approx(expected, abs=1e-9)
        assert not sim.uavs[0].sensor_ok[0]

    def test_multiplicative_weather_is_non_increasing(self):
        sc = make_scenario(
            events=[
                {
                    "time": 20,
                    "type": "weather",
                    "mode": "multiplicative",
                    "omega": {"VL": 0.5, "IR": 1.0, "RD": 1.0},
                },
                {
                    "time": 30,
                    "type": "weather",
                    "mode": "multiplicative",
                    "omega": {"VL": 0.5, "IR": 1.0, "RD": 1.0},
                },
            ]
        )
        sim = make_sim(sc)
        sim.assign(0)
        sim.run_until_decision()
        assert sim.weather[0] == pytest.approx(0.25)

    def test_target_move_redirects_transit(self):
        sc = make_scenario(
            targets=[
                {"id": "T1", "value": 50, "observation_time": 10, "position": [1000.0, 0.0, 100.0]}
            ],
            events=[
                {"time": 50, "type": "target_move", "target": "T1", "to": [500.0, 500.0, 100.0]}
            ],
        )
        sim = make_sim(sc)
        sim.assign(0)
        sim.run_until_decision()
        # At t=50 the UAV is at (500,0,100); remaining 500 m -> arrival at 100 s.
        assert sim.terminated
        assert sim.targets[0].completion_time == pytest.approx(110.0)
        assert sim.uavs[0].distance_flown == pytest.approx(1000.0)

    def test_target_move_interrupts_observation(self):
        sc = make_scenario(
            targets=[
                {"id": "T1", "value": 50, "observation_time": 10, "position": [1000.0, 0.0, 100.0]}
            ],
            events=[
                {"time": 105, "type": "target_move", "target": "T1", "to": [1000.0, 300.0, 100.0]}
            ],
        )
        sim = make_sim(sc)
        sim.assign(0)
        sim.run_until_decision()
        # Re-arrival at 105 + 30 = 135, observation restarts and ends at 145.
        assert sim.totals.interrupted_observations == 1
        assert sim.targets[0].completion_time == pytest.approx(145.0)

    def test_move_making_task_infeasible_aborts(self):
        sc = make_scenario(
            uav_types={
                "A": {
                    "speed": 10.0,
                    "endurance": 200.0,
                    "sensors": {"VL": 1, "IR": 0, "RD": 0},
                    "vulnerability": 1.0,
                }
            },
            targets=[
                {"id": "T1", "value": 50, "observation_time": 10, "position": [1000.0, 0.0, 100.0]}
            ],
            events=[
                {"time": 50, "type": "target_move", "target": "T1", "to": [9000.0, 0.0, 100.0]}
            ],
        )
        sim = make_sim(sc)
        sim.assign(0)
        sim.run_until_decision()
        assert sim.totals.aborted_tasks == 1
        assert sim.targets[0].locked_by is None
        assert sim.terminated and sim.termination_reason == "no_feasible_target"

    def test_new_target_while_all_busy(self):
        sc = make_scenario(
            targets=[
                {"id": "T1", "value": 50, "observation_time": 10, "position": [1000.0, 0.0, 100.0]}
            ]
            + [
                {
                    "id": "N1",
                    "value": 60,
                    "observation_time": 5,
                    "position": [0.0, 1000.0, 100.0],
                    "initial": False,
                }
            ],
            events=[{"time": 40, "type": "new_target", "target": "N1"}],
        )
        sim = make_sim(sc)
        sim.assign(0)
        sim.run_until_decision()
        assert len(sim.targets) == 2 and sim.targets[1].slot == 1
        assert sim.targets[1].reference_time == 40.0
        assert sim.t == pytest.approx(110.0) and sim.decision_uav == 0
        assert sim.valid_targets(0)[1]

    def test_value_change_on_completed_target_has_no_effect(self):
        sc = make_scenario(
            targets=[
                {"id": "T1", "value": 50, "observation_time": 10, "position": [1000.0, 0.0, 100.0]},
                {"id": "T2", "value": 80, "observation_time": 20, "position": [3000.0, 0.0, 100.0]},
            ],
            events=[{"time": 150, "type": "value_change", "target": "T1", "to": 10}],
        )
        sim = make_sim(sc)
        step_to_done(sim)
        rec = [e for e in sim.event_log if e["type"] == "value_change"][0]
        assert rec["effect"] == "no_effect"
        assert sim.targets[0].value == 50

    def test_expected_from_mismatch_is_logged(self):
        sc = make_scenario(
            events=[{"time": 5, "type": "value_change", "target": "T2", "from": 99, "to": 10}]
        )
        sim = make_sim(sc)
        step_to_done(sim)
        assert any("expected 99" in w for w in sim.warnings)
        assert sim.targets[1].value == 10

    def test_selector_crash_respects_min_operational(self):
        from uavta.events.types import UAVCrash
        from uavta.scenarios.scenario import Scenario

        base = make_scenario()
        sc = Scenario(
            name="sel",
            airspace=base.airspace,
            lambda_obse=base.lambda_obse,
            lambda_risk=base.lambda_risk,
            uavs=base.uavs,
            targets=base.targets,
            events=(UAVCrash(time=10.0, uav_selector=0.5),),
        )
        sim = make_sim(sc, min_operational_uavs=1)
        step_to_done(sim)
        assert sim.event_log[0]["effect"] == "skipped_min_operational"
        assert sim.termination_reason == "all_targets_completed"


class TestLifecycle:
    def test_truncation_at_horizon(self):
        sim = make_sim(horizon=150.0)
        step_to_done(sim)
        assert sim.truncated and sim.termination_reason == "horizon"
        assert sim.t == pytest.approx(150.0)

    def test_idle_depletion(self):
        # Only T1 feasible; afterwards the idle UAV hovers until a future event, depleting.
        sc = make_scenario(
            uav_types={
                "A": {
                    "speed": 10.0,
                    "endurance": 300.0,
                    "sensors": {"VL": 1, "IR": 0, "RD": 0},
                    "vulnerability": 1.0,
                }
            },
            events=[{"time": 1000, "type": "weather", "omega": {"VL": 0.9, "IR": 0.9, "RD": 0.9}}],
        )
        sim = make_sim(sc)
        step_to_done(sim)
        assert sim.uavs[0].status == UAVStatus.DEPLETED
        assert sim.uavs[0].energy == 0.0
        assert sim.t == pytest.approx(300.0)
        assert sim.termination_reason == "no_operational_uav"

    def test_no_targets(self):
        sim = make_sim(make_scenario(targets=[]))
        assert sim.terminated and sim.termination_reason == "all_targets_completed"

    def test_determinism(self):
        from uavta.scenarios.generator import ScenarioGenerator

        sc = ScenarioGenerator().generate(1234)
        a, b = make_sim(sc, max_targets=24), make_sim(sc, max_targets=24)
        step_to_done(a)
        step_to_done(b)
        assert a.episode_summary() == b.episode_summary()
        assert a.event_log == b.event_log

    def test_simulator_does_not_mutate_scenario(self):
        sc = make_scenario()
        sim = make_sim(sc)
        step_to_done(sim)
        assert sc.targets[0].position == (1000.0, 0.0, 100.0)

    def test_too_many_targets_for_slots(self):
        with pytest.raises(ValueError):
            make_sim(make_scenario(), max_targets=1)

    @pytest.mark.parametrize("seed", range(20))
    def test_random_rollouts_keep_invariants(self, seed):
        from uavta.scenarios.generator import EventRates, GeneratorConfig, ScenarioGenerator

        gen = ScenarioGenerator(GeneratorConfig(event_rates=EventRates().scaled(3.0)))
        sim = make_sim(gen.generate(seed), max_targets=24)
        rng = np.random.default_rng(seed)
        step_to_done(
            sim, lambda s: int(rng.choice(np.flatnonzero(s.valid_targets(s.decision_uav))))
        )
        s = sim.episode_summary()
        assert 0.0 <= s["value_fraction"] <= 1.0
        assert all(u.energy >= -1e-9 for u in sim.uavs)
        assert all(tg.completed or tg.locked_by is None for tg in sim.targets)
