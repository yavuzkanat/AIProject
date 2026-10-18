"""Event-driven mission simulator (no Gymnasium, no PyTorch).

The simulator advances continuous time from one state change to the next: a UAV arriving at
its target, an observation finishing, an idle UAV running out of energy, an exogenous event,
or the horizon. It stops whenever an operational, unassigned UAV has at least one valid target
and therefore needs an allocation decision (docs/phase1_design.md §B.4, §G.3).

Processing order at a time instant t (identical in every situation):

1. arrivals (transit → observing),
2. observation completions (value realized, target completed, UAV freed),
3. energy depletion of idle UAVs,
4. exogenous events in type order (crash → sensor → weather → move → value → new → preference),
   each applied to the state left by the previous one,
5. termination / truncation checks and decision-epoch detection.

Physical completions precede events at the same instant, so data already collected is never
destroyed retroactively.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from uavta.dynamics.energy import is_energy_feasible, residual_energy, time_to_depletion
from uavta.dynamics.kinematics import advance_toward, clip_to_airspace, distance
from uavta.dynamics.sensing import sensing_capability
from uavta.dynamics.survival import segment_survival
from uavta.dynamics.value import decay_factor, expected_collected_value
from uavta.entities.sensors import SENSORS, sensor_index
from uavta.entities.target import TargetState
from uavta.entities.uav import UAVState, UAVStatus
from uavta.events.types import (
    Event,
    EventType,
    NewTarget,
    PreferenceShift,
    SensorDamage,
    TargetMove,
    UAVCrash,
    ValueChange,
    WeatherChange,
)
from uavta.scenarios.scenario import Scenario

TIME_EPS = 1e-9
ENERGY_EPS = 1e-12
MAX_ITERATIONS_WITHOUT_PROGRESS = 10_000


class SimulationError(RuntimeError):
    """An internal invariant was violated (a bug, never a legitimate agent mistake)."""


class InvalidActionError(ValueError):
    """An allocation decision violates the action mask."""


@dataclass(frozen=True)
class SimulatorConfig:
    """Simulation rules that are design choices rather than paper facts.

    Attributes:
        horizon: T_max in s; reaching it truncates the episode (J10).
        energy_reserve: E_min in Eq. 11-style feasibility checks (0 = paper's strict "> 0").
        idle_consumes_energy: Hovering idle UAVs consume γᵢ per second (J5).
        allow_voluntary_wait: Allow WAIT while valid targets exist (J3b ablation only).
        min_operational_uavs: Selector-based (stochastic) crashes are skipped when they
            would leave fewer operational UAVs than this. Named crashes always apply.
    """

    horizon: float = 2400.0
    energy_reserve: float = 0.0
    idle_consumes_energy: bool = True
    allow_voluntary_wait: bool = False
    min_operational_uavs: int = 1

    def __post_init__(self) -> None:
        if self.horizon <= 0:
            raise ValueError("horizon must be > 0")
        if not 0.0 <= self.energy_reserve < 1.0:
            raise ValueError("energy_reserve must lie in [0, 1)")
        if self.min_operational_uavs < 0:
            raise ValueError("min_operational_uavs must be >= 0")


@dataclass
class Activity:
    """One contiguous activity of a UAV, recorded for Gantt charts and route maps."""

    uav: int
    kind: str  # "idle" | "transit" | "observe" | "failed" | "depleted"
    t_start: float
    p_start: np.ndarray
    slot: int | None = None
    t_end: float | None = None
    p_end: np.ndarray | None = None


@dataclass
class StepAccumulator:
    """Quantities accrued since the last decision, consumed by the reward function."""

    collected_value: float = 0.0
    active_time: float = 0.0
    energy_used: float = 0.0
    completions: int = 0

    def reset(self) -> None:
        self.collected_value = 0.0
        self.active_time = 0.0
        self.energy_used = 0.0
        self.completions = 0


@dataclass
class SimulatorTotals:
    """Episode-level running totals."""

    collected_value: float = 0.0
    active_time: float = 0.0
    completions: int = 0
    decisions: int = 0
    waits: int = 0
    aborted_tasks: int = 0
    interrupted_observations: int = 0
    last_completion_time: float = 0.0
    events_applied: int = 0
    events_without_effect: int = 0


@dataclass
class Simulator:
    """Mutable mission state plus the transition logic.

    Args:
        scenario: The mission to simulate.
        config: Simulation rules.
        max_targets: Number of target slots (Z_max). Defaults to ``scenario.max_targets``.
    """

    scenario: Scenario
    config: SimulatorConfig = field(default_factory=SimulatorConfig)
    max_targets: int | None = None

    def __post_init__(self) -> None:
        if self.max_targets is None:
            self.max_targets = self.scenario.max_targets
        if self.scenario.max_targets > self.max_targets:
            raise ValueError(
                f"scenario {self.scenario.name} needs {self.scenario.max_targets} target slots "
                f"but only {self.max_targets} are configured"
            )
        sc = self.scenario
        self.airspace = np.asarray(sc.airspace, dtype=np.float64)
        self.t = 0.0
        self.uavs: list[UAVState] = [UAVState.from_spec(s) for s in sc.uavs]
        self._uav_index = {u.spec.id: i for i, u in enumerate(self.uavs)}
        self.targets: list[TargetState] = []
        self._target_index: dict[str, int] = {}
        for spec in sc.targets:
            self._add_target(spec, appear_time=0.0)
        self.initial_target_value = float(sum(t.value for t in self.targets))
        self.weather = np.asarray(sc.initial_weather, dtype=np.float64).copy()
        self.preference = tuple(sc.initial_preference)
        self._events: list[Event] = list(sc.events)
        self._next_event = 0
        self.waiting = np.zeros(len(self.uavs), dtype=bool)
        self.decision_uav: int | None = None
        self.terminated = False
        self.truncated = False
        self.termination_reason: str | None = None
        self.step_acc = StepAccumulator()
        self.totals = SimulatorTotals()
        self.event_log: list[dict[str, Any]] = []
        self.assignment_log: list[dict[str, Any]] = []
        self.warnings: list[str] = []
        self.last_event_type: EventType | None = None
        self.last_event_time: float | None = None
        self.activities: list[Activity] = [
            Activity(uav=i, kind="idle", t_start=0.0, p_start=u.position.copy())
            for i, u in enumerate(self.uavs)
        ]
        self._open_activity: list[int] = list(range(len(self.uavs)))
        self._check_end_or_decision()
        if not self.done and self.decision_uav is None:
            self.run_until_decision()

    # ------------------------------------------------------------------ queries

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated

    @property
    def num_uavs(self) -> int:
        return len(self.uavs)

    def eta(self, i: int) -> float:
        """Current sensing capability ηᵢ(t) (Eq. 5)."""
        return sensing_capability(self.uavs[i].sensor_ok, self.weather)

    def has_pending_events(self) -> bool:
        """Privileged: whether the schedule holds future events. Used only for termination."""
        return self._next_event < len(self._events)

    def task_duration(self, i: int, slot: int) -> tuple[float, float]:
        """(distance m, travel + observation time s) for idle UAV i taking target ``slot`` now."""
        u, tg = self.uavs[i], self.targets[slot]
        d = distance(u.position, tg.position)
        return d, d / u.spec.speed + tg.spec.observation_time

    def valid_targets(self, i: int) -> np.ndarray:
        """Boolean mask of length ``max_targets``: slots UAV i may be assigned now.

        A slot is valid iff it holds an appeared, unfinished, unlocked target, UAV i is
        operational and unassigned, and the task is energy-feasible (Eq. 11 for one task).
        """
        assert self.max_targets is not None
        mask = np.zeros(self.max_targets, dtype=bool)
        u = self.uavs[i]
        if not u.is_operational or u.has_task:
            return mask
        rate = u.spec.consumption_rate
        for tg in self.targets:
            if tg.completed or tg.locked_by is not None:
                continue
            _, duration = self.task_duration(i, tg.slot)
            mask[tg.slot] = is_energy_feasible(u.energy, rate, duration, self.config.energy_reserve)
        return mask

    def needs_decision(self, i: int) -> bool:
        u = self.uavs[i]
        if not u.is_operational or u.has_task or self.waiting[i]:
            return False
        return bool(self.valid_targets(i).any())

    def time_until_free(self, i: int) -> float:
        """Seconds until UAV i finishes its committed task (0 if unassigned)."""
        u = self.uavs[i]
        if u.task_slot is None:
            return 0.0
        tg = self.targets[u.task_slot]
        if u.status == UAVStatus.OBSERVING:
            assert u.observation_end_time is not None
            return max(u.observation_end_time - self.t, 0.0)
        return distance(u.position, tg.position) / u.spec.speed + tg.spec.observation_time

    def free_position(self, i: int) -> np.ndarray:
        """Where UAV i will be when it becomes free."""
        u = self.uavs[i]
        if u.task_slot is None:
            return u.position
        return self.targets[u.task_slot].position

    def expected_value_now(self, i: int, slot: int) -> float:
        """Expected collected value if idle UAV i flies to ``slot`` now (Eq. 4 summand).

        Uses the current η, value and weather; the realized value is evaluated at completion.
        """
        u, tg = self.uavs[i], self.targets[slot]
        d = distance(u.position, tg.position)
        arrival = self.t + d / u.spec.speed
        survival = u.survival * segment_survival(d, u.spec.vulnerability, self.scenario.lambda_risk)
        decay = decay_factor(arrival, tg.reference_time, self.scenario.lambda_obse)
        return expected_collected_value(tg.value, self.eta(i), survival, decay)

    def mission_active(self) -> bool:
        """True while some operational UAV holds a task (time cost accrues, phase1 §E)."""
        return any(u.is_operational and u.has_task for u in self.uavs)

    # ------------------------------------------------------------------ decisions

    def assign(self, slot: int | None) -> None:
        """Apply the decision for ``decision_uav``: a target slot, or ``None`` for WAIT.

        Raises:
            InvalidActionError: If the decision violates the mask.
            SimulationError: If no decision is pending.
        """
        if self.done or self.decision_uav is None:
            raise SimulationError("assign() called with no pending decision")
        i = self.decision_uav
        valid = self.valid_targets(i)
        if slot is None:
            if valid.any() and not self.config.allow_voluntary_wait:
                raise InvalidActionError(f"WAIT chosen for UAV {i} although valid targets exist")
            self.waiting[i] = True
            self.totals.waits += 1
        else:
            assert self.max_targets is not None
            if not 0 <= slot < self.max_targets or not valid[slot]:
                raise InvalidActionError(f"target slot {slot} is not valid for UAV {i}")
            u, tg = self.uavs[i], self.targets[slot]
            u.task_slot = slot
            tg.locked_by = i
            self._set_status(i, UAVStatus.TRANSIT, slot)
            self.assignment_log.append({"time": self.t, "uav": i, "slot": slot})
        self.totals.decisions += 1
        self.decision_uav = None
        self._check_end_or_decision()

    def run_until_decision(self) -> None:
        """Advance time until a decision is needed or the episode ends."""
        stalled = 0
        while not self.done and self.decision_uav is None:
            t_before = self.t
            self._advance_to(self._next_change_time())
            self._process_instant()
            self._check_end_or_decision()
            stalled = stalled + 1 if self.t <= t_before else 0
            if stalled > MAX_ITERATIONS_WITHOUT_PROGRESS:
                raise SimulationError(f"simulation stalled at t={self.t}")

    # ------------------------------------------------------------------ time advance

    def _next_change_time(self) -> float:
        candidates = [self.config.horizon]
        for u in self.uavs:
            if not u.is_operational:
                continue
            if u.status == UAVStatus.TRANSIT:
                assert u.task_slot is not None
                d = distance(u.position, self.targets[u.task_slot].position)
                candidates.append(self.t + d / u.spec.speed)
            elif u.status == UAVStatus.OBSERVING:
                assert u.observation_end_time is not None
                candidates.append(u.observation_end_time)
            elif self.config.idle_consumes_energy:
                candidates.append(self.t + time_to_depletion(u.energy, u.spec.consumption_rate))
        if self.has_pending_events():
            candidates.append(self._events[self._next_event].time)
        return max(min(candidates), self.t)

    def _advance_to(self, t_next: float) -> None:
        dt = t_next - self.t
        if dt < 0:
            raise SimulationError(f"time would move backwards: {self.t} -> {t_next}")
        if dt == 0:
            return
        if self.mission_active():
            self.step_acc.active_time += dt
            self.totals.active_time += dt
        for u in self.uavs:
            if not u.is_operational:
                continue
            if u.status == UAVStatus.TRANSIT:
                assert u.task_slot is not None
                dest = self.targets[u.task_slot].position
                new_pos, flown = advance_toward(u.position, dest, u.spec.speed, dt)
                u.position = new_pos
                u.distance_flown += flown
                u.survival *= segment_survival(
                    flown, u.spec.vulnerability, self.scenario.lambda_risk
                )
            if u.status != UAVStatus.IDLE or self.config.idle_consumes_energy:
                used = u.spec.consumption_rate * dt
                if u.status == UAVStatus.IDLE:
                    used = min(used, u.energy)  # absorbs rounding at the depletion instant
                u.energy -= used  # busy UAVs are never clamped: the invariant check catches bugs
                self.step_acc.energy_used += used
        self.t = t_next

    def _process_instant(self) -> None:
        t = self.t
        # 1. arrivals
        for i, u in enumerate(self.uavs):
            if u.is_operational and u.status == UAVStatus.TRANSIT:
                assert u.task_slot is not None
                tg = self.targets[u.task_slot]
                remaining = distance(u.position, tg.position)
                if remaining / u.spec.speed <= TIME_EPS:
                    u.position = tg.position.copy()
                    u.arrival_time = t
                    u.observation_end_time = t + tg.spec.observation_time
                    self._set_status(i, UAVStatus.OBSERVING, u.task_slot)
        # 2. completions
        for i, u in enumerate(self.uavs):
            if (
                u.is_operational
                and u.status == UAVStatus.OBSERVING
                and u.observation_end_time is not None
                and u.observation_end_time <= t + TIME_EPS
            ):
                self._complete_observation(i)
        # 3. depletion of idle UAVs
        if self.config.idle_consumes_energy:
            for i, u in enumerate(self.uavs):
                if u.is_operational and u.status == UAVStatus.IDLE and u.energy <= ENERGY_EPS:
                    u.energy = 0.0
                    self._set_status(i, UAVStatus.DEPLETED, None)
        # 4. exogenous events
        changed = False
        while self.has_pending_events() and self._events[self._next_event].time <= t + TIME_EPS:
            self._apply_event(self._events[self._next_event])
            self._next_event += 1
            changed = True
        if changed:
            self.waiting[:] = False
        self._check_invariants()

    def _complete_observation(self, i: int) -> None:
        u = self.uavs[i]
        assert u.task_slot is not None and u.arrival_time is not None
        tg = self.targets[u.task_slot]
        decay = decay_factor(u.arrival_time, tg.reference_time, self.scenario.lambda_obse)
        value = expected_collected_value(tg.value, self.eta(i), u.survival, decay)
        tg.completed = True
        tg.completion_time = self.t
        tg.collected_value = value
        tg.locked_by = None
        self.step_acc.collected_value += value
        self.step_acc.completions += 1
        self.totals.collected_value += value
        self.totals.completions += 1
        self.totals.last_completion_time = self.t
        u.task_slot = None
        u.arrival_time = None
        u.observation_end_time = None
        self._set_status(i, UAVStatus.IDLE, None)
        self.waiting[:] = False

    # ------------------------------------------------------------------ events

    def _apply_event(self, event: Event) -> None:
        record: dict[str, Any] = {
            "time": self.t,
            "scheduled_time": event.time,
            "type": event.event_type.name.lower(),
            "label": event.label,
            "uav": None,
            "target": None,
            "effect": "applied",
        }
        if isinstance(event, UAVCrash):
            self._event_crash(event, record)
        elif isinstance(event, SensorDamage):
            self._event_sensor(event, record)
        elif isinstance(event, WeatherChange):
            if event.mode == "absolute":
                self.weather = np.asarray(event.omega, dtype=np.float64).copy()
            else:
                self.weather = self.weather * np.asarray(event.omega, dtype=np.float64)
            record["omega"] = self.weather.tolist()
        elif isinstance(event, TargetMove):
            self._event_move(event, record)
        elif isinstance(event, ValueChange):
            self._event_value(event, record)
        elif isinstance(event, NewTarget):
            slot = self._add_target(event.target, appear_time=self.t)
            record["target"] = self.targets[slot].spec.id
            record["slot"] = slot
        elif isinstance(event, PreferenceShift):
            if event.expected_from is not None and not np.allclose(
                event.expected_from, self.preference
            ):
                self._warn(
                    f"t={self.t}: preference shift expected {event.expected_from}, "
                    f"current is {self.preference}"
                )
            self.preference = tuple(event.weights)
            record["weights"] = list(event.weights)
        else:  # pragma: no cover
            raise TypeError(f"unknown event {event!r}")
        if record["effect"] == "applied":
            self.totals.events_applied += 1
            self.last_event_type = event.event_type
            self.last_event_time = self.t
        else:
            self.totals.events_without_effect += 1
        self.event_log.append(record)

    def _resolve_uav(
        self, name: str | None, selector: float | None, eligible: list[int]
    ) -> int | None:
        if name is not None:
            i = self._uav_index[name]
            return i if i in eligible else None
        if not eligible:
            return None
        assert selector is not None
        return eligible[int(selector * len(eligible))]

    def _resolve_target(self, name: str | None, selector: float | None) -> int | None:
        unfinished = [tg.slot for tg in self.targets if not tg.completed]
        if name is not None:
            slot = self._target_index.get(name)
            return slot if slot in unfinished else None
        if not unfinished:
            return None
        assert selector is not None
        return unfinished[int(selector * len(unfinished))]

    def _event_crash(self, event: UAVCrash, record: dict[str, Any]) -> None:
        operational = [i for i, u in enumerate(self.uavs) if u.is_operational]
        if event.uav is None and len(operational) <= self.config.min_operational_uavs:
            record["effect"] = "skipped_min_operational"
            return
        i = self._resolve_uav(event.uav, event.uav_selector, operational)
        if i is None:
            record["effect"] = "no_effect"
            return
        u = self.uavs[i]
        record["uav"] = u.spec.id
        if u.task_slot is not None:
            tg = self.targets[u.task_slot]
            tg.locked_by = None
            record["released_target"] = tg.spec.id
            if u.status == UAVStatus.OBSERVING:
                self.totals.interrupted_observations += 1
            self.totals.aborted_tasks += 1
        u.task_slot = None
        u.arrival_time = None
        u.observation_end_time = None
        self._set_status(i, UAVStatus.FAILED, None)

    def _event_sensor(self, event: SensorDamage, record: dict[str, Any]) -> None:
        eligible = [i for i, u in enumerate(self.uavs) if u.is_operational and u.sensor_ok.any()]
        i = self._resolve_uav(event.uav, event.uav_selector, eligible)
        if i is None:
            record["effect"] = "no_effect"
            return
        u = self.uavs[i]
        record["uav"] = u.spec.id
        if event.sensor is not None:
            m = sensor_index(event.sensor)
        else:
            working = np.flatnonzero(u.sensor_ok)
            assert event.sensor_selector is not None
            m = int(working[int(event.sensor_selector * len(working))])
        record["sensor"] = SENSORS[m]
        if not u.sensor_ok[m]:
            record["effect"] = "no_effect"
            return
        u.sensor_ok[m] = False

    def _event_move(self, event: TargetMove, record: dict[str, Any]) -> None:
        slot = self._resolve_target(event.target, event.target_selector)
        if slot is None:
            record["effect"] = "no_effect"
            record["target"] = event.target
            return
        tg = self.targets[slot]
        record["target"] = tg.spec.id
        if event.expected_from is not None and not np.allclose(event.expected_from, tg.position):
            self._warn(
                f"t={self.t}: move of {tg.spec.id} expected from {event.expected_from}, "
                f"current position {tg.position.tolist()}"
            )
        if event.destination is not None:
            dest = np.asarray(event.destination, dtype=np.float64)
        else:
            assert event.displacement is not None
            dest = tg.position + np.asarray(event.displacement, dtype=np.float64)
        tg.position = clip_to_airspace(dest, self.airspace)
        record["position"] = tg.position.tolist()
        if tg.locked_by is None:
            return
        i = tg.locked_by
        u = self.uavs[i]
        if u.status == UAVStatus.OBSERVING:
            # Observation interrupted: fly to the new position and restart it there (G7).
            self.totals.interrupted_observations += 1
            u.arrival_time = None
            u.observation_end_time = None
            record["interrupted_uav"] = u.spec.id
        d = distance(u.position, tg.position)
        duration = d / u.spec.speed + tg.spec.observation_time
        if is_energy_feasible(
            u.energy, u.spec.consumption_rate, duration, self.config.energy_reserve
        ):
            record["redirected_uav"] = u.spec.id
            self._set_status(i, UAVStatus.TRANSIT, slot)  # new straight segment from here
        else:
            tg.locked_by = None
            u.task_slot = None
            self.totals.aborted_tasks += 1
            self._set_status(i, UAVStatus.IDLE, None)
            record["aborted_uav"] = u.spec.id

    def _event_value(self, event: ValueChange, record: dict[str, Any]) -> None:
        slot = self._resolve_target(event.target, event.target_selector)
        if slot is None:
            record["effect"] = "no_effect"
            record["target"] = event.target
            return
        tg = self.targets[slot]
        record["target"] = tg.spec.id
        if event.expected_from is not None and not math.isclose(event.expected_from, tg.value):
            self._warn(
                f"t={self.t}: value of {tg.spec.id} expected {event.expected_from}, "
                f"current {tg.value}"
            )
        tg.value = (
            float(event.new_value)
            if event.new_value is not None
            else tg.value
            * float(
                event.factor  # type: ignore[arg-type]
            )
        )
        record["value"] = tg.value

    # ------------------------------------------------------------------ bookkeeping

    def _add_target(self, spec: Any, appear_time: float) -> int:
        assert self.max_targets is not None
        slot = len(self.targets)
        if slot >= self.max_targets:
            raise SimulationError(f"target {spec.id} needs slot {slot} >= max_targets")
        if spec.id in self._target_index:
            raise SimulationError(f"target id {spec.id} already present")
        self.targets.append(TargetState.from_spec(spec, slot=slot, appear_time=appear_time))
        self._target_index[spec.id] = slot
        return slot

    def _set_status(self, i: int, status: UAVStatus, slot: int | None) -> None:
        u = self.uavs[i]
        u.status = status
        act = self.activities[self._open_activity[i]]
        act.t_end = self.t
        act.p_end = u.position.copy()
        kind = {
            UAVStatus.IDLE: "idle",
            UAVStatus.TRANSIT: "transit",
            UAVStatus.OBSERVING: "observe",
            UAVStatus.FAILED: "failed",
            UAVStatus.DEPLETED: "depleted",
        }[status]
        self.activities.append(
            Activity(uav=i, kind=kind, t_start=self.t, p_start=u.position.copy(), slot=slot)
        )
        self._open_activity[i] = len(self.activities) - 1

    def close_activities(self) -> None:
        """Close the open activity of every UAV at the current time (call at episode end)."""
        for i, idx in enumerate(self._open_activity):
            act = self.activities[idx]
            if act.t_end is None:
                act.t_end = self.t
                act.p_end = self.uavs[i].position.copy()

    def _warn(self, message: str) -> None:
        self.warnings.append(message)

    def _check_end_or_decision(self) -> None:
        """Set termination/truncation flags or select the next decision UAV."""
        self.decision_uav = None
        if self.done:
            return
        operational = [i for i, u in enumerate(self.uavs) if u.is_operational]
        if not operational:
            self.terminated, self.termination_reason = True, "no_operational_uav"
            return
        if self.t >= self.config.horizon - TIME_EPS:
            self.truncated, self.termination_reason = True, "horizon"
            return
        for i in operational:
            if self.needs_decision(i):
                self.decision_uav = i
                return
        anyone_busy = any(self.uavs[i].has_task for i in operational)
        if anyone_busy or self.has_pending_events():
            return
        if self.config.allow_voluntary_wait and any(
            self.valid_targets(i).any() for i in operational
        ):
            return  # waiting UAVs keep hovering until they deplete
        all_done = all(tg.completed for tg in self.targets)
        self.terminated = True
        self.termination_reason = "all_targets_completed" if all_done else "no_feasible_target"

    def _check_invariants(self) -> None:
        holders: dict[int, int] = {}
        for i, u in enumerate(self.uavs):
            if u.energy < -1e-9:
                raise SimulationError(f"UAV {u.spec.id} energy negative: {u.energy}")
            if u.task_slot is not None:
                if not u.is_operational:
                    raise SimulationError(f"non-operational UAV {u.spec.id} holds a task")
                if u.task_slot in holders:
                    raise SimulationError(f"target slot {u.task_slot} held by two UAVs")
                holders[u.task_slot] = i
                tg = self.targets[u.task_slot]
                if tg.completed or tg.locked_by != i:
                    raise SimulationError(f"inconsistent lock on target {tg.spec.id}")
        for tg in self.targets:
            if tg.locked_by is not None and holders.get(tg.slot) != tg.locked_by:
                raise SimulationError(f"dangling lock on target {tg.spec.id}")

    # ------------------------------------------------------------------ metrics

    def residual_energy_after(self, i: int, duration: float) -> float:
        u = self.uavs[i]
        return residual_energy(u.energy, u.spec.consumption_rate, duration)

    def episode_summary(self) -> dict[str, Any]:
        """Realized mission outcomes (phase1 §I.3). Privileged; for evaluation only."""
        # Values cannot change after completion, so tg.value is the value at completion for
        # completed targets and the current value for unfinished ones.
        appeared_value = sum(tg.value for tg in self.targets)
        n_targets = len(self.targets)
        energies = np.array([u.energy for u in self.uavs])
        used = 1.0 - energies
        survival = np.array([u.survival for u in self.uavs])
        return {
            "t_end": self.t,
            "termination_reason": self.termination_reason,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "collected_value": self.totals.collected_value,
            "appeared_value": appeared_value,
            "initial_value": self.initial_target_value,
            "value_fraction": self.totals.collected_value / appeared_value
            if appeared_value > 0
            else 0.0,
            "residual_value_f1": appeared_value - self.totals.collected_value,
            "n_targets_appeared": n_targets,
            "n_completed": self.totals.completions,
            "completion_rate": self.totals.completions / n_targets if n_targets else 1.0,
            "makespan": self.totals.last_completion_time,
            "active_time": self.totals.active_time,
            "energy_used_total": float(used.sum()),
            "energy_used_per_uav": used.tolist(),
            "min_residual_energy": float(energies.min()) if len(energies) else 0.0,
            "risk_exposure": float(1.0 - survival.mean()) if len(survival) else 0.0,
            "n_operational_end": sum(u.is_operational for u in self.uavs),
            "n_decisions": self.totals.decisions,
            "n_waits": self.totals.waits,
            "aborted_tasks": self.totals.aborted_tasks,
            "interrupted_observations": self.totals.interrupted_observations,
            "events_applied": self.totals.events_applied,
            "events_without_effect": self.totals.events_without_effect,
            "unfinished_feasible_at_end": self._unfinished_feasible(),
        }

    def _unfinished_feasible(self) -> int:
        """Unfinished targets still energy-feasible for some operational idle UAV."""
        count = 0
        for tg in self.targets:
            if tg.completed:
                continue
            for i, u in enumerate(self.uavs):
                if not u.is_operational:
                    continue
                d = distance(self.free_position(i), tg.position)
                dur = self.time_until_free(i) + d / u.spec.speed + tg.spec.observation_time
                if is_energy_feasible(u.energy, u.spec.consumption_rate, dur):
                    count += 1
                    break
        return count
