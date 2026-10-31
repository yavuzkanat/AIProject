"""Fixed-shape observation vector (docs/assumptions.md §8).

Layout: [UAV block (max_uavs × UAV_FEATURES) | target block (max_targets × TARGET_FEATURES) |
global block (GLOBAL_FEATURES)]. Every feature lies in [-1, 1]; padded slots are all zero.
The observation contains only information the control center has at time t: no future
event times, types or parameters, and no RNG state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from uavta.dynamics.kinematics import distance
from uavta.dynamics.value import decay_factor
from uavta.entities.sensors import sensor_index
from uavta.environment.simulator import Simulator
from uavta.events.types import EventType

UAV_FEATURE_NAMES: tuple[str, ...] = (
    "present",
    "operational",
    "status_idle",
    "status_transit",
    "status_observing",
    "status_failed",
    "status_depleted",
    "pos_x",
    "pos_y",
    "pos_z",
    "energy",
    "endurance_norm",
    "speed_norm",
    "vulnerability_norm",
    "sensor_vl",
    "sensor_ir",
    "sensor_rd",
    "eta",
    "survival",
    "time_until_free",
    "is_decision_uav",
)
TARGET_FEATURE_NAMES: tuple[str, ...] = (
    "present",
    "completed",
    "locked",
    "pos_x",
    "pos_y",
    "pos_z",
    "value_norm",
    "decay_now",
    "observation_time_norm",
    "age",
    # relative to the decision UAV
    "distance",
    "energy_margin",
    "expected_value_now",
    "travel_time",
    "competitor_arrival",
    "valid",
)
GLOBAL_FEATURE_NAMES: tuple[str, ...] = (
    "time",
    "weather_vl",
    "weather_ir",
    "weather_rd",
    "preference_value",
    "preference_time",
    "operational_fraction",
    "unfinished_fraction",
    "appeared_fraction",
    *(f"last_event_{e.name.lower()}" for e in EventType),
    "time_since_last_event",
)
# Appended to every target row only when ``ObservationSpec.extended`` is True (extended
# environment with urgent targets and sensor requirements, docs/assumptions.md §12). The v1
# observation is unchanged, so v1 checkpoints keep working.
EXTRA_TARGET_FEATURE_NAMES: tuple[str, ...] = (
    "decay_rate",
    "requires_vl",
    "requires_ir",
    "requires_rd",
    "capable_uav_fraction",
    "has_deadline",
    "time_to_deadline",
    # relative to the decision UAV: (deadline − arrival time) / T_max, clipped to [-1, 1]
    "deadline_slack",
)
DECAY_RATE_REF = 0.02  # 1/s; decay rates are divided by this and clipped to [0, 1]
UAV_FEATURES = len(UAV_FEATURE_NAMES)
TARGET_FEATURES = len(TARGET_FEATURE_NAMES)
EXTENDED_TARGET_FEATURES = TARGET_FEATURES + len(EXTRA_TARGET_FEATURE_NAMES)
GLOBAL_FEATURES = len(GLOBAL_FEATURE_NAMES)


@dataclass(frozen=True)
class ObservationSpec:
    """Normalization constants, fixed per configuration (never per episode).

    Attributes:
        max_uavs, max_targets: Entity caps S_max, Z_max.
        horizon: T_max in s.
        airspace: (Lx, Ly, Lz) in m.
        value_max: V_max used to scale values.
        speed_ref: Speed scale in m/s.
        observation_time_ref: Observation-time scale in s.
        vulnerability_ref: Vulnerability scale.
        extended: Append ``EXTRA_TARGET_FEATURE_NAMES`` to every target row.
    """

    max_uavs: int
    max_targets: int
    horizon: float
    airspace: tuple[float, float, float]
    value_max: float = 100.0
    speed_ref: float = 30.0
    observation_time_ref: float = 60.0
    vulnerability_ref: float = 2.0
    extended: bool = False

    @property
    def target_features(self) -> int:
        return EXTENDED_TARGET_FEATURES if self.extended else TARGET_FEATURES

    @property
    def size(self) -> int:
        return (
            self.max_uavs * UAV_FEATURES
            + self.max_targets * self.target_features
            + GLOBAL_FEATURES
        )


def _clip01(x: float) -> float:
    return min(max(x, 0.0), 1.0)


def build_observation(sim: Simulator, spec: ObservationSpec) -> np.ndarray:
    """Encode the simulator state at a decision epoch as a float32 vector of ``spec.size``."""
    if sim.num_uavs > spec.max_uavs or len(sim.targets) > spec.max_targets:
        raise ValueError("simulator holds more entities than the observation spec allows")
    T = spec.horizon
    box = np.asarray(spec.airspace, dtype=np.float64)
    d_max = float(np.linalg.norm(box))
    uav_block = np.zeros((spec.max_uavs, UAV_FEATURES), dtype=np.float64)
    tgt_block = np.zeros((spec.max_targets, spec.target_features), dtype=np.float64)
    dec = sim.decision_uav

    for i, u in enumerate(sim.uavs):
        row = uav_block[i]
        row[0] = 1.0
        row[1] = float(u.is_operational)
        row[2 + int(u.status)] = 1.0
        row[7:10] = u.position / box
        row[10] = u.energy
        row[11] = _clip01(u.spec.endurance / T)
        row[12] = _clip01(u.spec.speed / spec.speed_ref)
        row[13] = _clip01(u.spec.vulnerability / spec.vulnerability_ref)
        row[14:17] = u.sensor_ok
        row[17] = sim.eta(i) if u.is_operational else 0.0
        row[18] = u.survival
        row[19] = _clip01(sim.time_until_free(i) / T) if u.is_operational else 0.0
        row[20] = float(i == dec)

    valid = sim.valid_targets(dec) if dec is not None else None
    others = [i for i, u in enumerate(sim.uavs) if u.is_operational and i != dec]
    operational = [i for i, u in enumerate(sim.uavs) if u.is_operational]
    for tg in sim.targets:
        row = tgt_block[tg.slot]
        row[0] = 1.0
        row[1] = float(tg.completed)
        row[2] = float(tg.locked_by is not None)
        row[3:6] = tg.position / box
        row[6] = _clip01(tg.value / spec.value_max)
        row[7] = decay_factor(sim.t, tg.reference_time, sim.decay_rate(tg.slot))
        row[8] = _clip01(tg.spec.observation_time / spec.observation_time_ref)
        row[9] = _clip01((sim.t - tg.appear_time) / T)
        if spec.extended:
            k = TARGET_FEATURES
            row[k] = _clip01(sim.decay_rate(tg.slot) / DECAY_RATE_REF)
            if tg.spec.required_sensor is not None:
                row[k + 1 + sensor_index(tg.spec.required_sensor)] = 1.0
            capable = sum(sim.can_sense(j, tg.slot) for j in operational)
            row[k + 4] = capable / len(operational) if operational else 0.0
            dl = tg.deadline_time
            if dl is not None and not tg.completed:
                row[k + 5] = 1.0
                row[k + 6] = _clip01((dl - sim.t) / T)
                if dec is not None:
                    u = sim.uavs[dec]
                    arrival = sim.t + distance(u.position, tg.position) / u.spec.speed
                    row[k + 7] = min(max((dl - arrival) / T, -1.0), 1.0)
        if dec is None or tg.completed:
            continue
        u = sim.uavs[dec]
        d = distance(u.position, tg.position)
        travel = d / u.spec.speed
        row[10] = _clip01(d / d_max)
        margin = u.energy - u.spec.consumption_rate * (travel + tg.spec.observation_time)
        row[11] = min(max(margin, -1.0), 1.0)
        row[12] = _clip01(sim.expected_value_now(dec, tg.slot) / spec.value_max)
        row[13] = _clip01(travel / T)
        best = math.inf
        for j in others:
            if tg.locked_by is not None and tg.locked_by != j:
                continue
            if not sim.can_sense(j, tg.slot):
                continue  # this UAV could never observe the target
            o = sim.uavs[j]
            arrival = (
                sim.time_until_free(j) + distance(sim.free_position(j), tg.position) / o.spec.speed
            )
            best = min(best, arrival)
        row[14] = _clip01(best / T) if math.isfinite(best) else 1.0
        assert valid is not None
        row[15] = float(valid[tg.slot])

    glob = np.zeros(GLOBAL_FEATURES, dtype=np.float64)
    glob[0] = _clip01(sim.t / T)
    glob[1:4] = sim.weather
    glob[4:6] = sim.preference
    glob[6] = sum(u.is_operational for u in sim.uavs) / max(sim.num_uavs, 1)
    n_app = len(sim.targets)
    glob[7] = sum(not tg.completed for tg in sim.targets) / n_app if n_app else 0.0
    glob[8] = n_app / spec.max_targets
    if sim.last_event_type is not None and sim.last_event_time is not None:
        glob[9 + int(sim.last_event_type)] = 1.0
        glob[9 + len(EventType)] = _clip01((sim.t - sim.last_event_time) / T)
    else:
        glob[9 + len(EventType)] = 1.0

    obs = np.concatenate([uav_block.ravel(), tgt_block.ravel(), glob]).astype(np.float32)
    return obs
