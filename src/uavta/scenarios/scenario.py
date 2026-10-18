"""Immutable scenario description consumed by the simulator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from uavta.entities.sensors import sensor_index
from uavta.entities.target import TargetSpec
from uavta.entities.uav import UAVSpec
from uavta.events.types import (
    Event,
    NewTarget,
    SensorDamage,
    TargetMove,
    UAVCrash,
    ValueChange,
    sort_events,
)


class ScenarioError(ValueError):
    """A scenario definition is internally inconsistent."""


@dataclass(frozen=True)
class Scenario:
    """A fully specified mission: entities, model constants and the event schedule.

    Attributes:
        name: Human-readable name, e.g. ``"paper_S3"``.
        airspace: (Lx, Ly, Lz) in m.
        lambda_obse: Value decay constant in 1/s.
        lambda_risk: Flight risk constant in 1/m.
        uavs: UAV specifications in index order.
        targets: Targets present at t = 0, in slot order.
        events: Event schedule sorted by (time, type priority).
        initial_weather: ω⁰ per sensor.
        initial_preference: (w₁, w₂) at t = 0.
    """

    name: str
    airspace: tuple[float, float, float]
    lambda_obse: float
    lambda_risk: float
    uavs: tuple[UAVSpec, ...]
    targets: tuple[TargetSpec, ...]
    events: tuple[Event, ...]
    initial_weather: tuple[float, ...] = (1.0, 1.0, 1.0)
    initial_preference: tuple[float, float] = (0.5, 0.5)

    def __post_init__(self) -> None:
        object.__setattr__(self, "events", tuple(sort_events(list(self.events))))
        self.validate()

    @property
    def num_new_targets(self) -> int:
        """Number of targets that appear through events."""
        return sum(isinstance(e, NewTarget) for e in self.events)

    @property
    def max_targets(self) -> int:
        """Total number of target slots the scenario can occupy."""
        return len(self.targets) + self.num_new_targets

    def validate(self) -> None:
        """Check cross-references and physical bounds.

        Raises:
            ScenarioError: On duplicate ids, positions outside the airspace, or events that
                reference unknown or not-yet-appeared entities.
        """
        box = np.asarray(self.airspace, dtype=np.float64)
        if np.any(box <= 0):
            raise ScenarioError(f"{self.name}: airspace must be positive, got {self.airspace}")

        def inside(p: tuple[float, ...]) -> bool:
            q = np.asarray(p, dtype=np.float64)
            return bool(np.all(q >= 0) and np.all(q <= box))

        uav_ids = [u.id for u in self.uavs]
        if len(set(uav_ids)) != len(uav_ids):
            raise ScenarioError(f"{self.name}: duplicate UAV ids {uav_ids}")
        for u in self.uavs:
            if not inside(u.start_position):
                raise ScenarioError(f"{self.name}: UAV {u.id} starts outside the airspace")

        known_targets = {t.id for t in self.targets}
        if len(known_targets) != len(self.targets):
            raise ScenarioError(f"{self.name}: duplicate initial target ids")
        for t in self.targets:
            if not inside(t.position):
                raise ScenarioError(f"{self.name}: target {t.id} lies outside the airspace")

        uav_by_id = {u.id: u for u in self.uavs}
        for e in self.events:
            if isinstance(e, NewTarget):
                if e.target.id in known_targets:
                    raise ScenarioError(f"{self.name}: target {e.target.id} appears twice")
                if not inside(e.target.position):
                    raise ScenarioError(f"{self.name}: new target {e.target.id} outside airspace")
                known_targets.add(e.target.id)
            elif isinstance(e, (UAVCrash, SensorDamage)) and e.uav is not None:
                if e.uav not in uav_by_id:
                    raise ScenarioError(
                        f"{self.name}: event at t={e.time} names unknown UAV {e.uav}"
                    )
                if (
                    isinstance(e, SensorDamage)
                    and e.sensor is not None
                    and not uav_by_id[e.uav].sensors[sensor_index(e.sensor)]
                ):
                    raise ScenarioError(
                        f"{self.name}: UAV {e.uav} does not carry sensor {e.sensor}"
                    )
            elif isinstance(e, (TargetMove, ValueChange)) and e.target is not None:
                if e.target not in known_targets:
                    raise ScenarioError(
                        f"{self.name}: event at t={e.time} names unknown or future "
                        f"target {e.target}"
                    )
                if (
                    isinstance(e, TargetMove)
                    and e.destination is not None
                    and not inside(e.destination)
                ):
                    raise ScenarioError(f"{self.name}: move destination outside airspace")
