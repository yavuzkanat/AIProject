"""UAV specification (paper Table 1, Table 3) and runtime state."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

from uavta.entities.sensors import NUM_SENSORS


class UAVStatus(IntEnum):
    """Operational status of a UAV.

    ``DEPLETED`` is not in the paper; it marks a UAV whose energy reached zero while
    hovering idle (docs/assumptions.md, N1).
    """

    IDLE = 0
    TRANSIT = 1
    OBSERVING = 2
    FAILED = 3
    DEPLETED = 4


@dataclass(frozen=True)
class UAVSpec:
    """Static UAV attributes that do not change during a mission.

    Attributes:
        id: Identifier, e.g. ``"U1"``.
        uav_type: Type label, e.g. ``"A"``.
        speed: Flight speed vᵢ in m/s.
        endurance: Maximum endurance t_endu,i in s.
        sensors: Carried sensors as booleans in ``SENSORS`` order (VL, IR, RD).
        vulnerability: Vulnerability factor ξᵢ (dimensionless, > 0).
        start_position: Initial position in m.
    """

    id: str
    uav_type: str
    speed: float
    endurance: float
    sensors: tuple[bool, ...]
    vulnerability: float
    start_position: tuple[float, float, float]

    def __post_init__(self) -> None:
        if self.speed <= 0:
            raise ValueError(f"{self.id}: speed must be > 0, got {self.speed}")
        if self.endurance <= 0:
            raise ValueError(f"{self.id}: endurance must be > 0, got {self.endurance}")
        if self.vulnerability <= 0:
            raise ValueError(f"{self.id}: vulnerability must be > 0, got {self.vulnerability}")
        if len(self.sensors) != NUM_SENSORS:
            raise ValueError(f"{self.id}: expected {NUM_SENSORS} sensor flags, got {self.sensors}")
        if len(self.start_position) != 3:
            raise ValueError(f"{self.id}: start_position must have 3 coordinates")

    @property
    def consumption_rate(self) -> float:
        """Energy consumption rate γᵢ = 1 / t_endu,i in energy fraction per second (Eq. 11)."""
        return 1.0 / self.endurance


@dataclass
class UAVState:
    """Mutable UAV state inside the simulator.

    Attributes:
        spec: The static specification.
        position: Current position in m, shape (3,).
        energy: Remaining energy Eᵢ as a fraction of a full battery.
        status: Current operational status.
        sensor_ok: Sensor availability Iᵢ (carried and not damaged), shape (3,).
        survival: Cumulative survival probability p_surv,i since mission start (Eq. 7, J1).
        task_slot: Target slot of the committed task, or ``None``.
        arrival_time: Time the UAV reached its current task target, or ``None``.
        observation_end_time: Scheduled observation completion time, or ``None``.
        distance_flown: Total distance flown in m.
    """

    spec: UAVSpec
    position: np.ndarray
    energy: float = 1.0
    status: UAVStatus = UAVStatus.IDLE
    sensor_ok: np.ndarray = field(default_factory=lambda: np.zeros(NUM_SENSORS, dtype=bool))
    survival: float = 1.0
    task_slot: int | None = None
    arrival_time: float | None = None
    observation_end_time: float | None = None
    distance_flown: float = 0.0

    @classmethod
    def from_spec(cls, spec: UAVSpec) -> UAVState:
        """Create the state at mission start: full energy, all carried sensors working."""
        return cls(
            spec=spec,
            position=np.asarray(spec.start_position, dtype=np.float64).copy(),
            sensor_ok=np.asarray(spec.sensors, dtype=bool).copy(),
        )

    @property
    def is_operational(self) -> bool:
        """True unless the UAV has crashed or depleted its energy."""
        return self.status not in (UAVStatus.FAILED, UAVStatus.DEPLETED)

    @property
    def has_task(self) -> bool:
        """True while the UAV is committed to a target (transit or observation)."""
        return self.task_slot is not None
