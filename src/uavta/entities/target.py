"""Target specification (paper Table 2, Table 4) and runtime state."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from uavta.entities.sensors import sensor_index


@dataclass(frozen=True)
class TargetSpec:
    """Static target attributes as defined by a scenario.

    Attributes:
        id: Identifier, e.g. ``"T1"`` or ``"NT1"``.
        value: Initial observational value Vⱼ (value units, > 0).
        observation_time: Required observation time t_obse,j in s.
        position: Initial position in m.
        initial: True if present at t = 0; False if it appears through an event.
        decay_rate: Target-specific timeliness decay constant in 1/s ("urgent" targets of the
            extended environment, docs/assumptions.md §12). ``None`` uses the scenario's
            λ_obse, as in the paper.
        required_sensor: Sensor that must observe this target (extended environment). ``None``
            means any sensor, combined by the paper's fusion model (Eq. 5).
        deadline: Seconds after the target's appearance by which an observation must START
            (extended environment, "critical" targets). ``None`` means no deadline.
        late_factor: Fraction of the value still obtained when the observation starts after
            the deadline (0 = worthless; such targets are removed from the action mask once
            no UAV can reach them in time).
    """

    id: str
    value: float
    observation_time: float
    position: tuple[float, float, float]
    initial: bool = True
    decay_rate: float | None = None
    required_sensor: str | None = None
    deadline: float | None = None
    late_factor: float = 0.0

    def __post_init__(self) -> None:
        if self.deadline is not None and self.deadline <= 0:
            raise ValueError(f"{self.id}: deadline must be > 0, got {self.deadline}")
        if not 0.0 <= self.late_factor <= 1.0:
            raise ValueError(f"{self.id}: late_factor must lie in [0, 1], got {self.late_factor}")
        if self.decay_rate is not None and self.decay_rate <= 0:
            raise ValueError(f"{self.id}: decay_rate must be > 0, got {self.decay_rate}")
        if self.required_sensor is not None:
            sensor_index(self.required_sensor)  # raises ValueError for unknown names
        if self.value <= 0:
            raise ValueError(f"{self.id}: value must be > 0, got {self.value}")
        if self.observation_time <= 0:
            raise ValueError(
                f"{self.id}: observation_time must be > 0, got {self.observation_time}"
            )
        if len(self.position) != 3:
            raise ValueError(f"{self.id}: position must have 3 coordinates")


@dataclass
class TargetState:
    """Mutable target state inside the simulator.

    Attributes:
        spec: The static specification.
        slot: Stable observation slot index; never reused within an episode.
        position: Current position in m, shape (3,); changes on target-move events.
        value: Current value Vⱼᵗ; changes on value-change events.
        appear_time: Time the target became known to the controller.
        reference_time: t_ref,j for the timeliness decay (J1).
        completed: Completion status cⱼ.
        completion_time: Time of completion, or ``None``.
        locked_by: Index of the UAV committed to this target, or ``None``.
        collected_value: Expected value realized at completion (Eq. 4 term), 0 until then.
    """

    spec: TargetSpec
    slot: int
    position: np.ndarray
    value: float
    appear_time: float
    reference_time: float
    completed: bool = False
    completion_time: float | None = None
    locked_by: int | None = None
    collected_value: float = 0.0

    @classmethod
    def from_spec(cls, spec: TargetSpec, slot: int, appear_time: float) -> TargetState:
        """Create the state of a target that appears at ``appear_time`` in ``slot``.

        The decay reference time equals the appearance time (0 for initial targets), per J1.
        """
        return cls(
            spec=spec,
            slot=slot,
            position=np.asarray(spec.position, dtype=np.float64).copy(),
            value=float(spec.value),
            appear_time=appear_time,
            reference_time=appear_time,
        )

    @property
    def deadline_time(self) -> float | None:
        """Absolute time by which an observation must start, or ``None``."""
        if self.spec.deadline is None:
            return None
        return self.appear_time + self.spec.deadline

    def timeliness(self, arrival_time: float) -> float:
        """Deadline factor of an observation starting at ``arrival_time``: 1 or late_factor."""
        dl = self.deadline_time
        if dl is None or arrival_time <= dl + 1e-9:
            return 1.0
        return self.spec.late_factor

    @property
    def is_available(self) -> bool:
        """True if the target is unfinished and not committed to any UAV."""
        return not self.completed and self.locked_by is None
