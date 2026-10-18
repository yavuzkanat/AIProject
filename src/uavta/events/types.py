"""Dynamic event types (paper §3.1, Table 5).

Events are immutable records. Each event either names its affected entity explicitly
(scheduled scenarios such as S1–S3) or carries a pre-drawn ``selector`` in [0, 1) that the
simulator maps to an eligible entity when the event fires (stochastic schedules). Drawing
selectors at reset fixes the whole event sequence for a seed, so every policy evaluated on
that seed faces the same disruptions.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import ClassVar, Literal

from uavta.entities.sensors import SENSORS
from uavta.entities.target import TargetSpec


class EventType(IntEnum):
    """Event categories. The integer value is the application order for simultaneous events."""

    UAV_CRASH = 0
    SENSOR_DAMAGE = 1
    WEATHER = 2
    TARGET_MOVE = 3
    VALUE_CHANGE = 4
    NEW_TARGET = 5
    PREFERENCE_SHIFT = 6


WeatherMode = Literal["absolute", "multiplicative"]


def _check_selector(name: str, explicit: object, selector: float | None) -> None:
    if (explicit is None) == (selector is None):
        raise ValueError(f"exactly one of {name!r} and its selector must be given")
    if selector is not None and not 0.0 <= selector < 1.0:
        raise ValueError(f"selector must lie in [0, 1), got {selector}")


@dataclass(frozen=True, kw_only=True)
class BaseEvent:
    """Fields shared by all events.

    Attributes:
        time: Occurrence time in s (> 0).
        label: Optional provenance tag, e.g. the paper's event number.
    """

    event_type: ClassVar[EventType]
    time: float
    label: str | None = None

    def __post_init__(self) -> None:
        if self.time <= 0:
            raise ValueError(f"event time must be > 0, got {self.time}")


@dataclass(frozen=True, kw_only=True)
class UAVCrash(BaseEvent):
    """A UAV is permanently lost."""

    event_type: ClassVar[EventType] = EventType.UAV_CRASH
    uav: str | None = None
    uav_selector: float | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        _check_selector("uav", self.uav, self.uav_selector)


@dataclass(frozen=True, kw_only=True)
class SensorDamage(BaseEvent):
    """One carried sensor of a UAV stops working."""

    event_type: ClassVar[EventType] = EventType.SENSOR_DAMAGE
    uav: str | None = None
    uav_selector: float | None = None
    sensor: str | None = None
    sensor_selector: float | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        _check_selector("uav", self.uav, self.uav_selector)
        _check_selector("sensor", self.sensor, self.sensor_selector)
        if self.sensor is not None and self.sensor not in SENSORS:
            raise ValueError(f"unknown sensor {self.sensor!r}")


@dataclass(frozen=True, kw_only=True)
class WeatherChange(BaseEvent):
    """Weather effectiveness factors change.

    ``absolute`` sets ω directly (S1–S3 replay, J6); ``multiplicative`` multiplies the current
    ω, which keeps ω non-increasing as the paper's text describes.
    """

    event_type: ClassVar[EventType] = EventType.WEATHER
    omega: tuple[float, ...]
    mode: WeatherMode = "absolute"

    def __post_init__(self) -> None:
        super().__post_init__()
        if len(self.omega) != len(SENSORS):
            raise ValueError(f"omega needs {len(SENSORS)} factors, got {self.omega}")
        if any(not 0.0 < w <= 1.0 for w in self.omega):
            raise ValueError(f"omega factors must lie in (0, 1], got {self.omega}")
        if self.mode not in ("absolute", "multiplicative"):
            raise ValueError(f"unknown weather mode {self.mode!r}")


@dataclass(frozen=True, kw_only=True)
class TargetMove(BaseEvent):
    """An unfinished target relocates.

    Scheduled form: ``target`` and absolute ``destination`` (``expected_from`` documents the
    paper's "from" position). Stochastic form: ``target_selector`` and a ``displacement``
    vector added to the current position and clipped to the airspace.
    """

    event_type: ClassVar[EventType] = EventType.TARGET_MOVE
    target: str | None = None
    target_selector: float | None = None
    destination: tuple[float, float, float] | None = None
    displacement: tuple[float, float, float] | None = None
    expected_from: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        _check_selector("target", self.target, self.target_selector)
        if (self.destination is None) == (self.displacement is None):
            raise ValueError("exactly one of 'destination' and 'displacement' must be given")


@dataclass(frozen=True, kw_only=True)
class ValueChange(BaseEvent):
    """An unfinished target's observational value changes.

    Scheduled form sets ``new_value``; stochastic form multiplies by ``factor``.
    """

    event_type: ClassVar[EventType] = EventType.VALUE_CHANGE
    target: str | None = None
    target_selector: float | None = None
    new_value: float | None = None
    factor: float | None = None
    expected_from: float | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        _check_selector("target", self.target, self.target_selector)
        if (self.new_value is None) == (self.factor is None):
            raise ValueError("exactly one of 'new_value' and 'factor' must be given")
        if self.new_value is not None and self.new_value <= 0:
            raise ValueError(f"new_value must be > 0, got {self.new_value}")
        if self.factor is not None and self.factor <= 0:
            raise ValueError(f"factor must be > 0, got {self.factor}")


@dataclass(frozen=True, kw_only=True)
class NewTarget(BaseEvent):
    """A previously unknown target appears and takes the next free slot."""

    event_type: ClassVar[EventType] = EventType.NEW_TARGET
    target: TargetSpec


@dataclass(frozen=True, kw_only=True)
class PreferenceShift(BaseEvent):
    """The decision maker's preference weights change (logged only under J4a)."""

    event_type: ClassVar[EventType] = EventType.PREFERENCE_SHIFT
    weights: tuple[float, float]
    expected_from: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if len(self.weights) != 2 or any(w < 0 for w in self.weights):
            raise ValueError(f"weights must be two non-negative numbers, got {self.weights}")
        if abs(sum(self.weights) - 1.0) > 1e-9:
            raise ValueError(f"weights must sum to 1, got {self.weights}")


Event = (
    UAVCrash | SensorDamage | WeatherChange | TargetMove | ValueChange | NewTarget | PreferenceShift
)


def sort_events(events: list[Event]) -> list[Event]:
    """Order events by time, then by type priority; stable within equal (time, type)."""
    return sorted(events, key=lambda e: (e.time, int(e.event_type)))
