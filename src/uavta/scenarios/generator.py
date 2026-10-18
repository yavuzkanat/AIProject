"""Random scenario generator for training and generalization tests.

Layouts and parameter ranges bracket the paper's S1–S3 (§5.1.1, Tables 3–4), but they are
design choices, not paper values (docs/assumptions.md §7). A scenario is a pure function of
its seed: layout and events come from independent child streams of that seed, so changing
the event model never changes the layout drawn for a seed.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from uavta.entities.sensors import SENSORS
from uavta.entities.target import TargetSpec
from uavta.entities.uav import UAVSpec
from uavta.events.types import (
    Event,
    NewTarget,
    SensorDamage,
    TargetMove,
    UAVCrash,
    ValueChange,
    WeatherChange,
)
from uavta.scenarios.scenario import Scenario
from uavta.scenarios.schema import UAVTypeModel
from uavta.utils.seeding import make_streams

# Disjoint seed ranges (phase1 §I.2). Training seeds are drawn below VALIDATION_SEED_BASE.
VALIDATION_SEED_BASE = 1_000_000
TEST_SEED_BASE = 2_000_000
OOD_SEED_BASE = 3_000_000
TRAIN_SEED_UPPER = VALIDATION_SEED_BASE

Range = tuple[float, float]


def _paper_uav_types() -> dict[str, UAVTypeModel]:
    return {
        "A": UAVTypeModel(
            speed=15.0, endurance=1800.0, sensors={"VL": 1, "IR": 1, "RD": 0}, vulnerability=0.9
        ),
        "B": UAVTypeModel(
            speed=15.0, endurance=2400.0, sensors={"VL": 1, "IR": 1, "RD": 1}, vulnerability=1.0
        ),
    }


class EventRates(BaseModel):
    """Poisson rates in events per 1000 s for each stochastic event class."""

    model_config = ConfigDict(extra="forbid")

    new_target: float = Field(3.0, ge=0)
    uav_crash: float = Field(1.5, ge=0)
    sensor_damage: float = Field(3.0, ge=0)
    weather: float = Field(3.0, ge=0)
    target_move: float = Field(3.0, ge=0)
    value_change: float = Field(3.0, ge=0)

    def scaled(self, factor: float) -> EventRates:
        return EventRates(**{k: v * factor for k, v in self.model_dump().items()})

    def only(self, name: str) -> EventRates:
        """Keep one class at its rate, set the others to zero (single-class stress tests)."""
        data = {k: 0.0 for k in self.model_dump()}
        data[name] = getattr(self, name)
        return EventRates(**data)


class GeneratorConfig(BaseModel):
    """Distribution of generated scenarios. Defaults approximate the paper's S1–S3 setting."""

    model_config = ConfigDict(extra="forbid")

    airspace: tuple[float, float, float] = (8000.0, 8000.0, 200.0)
    lambda_obse: float = Field(0.001, gt=0)
    lambda_risk: float = Field(0.0002, gt=0)
    n_uavs: tuple[int, int] = (3, 5)
    n_targets: tuple[int, int] = (8, 18)
    uav_types: dict[str, UAVTypeModel] = Field(default_factory=_paper_uav_types)
    uav_type_probs: dict[str, float] = Field(default_factory=lambda: {"A": 0.6, "B": 0.4})
    uav_start_box: tuple[Range, Range, Range] = ((0.0, 1500.0), (0.0, 1500.0), (70.0, 80.0))
    target_z: Range = (75.0, 140.0)
    value_distribution: Literal["uniform", "bimodal", "skewed"] = "uniform"
    value_range: Range = (50.0, 100.0)
    observation_time_range: Range = (10.0, 20.0)
    endurance_scale: float = Field(1.0, gt=0)
    event_rates: EventRates = Field(default_factory=EventRates)
    event_window: float = Field(800.0, gt=0)
    max_new_targets: int | None = None
    max_total_targets: int = Field(24, gt=0)
    move_distance: Range = (300.0, 1500.0)
    move_dz: float = Field(30.0, ge=0)
    value_factor: Range = (0.4, 0.7)
    weather_factor: Range = (0.5, 1.0)

    @model_validator(mode="after")
    def _check(self) -> GeneratorConfig:
        for name in ("n_uavs", "n_targets"):
            lo, hi = getattr(self, name)
            if not 1 <= lo <= hi:
                raise ValueError(f"{name} must satisfy 1 <= low <= high, got {(lo, hi)}")
        if self.n_targets[1] > self.max_total_targets:
            raise ValueError("n_targets upper bound exceeds max_total_targets")
        if set(self.uav_type_probs) - set(self.uav_types):
            raise ValueError("uav_type_probs names an unknown UAV type")
        if abs(sum(self.uav_type_probs.values()) - 1.0) > 1e-9:
            raise ValueError("uav_type_probs must sum to 1")
        lo, hi = self.weather_factor
        if not 0.0 < lo <= hi <= 1.0:
            raise ValueError("weather_factor must lie in (0, 1]")
        return self


class ScenarioGenerator:
    """Draw scenarios from a :class:`GeneratorConfig`."""

    def __init__(self, config: GeneratorConfig | None = None, name: str = "gen") -> None:
        self.config = config or GeneratorConfig()
        self.name = name

    def generate(self, seed: int) -> Scenario:
        """Return the scenario for ``seed`` (deterministic)."""
        cfg = self.config
        streams = make_streams(seed)
        layout, ev_rng = streams["layout"], streams["events"]
        box = np.asarray(cfg.airspace)

        n_uavs = int(layout.integers(cfg.n_uavs[0], cfg.n_uavs[1] + 1))
        n_targets = int(layout.integers(cfg.n_targets[0], cfg.n_targets[1] + 1))
        type_names = sorted(cfg.uav_type_probs)
        probs = np.array([cfg.uav_type_probs[k] for k in type_names])
        uavs = []
        for k in range(n_uavs):
            tname = type_names[int(layout.choice(len(type_names), p=probs))]
            tm = cfg.uav_types[tname]
            pos = tuple(float(layout.uniform(lo, hi)) for lo, hi in cfg.uav_start_box)
            uavs.append(
                UAVSpec(
                    id=f"U{k + 1}",
                    uav_type=tname,
                    speed=tm.speed,
                    endurance=tm.endurance * cfg.endurance_scale,
                    sensors=tuple(bool(tm.sensors[s]) for s in SENSORS),
                    vulnerability=tm.vulnerability,
                    start_position=(pos[0], pos[1], pos[2]),
                )
            )
        targets = [self._target(layout, f"T{k + 1}", box, initial=True) for k in range(n_targets)]
        events = self._events(ev_rng, box, max_new=cfg.max_total_targets - n_targets)
        return Scenario(
            name=f"{self.name}_{seed}",
            airspace=cfg.airspace,
            lambda_obse=cfg.lambda_obse,
            lambda_risk=cfg.lambda_risk,
            uavs=tuple(uavs),
            targets=tuple(targets),
            events=tuple(events),
        )

    def _value(self, rng: np.random.Generator) -> float:
        lo, hi = self.config.value_range
        dist = self.config.value_distribution
        if dist == "uniform":
            return float(rng.uniform(lo, hi))
        if dist == "bimodal":
            span = 0.2 * (hi - lo)
            return float(
                rng.uniform(lo, lo + span) if rng.random() < 0.5 else rng.uniform(hi - span, hi)
            )
        # skewed: most targets near the low end, a few very valuable ones
        return float(lo + (hi - lo) * rng.beta(0.7, 3.0))

    def _target(
        self, rng: np.random.Generator, tid: str, box: np.ndarray, initial: bool
    ) -> TargetSpec:
        cfg = self.config
        x = float(rng.uniform(0.0, box[0]))
        y = float(rng.uniform(0.0, box[1]))
        z = float(rng.uniform(*cfg.target_z))
        return TargetSpec(
            id=tid,
            value=self._value(rng),
            observation_time=float(rng.uniform(*cfg.observation_time_range)),
            position=(x, y, min(z, float(box[2]))),
            initial=initial,
        )

    def _times(self, rng: np.random.Generator, rate: float) -> np.ndarray:
        window = self.config.event_window
        n = int(rng.poisson(rate * window / 1000.0))
        # Strictly positive times: events at t = 0 would precede the first decision.
        return np.sort(rng.uniform(1e-3, window, size=n))

    def _events(self, rng: np.random.Generator, box: np.ndarray, max_new: int) -> list[Event]:
        cfg = self.config
        rates = cfg.event_rates
        events: list[Event] = []
        # Fixed class order keeps the draw sequence (and thus every schedule) reproducible.
        new_times = self._times(rng, rates.new_target)
        cap = max_new if cfg.max_new_targets is None else min(max_new, cfg.max_new_targets)
        for k, t in enumerate(new_times[: max(cap, 0)]):
            events.append(
                NewTarget(time=float(t), target=self._target(rng, f"N{k + 1}", box, initial=False))
            )
        for t in self._times(rng, rates.uav_crash):
            events.append(UAVCrash(time=float(t), uav_selector=float(rng.random())))
        for t in self._times(rng, rates.sensor_damage):
            events.append(
                SensorDamage(
                    time=float(t),
                    uav_selector=float(rng.random()),
                    sensor_selector=float(rng.random()),
                )
            )
        for t in self._times(rng, rates.weather):
            omega = tuple(float(rng.uniform(*cfg.weather_factor)) for _ in SENSORS)
            events.append(WeatherChange(time=float(t), omega=omega, mode="multiplicative"))
        for t in self._times(rng, rates.target_move):
            r = float(rng.uniform(*cfg.move_distance))
            theta = float(rng.uniform(0.0, 2 * np.pi))
            dz = float(rng.uniform(-cfg.move_dz, cfg.move_dz))
            events.append(
                TargetMove(
                    time=float(t),
                    target_selector=float(rng.random()),
                    displacement=(r * np.cos(theta), r * np.sin(theta), dz),
                )
            )
        for t in self._times(rng, rates.value_change):
            events.append(
                ValueChange(
                    time=float(t),
                    target_selector=float(rng.random()),
                    factor=float(rng.uniform(*cfg.value_factor)),
                )
            )
        return events
