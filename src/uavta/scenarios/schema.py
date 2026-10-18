"""Pydantic models describing the scenario YAML format (see configs/scenarios/paper.yaml).

These models only validate file structure. Cross-references (e.g. an event naming an
unknown UAV) are checked when the file is converted into a :class:`Scenario`.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveFloat

Vec3 = Annotated[list[float], Field(min_length=3, max_length=3)]
Weights = Annotated[list[float], Field(min_length=2, max_length=2)]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class UAVTypeModel(_Strict):
    speed: PositiveFloat
    endurance: PositiveFloat
    sensors: dict[str, Literal[0, 1]]
    vulnerability: PositiveFloat


class UAVModel(_Strict):
    id: str
    type: str
    position: Vec3
    scenarios: list[str] | None = None


class TargetModel(_Strict):
    id: str
    value: PositiveFloat
    observation_time: PositiveFloat
    position: Vec3
    initial: bool = True
    scenarios: list[str] | None = None


class _EventModel(_Strict):
    paper_no: int | None = None
    time: PositiveFloat
    scenarios: list[str] | None = None


class UAVCrashModel(_EventModel):
    type: Literal["uav_crash"]
    uav: str


class SensorDamageModel(_EventModel):
    type: Literal["sensor_damage"]
    uav: str
    sensor: str


class WeatherModel(_EventModel):
    type: Literal["weather"]
    omega: dict[str, float]
    mode: Literal["absolute", "multiplicative"] = "absolute"


class TargetMoveModel(_EventModel):
    type: Literal["target_move"]
    target: str
    to: Vec3
    from_: Vec3 | None = Field(default=None, alias="from")


class ValueChangeModel(_EventModel):
    type: Literal["value_change"]
    target: str
    to: PositiveFloat
    from_: PositiveFloat | None = Field(default=None, alias="from")


class NewTargetModel(_EventModel):
    type: Literal["new_target"]
    target: str
    position: Vec3 | None = None


class PreferenceShiftModel(_EventModel):
    type: Literal["preference_shift"]
    to: Weights
    from_: Weights | None = Field(default=None, alias="from")


EventModel = Annotated[
    UAVCrashModel
    | SensorDamageModel
    | WeatherModel
    | TargetMoveModel
    | ValueChangeModel
    | NewTargetModel
    | PreferenceShiftModel,
    Field(discriminator="type"),
]


class ScenarioFile(_Strict):
    name: str
    variants: list[str] | None = None
    airspace: Vec3
    lambda_obse: PositiveFloat
    lambda_risk: PositiveFloat
    initial_weather: dict[str, float] = Field(
        default_factory=lambda: {"VL": 1.0, "IR": 1.0, "RD": 1.0}
    )
    initial_preference: Weights = Field(default_factory=lambda: [0.5, 0.5])
    uav_types: dict[str, UAVTypeModel]
    uavs: list[UAVModel]
    targets: list[TargetModel]
    events: list[EventModel] = Field(default_factory=list)
