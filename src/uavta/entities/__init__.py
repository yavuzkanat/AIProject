"""Static specifications and mutable runtime state of UAVs and targets."""

from uavta.entities.sensors import NUM_SENSORS, SENSORS
from uavta.entities.target import TargetSpec, TargetState
from uavta.entities.uav import UAVSpec, UAVState, UAVStatus

__all__ = [
    "NUM_SENSORS",
    "SENSORS",
    "TargetSpec",
    "TargetState",
    "UAVSpec",
    "UAVState",
    "UAVStatus",
]
