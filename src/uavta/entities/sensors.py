"""Sensor modalities of the paper (§3.1): visible light, infrared and radar."""

SENSORS: tuple[str, ...] = ("VL", "IR", "RD")
NUM_SENSORS: int = len(SENSORS)


def sensor_index(name: str) -> int:
    """Return the array index of a sensor name, raising ``ValueError`` for unknown names."""
    try:
        return SENSORS.index(name)
    except ValueError:
        raise ValueError(f"unknown sensor {name!r}; expected one of {SENSORS}") from None
