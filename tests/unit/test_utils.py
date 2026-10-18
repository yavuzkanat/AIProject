"""Seeding, configuration, device selection and provenance helpers."""

import json

import numpy as np
import pytest

from uavta.utils import device
from uavta.utils.config import deep_merge, load_yaml, save_yaml
from uavta.utils.provenance import write_versions
from uavta.utils.seeding import STREAM_NAMES, make_streams


def test_streams_reproducible():
    a, b = make_streams(123), make_streams(123)
    for name in STREAM_NAMES:
        np.testing.assert_array_equal(a[name].random(5), b[name].random(5))


def test_streams_independent():
    s = make_streams(123)
    assert not np.array_equal(s["layout"].random(5), s["events"].random(5))


def test_different_seeds_differ():
    assert make_streams(1)["events"].random() != make_streams(2)["events"].random()


def test_deep_merge():
    base = {"a": 1, "b": {"c": 2, "d": 3}}
    out = deep_merge(base, {"b": {"d": 4}, "e": 5})
    assert out == {"a": 1, "b": {"c": 2, "d": 4}, "e": 5}
    assert base["b"]["d"] == 3


def test_yaml_roundtrip(tmp_path):
    save_yaml({"x": [1, 2], "y": {"z": "w"}}, tmp_path / "c.yaml")
    assert load_yaml(tmp_path / "c.yaml") == {"x": [1, 2], "y": {"z": "w"}}


def test_yaml_requires_mapping(tmp_path):
    (tmp_path / "bad.yaml").write_text("- 1\n- 2\n")
    with pytest.raises(ValueError):
        load_yaml(tmp_path / "bad.yaml")


def test_device_cpu_default():
    assert device.resolve_device() == "cpu"


def test_device_auto_small_network_stays_on_cpu():
    assert device.resolve_device("auto", accelerator_worthy=False) == "cpu"


def test_device_mps_fallback(monkeypatch):
    monkeypatch.setattr(device, "mps_available", lambda: False)
    assert device.resolve_device("mps") == "cpu"
    assert device.resolve_device("auto", accelerator_worthy=True) == "cpu"


def test_device_mps_selected_when_available(monkeypatch):
    monkeypatch.setattr(device, "mps_available", lambda: True)
    assert device.resolve_device("mps") == "mps"


def test_device_rejects_cuda():
    with pytest.raises(ValueError):
        device.resolve_device("cuda")  # type: ignore[arg-type]


def test_write_versions(tmp_path):
    path = write_versions(tmp_path, {"device": "cpu"})
    data = json.loads(path.read_text())
    assert data["device"] == "cpu"
    assert data["packages"]["numpy"] is not None
