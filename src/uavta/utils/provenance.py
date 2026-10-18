"""Record the software environment of a run for reproducibility."""

from __future__ import annotations

import json
import platform
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

TRACKED_PACKAGES: tuple[str, ...] = (
    "uavta",
    "numpy",
    "gymnasium",
    "pydantic",
    "PyYAML",
    "torch",
    "stable-baselines3",
    "sb3-contrib",
    "pandas",
    "scipy",
    "matplotlib",
)


def collect_versions() -> dict[str, Any]:
    """Return Python, platform, package versions and torch backend availability."""
    packages: dict[str, str | None] = {}
    for name in TRACKED_PACKAGES:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    info: dict[str, Any] = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": packages,
    }
    if packages["torch"] is not None:
        import torch

        info["torch_mps_available"] = bool(torch.backends.mps.is_available())
        info["torch_cuda_available"] = bool(torch.cuda.is_available())
    return info


def write_versions(run_dir: str | Path, extra: dict[str, Any] | None = None) -> Path:
    """Write ``versions.json`` into ``run_dir``, merging in ``extra`` (e.g. the resolved device)."""
    out = Path(run_dir) / "versions.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = collect_versions() | (extra or {})
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out
