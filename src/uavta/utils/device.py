"""Neural-network device selection for Apple Silicon and CPU-only machines.

The simulator always runs on the CPU in NumPy; this module only concerns PyTorch.
Stable-Baselines3's ``device="auto"`` never selects MPS, and for small MLP policies MPS was
measured to be about 15× slower than CPU on the development M3 (docs/phase1_design.md §H).
CPU is therefore the default.
"""

from __future__ import annotations

import logging
from typing import Literal

DeviceRequest = Literal["cpu", "mps", "auto"]

logger = logging.getLogger(__name__)


def mps_available() -> bool:
    """True if this PyTorch build can use Apple's Metal Performance Shaders backend."""
    import torch

    return bool(torch.backends.mps.is_available())


def resolve_device(request: DeviceRequest = "cpu", accelerator_worthy: bool = False) -> str:
    """Translate a configured device request into a concrete PyTorch device string.

    Args:
        request: ``"cpu"``, ``"mps"`` or ``"auto"``.
        accelerator_worthy: For ``"auto"``, whether the network is large enough to benefit
            from MPS. Small MLPs are not.

    Returns:
        ``"cpu"`` or ``"mps"``. Falls back to ``"cpu"`` with a warning if MPS was requested
        but is unavailable. CUDA is never chosen implicitly.
    """
    if request not in ("cpu", "mps", "auto"):
        raise ValueError(f"unknown device request {request!r}")
    if request == "cpu":
        resolved = "cpu"
    elif request == "mps":
        if mps_available():
            resolved = "mps"
        else:
            logger.warning("device 'mps' requested but unavailable; falling back to cpu")
            resolved = "cpu"
    else:
        resolved = "mps" if accelerator_worthy and mps_available() else "cpu"
    logger.info("device request %r resolved to %r", request, resolved)
    return resolved
