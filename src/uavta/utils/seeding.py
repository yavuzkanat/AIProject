"""Seeded random-number streams.

Each source of randomness gets its own generator derived from one root seed, so that, for
example, changing the event model does not change the scenario layout drawn for a seed.
"""

from __future__ import annotations

import random

import numpy as np

STREAM_NAMES: tuple[str, ...] = ("layout", "events", "policy")


def make_streams(seed: int | np.random.SeedSequence) -> dict[str, np.random.Generator]:
    """Return independent generators for every name in ``STREAM_NAMES``.

    The mapping from name to child sequence is fixed by tuple order; append new names at
    the end to keep existing streams unchanged.
    """
    root = seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(seed)
    children = root.spawn(len(STREAM_NAMES))
    return {
        name: np.random.default_rng(child)
        for name, child in zip(STREAM_NAMES, children, strict=True)
    }


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy's legacy global RNG and (if installed) PyTorch.

    Only training entry points should call this. Simulator code never uses global RNGs.
    """
    random.seed(seed)
    np.random.seed(seed % 2**32)  # noqa: NPY002 - seeds third-party code using the legacy RNG
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
