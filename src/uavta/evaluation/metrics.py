"""Per-episode evaluation metrics derived from the simulator logs (phase1 §I.3)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from uavta.environment.simulator import Simulator

# Direction of improvement for reporting.
MAXIMIZE = (
    "weighted_score",
    "value_fraction",
    "collected_value",
    "completion_rate",
    "episode_return",
    "critical_on_time_rate",
)
MINIMIZE = (
    "makespan",
    "residual_value_f1",
    "energy_used_total",
    "risk_exposure",
    "mean_response_delay",
    "latency_median_ms",
    "failed_mission",
)


def response_delays(sim: Simulator) -> dict[str, float]:
    """Simulated seconds from each applied event to the next assignment made at or after it.

    Also reports, for crashes that released a task, the delay until the released target was
    re-assigned. NaN when no such assignment occurred.
    """
    times = np.array([a["time"] for a in sim.assignment_log])
    delays, reassign = [], []
    for ev in sim.event_log:
        if ev["effect"] != "applied":
            continue
        later = times[times >= ev["time"] - 1e-9] if times.size else times
        delays.append(float(later.min() - ev["time"]) if later.size else math.nan)
        released = ev.get("released_target")
        if released is not None:
            slot = sim._target_index[released]
            again = [
                a["time"]
                for a in sim.assignment_log
                if a["slot"] == slot and a["time"] >= ev["time"]
            ]
            reassign.append(min(again) - ev["time"] if again else math.nan)
    finite = [d for d in delays if not math.isnan(d)]
    finite_r = [d for d in reassign if not math.isnan(d)]
    return {
        "mean_response_delay": float(np.mean(finite)) if finite else math.nan,
        "mean_crash_reassign_delay": float(np.mean(finite_r)) if finite_r else math.nan,
        "n_events_unanswered": len(delays) - len(finite),
    }


def latency_stats(latencies_s: list[float]) -> dict[str, float]:
    arr = np.asarray(latencies_s) * 1000.0
    if arr.size == 0:
        return {
            "latency_median_ms": math.nan,
            "latency_p95_ms": math.nan,
            "latency_mean_ms": math.nan,
        }
    return {
        "latency_median_ms": float(np.median(arr)),
        "latency_p95_ms": float(np.percentile(arr, 95)),
        "latency_mean_ms": float(arr.mean()),
    }


def finalize_metrics(
    mission: dict[str, Any], sim: Simulator, latencies_s: list[float]
) -> dict[str, Any]:
    """Merge the mission summary with response and latency metrics; drop list-valued fields."""
    row = {k: v for k, v in mission.items() if not isinstance(v, list)}
    row["energy_used_max_uav"] = float(max(mission["energy_used_per_uav"], default=0.0))
    row["failed_mission"] = int(mission["unfinished_feasible_at_end"] > 0)
    row.update(response_delays(sim))
    row.update(latency_stats(latencies_s))
    return row
