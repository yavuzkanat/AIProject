"""Matplotlib figures. Pure functions of evaluation records / data frames; no simulator logic."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from uavta.evaluation.runner import EpisodeRecord  # noqa: E402
from uavta.evaluation.stats import bootstrap_ci  # noqa: E402

KIND_COLORS = {
    "transit": "#4C78A8",
    "observe": "#F58518",
    "idle": "#D9D9D9",
    "failed": "#E45756",
    "depleted": "#9D755D",
}
EVENT_MARKERS = {
    "uav_crash": "X",
    "sensor_damage": "v",
    "weather": "*",
    "target_move": ">",
    "value_change": "d",
    "new_target": "P",
    "preference_shift": "s",
}
METHOD_ORDER = ["random", "greedy_nearest", "greedy_value_rate", "rh_rk", "ppo", "ppo_entity"]


def _save(fig: plt.Figure, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _methods(df: pd.DataFrame) -> list[str]:
    present = list(df["method"].unique())
    return [m for m in METHOD_ORDER if m in present] + [m for m in present if m not in METHOD_ORDER]


def gantt(record: EpisodeRecord, uav_ids: Sequence[str], title: str, path: str | Path) -> Path:
    """Per-UAV timeline (transit / observe / idle / failed) with event markers, cf. paper Fig. 5."""
    fig, ax = plt.subplots(figsize=(10, 0.6 * len(uav_ids) + 1.5))
    slot_name = {v["slot"]: k for k, v in record.target_positions.items()}
    for act in record.activities:
        if act.t_end is None or act.t_end <= act.t_start:
            continue
        ax.barh(
            act.uav,
            act.t_end - act.t_start,
            left=act.t_start,
            color=KIND_COLORS[act.kind],
            edgecolor="white",
            height=0.6,
        )
        if act.kind == "observe" and act.slot is not None:
            ax.text(act.t_start, act.uav + 0.38, slot_name.get(act.slot, ""), fontsize=6)
    for ev in record.event_log:
        if ev["effect"] != "applied":
            continue
        ax.axvline(ev["time"], color="black", lw=0.6, ls=":")
    ax.set_yticks(range(len(uav_ids)), uav_ids)
    ax.invert_yaxis()
    ax.set_xlabel("time (s)")
    ax.set_title(title)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in KIND_COLORS.values()]
    ax.legend(handles, KIND_COLORS.keys(), ncol=5, fontsize=7, loc="upper right")
    return _save(fig, path)


def route_map(
    record: EpisodeRecord, airspace: Sequence[float], title: str, path: str | Path
) -> Path:
    """Top-down view of UAV routes, targets coloured by value, completion state and moves."""
    fig, ax = plt.subplots(figsize=(6.5, 6))
    cmap = plt.get_cmap("tab10")
    for act in record.activities:
        if act.kind != "transit" or act.p_end is None:
            continue
        ax.plot(
            [act.p_start[0], act.p_end[0]],
            [act.p_start[1], act.p_end[1]],
            color=cmap(act.uav % 10),
            lw=1.2,
            alpha=0.85,
        )
    for act in record.activities:
        if act.kind == "failed":
            ax.scatter(
                act.p_start[0],
                act.p_start[1],
                marker="X",
                s=90,
                color=cmap(act.uav % 10),
                edgecolor="k",
                zorder=5,
            )
    vals = [v["value"] for v in record.target_positions.values()]
    vmin, vmax = min(vals, default=0), max(vals, default=1)
    for tid, info in record.target_positions.items():
        p, p0 = info["position"], info["initial_position"]
        if p != p0:
            ax.annotate(
                "", xy=p[:2], xytext=p0[:2], arrowprops={"arrowstyle": "->", "color": "grey"}
            )
        ax.scatter(
            p[0],
            p[1],
            c=[info["value"]],
            cmap="coolwarm",
            vmin=vmin,
            vmax=vmax,
            s=60,
            marker="o" if info["completed"] else "s",
            edgecolor="k",
            zorder=4,
        )
        ax.text(p[0] + 80, p[1] + 80, tid, fontsize=7)
    starts = {a.uav: a.p_start for a in record.activities if a.t_start == 0.0}
    for u, p in starts.items():
        ax.scatter(p[0], p[1], marker="^", s=70, color=cmap(u % 10), edgecolor="k", zorder=5)
    ax.set_xlim(0, airspace[0])
    ax.set_ylim(0, airspace[1])
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(
        title + "\n(○ completed, □ unfinished, ▲ start, ✕ crash, → target move)", fontsize=9
    )
    return _save(fig, path)


def value_timeline(records: dict[str, EpisodeRecord], title: str, path: str | Path) -> Path:
    """Cumulative expected collected value over time per method, with event times."""
    fig, ax = plt.subplots(figsize=(8, 4))
    first = next(iter(records.values()))
    for name, rec in records.items():
        done = sorted(
            (v["completion_time"], v["collected_value"])
            for v in rec.target_positions.values()
            if v["completed"]
        )
        t = [0.0] + [d[0] for d in done]
        c = np.cumsum([0.0] + [d[1] for d in done])
        ax.step(t, c, where="post", label=name)
    for ev in first.event_log:
        if ev["effect"] == "applied":
            ax.axvline(ev["time"], color="grey", lw=0.5, ls=":")
            ax.text(
                ev["time"], ax.get_ylim()[1] * 0.02, EVENT_MARKERS.get(ev["type"], "?"), fontsize=8
            )
    ax.set_xlabel("time (s)")
    ax.set_ylabel("cumulative expected value")
    ax.set_title(title)
    ax.legend(fontsize=8)
    return _save(fig, path)


def training_curves(
    curves: dict[str, list[pd.DataFrame]],
    metric: str,
    baselines: dict[str, float],
    path: str | Path,
) -> Path:
    """Validation metric vs. environment steps: mean and min–max band over training seeds."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, frames in curves.items():
        grid = np.unique(np.concatenate([f["timesteps"].to_numpy() for f in frames]))
        ys = np.stack([np.interp(grid, f["timesteps"], f[metric]) for f in frames])
        line = ax.plot(grid, ys.mean(0), label=f"{name} (n={len(frames)})")[0]
        ax.fill_between(grid, ys.min(0), ys.max(0), color=line.get_color(), alpha=0.2)
    for name, value in baselines.items():
        ax.axhline(value, ls="--", lw=1, label=name, color="grey" if "random" in name else None)
    ax.set_xlabel("environment steps")
    ax.set_ylabel(f"validation {metric}")
    ax.legend(fontsize=7)
    return _save(fig, path)


def method_boxplots(df: pd.DataFrame, metrics: Sequence[str], title: str, path: str | Path) -> Path:
    methods = _methods(df)
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 4))
    for ax, metric in zip(np.atleast_1d(axes), metrics, strict=True):
        data = [df.loc[df["method"] == m, metric].dropna().to_numpy() for m in methods]
        ax.boxplot(data, showfliers=False)
        ax.set_xticks(range(1, len(methods) + 1), methods, rotation=40, ha="right", fontsize=8)
        ax.set_title(metric)
    fig.suptitle(title)
    return _save(fig, path)


def delta_bars(deltas: pd.DataFrame, title: str, path: str | Path) -> Path:
    """Bar chart of mean paired difference with 95 % CI per condition (columns: condition, mean,
    ci_low, ci_high)."""
    fig, ax = plt.subplots(figsize=(max(6, 0.55 * len(deltas)), 4))
    x = np.arange(len(deltas))
    err = np.vstack([deltas["mean"] - deltas["ci_low"], deltas["ci_high"] - deltas["mean"]])
    colors = [
        "#54A24B" if lo > 0 else "#E45756" if hi < 0 else "#BAB0AC"
        for lo, hi in zip(deltas["ci_low"], deltas["ci_high"], strict=True)
    ]
    ax.bar(x, deltas["mean"], yerr=err, color=colors, capsize=3)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x, deltas["condition"], rotation=50, ha="right", fontsize=8)
    ax.set_ylabel("ΔJ (mean, 95 % CI)")
    ax.set_title(title)
    return _save(fig, path)


def tradeoff_scatter(points: pd.DataFrame, path: str | Path) -> Path:
    """Realized (value fraction, makespan) per method/weight (columns: label, value_fraction,
    makespan)."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for label, grp in points.groupby("label"):
        ax.scatter(grp["makespan"], grp["value_fraction"], label=label, s=40)
    ax.set_xlabel("realized makespan (s)  ↓")
    ax.set_ylabel("collected value fraction  ↑")
    ax.legend(fontsize=7)
    return _save(fig, path)


def mean_ci(values: Any) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    lo, hi = bootstrap_ci(arr)
    return float(np.nanmean(arr)), lo, hi
