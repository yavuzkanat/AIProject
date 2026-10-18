"""Generate all report figures from saved runs and evaluation CSVs.

Example:
    python scripts/make_figures.py
Writes results/figures/*.png.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import pandas as pd

from uavta.agents.ppo_factory import PPOPolicy
from uavta.baselines.heuristics import GreedyValueRatePolicy
from uavta.baselines.rh_rk import RollingHorizonRandomKeyPolicy
from uavta.environment.env import MultiUAVDynamicEnv
from uavta.evaluation.report import compare
from uavta.evaluation.runner import ScenarioRef, run_episode
from uavta.scenarios.loader import load_paper_scenario
from uavta.visualization import plots


def curves(pattern: str) -> list[pd.DataFrame]:
    return [pd.read_csv(p) for p in sorted(glob.glob(f"{pattern}/validation.csv"))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/figures")
    parser.add_argument("--main", nargs="+", default=["results/eval/main.csv"])
    parser.add_argument("--gen", nargs="*", default=["results/eval/generalization.csv"])
    parser.add_argument("--runs", default="results/runs")
    args = parser.parse_args()
    out = Path(args.out)
    mains = [p for p in args.main if Path(p).exists()]
    main_df = pd.concat([pd.read_csv(p) for p in mains], ignore_index=True) if mains else None

    # Training curves with baseline reference lines (validation split).
    groups = {
        "ppo (MLP)": curves(f"{args.runs}/ppo_default_seed*"),
        "ppo_entity": curves(f"{args.runs}/ppo_entity_seed*"),
    }
    groups = {k: v for k, v in groups.items() if v}
    abl = {
        name: curves(f"{args.runs}/{name}_seed*")
        for name in ("abl_value_only", "abl_w25", "abl_w75", "abl_energy", "abl_unmasked")
    }
    if groups:
        plots.training_curves(groups, "weighted_score", {}, out / "training_curves_J.png")
        plots.training_curves(groups, "value_fraction", {}, out / "training_curves_value.png")
        plots.training_curves(groups, "makespan", {}, out / "training_curves_makespan.png")
    if any(abl.values()):
        merged = {**{k: v for k, v in groups.items()}, **{k: v for k, v in abl.items() if v}}
        plots.training_curves(merged, "weighted_score", {}, out / "training_curves_ablations.png")

    if main_df is not None:
        test = main_df[main_df["condition"] == "test_id"]
        plots.method_boxplots(
            test,
            ["weighted_score", "value_fraction", "makespan"],
            "Held-out test scenarios (n=500)",
            out / "test_id_boxplots.png",
        )
        plots.method_boxplots(
            test,
            ["energy_used_total", "risk_exposure", "latency_median_ms"],
            "Held-out test scenarios (n=500)",
            out / "test_id_boxplots_extra.png",
        )

    gens = [p for p in args.gen if Path(p).exists()]
    if gens:
        gen = pd.concat([pd.read_csv(p) for p in gens], ignore_index=True)
        for p in ("ppo", "ppo_entity"):
            if p not in set(gen["method"]):
                continue
            for b in ("greedy_value_rate", "rh_rk"):
                rows = [compare(gen, c, p, b, "weighted_score") for c in gen["condition"].unique()]
                d = pd.DataFrame(rows).rename(columns={"mean_diff": "mean"})
                plots.delta_bars(
                    d,
                    f"{p} − {b}: ΔJ per condition (100 unseen scenarios each)",
                    out / f"generalization_delta_{p}_vs_{b}.png",
                )

    # Paper scenarios: Gantt, route maps and value timelines for representative policies.
    env = MultiUAVDynamicEnv()
    policies = {
        "greedy_value_rate": GreedyValueRatePolicy(),
        "rh_rk": RollingHorizonRandomKeyPolicy(),
    }
    for name, pattern in (("ppo", "ppo_default_seed0"), ("ppo_entity", "ppo_entity_seed0")):
        path = Path(args.runs) / pattern / "best_model.zip"
        if path.exists():
            policies[name] = PPOPolicy.load(path, name=name)
    for variant in ("S1", "S2", "S3"):
        sc = load_paper_scenario(variant)
        ref = ScenarioRef(f"paper_{variant}", scenario=sc)
        recs = {
            n: run_episode(env, p, ref, policy_seed=0, keep_logs=True) for n, p in policies.items()
        }
        ids = [u.id for u in sc.uavs]
        for n, rec in recs.items():
            m = rec.metrics
            title = (
                f"{n} on {variant}: J={m['weighted_score']:.3f}, "
                f"value frac={m['value_fraction']:.3f}, "
                f"makespan={m['makespan']:.0f} s"
            )
            plots.gantt(rec, ids, title, out / f"gantt_{variant}_{n}.png")
            plots.route_map(rec, sc.airspace, title, out / f"map_{variant}_{n}.png")
        plots.value_timeline(
            recs, f"Cumulative expected value on {variant}", out / f"value_timeline_{variant}.png"
        )

    # Reward-weight trade-off (realized outcomes on test_id) if the weight sweep was evaluated.
    tradeoff = Path("results/eval/ablations.csv")
    if tradeoff.exists():
        abl_df = pd.read_csv(tradeoff)
        pts = (
            abl_df.groupby(["method", "train_seed"])[["value_fraction", "makespan"]]
            .mean()
            .reset_index()
        )
        pts["label"] = pts["method"]
        plots.tradeoff_scatter(pts, out / "tradeoff_value_vs_makespan.png")
    print(f"figures written to {out}")


if __name__ == "__main__":
    main()
