"""Turn evaluation CSVs into report tables and hypothesis tests.

Example:
    python scripts/analyze.py --main results/eval/main.csv --gen results/eval/generalization.csv
Writes results/tables/*.md and results/tables/*.csv.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from uavta.evaluation.report import compare_family, fmt_comparisons, fmt_summary, method_summary

METRICS = ["weighted_score", "value_fraction", "makespan", "completion_rate"]
EXTRA = [
    "energy_used_total",
    "risk_exposure",
    "mean_response_delay",
    "latency_median_ms",
    "failed_mission",
    "n_invalid_actions",
    "critical_on_time_rate",
]
H3_MARGIN = 0.05  # J11, fixed before the experiments


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main", nargs="+", default=["results/eval/main.csv"])
    parser.add_argument("--gen", nargs="*", default=["results/eval/generalization.csv"])
    parser.add_argument("--out", default="results/tables")
    parser.add_argument("--ppo", nargs="*", default=["ppo", "ppo_entity"])
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    md: list[str] = []
    hyp: dict[str, object] = {}

    main_df = pd.concat([pd.read_csv(p) for p in args.main], ignore_index=True)
    extra = [m for m in EXTRA if m in main_df.columns]  # older CSVs lack newer metrics
    ppo_methods = [m for m in args.ppo if m in set(main_df["method"])]
    for cond in main_df["condition"].unique():
        s = method_summary(main_df, cond, METRICS + extra)
        s.to_csv(out / f"summary_{cond}.csv", index=False)
        md += [f"## {cond}", "", fmt_summary(s, METRICS), "", fmt_summary(s, extra), ""]

    # H1, H2: test_id, paired on 500 scenarios; one Holm family per PPO variant.
    pairs = []
    for p in ppo_methods:
        for b in ("random", "greedy_value_rate", "greedy_nearest", "rh_rk"):
            pairs.append(("test_id", p, b, "weighted_score"))
    comp = compare_family(main_df, pairs)
    comp.to_csv(out / "comparisons_test_id.csv", index=False)
    md += ["## Paired comparisons on test_id (weighted score J)", "", fmt_comparisons(comp), ""]
    for p in ppo_methods:
        c = comp[comp["a"] == p].set_index("b")
        hyp[f"H1_{p}_gt_random"] = bool(
            c.loc["random", "mean_diff"] > 0 and c.loc["random", "p_holm"] < 0.05
        )
        for g in ("greedy_value_rate", "greedy_nearest", "rh_rk"):
            # RH-RK is not part of H2 (a re-planner, not a greedy rule); same statistics.
            key = f"CMP_{p}_vs_{g}" if g == "rh_rk" else f"H2_{p}_ge_{g}"
            hyp[key] = {
                "mean_diff": float(c.loc[g, "mean_diff"]),
                "ci": [float(c.loc[g, "ci_low"]), float(c.loc[g, "ci_high"])],
                "p_holm": float(c.loc[g, "p_holm"]),
                "supported_superior": bool(c.loc[g, "mean_diff"] > 0 and c.loc[g, "p_holm"] < 0.05),
                "supported_noninferior": bool(c.loc[g, "ci_low"] > -0.005),
            }

    # H3: paper scenarios, non-inferiority to RH-RK with margin 5 % of |J_RHRK|.
    for p in ppo_methods:
        for cond in ("paper_S1", "paper_S2", "paper_S3"):
            sub = main_df[main_df["condition"] == cond]
            if sub.empty:
                continue
            seed_j = sub[sub["method"] == p].groupby("train_seed")["weighted_score"].mean()
            rh = sub[sub["method"] == "rh_rk"]["weighted_score"].mean()
            thr = rh - H3_MARGIN * abs(rh)
            hyp[f"H3_{p}_{cond}"] = {
                "ppo_mean": float(seed_j.mean()),
                "ppo_seed_values": [float(v) for v in seed_j],
                "rh_rk_mean": float(rh),
                "threshold": float(thr),
                "seeds_meeting_threshold": int((seed_j >= thr).sum()),
                "mean_meets_threshold": bool(seed_j.mean() >= thr),
            }

    # H4: latency. Prefer the controlled benchmark (idle machine) over latencies recorded during
    # the main evaluation, which may have run alongside training jobs.
    bench = out / "latency.json"
    if bench.exists():
        data = json.loads(bench.read_text())
        lat = pd.Series({k: v["median_of_episode_median_ms"] for k, v in data.items()})
        hyp["H4_source"] = str(bench)
    else:
        test = main_df[main_df["condition"] == "test_id"]
        lat = test.groupby("method")["latency_median_ms"].median()
        hyp["H4_source"] = "main evaluation (uncontrolled load)"
    for p in ppo_methods:
        if "rh_rk" in lat and p in lat:
            hyp[f"H4_{p}_latency_ratio_rhrk_over_ppo"] = float(lat["rh_rk"] / lat[p])
            hyp[f"H4_{p}_supported"] = bool(lat["rh_rk"] / lat[p] >= 10.0)

    gen_paths = [Path(p) for p in args.gen if Path(p).exists()]
    if gen_paths:
        gen = pd.concat([pd.read_csv(p) for p in gen_paths], ignore_index=True)
        pairs = []
        for cond in gen["condition"].unique():
            for p in ppo_methods:
                for b in ("greedy_value_rate", "greedy_nearest", "rh_rk"):
                    pairs.append((cond, p, b, "weighted_score"))
        gcomp = compare_family(gen, pairs)
        gcomp.to_csv(out / "comparisons_generalization.csv", index=False)
        md += ["## Generalization grid: paired ΔJ", "", fmt_comparisons(gcomp), ""]
        for cond in gen["condition"].unique():
            s = method_summary(gen, cond, METRICS)
            s.to_csv(out / f"summary_gen_{cond}.csv", index=False)
            md += [f"### {cond}", "", fmt_summary(s, METRICS), ""]

    (out / "summary.md").write_text("\n".join(md), encoding="utf-8")
    (out / "hypotheses.json").write_text(json.dumps(hyp, indent=2), encoding="utf-8")
    print(json.dumps(hyp, indent=2))


if __name__ == "__main__":
    main()
