"""Tables for the post-study improvement experiments (kept separate from the pre-registered
hypothesis family in scripts/analyze.py, so adding them does not change those corrections).

Example:
    python scripts/analyze_improvements.py
Writes results/tables/improvements.md, comparisons_improvements.csv and summary_improvements.csv.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from uavta.evaluation.report import compare_family, fmt_comparisons, fmt_summary, method_summary

FILES = [
    "main_baselines.csv",
    "main_ppo.csv",
    "main_entity.csv",
    "improvements_lookahead.csv",
    "improvements_attention.csv",
]
NEW = ["ppo_entity_lookahead", "ppo_attention"]
METRICS = ["weighted_score", "value_fraction", "makespan", "completion_rate"]


def main() -> None:
    out = Path("results/tables")
    out.mkdir(parents=True, exist_ok=True)
    df = pd.concat(
        [
            pd.read_csv(Path("results/eval") / f)
            for f in FILES
            if (Path("results/eval") / f).exists()
        ],
        ignore_index=True,
    )
    md = ["# Improvement experiments", ""]
    summary = method_summary(df, "test_id", METRICS + ["risk_exposure"])
    summary.to_csv(out / "summary_improvements.csv", index=False)
    md += ["## Held-out test set (500 scenarios)", "", fmt_summary(summary, METRICS), ""]
    pairs = [
        ("test_id", a, b, "weighted_score")
        for a in NEW
        for b in ("ppo_entity", "rh_rk", "greedy_nearest")
        if a in set(df["method"])
    ]
    comp = compare_family(df, pairs)
    comp.to_csv(out / "comparisons_improvements.csv", index=False)
    md += ["## Paired comparisons (Holm within this family)", "", fmt_comparisons(comp), ""]
    for cond in ("paper_S1", "paper_S2", "paper_S3"):
        s = method_summary(df, cond, ["weighted_score"])
        md += [f"## {cond}", "", fmt_summary(s, ["weighted_score"]), ""]
    lat = out / "latency.json"
    if lat.exists():
        data = json.loads(lat.read_text())
        md += ["## Decision latency (ms, median)", ""]
        md += [f"* {k}: {v['median_of_episode_median_ms']:.3f}" for k, v in data.items()]
    (out / "improvements.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print((out / "improvements.md").read_text())


if __name__ == "__main__":
    main()
