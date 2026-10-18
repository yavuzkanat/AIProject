"""Aggregate evaluation CSVs into tables and hypothesis tests (phase1 §B.1, §I.4).

Aggregation rules:
* PPO: each training seed is evaluated deterministically on all scenarios. Reported spread is
  the std across training seeds of the per-seed mean (n = number of seeds).
* Stochastic baselines (random, RH-RK) are averaged over repeats per scenario first.
* Paired tests compare per-scenario means (PPO averaged over seeds) on identical scenarios.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from uavta.evaluation.stats import bootstrap_ci, holm_correction, iqm, paired_wilcoxon


def per_scenario(df: pd.DataFrame, condition: str, method: str, metric: str) -> pd.Series:
    sub = df[(df["condition"] == condition) & (df["method"] == method)]
    return sub.groupby("scenario_ref")[metric].mean().sort_index()


def method_summary(df: pd.DataFrame, condition: str, metrics: Sequence[str]) -> pd.DataFrame:
    rows = []
    sub = df[df["condition"] == condition]
    for method, grp in sub.groupby("method", sort=False):
        row: dict[str, object] = {"method": method}
        seeds = sorted(s for s in grp["train_seed"].unique() if s >= 0)
        row["n_seeds"] = len(seeds) if seeds else 0
        row["n_scenarios"] = grp["scenario_ref"].nunique()
        for metric in metrics:
            scen = grp.groupby("scenario_ref")[metric].mean()
            if seeds:
                seed_means = grp.groupby("train_seed")[metric].mean()
                spread = float(seed_means.std(ddof=1)) if len(seed_means) > 1 else 0.0
            else:
                spread = float(scen.std(ddof=1)) if len(scen) > 1 else 0.0
            lo, hi = bootstrap_ci(scen.to_numpy())
            row[metric] = float(scen.mean())
            row[f"{metric}_spread"] = spread
            row[f"{metric}_ci"] = (lo, hi)
            row[f"{metric}_iqm"] = iqm(scen.to_numpy())
        rows.append(row)
    return pd.DataFrame(rows)


def compare(df: pd.DataFrame, condition: str, a: str, b: str, metric: str) -> dict[str, Any]:
    sa, sb = per_scenario(df, condition, a, metric), per_scenario(df, condition, b, metric)
    common = sa.index.intersection(sb.index)
    test = paired_wilcoxon(sa[common].to_numpy(), sb[common].to_numpy())
    diff = sa[common].to_numpy() - sb[common].to_numpy()
    lo, hi = bootstrap_ci(diff)
    return {
        "condition": condition,
        "a": a,
        "b": b,
        "metric": metric,
        "n": test.n,
        "mean_a": float(sa[common].mean()),
        "mean_b": float(sb[common].mean()),
        "mean_diff": test.mean_difference,
        "median_diff": test.median_difference,
        "ci_low": lo,
        "ci_high": hi,
        "p_value": test.p_value,
        "rank_biserial": test.rank_biserial,
    }


def compare_family(df: pd.DataFrame, pairs: Sequence[tuple[str, str, str, str]]) -> pd.DataFrame:
    """Run (condition, a, b, metric) comparisons and add Holm-adjusted p-values."""
    rows = [compare(df, *p) for p in pairs]
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = holm_correction(out["p_value"].tolist())
    return out


def fmt_summary(summary: pd.DataFrame, metrics: Sequence[str], digits: int = 3) -> str:
    """Markdown table: mean ± spread [95 % CI]."""
    header = "| method | seeds | scenarios | " + " | ".join(metrics) + " |"
    sep = "|---|---|---|" + "---|" * len(metrics)
    lines = [header, sep]
    for _, r in summary.iterrows():
        cells = []
        for m in metrics:
            lo, hi = r[f"{m}_ci"]
            mean, spread = r[m], r[f"{m}_spread"]
            cells.append(
                f"{mean:.{digits}f} ± {spread:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"
            )
        seeds = r["n_seeds"] if r["n_seeds"] else "–"
        lines.append(
            f"| {r['method']} | {seeds} | {r['n_scenarios']} | " + " | ".join(cells) + " |"
        )
    return "\n".join(lines)


def fmt_comparisons(comp: pd.DataFrame, digits: int = 4) -> str:
    header = (
        "| condition | A | B | metric | n | mean A | mean B | mean Δ [95 % CI] | r_rb | p "
        "| p (Holm) |"
    )
    lines = [header, "|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in comp.iterrows():
        lines.append(
            f"| {r['condition']} | {r['a']} | {r['b']} | {r['metric']} | {r['n']} | "
            f"{r['mean_a']:.{digits}f} | {r['mean_b']:.{digits}f} | "
            f"{r['mean_diff']:+.{digits}f} "
            f"[{r['ci_low']:+.{digits}f}, {r['ci_high']:+.{digits}f}] | "
            f"{r['rank_biserial']:+.2f} | {r['p_value']:.2g} | {r.get('p_holm', np.nan):.2g} |"
        )
    return "\n".join(lines)
