"""Statistics for comparing methods across scenarios and training seeds (phase1 §I.4)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats


def bootstrap_ci(
    values: ArrayLike, *, confidence: float = 0.95, n_resamples: int = 10_000, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval of the mean."""
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = rng.choice(arr, size=(n_resamples, arr.size), replace=True).mean(axis=1)
    alpha = (1 - confidence) / 2
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))


def iqm(values: ArrayLike) -> float:
    """Interquartile mean: mean of the middle 50 % of values."""
    arr = np.sort(np.asarray(values, dtype=float))
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return float("nan")
    lo, hi = int(np.floor(0.25 * arr.size)), int(np.ceil(0.75 * arr.size))
    return float(arr[lo:hi].mean()) if hi > lo else float(arr.mean())


@dataclass(frozen=True)
class PairedTest:
    n: int
    median_difference: float
    mean_difference: float
    statistic: float
    p_value: float
    rank_biserial: float


def paired_wilcoxon(a: ArrayLike, b: ArrayLike) -> PairedTest:
    """Two-sided Wilcoxon signed-rank test of a − b with the matched-pairs rank-biserial r.

    r = (W⁺ − W⁻) / (W⁺ + W⁻); positive values mean ``a`` tends to be larger.
    """
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if x.shape != y.shape:
        raise ValueError("paired samples must have the same length")
    keep = ~(np.isnan(x) | np.isnan(y))
    d = x[keep] - y[keep]
    nz = d[d != 0]
    if nz.size == 0:
        return PairedTest(int(d.size), 0.0, 0.0, 0.0, 1.0, 0.0)
    ranks = stats.rankdata(np.abs(nz))
    w_plus, w_minus = ranks[nz > 0].sum(), ranks[nz < 0].sum()
    res = stats.wilcoxon(nz, zero_method="wilcox", alternative="two-sided")
    return PairedTest(
        n=int(d.size),
        median_difference=float(np.median(d)),
        mean_difference=float(d.mean()),
        statistic=float(res.statistic),
        p_value=float(res.pvalue),
        rank_biserial=float((w_plus - w_minus) / (w_plus + w_minus)),
    )


def holm_correction(p_values: ArrayLike) -> list[float]:
    """Holm–Bonferroni adjusted p-values (same order as the input)."""
    p = np.asarray(p_values, dtype=float)
    m = p.size
    order = np.argsort(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adjusted[idx] = min(running, 1.0)
    return adjusted.tolist()


def summarize_across_seeds(per_seed_means: ArrayLike) -> dict[str, float]:
    """Mean ± std over training seeds of each seed's mean, plus a bootstrap CI."""
    arr = np.asarray(per_seed_means, dtype=float)
    lo, hi = bootstrap_ci(arr)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "ci_low": lo,
        "ci_high": hi,
        "n_seeds": int(arr.size),
    }
