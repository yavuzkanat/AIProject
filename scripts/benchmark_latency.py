"""Controlled decision-latency benchmark (H4). Run on an otherwise idle machine.

Measures wall-clock time per decision (policy.act) for every method on the same scenarios,
single-threaded on the CPU. Training time is reported separately (training_summary.json).

Example:
    python scripts/benchmark_latency.py --count 50
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from uavta.agents.ppo_factory import PPOPolicy
from uavta.environment.env import MultiUAVDynamicEnv
from uavta.evaluation.experiments import BASELINES
from uavta.evaluation.runner import evaluate_policy, seed_refs
from uavta.scenarios.generator import TEST_SEED_BASE


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--out", default="results/tables/latency.json")
    args = parser.parse_args()
    torch.set_num_threads(1)
    env = MultiUAVDynamicEnv()
    refs = seed_refs(TEST_SEED_BASE, args.count, "test_id")
    policies = {name: make() for name, make in BASELINES.items()}
    for name, run in (
        ("ppo", "ppo_default_seed0"),
        ("ppo_entity", "ppo_entity_seed0"),
        ("ppo_attention", "ppo_attention_seed0"),
    ):
        path = Path("results/runs") / run / "best_model.zip"
        if path.exists():
            policies[name] = PPOPolicy.load(path, name=name)
    if "ppo_entity" in policies:
        from uavta.agents.lookahead import LookaheadPolicy

        base = PPOPolicy.load(
            Path("results/runs/ppo_entity_seed0/best_model.zip"), name="ppo_entity"
        )
        policies["ppo_entity_lookahead"] = LookaheadPolicy(base, top_k=2, samples=0)
    result = {}
    for name, policy in policies.items():
        df = evaluate_policy(env, policy, refs, method=name)
        result[name] = {
            "median_of_episode_median_ms": float(np.median(df["latency_median_ms"])),
            "mean_of_episode_p95_ms": float(np.mean(df["latency_p95_ms"])),
            "mean_decisions_per_episode": float(df["n_decisions"].mean()),
        }
        print(name, result[name])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
