"""Evaluate trained PPO runs and baselines on the conditions of an experiment spec.

Example:
    python scripts/evaluate.py --spec configs/eval/main.yaml
"""

from __future__ import annotations

import argparse
import logging

from uavta.evaluation.experiments import ExperimentSpec, run_experiment
from uavta.utils.config import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--spec", required=True)
    parser.add_argument(
        "--only", nargs="*", default=None, help="evaluate only these condition names"
    )
    parser.add_argument("--output", default=None, help="override the output CSV path")
    parser.add_argument("--skip-ppo", action="store_true", help="evaluate baselines only")
    parser.add_argument("--skip-baselines", action="store_true", help="evaluate PPO runs only")
    parser.add_argument("--ppo", nargs="*", default=None, help="evaluate only these PPO groups")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    spec = ExperimentSpec.model_validate(load_yaml(args.spec))
    if args.only:
        spec.conditions = [c for c in spec.conditions if c.name in set(args.only)]
    if args.output:
        spec.output = args.output
    if args.skip_ppo:
        spec.ppo_runs = {}
    if args.ppo is not None:
        spec.ppo_runs = {k: v for k, v in spec.ppo_runs.items() if k in set(args.ppo)}
    if args.skip_baselines:
        spec.baselines = []
        for cond in spec.conditions:
            cond.baselines = []
    df = run_experiment(spec)
    print(f"wrote {len(df)} rows to {spec.output}")


if __name__ == "__main__":
    main()
