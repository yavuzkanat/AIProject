"""Train a PPO dispatch policy.

Example:
    python scripts/train.py --config configs/train/ppo_default.yaml --seed 0
"""

from __future__ import annotations

import argparse
import json
import logging

from uavta.training.train import load_train_config, train


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--out", default="results/runs")
    parser.add_argument("--set", default=None, help="JSON dict merged into the config")
    args = parser.parse_args()
    overrides = json.loads(args.set) if args.set else {}
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.timesteps is not None:
        overrides["total_timesteps"] = args.timesteps
    if args.name is not None:
        overrides["name"] = args.name
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_dir = train(load_train_config(args.config, overrides), args.out)
    print(run_dir)


if __name__ == "__main__":
    main()
