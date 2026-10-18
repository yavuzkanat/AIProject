#!/usr/bin/env bash
# Phase 11 training runs (3 seeds, 1M steps each). Run from the repository root.
set -euo pipefail
PY=.venv/bin/python
mkdir -p results/logs
for s in 0 1 2; do
  $PY scripts/train.py --config configs/train/ppo_entity.yaml --seed $s --timesteps 1000000 --name abl_value_only \
    --set '{"env": {"reward": {"w_value": 1.0, "w_time": 0.0}}}' > results/logs/abl_value_only_seed$s.log 2>&1
  $PY scripts/train.py --config configs/train/ppo_entity.yaml --seed $s --timesteps 1000000 --name abl_w25 \
    --set '{"env": {"reward": {"w_value": 0.25, "w_time": 0.75}}}' > results/logs/abl_w25_seed$s.log 2>&1
  $PY scripts/train.py --config configs/train/ppo_entity.yaml --seed $s --timesteps 1000000 --name abl_w75 \
    --set '{"env": {"reward": {"w_value": 0.75, "w_time": 0.25}}}' > results/logs/abl_w75_seed$s.log 2>&1
  $PY scripts/train.py --config configs/train/ppo_entity.yaml --seed $s --timesteps 1000000 --name abl_energy \
    --set '{"env": {"reward": {"w_energy": 0.5}}}' > results/logs/abl_energy_seed$s.log 2>&1
  $PY scripts/train.py --config configs/train/ppo_default.yaml --seed $s --timesteps 1000000 --name abl_unmasked \
    --set '{"env": {"invalid_action_mode": "penalize"}, "ppo": {"use_masking": false}}' > results/logs/abl_unmasked_seed$s.log 2>&1
done
