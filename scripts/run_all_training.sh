#!/usr/bin/env bash
# Every training run of the study, in order. Safe to re-run at any time:
#   * completed runs (training_summary.json present) are skipped,
#   * interrupted runs are cleaned and restarted from scratch (same seed => same result on CPU),
#   * a run already being trained by another live process is refused, not duplicated.
# Usage (from the repository root):  ./scripts/run_all_training.sh
set -uo pipefail
PY=.venv/bin/python
mkdir -p results/logs
failed=()

run() {  # run <log-name> <train.py args...>
  local name=$1; shift
  echo "$(date '+%H:%M:%S') ▶ $name"
  if ! $PY scripts/train.py "$@" >> "results/logs/$name.log" 2>&1; then
    echo "   ✗ $name failed (see results/logs/$name.log)"; failed+=("$name")
  fi
}

# Main study: 5 seeds x 2M steps for each architecture.
for s in 0 1 2 3 4; do run "ppo_default_seed$s" --config configs/train/ppo_default.yaml --seed $s; done
for s in 0 1 2 3 4; do run "ppo_entity_seed$s"  --config configs/train/ppo_entity.yaml  --seed $s; done

# Improvement 2: entity scorer + attention (5 seeds x 1M steps, see configs/train/ppo_attention.yaml).
for s in 0 1 2 3 4; do run "ppo_attention_seed$s" --config configs/train/ppo_attention.yaml --seed $s; done

# Phase 11 ablations: 3 seeds x 1M steps.
ABL_STEPS=${ABL_STEPS:-1000000}
for s in 0 1 2; do
  run "abl_value_only_seed$s" --config configs/train/ppo_entity.yaml --seed $s --timesteps $ABL_STEPS \
      --name abl_value_only --set '{"env": {"reward": {"w_value": 1.0, "w_time": 0.0}}}'
  run "abl_w25_seed$s" --config configs/train/ppo_entity.yaml --seed $s --timesteps $ABL_STEPS \
      --name abl_w25 --set '{"env": {"reward": {"w_value": 0.25, "w_time": 0.75}}}'
  run "abl_w75_seed$s" --config configs/train/ppo_entity.yaml --seed $s --timesteps $ABL_STEPS \
      --name abl_w75 --set '{"env": {"reward": {"w_value": 0.75, "w_time": 0.25}}}'
  run "abl_energy_seed$s" --config configs/train/ppo_entity.yaml --seed $s --timesteps $ABL_STEPS \
      --name abl_energy --set '{"env": {"reward": {"w_energy": 0.5}}}'
  run "abl_unmasked_seed$s" --config configs/train/ppo_default.yaml --seed $s --timesteps $ABL_STEPS \
      --name abl_unmasked --set '{"env": {"invalid_action_mode": "penalize"}, "ppo": {"use_masking": false}}'
done

if ((${#failed[@]})); then echo "failed runs: ${failed[*]}"; exit 1; fi
echo "all training runs complete"
