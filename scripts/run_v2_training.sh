#!/usr/bin/env bash
# Extended environment v2: train the entity policy for 5 seeds, PARALLEL runs at a time.
# Safe to re-run: completed runs are skipped, interrupted runs restart cleanly (same seed =>
# same result on CPU). Usage (from the repository root):  PARALLEL=2 ./scripts/run_v2_training.sh
set -uo pipefail
PY=.venv/bin/python
PARALLEL=${PARALLEL:-2}
SEEDS=${SEEDS:-"0 1 2 3 4"}
mkdir -p results/logs
for s in $SEEDS; do
  while [ "$(jobs -rp | wc -l)" -ge "$PARALLEL" ]; do sleep 5; done
  echo "$(date '+%H:%M:%S') start ppo_entity_hard_seed$s"
  $PY scripts/train.py --config configs/train/ppo_entity_hard.yaml --seed "$s" \
    >> "results/logs/ppo_entity_hard_seed$s.log" 2>&1 \
    && echo "$(date '+%H:%M:%S') done ppo_entity_hard_seed$s" \
    || echo "$(date '+%H:%M:%S') FAILED ppo_entity_hard_seed$s (see results/logs)" &
done
wait
