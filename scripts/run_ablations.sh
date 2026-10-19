#!/usr/bin/env bash
# Kept for compatibility: the ablations are part of scripts/run_all_training.sh, which skips
# completed runs, so running it trains whatever is still missing.
exec "$(dirname "$0")/run_all_training.sh" "$@"
