#!/usr/bin/env bash
set -euo pipefail

TRAIN_PID="${1:-}"
CHECKPOINT_DIR="/path/to/numAnchor_artifact/addressaware_runs/trex_stateformer_x64_20260414/addressaware_aa_ckpt"
CHECKPOINT_PATH="${CHECKPOINT_DIR}/checkpoint_best.pt"
LOG_PATH="/path/to/numAnchor_artifact/addressaware_runs/trex_stateformer_x64_20260414/logs/addressaware_aa_posteval.log"

if [[ -n "${TRAIN_PID}" ]]; then
  while kill -0 "${TRAIN_PID}" 2>/dev/null; do
    sleep 60
  done
else
  until [[ -f "${CHECKPOINT_DIR}/checkpoint_last.pt" ]]; do
    sleep 60
  done
fi

until [[ -f "${CHECKPOINT_PATH}" ]]; do
  sleep 30
done

bash /path/to/numAnchor/trex/command/stateformer_x64/evaluate_addressaware_aa_pools.sh "${CHECKPOINT_PATH}" >> "${LOG_PATH}" 2>&1
