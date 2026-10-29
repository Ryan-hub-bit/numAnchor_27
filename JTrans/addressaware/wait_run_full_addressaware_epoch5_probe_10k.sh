#!/usr/bin/env zsh
set -euo pipefail

CHECKPOINT=/path/to/numAnchor_artifact/numAnchor/jtrans/paper_full_addressaware/pretrain_out/checkpoint_epoch_5
SCRIPT=/path/to/numAnchor/JTrans/addressaware/run_full_addressaware_epoch5_probe_10k.sh
SUMMARY=/path/to/numAnchor_artifact/numAnchor/jtrans/paper_full_addressaware/summaries/pool_summary_epoch5_probe_10k.json

while [[ ! -d "${CHECKPOINT}" ]]; do
  sleep 300
done

while true; do
  if [[ -f "${SUMMARY}" ]]; then
    exit 0
  fi
  GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 0 | tr -d ' ')
  if [[ "${GPU_UTIL}" -le 15 ]]; then
    break
  fi
  sleep 300
done

export CUDA_VISIBLE_DEVICES=0
exec "${SCRIPT}"
