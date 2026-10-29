#!/usr/bin/env zsh
set -euo pipefail

SCRIPT=/path/to/numAnchor/JTrans/addressaware/run_full_baseline_epoch6_probe_10k.sh
SUMMARY=/path/to/numAnchor_artifact/numAnchor/jtrans/paper_full_baseline/summaries/pool_summary_epoch6_probe_10k.json

while true; do
  if [[ -f "${SUMMARY}" ]]; then
    exit 0
  fi
  GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 1 | tr -d ' ')
  GPU_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 | tr -d ' ')
  if [[ "${GPU_UTIL}" -le 15 && "${GPU_MEM}" -le 5000 ]]; then
    break
  fi
  sleep 120
done

export CUDA_VISIBLE_DEVICES=1
exec "${SCRIPT}"
