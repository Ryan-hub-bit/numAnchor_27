#!/usr/bin/env bash
set -euo pipefail

AA_SUMMARY=/path/to/numAnchor_artifact/numAnchor/jtrans/paper_full_addressaware/summaries/pool_summary_epoch2_probe_10k.json
BASE_SUMMARY=/path/to/numAnchor_artifact/numAnchor/jtrans/paper_full_baseline/summaries/pool_summary_epoch2_probe_10k.json

wait_for_gpu() {
  while true; do
    for gpu in 0 1; do
      util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$gpu" | tr -d ' ')
      mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu" | tr -d ' ')
      if [[ "${util}" -lt 10 && "${mem}" -lt 1000 ]]; then
        echo "$gpu"
        return 0
      fi
    done
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] waiting for a free GPU" >&2
    sleep 120
  done
}

if [[ ! -f "${AA_SUMMARY}" ]]; then
  GPU=$(wait_for_gpu)
  echo "Launching AA epoch2 probe on GPU ${GPU} at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  CUDA_VISIBLE_DEVICES="${GPU}" /path/to/numAnchor/JTrans/addressaware/run_full_addressaware_epoch2_probe_10k.sh
fi

if [[ ! -f "${BASE_SUMMARY}" ]]; then
  GPU=$(wait_for_gpu)
  echo "Launching baseline epoch2 probe on GPU ${GPU} at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  CUDA_VISIBLE_DEVICES="${GPU}" /path/to/numAnchor/JTrans/addressaware/run_full_baseline_epoch2_probe_10k.sh
fi
