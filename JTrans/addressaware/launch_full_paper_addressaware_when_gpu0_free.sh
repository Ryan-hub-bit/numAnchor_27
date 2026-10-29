#!/usr/bin/env bash
set -euo pipefail

LOG_DIR="/path/to/numAnchor_artifact/numAnchor/jtrans/paper_full_addressaware/logs"
LOG_FILE="${LOG_DIR}/run_$(date -u +%Y%m%d_%H%M%S).log"
mkdir -p "${LOG_DIR}"

while true; do
  util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 0 | tr -d ' ')
  if [[ "${util}" -lt 10 ]]; then
    echo "[$(date -u +%FT%TZ)] GPU 0 is free enough, launching address-aware pipeline" | tee -a "${LOG_FILE}"
    CUDA_VISIBLE_DEVICES=0 /path/to/numAnchor/JTrans/addressaware/run_full_paper_addressaware.sh 2>&1 | tee -a "${LOG_FILE}"
    exit 0
  fi
  echo "[$(date -u +%FT%TZ)] GPU 0 still busy" | tee -a "${LOG_FILE}"
  sleep 300
done
