#!/usr/bin/env bash
set -euo pipefail

TARGET_GPU="${TARGET_GPU:-0}"
POLL_SECONDS="${POLL_SECONDS:-300}"
RUN_SCRIPT="/path/to/numAnchor/JTrans/addressaware/run_full_cfgsplit_addraware.sh"

echo "[$(date -u +%FT%TZ)] waiting for GPU ${TARGET_GPU} to become free"
while nvidia-smi --id="${TARGET_GPU}" --query-compute-apps=pid --format=csv,noheader | grep -q "[0-9]"; do
  echo "[$(date -u +%FT%TZ)] GPU ${TARGET_GPU} still busy"
  sleep "${POLL_SECONDS}"
done

echo "[$(date -u +%FT%TZ)] GPU ${TARGET_GPU} is free, starting ${RUN_SCRIPT}"
exec "${RUN_SCRIPT}"
