#!/usr/bin/env zsh
set -euo pipefail

CHECKPOINT=/path/to/numAnchor_artifact/numAnchor/jtrans/paper_full_addressaware/pretrain_out/checkpoint_epoch_5
SCRIPT=/path/to/numAnchor/JTrans/addressaware/run_full_addressaware_epoch5_probe_10k_os.sh
SUMMARY=/path/to/numAnchor_artifact/numAnchor/jtrans/paper_full_addressaware/summaries/pool_summary_epoch5_probe_10k_os.json

while [[ ! -d "${CHECKPOINT}" ]]; do
  sleep 300
done

while pgrep -f "/path/to/numAnchor_artifact/numAnchor/jtrans/paper_full_baseline/finetune_out_epoch5_probe_10k_os" >/dev/null; do
  sleep 120
done

while true; do
  if [[ -f "${SUMMARY}" ]]; then
    exit 0
  fi
  GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 1 | tr -d ' ')
  GPU_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 | tr -d ' ')
  if [[ "${GPU_UTIL}" -le 15 && "${GPU_MEM}" -le 2000 ]]; then
    break
  fi
  sleep 120
done

export CUDA_VISIBLE_DEVICES=1
exec "${SCRIPT}"
