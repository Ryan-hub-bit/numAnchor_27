#!/usr/bin/env bash
set -euo pipefail

choose_free_gpu() {
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \
    | awk -F', *' '$2+0 < 5000 && $3+0 < 20 {print $1; exit}'
}

while true; do
  GPU="$(choose_free_gpu || true)"
  if [[ -n "${GPU}" ]]; then
    break
  fi
  sleep 60
done

echo "Using GPU ${GPU}"
export CUDA_VISIBLE_DEVICES="${GPU}"
exec /path/to/numAnchor/PalmTree/addressaware/scripts/run_funcsim_baseline_ep9resume_full_ep3.sh
