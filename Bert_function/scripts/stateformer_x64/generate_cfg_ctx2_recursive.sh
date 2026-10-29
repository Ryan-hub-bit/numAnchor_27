#!/bin/bash

set -euo pipefail

RAW_ROOT="/path/to/numAnchor_artifact/stateformer_x64_dataset/raw_input/x86-64"
CFG_ROOT="/path/to/numAnchor_artifact/stateformer_x64_dataset/cfg_ctx2"
LOG_FILE="/path/to/numAnchor_artifact/log/Palmtree/generate_cfg_ctx2_recursive.log"
RUNNER="/path/to/numAnchor/PalmTree/data_generator/run_cfg_batch.sh"

mkdir -p "${CFG_ROOT}" "$(dirname "${LOG_FILE}")"

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" | tee -a "${LOG_FILE}"
}

if [ ! -d "${RAW_ROOT}" ]; then
  log "ERROR: raw root not found: ${RAW_ROOT}"
  exit 1
fi

log "Starting recursive ctx=2 CFG generation from ${RAW_ROOT}"

for subdir in "${RAW_ROOT}"/*; do
  if [ ! -d "${subdir}" ]; then
    continue
  fi
  name="$(basename "${subdir}")"
  file_count="$(find "${subdir}" -maxdepth 1 -type f | wc -l)"
  if [ "${file_count}" -eq 0 ]; then
    log "SKIP ${name}: no files at top level"
    continue
  fi
  out_dir="${CFG_ROOT}/${name}"
  log "RUN ${name}: ${file_count} files -> ${out_dir}"
  bash "${RUNNER}" 2 "${subdir}" "${out_dir}" 2>&1 | tee -a "${LOG_FILE}"
done

log "Finished recursive ctx=2 CFG generation"
