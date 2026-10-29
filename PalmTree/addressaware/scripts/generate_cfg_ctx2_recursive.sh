#!/bin/bash

set -euo pipefail

RAW_ROOT="${RAW_ROOT:-/path/to/numAnchor_artifact/stateformer_x64_dataset/raw_input/x86-64}"
CFG_ROOT="${CFG_ROOT:-/path/to/numAnchor_artifact/addressaware/cfg_ctx2_unique}"
LOG_FILE="${LOG_FILE:-/path/to/numAnchor_artifact/log/Palmtree/addressaware_generate_cfg_ctx2_unique.log}"
RUNNER="${RUNNER:-/path/to/numAnchor/PalmTree/addressaware/data_generator/run_cfg_batch.sh}"
SEG_LEN="${SEG_LEN:-2}"
MAX_DIRS="${MAX_DIRS:-0}"
DIR_FILTER="${DIR_FILTER:-}"

mkdir -p "${CFG_ROOT}" "$(dirname "${LOG_FILE}")"

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" | tee -a "${LOG_FILE}"
}

if [ ! -d "${RAW_ROOT}" ]; then
  log "ERROR: raw root not found: ${RAW_ROOT}"
  exit 1
fi

log "Starting recursive CFG generation from ${RAW_ROOT}"
log "SEG_LEN=${SEG_LEN}, CFG_ROOT=${CFG_ROOT}, DIR_FILTER=${DIR_FILTER:-<none>}, MAX_DIRS=${MAX_DIRS}"

processed_dirs=0

for subdir in "${RAW_ROOT}"/*; do
  if [ ! -d "${subdir}" ]; then
    continue
  fi

  name="$(basename "${subdir}")"

  if [ -n "${DIR_FILTER}" ] && [[ "${name}" != *"${DIR_FILTER}"* ]]; then
    continue
  fi

  file_count="$(find "${subdir}" -maxdepth 1 -type f | wc -l)"
  if [ "${file_count}" -eq 0 ]; then
    log "SKIP ${name}: no files at top level"
    continue
  fi

  out_dir="${CFG_ROOT}/${name}"
  log "RUN ${name}: ${file_count} files -> ${out_dir}"
  bash "${RUNNER}" "${SEG_LEN}" "${subdir}" "${out_dir}" 2>&1 | tee -a "${LOG_FILE}"

  processed_dirs=$((processed_dirs + 1))
  if [ "${MAX_DIRS}" -gt 0 ] && [ "${processed_dirs}" -ge "${MAX_DIRS}" ]; then
    log "Reached MAX_DIRS=${MAX_DIRS}; stopping"
    break
  fi
done

log "Finished recursive CFG generation; processed_dirs=${processed_dirs}"
