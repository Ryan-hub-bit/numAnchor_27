#!/bin/bash

set -euo pipefail

X64_FOLDER_ID="${X64_FOLDER_ID:-1gvGb_d9Fqkam2fm3LKgWhhWfhPxa_UsM}"
ROOT_NAME="${ROOT_NAME:-x86-64}"
TARGET_ROOT="${TARGET_ROOT:-/path/to/numAnchor_artifact/stateformer_x64_dataset/raw_input}"
TARGET_DIR="${TARGET_ROOT}/${ROOT_NAME}"
LOG_DIR="${LOG_DIR:-/path/to/numAnchor_artifact/log/Palmtree}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/download_stateformer_x64.log}"
VENV_DIR="${VENV_DIR:-/path/to/numAnchor_artifact/stateformer_x64_dataset/.gdown_venv}"

mkdir -p "${TARGET_ROOT}" "${LOG_DIR}"

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" >> "${LOG_FILE}"
}

log "Preparing Google Drive downloader venv at ${VENV_DIR}"

if [ ! -x "${VENV_DIR}/bin/gdown" ]; then
  python3 -m venv "${VENV_DIR}"
  "${VENV_DIR}/bin/pip" install gdown | tee -a "${LOG_FILE}"
fi

log "Downloading only the x86-64 folder (folder id: ${X64_FOLDER_ID})"
log "Target root: ${TARGET_ROOT}"
mkdir -p "${TARGET_ROOT}"

"${VENV_DIR}/bin/python" /path/to/numAnchor/PalmTree/scripts/stateformer_x64/drive_recursive_download.py \
  --folder_id "${X64_FOLDER_ID}" \
  --output_root "${TARGET_ROOT}" \
  --root_name "${ROOT_NAME}" >> "${LOG_FILE}" 2>&1

if [ ! -d "${TARGET_DIR}" ]; then
  log "ERROR: expected download directory missing: ${TARGET_DIR}"
  exit 1
fi

log "Download completed successfully"
find "${TARGET_DIR}" -maxdepth 2 | sed -n '1,120p' >> "${LOG_FILE}" 2>&1
