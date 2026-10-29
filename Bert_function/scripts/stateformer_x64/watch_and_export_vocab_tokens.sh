#!/usr/bin/env bash
set -euo pipefail

VOCAB_PATH="${1:?usage: watch_and_export_vocab_tokens.sh <vocab.pkl> <indexed.txt> <tokens_only.txt> [log_file] [poll_seconds] }"
INDEXED_OUT="${2:?usage: watch_and_export_vocab_tokens.sh <vocab.pkl> <indexed.txt> <tokens_only.txt> [log_file] [poll_seconds] }"
TOKENS_ONLY_OUT="${3:?usage: watch_and_export_vocab_tokens.sh <vocab.pkl> <indexed.txt> <tokens_only.txt> [log_file] [poll_seconds] }"
LOG_FILE="${4:-/path/to/numAnchor_artifact/log/Palmtree/export_vocab_tokens_stateformer_x64_ctx2.log}"
POLL_SECONDS="${5:-30}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

mkdir -p "$(dirname "${LOG_FILE}")"

{
  echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] Waiting for vocab: ${VOCAB_PATH}"
  while [[ ! -f "${VOCAB_PATH}" ]]; do
    sleep "${POLL_SECONDS}"
    echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] Still waiting..."
  done

  echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] Vocab found, exporting tokens"

  cd "${REPO_ROOT}"

  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  else
    echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] ERROR: no python interpreter found"
    exit 1
  fi

  "${PYTHON_BIN}" "${SCRIPT_DIR}/export_vocab_tokens.py" \
    --vocab "${VOCAB_PATH}" \
    --output "${INDEXED_OUT}" \
    --tokens-only-output "${TOKENS_ONLY_OUT}"

  echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] Export finished"
} >> "${LOG_FILE}" 2>&1
