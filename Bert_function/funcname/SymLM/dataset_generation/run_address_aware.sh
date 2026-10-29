#!/bin/bash

set -euo pipefail

# Address-aware SymLM dataset generation:
# 1. use IDA Pro to export function bodies + caller/callee metadata
# 2. convert the export into an address-aware SymLM dataset shard

IDA_PATH="/path/to/ida-pro-9.0/idat"
BINARY_PATH=""
IDA_EXPORT_DIR="./ida_export"
DATASET_OUTPUT_DIR="./addressaware_dataset"
SYMBOLS_FILE="/path/to/numAnchor_artifact/addressaware/cfg_train/top_symbols_cfg_8inline.txt"
ENABLE_SMART_MERGE="true"
TOPK=2

if [[ -z "${BINARY_PATH}" ]]; then
  echo "Please set BINARY_PATH in dataset_generation/run_address_aware.sh"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"${IDA_PATH}" -A -S"${SCRIPT_DIR}/process_address_aware_ida.py ${IDA_EXPORT_DIR} ${SYMBOLS_FILE} ${ENABLE_SMART_MERGE}" "${BINARY_PATH}"

python3 "${SCRIPT_DIR}/prepare_address_aware_dataset.py" \
  --input-dir "${IDA_EXPORT_DIR}" \
  --output-dir "${DATASET_OUTPUT_DIR}" \
  --topK "${TOPK}"
