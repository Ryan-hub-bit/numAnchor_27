#!/bin/bash
# Wrapper script to generate CFG with IDA Pro batch mode
# Automatically cleans up IDA-generated database files

IDA_PATH="/path/to/ida-pro-9.0"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CFG_SCRIPT="$SCRIPT_DIR/cfg_gen.py"

# Default parameters
SEG_LEN=${1:-8}
BIN_FOLDER=${2:-"/path/to/numAnchor_artifact/addressaware/addressbin_train"}
OUTPUT_DIR=${3:-"/path/to/numAnchor_artifact/addressaware/cfg_train/${SEG_LEN}"}

export OUTPUT_DIR
export SEG_LEN

echo "=========================================="
echo "IDA Pro CFG Generation"
echo "=========================================="
echo "IDA Path: $IDA_PATH"
echo "CFG Script: $CFG_SCRIPT"
echo "SEG_LEN: $SEG_LEN"
echo "Binary Folder: $BIN_FOLDER"
echo "Output Dir: $OUTPUT_DIR"
echo "=========================================="

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Create log file
LOG_FILE="$OUTPUT_DIR/ida_cfg_generation_$(date +%Y%m%d_%H%M%S).log"
echo "Log file: $LOG_FILE"
echo ""

# Function to log message (both to console and file)
log() {
  echo "$1" | tee -a "$LOG_FILE"
}

# Function to clean up IDA database files
cleanup_ida_files() {
  local binary="$1"
  rm -f "${binary}.idb" "${binary}.i64" "${binary}.id0" "${binary}.id1" "${binary}.id2" "${binary}.nam" "${binary}.til" 2>/dev/null
}

# Count total binaries
total=$(find "$BIN_FOLDER" -maxdepth 1 -type f ! -name "*.txt" ! -name "*.idb" ! -name "*.i64" ! -name "*.id0" ! -name "*.id1" ! -name "*.id2" ! -name "*.nam" ! -name "*.til" | wc -l)

if [ "$total" -eq 0 ]; then
  log "[ERROR] No binary files found in $BIN_FOLDER"
  exit 1
fi

log "[INFO] Found $total binaries to process"
log ""

# Process each binary file
count=0
success=0
failed=0
skipped=0

for binary in "$BIN_FOLDER"/*; do
  # Skip non-files
  if [ ! -f "$binary" ]; then
    continue
  fi

  basename=$(basename "$binary")

  # Skip text files and IDA database files
  if [[ "$basename" == *.txt ]] ||
    [[ "$basename" == *.idb ]] ||
    [[ "$basename" == *.i64 ]] ||
    [[ "$basename" == *.id0 ]] ||
    [[ "$basename" == *.id1 ]] ||
    [[ "$basename" == *.id2 ]] ||
    [[ "$basename" == *.nam ]] ||
    [[ "$basename" == *.til ]]; then
    continue
  fi

  count=$((count + 1))
  log "======================================================================"
  log "Processing $count/$total: $basename"
  log "======================================================================"

  # Start timing
  start_time=$(date +%s)

  # Check if output file already exists
  cfg_file="$OUTPUT_DIR/${basename}_cfg_${SEG_LEN}_inline.txt"

  if [ -f "$cfg_file" ]; then
    log "[SKIP] CFG file already exists for $basename"
    skipped=$((skipped + 1))
    log ""
    continue
  fi

  # ==========================================
  # Generate CFG
  # ==========================================
  log "[INFO] Generating CFG for $basename..."
  cfg_start=$(date +%s)
  
  # Run IDA in batch mode (passing output dir as script argument)
  "$IDA_PATH/idat" -A -S"$CFG_SCRIPT $OUTPUT_DIR" "$binary" >/dev/null 2>&1
  exit_code=$?
  
  cfg_end=$(date +%s)
  cfg_time=$((cfg_end - cfg_start))

  # Clean up IDA database files
  cleanup_ida_files "$binary"

  # Verify output file was created
  if [ -f "$cfg_file" ]; then
    cfg_lines=$(wc -l <"$cfg_file")
    log "[SUCCESS] CFG generation completed in ${cfg_time}s"
    log "  → CFG: $cfg_lines sequences"
    success=$((success + 1))
  else
    log "[FAILED] CFG generation failed (${cfg_time}s)"
    log "  → CFG: [MISSING]"
    failed=$((failed + 1))
  fi

  # Calculate and display total time
  end_time=$(date +%s)
  total_time=$((end_time - start_time))
  log "  → Total time: ${total_time}s"

  log ""
done

log "======================================================================"
log "All binaries processed!"
log "======================================================================"
log "Output location: $OUTPUT_DIR"
log ""

# Summary statistics
cfg_count=$(find "$OUTPUT_DIR" -name "*_cfg_${SEG_LEN}_inline.txt" -type f | wc -l)
total_size=$(du -sh "$OUTPUT_DIR" 2>/dev/null | cut -f1)

log "Summary:"
log "  Total binaries: $count"
log "  Success: $success"
log "  Failed: $failed"
log "  Skipped (already exists): $skipped"
log "  CFG files in output: $cfg_count"
log "  Total output size: ${total_size:-N/A}"
log "======================================================================"
log "Log saved to: $LOG_FILE"
