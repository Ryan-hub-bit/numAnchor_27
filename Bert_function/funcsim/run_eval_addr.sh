#!/bin/bash
# ============================================================
# Evaluate Function Similarity Model
#
# Computes Recall@1/5/10/20 and MRR on a retrieval pool.
#
# Usage:
#   ./run_eval_addr.sh             # default: pool_10k
#   ./run_eval_addr.sh 100         # pool_100  (quick sanity check)
#   ./run_eval_addr.sh 1k          # pool_1k
#   ./run_eval_addr.sh 10k         # pool_10k  (full eval)
# ============================================================
set -e

export CUDA_VISIBLE_DEVICES=1    # use GPU 1 for evaluation

# ── Pool size (from first argument, default=10k) ──────────────
POOL_TAG="${1:-10k}"

# ── Paths ────────────────────────────────────────────────────
FUNCTION_BLOCKS="/path/to/numAnchor_artifact/addressaware/funcsim_block/function_blocks.json"
FUNCSIM_PAIRS="/path/to/numAnchor_artifact/addressaware/funcsim_block/funcsim_pairs.json"
VOCAB="/path/to/numAnchor/PalmTree/vocab.pkl"

# Trained model checkpoint (produced by run_funcsim_train.sh)
CHECKPOINT="../../output/funcsim/mlm_addr_var_moredatafinetune/best_model.pt"

# Pool file (set by POOL_TAG)
EVAL_POOL="/path/to/numAnchor_artifact/addressaware/funcsim_block/pool_ids_${POOL_TAG}.json"

# ── Output ───────────────────────────────────────────────────
OUTPUT="../../output/funcsim/mlm_addr_var/eval_results_${POOL_TAG}.json"
LOG_DIR="../../log/funcsim/mlm_addr_var"
TASK_NAME="mlm_addr_var_${POOL_TAG}"   # used for embedding cache filename

# ── Model config (must match training) ───────────────────────
HIDDEN=768
N_LAYERS=12
ATTN_HEADS=12
EMBEDDING_DIM=256
SEQ_LEN=420
MAX_INSTRUCTIONS=50

# ── Evaluation config ────────────────────────────────────────
BATCH_SIZE=64     # encoding batch size (no gradients, can be larger)

# ── Pre-flight checks ────────────────────────────────────────
echo "========================================================"
echo "  Function Similarity Evaluation  [pool=${POOL_TAG}]"
echo "========================================================"
echo "  Checkpoint : ${CHECKPOINT}"
echo "  Pool file  : ${EVAL_POOL}"
echo "  Output     : ${OUTPUT}"
echo "========================================================"

for f in "$CHECKPOINT" "$FUNCSIM_PAIRS" "$VOCAB" "$EVAL_POOL"; do
  if [ ! -f "$f" ]; then
    echo "[ERROR] File not found: $f"
    exit 1
  fi
done

if [ ! -f "$FUNCTION_BLOCKS" ]; then
  echo "[ERROR] Function blocks not found: $FUNCTION_BLOCKS"
  exit 1
fi

mkdir -p "${LOG_DIR}" "$(dirname ${OUTPUT})"

# ── Run ──────────────────────────────────────────────────────
python3 evaluate.py \
  --function_blocks "${FUNCTION_BLOCKS}" \
  --funcsim_pairs   "${FUNCSIM_PAIRS}"   \
  --vocab           "${VOCAB}"           \
  --checkpoint      "${CHECKPOINT}"      \
  --hidden          ${HIDDEN}            \
  --n_layers        ${N_LAYERS}          \
  --attn_heads      ${ATTN_HEADS}        \
  --embedding_dim   ${EMBEDDING_DIM}     \
  --seq_len         ${SEQ_LEN}           \
  --max_instructions ${MAX_INSTRUCTIONS} \
  --batch_size      ${BATCH_SIZE}        \
  --task_name       "${TASK_NAME}"       \
  --output          "${OUTPUT}"          \
  --log_dir         "${LOG_DIR}"         \
  --device          cuda                 \
  --eval_pool       "${EVAL_POOL}"

echo ""
echo "========================================================"
echo "  Evaluation complete!"
echo "  Results : ${OUTPUT}"
echo "========================================================"
