#!/bin/bash
# ============================================================
# Fine-tune MLM-only BERT for Function Similarity
#
# Single GPU:
#   python train.py <args>
#
# Two GPUs (DDP via torchrun):
#   ./run_funcsim_train_mlm.sh
# ============================================================
set -e

export CUDA_VISIBLE_DEVICES=0,1 # expose GPU 0 and GPU 1

# ── Paths ────────────────────────────────────────────────────
FUNCTION_BLOCKS="/path/to/numAnchor_artifact/addressaware/funcsim_block/function_blocks.json"
FUNCSIM_PAIRS="/path/to/numAnchor_artifact/addressaware/funcsim_block/funcsim_pairs.json"
VOCAB="/path/to/numAnchor/PalmTree/vocab.pkl"

# Pre-trained BERT checkpoint (MLM only — no address/var embeddings).
PRETRAINED_BERT="/path/to/numAnchor/output/mlm/latest_bert.pt"

# Resume from an existing funcsim checkpoint (loads full model + optimizer).
# Set to "" for a fresh fine-tune from PRETRAINED_BERT.
# Set to best_model.pt path to continue for more epochs.
RESUME_CHECKPOINT=""

# ── Output ───────────────────────────────────────────────────
OUTPUT_DIR="../../output/funcsim/mlm"
LOG_DIR="../../log/funcsim/mlm"
EXPERIMENT_NAME="funcsim_mlm"
TASK_NAME="mlm"

# ── Model config (must match pre-trained BERT) ───────────────
HIDDEN=768
N_LAYERS=12
ATTN_HEADS=12
EMBEDDING_DIM=256 # output embedding dimension

# ── Training config ──────────────────────────────────────────
BATCH_SIZE=32       # per-GPU; effective = 32 × 2 GPUs = 64
EPOCHS=5            # number of fine-tuning epochs
LR=3e-5             # peak LR for projection head (randomly initialised)
LR_BERT=3e-6        # peak LR for BERT backbone (10× lower — preserve pre-trained weights)
WARMUP_RATIO=0.06   # 6% of total steps used for linear warmup
SEQ_LEN=420         # max tokens per function (50 instrs × ~8 tokens + 2 special)
MAX_INSTRUCTIONS=50 # max instructions per function
LOSS_TYPE=infonce   # 'infonce' (recommended) or 'contrastive'
TEMPERATURE=0.07    # InfoNCE temperature — lower = harder, 0.07 is a standard default
MARGIN=1.0          # contrastive margin on cosine dist [0,2] (only used if LOSS_TYPE=contrastive)
NEGATIVE_SAMPLES=0  # InfoNCE uses in-batch negatives; set >0 only for ContrastiveLoss

# Dataset sampling (set DATASET_FRACTION < 1.0 for quick experiments)
DATASET_FRACTION=0.004 # ~55k samples — sanity-check run; set to 1.0 for full training
# Validation split — hold out 10% to monitor overfitting
TRAIN_SPLIT=0.9
VAL_SPLIT=0.1

# Memory
GRADIENT_CHECKPOINTING=true # set to false to disable

# Misc
DEVICE="cuda"
NUM_WORKERS=4

# ── Pre-flight checks ────────────────────────────────────────
echo "========================================================"
echo "  Function Similarity Fine-tuning  (MLM only)"
echo "========================================================"
echo "  Pre-trained BERT : ${PRETRAINED_BERT}"
echo "  Function blocks  : ${FUNCTION_BLOCKS}"
echo "  Funcsim pairs    : ${FUNCSIM_PAIRS}"
echo "  Output dir       : ${OUTPUT_DIR}"
echo "========================================================"

for f in "$FUNCTION_BLOCKS" "$FUNCSIM_PAIRS" "$VOCAB"; do
  if [ ! -f "$f" ]; then
    echo "[ERROR] File not found: $f"
    exit 1
  fi
done

# Only require pretrained BERT when not resuming from a funcsim checkpoint
if [ -z "${RESUME_CHECKPOINT}" ] && [ ! -f "${PRETRAINED_BERT}" ]; then
  echo "[ERROR] File not found: ${PRETRAINED_BERT}"
  exit 1
fi

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

# ── Build argument list ──────────────────────────────────────
EXTRA_ARGS=""
if [ "${GRADIENT_CHECKPOINTING}" = "true" ]; then
  EXTRA_ARGS="--gradient_checkpointing"
fi
if [ -n "${RESUME_CHECKPOINT}" ] && [ -f "${RESUME_CHECKPOINT}" ]; then
  EXTRA_ARGS="${EXTRA_ARGS} --resume_checkpoint ${RESUME_CHECKPOINT}"
  echo "  Resuming   : ${RESUME_CHECKPOINT}"
fi

# ── Launch ───────────────────────────────────────────────────
torchrun --standalone --nproc_per_node=2 train.py \
  --function_blocks "${FUNCTION_BLOCKS}" \
  --funcsim_pairs "${FUNCSIM_PAIRS}" \
  --vocab "${VOCAB}" \
  --pretrained_bert "${PRETRAINED_BERT}" \
  --hidden ${HIDDEN} \
  --n_layers ${N_LAYERS} \
  --attn_heads ${ATTN_HEADS} \
  --embedding_dim ${EMBEDDING_DIM} \
  --seq_len ${SEQ_LEN} \
  --max_instructions ${MAX_INSTRUCTIONS} \
  --batch_size ${BATCH_SIZE} \
  --epochs ${EPOCHS} \
  --lr ${LR} \
  --lr_bert ${LR_BERT} \
  --warmup_ratio ${WARMUP_RATIO} \
  --loss ${LOSS_TYPE} \
  --temperature ${TEMPERATURE} \
  --margin ${MARGIN} \
  --negative_samples ${NEGATIVE_SAMPLES} \
  --dataset_fraction ${DATASET_FRACTION} \
  --train_split ${TRAIN_SPLIT} \
  --val_split ${VAL_SPLIT} \
  --output_dir "${OUTPUT_DIR}" \
  --log_dir "${LOG_DIR}" \
  --experiment_name "${EXPERIMENT_NAME}" \
  --task_name "${TASK_NAME}" \
  --device "${DEVICE}" \
  --num_workers ${NUM_WORKERS} \
  ${EXTRA_ARGS}

echo ""
echo "========================================================"
echo "  Training complete!"
echo "  Best model : ${OUTPUT_DIR}/best_model.pt"
echo "  Logs       : ${LOG_DIR}/"
echo "========================================================"
