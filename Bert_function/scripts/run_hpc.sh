#!/bin/bash

# Run Instruction Masking Training  (HPC version)
# GPU assignment is managed by SLURM — do NOT set CUDA_VISIBLE_DEVICES here.

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "========================================"
echo "Instruction Masking Training (HPC)"
echo "========================================"
echo "Node:               $(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "SLURM_JOB_ID:       $SLURM_JOB_ID"
echo "SLURM_GPUS:         $SLURM_GPUS"
echo ""

# =============================================================================
# Experiment Selection
# =============================================================================
# Uncomment ONE of the following experiment blocks:
# Experiment name is auto-generated from TASKS + ADDRESS + VAR settings

# -----------------------------------------------------------------------------
# Experiment 1: IMC + MLM + Address + Var (Full features)
# -> Auto-generates: imc_mlm_addr_var
# -----------------------------------------------------------------------------
# TASKS="--enable_imc --enable_mlm"
TASKS="--enable_mlm"
USE_ADDRESS_EMBEDDING=true
USE_VAR_EMBEDDING=true

# -----------------------------------------------------------------------------
# Experiment 2: IMC + MLM + Address (No Var embedding)
# -> Auto-generates: imc_mlm_addr
# -----------------------------------------------------------------------------
# TASKS="--enable_imc --enable_mlm"
# USE_ADDRESS_EMBEDDING=true
# USE_VAR_EMBEDDING=false

# -----------------------------------------------------------------------------
# Experiment 3: IMC + MLM + Var (No Address embedding)
# -> Auto-generates: imc_mlm_var
# -----------------------------------------------------------------------------
# TASKS="--enable_imc --enable_mlm"
# USE_ADDRESS_EMBEDDING=false
# USE_VAR_EMBEDDING=true

# -----------------------------------------------------------------------------
# Experiment 4: IMC + MLM (Baseline - No Address, No Var)
# -> Auto-generates: imc_mlm
# -----------------------------------------------------------------------------
# TASKS="--enable_imc --enable_mlm"
# USE_ADDRESS_EMBEDDING=false
# USE_VAR_EMBEDDING=false

# -----------------------------------------------------------------------------
# Experiment 5: IMC + IMD + MLM + Address + Var (Full with DFG)
# -> Auto-generates: imc_imd_mlm_addr_var
# -----------------------------------------------------------------------------
# TASKS="--enable_imc --enable_imd --enable_mlm"
# USE_ADDRESS_EMBEDDING=true
# USE_VAR_EMBEDDING=true

# -----------------------------------------------------------------------------
# Experiment 6: IMC + MLM + Scope + Address + Var (Full with Scope)
# -> Auto-generates: imc_mlm_scope_addr_var
# -----------------------------------------------------------------------------
# TASKS="--enable_imc --enable_mlm --enable_scope"
# USE_ADDRESS_EMBEDDING=true
# USE_VAR_EMBEDDING=true

# =============================================================================
# Convert flags to command line arguments
# =============================================================================
if [ "$USE_ADDRESS_EMBEDDING" = true ]; then
  ADDRESS_FLAG=""
  ADDR_SUFFIX="_addr"
else
  ADDRESS_FLAG="--disable_address_embedding"
  ADDR_SUFFIX=""
fi

if [ "$USE_VAR_EMBEDDING" = true ]; then
  VAR_FLAG=""
  VAR_SUFFIX="_var"
else
  VAR_FLAG="--disable_var_embedding"
  VAR_SUFFIX=""
fi

# Auto-generate experiment name from TASKS and embeddings
TASK_NAME=""
if [[ "$TASKS" == *"--enable_imc"* ]];   then TASK_NAME="${TASK_NAME}_imc";   fi
if [[ "$TASKS" == *"--enable_imd"* ]];   then TASK_NAME="${TASK_NAME}_imd";   fi
if [[ "$TASKS" == *"--enable_mlm"* ]];   then TASK_NAME="${TASK_NAME}_mlm";   fi
if [[ "$TASKS" == *"--enable_scope"* ]]; then TASK_NAME="${TASK_NAME}_scope"; fi
TASK_NAME="${TASK_NAME#_}"

EXPERIMENT="${TASK_NAME}${ADDR_SUFFIX}${VAR_SUFFIX}"

# Configuration
MODEL_NAME="${EXPERIMENT}"
OUTPUT_DIR="/work/kliu14/Palmtreeoutput_8/${MODEL_NAME}"
LOG_DIR="/work/kliu14/Palmtreeoutput/log/${MODEL_NAME}"

# Model parameters
HIDDEN=768
LAYERS=12
ATTN_HEADS=12
SEQ_LEN=60

# Training parameters
BATCH_SIZE=16                # per-GPU batch size (SEQ_LEN=60 → attn memory ∝ 60²=9× vs 2-inline; 128→16)
GRADIENT_ACCUMULATION=8     # accumulate 8 steps → effective batch = 16×8 = 128 (same as 2-inline baseline)
GRADIENT_CHECKPOINTING=true # recompute BERT activations on backward (~40% less memory)
LR=1e-4
EPOCHS=10
WARMUP=1000
EARLY_STOPPING=10

# Masking parameters
TOKEN_MASK_PROB=0.15        # Standard MLM masking rate
INSTRUCTION_MASK_PROB=0.15  # Instruction-level masking rate

# Data parameters
TRAIN_PERCENTAGE=1          # Use 100% of training data on HPC

# Data paths
VOCAB_PATH="./vocab_8.pkl"
CFG_TRAIN="/work/kliu14/Palmtreedata/cfg_train/cfg_8inline_pretrain_nodup.txt"

# Multi-GPU via torchrun (SLURM allocates 4 GPUs via --gres=gpu:4)
# torchrun sets LOCAL_RANK automatically — no --cuda / --multi_gpu flags needed
NUM_GPUS=${SLURM_GPUS_ON_NODE:-4}
# Workers PER DDP PROCESS: divide total CPUs by GPU count to avoid RAM explosion
# (each worker forks the parent process memory; too many workers → CPU OOM)
NUM_WORKERS=$(( (${SLURM_CPUS_PER_TASK:-8} / ${NUM_GPUS}) > 0 ? (${SLURM_CPUS_PER_TASK:-8} / ${NUM_GPUS}) : 2 ))

echo "Configuration:"
echo "  Model:              ${MODEL_NAME}"
echo "  Output:             ${OUTPUT_DIR}"
echo "  Tasks:              ${TASKS}"
echo "  Token mask rate:    ${TOKEN_MASK_PROB}"
echo "  Instr mask rate:    ${INSTRUCTION_MASK_PROB}"
echo "  Batch size:         ${BATCH_SIZE}"
echo "  Learning rate:      ${LR}"
echo "  Epochs:             ${EPOCHS}"
echo "  Train data:         ${TRAIN_PERCENTAGE}"
echo "  Workers:            ${NUM_WORKERS}"
echo ""

# Create output and log directories
mkdir -p ${OUTPUT_DIR}
mkdir -p ${LOG_DIR}

echo "Starting training..."
echo ""

torchrun --nproc_per_node=${NUM_GPUS} --master_port=29500 train.py \
  --cfg_train "${CFG_TRAIN}" \
  --vocab "${VOCAB_PATH}" \
  --hidden ${HIDDEN} \
  --layers ${LAYERS} \
  --attn_heads ${ATTN_HEADS} \
  --seq_len ${SEQ_LEN} \
  --dropout 0.1 \
  --epochs ${EPOCHS} \
  --batch_size ${BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACCUMULATION} \
  $([ "${GRADIENT_CHECKPOINTING}" = "true" ] && echo "--gradient_checkpointing") \
  --lr ${LR} \
  --warmup_steps ${WARMUP} \
  --num_workers ${NUM_WORKERS} \
  --early_stopping_patience ${EARLY_STOPPING} \
  --token_mask_prob ${TOKEN_MASK_PROB} \
  --instruction_mask_prob ${INSTRUCTION_MASK_PROB} \
  --data_percentage ${TRAIN_PERCENTAGE} \
  --output_dir "${OUTPUT_DIR}" \
  --log_dir "${LOG_DIR}" \
  --resume \
  ${TASKS} \
  ${ADDRESS_FLAG} \
  ${VAR_FLAG}

if [ $? -eq 0 ]; then
  echo ""
  echo -e "${GREEN}========================================"
  echo "Training Completed Successfully!"
  echo "========================================${NC}"
  echo ""
  echo "Model saved to: ${OUTPUT_DIR}"
  echo "  - best_model.pt"
  echo "  - best_bert.pt"
  echo "  - checkpoint_latest.pt"
  echo ""
  echo "Logs saved to: ${LOG_DIR}"
else
  echo -e "${YELLOW}Training failed or was interrupted.${NC}"
  exit 1
fi