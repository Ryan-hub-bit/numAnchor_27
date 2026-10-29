#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/path/to/conda/bin/python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export CUDA_VISIBLE_DEVICES

TRAIN_DIR="/path/to/numAnchor_artifact/numAnchor/jtrans/official_small_train_aa_full_cfgsplit_plt30"
TEST_DIR="/path/to/numAnchor_artifact/numAnchor/jtrans/official_small_test_cfgsplit_plt30"

VOCAB="${TRAIN_DIR}/vocab.pkl"
TRAIN_CORPUS="${TRAIN_DIR}/pretrain_train_seq512.txt"
VAL_CORPUS="${TRAIN_DIR}/pretrain_val_seq512.txt"
FUNCTION_BLOCKS="${TRAIN_DIR}/funcsim_function_blocks.json"
FUNCSIM_PAIRS="${TRAIN_DIR}/funcsim_pairs.json"
TEST_FUNCTION_BLOCKS="${TEST_DIR}/funcsim_function_blocks.json"
TEST_FUNCSIM_PAIRS="${TEST_DIR}/funcsim_pairs.json"

PRETRAIN_OUT="${TRAIN_DIR}/pretrain_out_seq512_h384_l6_baseline"
PRETRAIN_LOG_DIR="${TRAIN_DIR}/logs_pretrain_seq512_h384_l6_baseline"

FUNC_EPOCHS="${FUNC_EPOCHS:-1}"
FUNC_BATCH_SIZE="${FUNC_BATCH_SIZE:-12}"
FUNC_LR="${FUNC_LR:-5e-6}"
FUNC_OUT="${TRAIN_DIR}/paper_triplet_out_seq512_h384_l6_baseline_nofreeze_lr5e6_fullcmp_e${FUNC_EPOCHS}"
FUNC_LOG_DIR="${TRAIN_DIR}/logs_paper_triplet_seq512_h384_l6_baseline_nofreeze_lr5e6_fullcmp_e${FUNC_EPOCHS}"
EVAL_JSON="${FUNC_OUT}/official_small_test_opt_eval.json"

mkdir -p "${PRETRAIN_OUT}" "${PRETRAIN_LOG_DIR}" "${FUNC_OUT}" "${FUNC_LOG_DIR}"

echo "[1/3] full baseline pretrain"
"${PYTHON_BIN}" "${SCRIPT_DIR}/pretrain.py" \
  --train_corpus "${TRAIN_CORPUS}" \
  --val_corpus "${VAL_CORPUS}" \
  --vocab "${VOCAB}" \
  --output_dir "${PRETRAIN_OUT}" \
  --log_dir "${PRETRAIN_LOG_DIR}" \
  --seq_len 512 \
  --hidden 384 \
  --n_layers 6 \
  --attn_heads 8 \
  --batch_size 8 \
  --epochs 6 \
  --lr 1e-4 \
  --device cuda \
  --gradient_checkpointing \
  --disable_address_embedding \
  --disable_var_embedding

echo "[2/3] full baseline downstream"
"${PYTHON_BIN}" "${SCRIPT_DIR}/train_funcsim_paper.py" \
  --function_blocks "${FUNCTION_BLOCKS}" \
  --funcsim_pairs "${FUNCSIM_PAIRS}" \
  --vocab "${VOCAB}" \
  --pretrained_bert "${PRETRAIN_OUT}/best_model.pt" \
  --output_dir "${FUNC_OUT}" \
  --log_dir "${FUNC_LOG_DIR}" \
  --hidden 384 \
  --n_layers 6 \
  --attn_heads 8 \
  --seq_len 512 \
  --batch_size "${FUNC_BATCH_SIZE}" \
  --epochs "${FUNC_EPOCHS}" \
  --lr "${FUNC_LR}" \
  --device cuda \
  --gradient_checkpointing \
  --experiment_name "seq512_h384_l6_baseline_nofreeze_lr5e6_fullcmp_e${FUNC_EPOCHS}" \
  --disable_address_embedding \
  --disable_var_embedding

echo "[3/3] full baseline evaluation"
"${PYTHON_BIN}" "${SCRIPT_DIR}/evaluate_opt_pairs_paper.py" \
  --function_blocks "${TEST_FUNCTION_BLOCKS}" \
  --funcsim_pairs "${TEST_FUNCSIM_PAIRS}" \
  --vocab "${VOCAB}" \
  --checkpoint "${FUNC_OUT}/best_model.pt" \
  --output "${EVAL_JSON}" \
  --batch_size 64 \
  --poolsizes "32,1000,10000" \
  --device cuda

echo "Full baseline pipeline complete:"
echo "  ${EVAL_JSON}"
