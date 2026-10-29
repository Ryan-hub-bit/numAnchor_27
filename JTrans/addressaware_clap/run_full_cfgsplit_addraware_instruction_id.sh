#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

NUMANCHOR_RUN_ROOT="${NUMANCHOR_RUN_ROOT:-/path/to/numanchor_runs}"
TRAIN_DIR="${TRAIN_DIR:-${NUMANCHOR_RUN_ROOT}/jtrans/official_train_numanchor_clap_instruction_id}"
TEST_DIR="${TEST_DIR:-${NUMANCHOR_RUN_ROOT}/jtrans/official_test_numanchor_clap_instruction_id}"
OUT_ROOT="${OUT_ROOT:-${NUMANCHOR_RUN_ROOT}/jtrans/addressaware_clap_instruction_id}"

SEQ_LEN="${SEQ_LEN:-512}"
SEGMENT_TYPES="${SEGMENT_TYPES:-513}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-6}"
PRETRAIN_BATCH_SIZE="${PRETRAIN_BATCH_SIZE:-8}"
PRETRAIN_LR="${PRETRAIN_LR:-1e-4}"
MAX_TRAIN_FUNCTIONS="${MAX_TRAIN_FUNCTIONS:-}"
MAX_VAL_FUNCTIONS="${MAX_VAL_FUNCTIONS:-}"
FUNC_EPOCHS="${FUNC_EPOCHS:-1}"
FUNC_BATCH_SIZE="${FUNC_BATCH_SIZE:-12}"
FUNC_LR="${FUNC_LR:-5e-6}"
DATASET_FRACTION="${DATASET_FRACTION:-1.0}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
POOLSIZES="${POOLSIZES:-32,1000,10000}"
MAX_INSTRUCTIONS="${MAX_INSTRUCTIONS:-100000}"

VOCAB="${VOCAB:-${TRAIN_DIR}/vocab.pkl}"
TRAIN_CORPUS="${TRAIN_CORPUS:-${TRAIN_DIR}/pretrain_train_seq512.txt}"
VAL_CORPUS="${VAL_CORPUS:-${TRAIN_DIR}/pretrain_val_seq512.txt}"
FUNCTION_BLOCKS="${FUNCTION_BLOCKS:-${TRAIN_DIR}/funcsim_function_blocks.json}"
FUNCSIM_PAIRS="${FUNCSIM_PAIRS:-${TRAIN_DIR}/funcsim_pairs.json}"
TEST_FUNCTION_BLOCKS="${TEST_FUNCTION_BLOCKS:-${TEST_DIR}/funcsim_function_blocks.json}"
TEST_FUNCSIM_PAIRS="${TEST_FUNCSIM_PAIRS:-${TEST_DIR}/funcsim_pairs.json}"

PRETRAIN_OUT="${OUT_ROOT}/pretrain_out_seq${SEQ_LEN}_h384_l6_addrtype_scale2048_instrid"
PRETRAIN_LOG_DIR="${OUT_ROOT}/logs_pretrain_seq${SEQ_LEN}_h384_l6_addrtype_scale2048_instrid"
FUNC_OUT="${OUT_ROOT}/paper_triplet_out_seq${SEQ_LEN}_h384_l6_addrtype_scale2048_instrid_e${FUNC_EPOCHS}"
FUNC_LOG_DIR="${OUT_ROOT}/logs_paper_triplet_seq${SEQ_LEN}_h384_l6_addrtype_scale2048_instrid_e${FUNC_EPOCHS}"
EVAL_JSON="${FUNC_OUT}/official_small_test_opt_eval.json"
SUMMARY_JSON="${FUNC_OUT}/official_small_test_opt_eval_summary.json"

mkdir -p "${PRETRAIN_OUT}" "${PRETRAIN_LOG_DIR}" "${FUNC_OUT}" "${FUNC_LOG_DIR}"

echo "[instruction-id AAE] ROOT=${SCRIPT_DIR}"
echo "[instruction-id AAE] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[instruction-id AAE] OUT_ROOT=${OUT_ROOT}"
echo "[instruction-id AAE] segment_types=${SEGMENT_TYPES}"
echo "[instruction-id AAE] max_train_functions=${MAX_TRAIN_FUNCTIONS:-all}"
echo "[instruction-id AAE] max_val_functions=${MAX_VAL_FUNCTIONS:-all}"
echo "[instruction-id AAE] dataset_fraction=${DATASET_FRACTION}"

PRETRAIN_ARGS=(
  --train_corpus "${TRAIN_CORPUS}"
  --val_corpus "${VAL_CORPUS}"
  --vocab "${VOCAB}"
  --output_dir "${PRETRAIN_OUT}"
  --log_dir "${PRETRAIN_LOG_DIR}"
  --seq_len "${SEQ_LEN}"
  --hidden 384
  --n_layers 6
  --attn_heads 8
  --batch_size "${PRETRAIN_BATCH_SIZE}"
  --epochs "${PRETRAIN_EPOCHS}"
  --lr "${PRETRAIN_LR}"
  --device cuda
  --gradient_checkpointing
  --address_scale 2048
  --segment_types "${SEGMENT_TYPES}"
)

if [[ -f "${PRETRAIN_OUT}/latest_model.pt" ]]; then
  PRETRAIN_ARGS+=(--resume_checkpoint "${PRETRAIN_OUT}/latest_model.pt")
fi
if [[ -n "${MAX_TRAIN_FUNCTIONS}" ]]; then
  PRETRAIN_ARGS+=(--max_train_functions "${MAX_TRAIN_FUNCTIONS}")
fi
if [[ -n "${MAX_VAL_FUNCTIONS}" ]]; then
  PRETRAIN_ARGS+=(--max_val_functions "${MAX_VAL_FUNCTIONS}")
fi

echo "[1/4] pretrain"
"${PYTHON_BIN}" "${SCRIPT_DIR}/pretrain.py" "${PRETRAIN_ARGS[@]}"

echo "[2/4] funcsim finetune"
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
  --seq_len "${SEQ_LEN}" \
  --max_instructions "${MAX_INSTRUCTIONS}" \
  --batch_size "${FUNC_BATCH_SIZE}" \
  --epochs "${FUNC_EPOCHS}" \
  --lr "${FUNC_LR}" \
  --device cuda \
  --gradient_checkpointing \
  --experiment_name "seq${SEQ_LEN}_h384_l6_addrtype_scale2048_instrid_e${FUNC_EPOCHS}" \
  --address_scale 2048 \
  --dataset_fraction "${DATASET_FRACTION}" \
  --segment_types "${SEGMENT_TYPES}"

echo "[3/4] opt-pair evaluation"
"${PYTHON_BIN}" "${SCRIPT_DIR}/evaluate_opt_pairs_paper.py" \
  --function_blocks "${TEST_FUNCTION_BLOCKS}" \
  --funcsim_pairs "${TEST_FUNCSIM_PAIRS}" \
  --vocab "${VOCAB}" \
  --checkpoint "${FUNC_OUT}/best_model.pt" \
  --output "${EVAL_JSON}" \
  --batch_size "${EVAL_BATCH_SIZE}" \
  --poolsizes "${POOLSIZES}" \
  --device cuda

echo "[4/4] summarize"
"${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_aae_pool_results.py" \
  --input "${EVAL_JSON}" \
  --output "${SUMMARY_JSON}"

echo "[instruction-id AAE] done"
echo "pretrain: ${PRETRAIN_OUT}"
echo "finetune: ${FUNC_OUT}"
echo "eval: ${EVAL_JSON}"
echo "summary: ${SUMMARY_JSON}"
