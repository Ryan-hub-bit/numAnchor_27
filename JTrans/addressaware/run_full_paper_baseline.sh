#!/usr/bin/env bash
set -euo pipefail

ROOT="/path/to/numAnchor/JTrans/addressaware"
AAE_ROOT="/path/to/numAnchor/JTrans"
OUT_ROOT="/path/to/numAnchor_artifact/numAnchor/jtrans/paper_full_baseline"
TRAIN_EXTRACT="/path/to/numAnchor_artifact/numAnchor/jtrans/official/small_train"
TEST_EXTRACT="/path/to/numAnchor_artifact/numAnchor/jtrans/official/small_test"

PRETRAIN_DIR="${OUT_ROOT}/pretrain"
TRAIN_DIR="${OUT_ROOT}/train_json"
TEST_DIR="${OUT_ROOT}/test_json"
POOL_DIR="${OUT_ROOT}/eval_pools"
FINETUNE_DIR="${OUT_ROOT}/finetune"
SUMMARY_DIR="${OUT_ROOT}/summaries"

PRETRAIN_CORPUS="${PRETRAIN_DIR}/baseline_pretrain_full.txt"
TRAIN_FUNC_BLOCKS="${TRAIN_DIR}/func_blocks_baseline.json"
TRAIN_GT="${TRAIN_DIR}/ground_truth.json"
TEST_FUNC_BLOCKS="${TEST_DIR}/func_blocks_baseline.json"
TEST_GT="${TEST_DIR}/ground_truth.json"
PRETRAIN_OUT="${OUT_ROOT}/pretrain_out"
FINETUNE_OUT="${OUT_ROOT}/finetune_out"
EVAL_JSON="${POOL_DIR}/evaluation_results.json"
SUMMARY_JSON="${SUMMARY_DIR}/pool_summary.json"

PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-10}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-10}"
PRETRAIN_BATCH="${PRETRAIN_BATCH:-32}"
FINETUNE_BATCH="${FINETUNE_BATCH:-32}"
FINETUNE_ACCUM="${FINETUNE_ACCUM:-2}"
EVAL_BATCH="${EVAL_BATCH:-64}"
BASELINE_RESUME_EPOCH="${BASELINE_RESUME_EPOCH:-auto}"

mkdir -p "${PRETRAIN_DIR}" "${TRAIN_DIR}" "${TEST_DIR}" "${POOL_DIR}" "${FINETUNE_DIR}" "${SUMMARY_DIR}"

source /path/to/conda/etc/profile.d/conda.sh
conda activate jtrans

echo "============================================================"
echo "Full paper-sized jTrans baseline"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Output root: ${OUT_ROOT}"
echo "============================================================"

if [[ ! -f "${PRETRAIN_CORPUS}" ]]; then
  echo "[baseline] Generating baseline pretrain corpus..."
  python "${AAE_ROOT}/datautils/convert_pkl_to_text.py" "${TRAIN_EXTRACT}" "${PRETRAIN_CORPUS}" --recursive
else
  echo "[baseline] Reusing ${PRETRAIN_CORPUS}"
fi

if [[ ! -f "${TRAIN_FUNC_BLOCKS}" ]]; then
  echo "[baseline] Building train func_blocks..."
  python "${ROOT}/build_baseline_func_blocks_from_extract.py" \
    --extract_root "${TRAIN_EXTRACT}" \
    --output "${TRAIN_FUNC_BLOCKS}"
fi

if [[ ! -f "${TRAIN_GT}" ]]; then
  echo "[baseline] Building train ground truth..."
  python "${ROOT}/group_variant_ids_to_ground_truth.py" \
    --func_blocks "${TRAIN_FUNC_BLOCKS}" \
    --output "${TRAIN_GT}"
fi

if [[ ! -f "${TEST_FUNC_BLOCKS}" ]]; then
  echo "[baseline] Building test func_blocks..."
  python "${ROOT}/build_baseline_func_blocks_from_extract.py" \
    --extract_root "${TEST_EXTRACT}" \
    --output "${TEST_FUNC_BLOCKS}"
fi

if [[ ! -f "${TEST_GT}" ]]; then
  echo "[baseline] Building test ground truth..."
  python "${ROOT}/group_variant_ids_to_ground_truth.py" \
    --func_blocks "${TEST_FUNC_BLOCKS}" \
    --output "${TEST_GT}"
fi

if [[ ! -f "${POOL_DIR}/pool_10000_O0_vs_O3_filtered.json" ]]; then
  echo "[baseline] Creating filtered eval pools..."
  python "${AAE_ROOT}/create_filtered_baseline_pools.py" \
    --func-blocks "${TEST_FUNC_BLOCKS}" \
    --ground-truth "${TEST_GT}" \
    --output-dir "${POOL_DIR}"
fi

if [[ ! -f "${PRETRAIN_OUT}/checkpoint_epoch_${PRETRAIN_EPOCHS}/pytorch_model.bin" ]]; then
  echo "[baseline] Starting pretrain..."
  python "${AAE_ROOT}/pretrain/baseline/train_baseline.py" \
    --train_path "${PRETRAIN_CORPUS}" \
    --tokenizer_path "${AAE_ROOT}/pretrain/baseline" \
    --output_dir "${PRETRAIN_OUT}" \
    --hidden_size 768 \
    --num_hidden_layers 12 \
    --num_attention_heads 12 \
    --intermediate_size 3072 \
    --max_len 512 \
    --batch_size "${PRETRAIN_BATCH}" \
    --learning_rate 1e-4 \
    --num_epochs "${PRETRAIN_EPOCHS}" \
    --mlm_probability 0.15 \
    --jtp_probability 0.20 \
    --save_every 1 \
    --num_workers 4 \
    --data_ratio 1.0 \
    --fair_token_budget \
    --resume_epoch "${BASELINE_RESUME_EPOCH}"
else
  echo "[baseline] Reusing pretrain checkpoint epoch ${PRETRAIN_EPOCHS}"
fi

if [[ ! -f "${FINETUNE_OUT}/finetune_epoch_${FINETUNE_EPOCHS}/pytorch_model.bin" ]]; then
  echo "[baseline] Starting finetune..."
  pushd "${AAE_ROOT}" >/dev/null
  python "${AAE_ROOT}/finetune.py" \
    --model_type baseline \
    --data_type json \
    --func_blocks "${TRAIN_FUNC_BLOCKS}" \
    --ground_truth "${TRAIN_GT}" \
    --tokenizer "${AAE_ROOT}/pretrain/baseline" \
    --model_path "${PRETRAIN_OUT}/checkpoint_epoch_${PRETRAIN_EPOCHS}" \
    --output_path "${FINETUNE_OUT}" \
    --batch_size "${FINETUNE_BATCH}" \
    --gradient_accumulation_steps "${FINETUNE_ACCUM}" \
    --eval_batch_size 128 \
    --lr 1e-5 \
    --epoch "${FINETUNE_EPOCHS}" \
    --weight_decay 1e-4 \
    --warmup 1000 \
    --loss_type triplet \
    --triplet_margin 0.2 \
    --freeze_cnt 10 \
    --pooling_type cls \
    --save_every 1 \
    --data_ratio 1.0 \
    --fair_token_budget
  popd >/dev/null
else
  echo "[baseline] Reusing finetune epoch ${FINETUNE_EPOCHS}"
fi

echo "[baseline] Running pool evaluation..."
pushd "${AAE_ROOT}" >/dev/null
python "${AAE_ROOT}/evaluate_baseline_pools.py" \
  --model "${FINETUNE_OUT}/finetune_epoch_${FINETUNE_EPOCHS}" \
  --vocab "${AAE_ROOT}/pretrain/baseline" \
  --pool-dir "${POOL_DIR}" \
  --func-blocks "${TEST_FUNC_BLOCKS}" \
  --max-len 512 \
  --batch-size "${EVAL_BATCH}" \
  --device cuda \
  --fair_token_budget
popd >/dev/null

python "${ROOT}/summarize_aae_pool_results.py" \
  --input "${EVAL_JSON}" \
  --output "${SUMMARY_JSON}"

echo "[baseline] Done"
echo "[baseline] Summary: ${SUMMARY_JSON}"
