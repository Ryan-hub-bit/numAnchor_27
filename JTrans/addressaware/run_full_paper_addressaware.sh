#!/usr/bin/env bash
set -euo pipefail

ROOT="/path/to/numAnchor/JTrans/addressaware"
AAE_ROOT="/path/to/numAnchor/JTrans"
OUT_ROOT="/path/to/numAnchor_artifact/numAnchor/jtrans/paper_full_addressaware"
TRAIN_AA_ROOT="/path/to/numAnchor_artifact/numAnchor/jtrans/official_small_train_aa_full_cfgsplit_plt30"
TEST_AA_ROOT="/path/to/numAnchor_artifact/numAnchor/jtrans/official_small_test_cfgsplit_plt30"

TRAIN_DIR="${OUT_ROOT}/train_json"
TEST_DIR="${OUT_ROOT}/test_json"
TOKENIZER_DIR="${OUT_ROOT}/tokenizer"
POOL_DIR="${OUT_ROOT}/eval_pools"
PRETRAIN_OUT="${OUT_ROOT}/pretrain_out"
FINETUNE_OUT="${OUT_ROOT}/finetune_out"
SUMMARY_DIR="${OUT_ROOT}/summaries"

TRAIN_FUNC_BLOCKS="${TRAIN_DIR}/func_blocks_addr.json"
TRAIN_GT="${TRAIN_DIR}/ground_truth.json"
TEST_FUNC_BLOCKS="${TEST_DIR}/func_blocks_addr.json"
TEST_GT="${TEST_DIR}/ground_truth.json"
VOCAB_PKL="${OUT_ROOT}/vocab_addr.pkl"
EVAL_JSON="${OUT_ROOT}/addressaware_eval_results.json"
SUMMARY_JSON="${SUMMARY_DIR}/pool_summary.json"

PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-10}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-10}"
PRETRAIN_BATCH="${PRETRAIN_BATCH:-32}"
FINETUNE_BATCH="${FINETUNE_BATCH:-16}"
FINETUNE_ACCUM="${FINETUNE_ACCUM:-4}"
EVAL_BATCH="${EVAL_BATCH:-32}"

mkdir -p "${TRAIN_DIR}" "${TEST_DIR}" "${TOKENIZER_DIR}" "${POOL_DIR}" "${PRETRAIN_OUT}" "${FINETUNE_OUT}" "${SUMMARY_DIR}"

source /path/to/conda/etc/profile.d/conda.sh
conda activate jtrans

echo "============================================================"
echo "Full paper-sized address-aware jTrans"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Output root: ${OUT_ROOT}"
echo "============================================================"

if [[ ! -f "${TRAIN_FUNC_BLOCKS}" ]]; then
  echo "[aa] Wrapping train func blocks..."
  python "${ROOT}/wrap_addressaware_func_blocks_for_aae.py" \
    --input "${TRAIN_AA_ROOT}/funcsim_function_blocks.json" \
    --output "${TRAIN_FUNC_BLOCKS}"
fi

if [[ ! -f "${TRAIN_GT}" ]]; then
  echo "[aa] Building train ground truth..."
  python "${ROOT}/group_variant_ids_to_ground_truth.py" \
    --func_blocks "${TRAIN_FUNC_BLOCKS}" \
    --output "${TRAIN_GT}"
fi

if [[ ! -f "${TEST_FUNC_BLOCKS}" ]]; then
  echo "[aa] Wrapping test func blocks..."
  python "${ROOT}/wrap_addressaware_func_blocks_for_aae.py" \
    --input "${TEST_AA_ROOT}/funcsim_function_blocks.json" \
    --output "${TEST_FUNC_BLOCKS}"
fi

if [[ ! -f "${TEST_GT}" ]]; then
  echo "[aa] Building test ground truth..."
  python "${ROOT}/group_variant_ids_to_ground_truth.py" \
    --func_blocks "${TEST_FUNC_BLOCKS}" \
    --output "${TEST_GT}"
fi

if [[ ! -f "${POOL_DIR}/pool_O0_vs_O3_10000.json" ]]; then
  echo "[aa] Creating filtered eval pools..."
  python "${AAE_ROOT}/create_filtered_addressaware_pools.py" \
    --func_blocks "${TEST_FUNC_BLOCKS}" \
    --ground_truth "${TEST_GT}" \
    --output_dir "${POOL_DIR}" \
    --pool_sizes 100 1000 10000
fi

if [[ ! -f "${VOCAB_PKL}" ]]; then
  echo "[aa] Building vocab..."
  python - <<PY
from pathlib import Path
import importlib.util
import sys
sys.path.insert(0, "${AAE_ROOT}/pretrain/address_aware")
module_path = Path("${AAE_ROOT}/pretrain/address_aware/create_vocab.py")
spec = importlib.util.spec_from_file_location("aae_create_vocab", module_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.create_vocab(
    ["${TRAIN_AA_ROOT}/pretrain_train_seq512.txt"],
    "${VOCAB_PKL}",
    max_size=10000,
    min_freq=2,
)
PY
else
  echo "[aa] Reusing ${VOCAB_PKL}"
fi

if [[ ! -f "${TOKENIZER_DIR}/vocab.txt" ]]; then
  echo "[aa] Exporting tokenizer dir..."
  python "${ROOT}/write_wordvocab_tokenizer_dir.py" \
    --vocab_pkl "${VOCAB_PKL}" \
    --output_dir "${TOKENIZER_DIR}" \
    --template_dir "${AAE_ROOT}/pretrain/address_aware"
fi

if [[ ! -f "${PRETRAIN_OUT}/checkpoint_epoch_${PRETRAIN_EPOCHS}/pytorch_model.bin" ]]; then
  echo "[aa] Starting pretrain..."
  python "${AAE_ROOT}/pretrain/address_aware/train_addressaware.py" \
    --train_path "${TRAIN_AA_ROOT}/pretrain_train_seq512.txt" \
    --vocab_path "${VOCAB_PKL}" \
    --output_dir "${PRETRAIN_OUT}" \
    --hidden_size 768 \
    --num_hidden_layers 12 \
    --num_attention_heads 12 \
    --max_len 512 \
    --batch_size "${PRETRAIN_BATCH}" \
    --learning_rate 1e-4 \
    --num_epochs "${PRETRAIN_EPOCHS}" \
    --token_mask_prob 0.15 \
    --instruction_mask_prob 0.0 \
    --masking_strategy token \
    --jtp_weight 1.0 \
    --val_split 0.05 \
    --save_every 1 \
    --num_workers 4 \
    --data_ratio 1.0
else
  echo "[aa] Reusing pretrain checkpoint epoch ${PRETRAIN_EPOCHS}"
fi

if [[ ! -f "${FINETUNE_OUT}/finetune_epoch_${FINETUNE_EPOCHS}/pytorch_model.bin" ]]; then
  echo "[aa] Starting finetune..."
  pushd "${AAE_ROOT}" >/dev/null
  python "${AAE_ROOT}/finetune.py" \
    --model_type addressaware \
    --data_type json \
    --func_blocks "${TRAIN_FUNC_BLOCKS}" \
    --ground_truth "${TRAIN_GT}" \
    --tokenizer "${TOKENIZER_DIR}" \
    --model_path "${PRETRAIN_OUT}/checkpoint_epoch_${PRETRAIN_EPOCHS}" \
    --output_path "${FINETUNE_OUT}" \
    --batch_size "${FINETUNE_BATCH}" \
    --gradient_accumulation_steps "${FINETUNE_ACCUM}" \
    --eval_batch_size 64 \
    --lr 1e-5 \
    --epoch "${FINETUNE_EPOCHS}" \
    --weight_decay 1e-4 \
    --warmup 1000 \
    --loss_type triplet \
    --triplet_margin 0.2 \
    --freeze_cnt 10 \
    --pooling_type cls \
    --no_projection \
    --save_every 1 \
    --data_ratio 1.0
  popd >/dev/null
else
  echo "[aa] Reusing finetune epoch ${FINETUNE_EPOCHS}"
fi

echo "[aa] Running pool evaluation..."
pushd "${AAE_ROOT}" >/dev/null
python "${AAE_ROOT}/evaluate_addressaware_pools.py" \
  --vocab_path "${TOKENIZER_DIR}" \
  --checkpoint "${FINETUNE_OUT}/finetune_epoch_${FINETUNE_EPOCHS}" \
  --func_blocks "${TEST_FUNC_BLOCKS}" \
  --ground_truth "${TEST_GT}" \
  --pool_dir "${POOL_DIR}" \
  --max_len 512 \
  --batch_size "${EVAL_BATCH}" \
  --device cuda \
  --output "${EVAL_JSON}"
popd >/dev/null

python "${ROOT}/summarize_aae_pool_results.py" \
  --input "${EVAL_JSON}" \
  --output "${SUMMARY_JSON}"

echo "[aa] Done"
echo "[aa] Summary: ${SUMMARY_JSON}"
