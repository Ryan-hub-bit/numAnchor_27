#!/usr/bin/env zsh
set -euo pipefail

source /path/to/conda/etc/profile.d/conda.sh
conda activate jtrans

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

AAE_ROOT=/path/to/numAnchor/JTrans
OUT_ROOT=/path/to/numAnchor_artifact/numAnchor/jtrans/paper_full_addressaware

PRETRAIN_OUT="${OUT_ROOT}/pretrain_out"
FINETUNE_OUT="${OUT_ROOT}/finetune_out_epoch7_ft3_probe_10k_os"
POOL_DIR="${OUT_ROOT}/eval_pools_epoch7_ft3_probe_10k_os"
SUMMARY_JSON="${OUT_ROOT}/summaries/pool_summary_epoch7_ft3_probe_10k_os.json"
EVAL_JSON="${OUT_ROOT}/addressaware_eval_results_epoch7_ft3_probe_10k_os.json"

mkdir -p "${POOL_DIR}"

if [[ ! -f "${POOL_DIR}/pool_O0_vs_Os_10000.json" ]]; then
  python "${AAE_ROOT}/create_filtered_addressaware_pools.py" \
    --func_blocks "${OUT_ROOT}/test_json/func_blocks_addr.json" \
    --ground_truth "${OUT_ROOT}/test_json/ground_truth.json" \
    --output_dir "${POOL_DIR}" \
    --pool_sizes 10000 \
    --query_ratio 0.2 \
    --min_instructions 11 \
    --opt_pairs O0:Os O1:Os O2:Os
fi

cd "${AAE_ROOT}"

python "${AAE_ROOT}/finetune.py" \
  --model_type addressaware \
  --data_type json \
  --func_blocks "${OUT_ROOT}/train_json/func_blocks_addr.json" \
  --ground_truth "${OUT_ROOT}/train_json/ground_truth.json" \
  --tokenizer "${OUT_ROOT}/tokenizer" \
  --model_path "${PRETRAIN_OUT}/checkpoint_epoch_7" \
  --output_path "${FINETUNE_OUT}" \
  --batch_size 4 \
  --gradient_accumulation_steps 16 \
  --eval_batch_size 16 \
  --lr 1e-5 \
  --epoch 3 \
  --weight_decay 1e-4 \
  --warmup 1000 \
  --loss_type triplet \
  --triplet_margin 0.2 \
  --freeze_cnt 10 \
  --pooling_type cls \
  --no_projection \
  --save_every 1 \
  --data_ratio 1.0 \
  --target_opt Os \
  --json_opt_levels O0,O1,O2,Os

python "${AAE_ROOT}/evaluate_addressaware_pools.py" \
  --vocab_path "${OUT_ROOT}/tokenizer" \
  --checkpoint "${FINETUNE_OUT}/finetune_epoch_3" \
  --func_blocks "${OUT_ROOT}/test_json/func_blocks_addr.json" \
  --ground_truth "${OUT_ROOT}/test_json/ground_truth.json" \
  --pool_dir "${POOL_DIR}" \
  --pool_size 10000 \
  --max_len 512 \
  --batch_size 16 \
  --device cuda \
  --output "${EVAL_JSON}"

python /path/to/numAnchor/JTrans/addressaware/summarize_aae_pool_results.py \
  --input "${EVAL_JSON}" \
  --output "${SUMMARY_JSON}"
