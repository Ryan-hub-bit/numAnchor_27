#!/usr/bin/env zsh
set -euo pipefail

source /path/to/conda/etc/profile.d/conda.sh
conda activate jtrans

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

AAE_ROOT=/path/to/numAnchor/JTrans
OUT_ROOT=/path/to/numAnchor_artifact/numAnchor/jtrans/paper_full_baseline
POOL_DIR="${OUT_ROOT}/eval_pools_epoch6_probe_10k"
PRETRAIN_OUT="${OUT_ROOT}/pretrain_out"
FINETUNE_OUT="${OUT_ROOT}/finetune_out_epoch6_probe_10k"
SUMMARY_JSON="${OUT_ROOT}/summaries/pool_summary_epoch6_probe_10k.json"

mkdir -p "${POOL_DIR}"

if [[ ! -f "${POOL_DIR}/pool_10000_O0_vs_O3_filtered.json" ]]; then
  cp "${OUT_ROOT}"/eval_pools/pool_10000_O*_vs_O3_filtered.json "${POOL_DIR}/"
  cp "${OUT_ROOT}"/eval_pools/query_2000_from_pool_10000_O*_vs_O3_filtered.json "${POOL_DIR}/"
fi

python "${AAE_ROOT}/pretrain/baseline/train_baseline.py" \
  --train_path "${OUT_ROOT}/pretrain/baseline_pretrain_full.txt" \
  --tokenizer_path "${AAE_ROOT}/pretrain/baseline" \
  --output_dir "${PRETRAIN_OUT}" \
  --hidden_size 768 \
  --num_hidden_layers 12 \
  --num_attention_heads 12 \
  --intermediate_size 3072 \
  --max_len 512 \
  --batch_size 32 \
  --learning_rate 1e-4 \
  --num_epochs 6 \
  --mlm_probability 0.15 \
  --jtp_probability 0.20 \
  --save_every 1 \
  --num_workers 4 \
  --data_ratio 1.0 \
  --fair_token_budget \
  --resume_epoch auto

cd "${AAE_ROOT}"

python "${AAE_ROOT}/finetune.py" \
  --model_type baseline \
  --data_type json \
  --func_blocks "${OUT_ROOT}/train_json/func_blocks_baseline.json" \
  --ground_truth "${OUT_ROOT}/train_json/ground_truth.json" \
  --tokenizer "${AAE_ROOT}/pretrain/baseline" \
  --model_path "${PRETRAIN_OUT}/checkpoint_epoch_6" \
  --output_path "${FINETUNE_OUT}" \
  --batch_size 4 \
  --gradient_accumulation_steps 16 \
  --eval_batch_size 16 \
  --lr 1e-5 \
  --epoch 1 \
  --weight_decay 1e-4 \
  --warmup 1000 \
  --loss_type triplet \
  --triplet_margin 0.2 \
  --freeze_cnt 10 \
  --pooling_type cls \
  --save_every 1 \
  --data_ratio 1.0 \
  --fair_token_budget

python "${AAE_ROOT}/evaluate_baseline_pools.py" \
  --model "${FINETUNE_OUT}/finetune_epoch_1" \
  --vocab "${AAE_ROOT}/pretrain/baseline" \
  --pool-dir "${POOL_DIR}" \
  --func-blocks "${OUT_ROOT}/test_json/func_blocks_baseline.json" \
  --pool-size 10000 \
  --max-len 512 \
  --batch-size 16 \
  --device cuda \
  --fair_token_budget

python /path/to/numAnchor/JTrans/addressaware/summarize_aae_pool_results.py \
  --input "${POOL_DIR}/evaluation_results.json" \
  --output "${SUMMARY_JSON}"
