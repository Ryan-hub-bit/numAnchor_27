#!/usr/bin/env zsh
set -euo pipefail

source /path/to/conda/etc/profile.d/conda.sh
conda activate jtrans

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

POOL_DIR=/path/to/numAnchor_artifact/numAnchor/jtrans/paper_full_baseline/eval_pools_epoch2_probe_10k
mkdir -p "${POOL_DIR}"

if [[ ! -f "${POOL_DIR}/pool_10000_O0_vs_O3_filtered.json" ]]; then
  cp /path/to/numAnchor_artifact/numAnchor/jtrans/paper_full_baseline/eval_pools/pool_10000_O*_vs_O3_filtered.json "${POOL_DIR}/"
  cp /path/to/numAnchor_artifact/numAnchor/jtrans/paper_full_baseline/eval_pools/query_2000_from_pool_10000_O*_vs_O3_filtered.json "${POOL_DIR}/"
fi

cd /path/to/numAnchor/JTrans

python /path/to/numAnchor/JTrans/finetune.py \
  --model_type baseline \
  --data_type json \
  --func_blocks /path/to/numAnchor_artifact/numAnchor/jtrans/paper_full_baseline/train_json/func_blocks_baseline.json \
  --ground_truth /path/to/numAnchor_artifact/numAnchor/jtrans/paper_full_baseline/train_json/ground_truth.json \
  --tokenizer /path/to/numAnchor/JTrans/pretrain/baseline \
  --model_path /path/to/numAnchor_artifact/numAnchor/jtrans/paper_full_baseline/pretrain_out/checkpoint_epoch_2 \
  --output_path /path/to/numAnchor_artifact/numAnchor/jtrans/paper_full_baseline/finetune_out_epoch2_probe_10k \
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

python /path/to/numAnchor/JTrans/evaluate_baseline_pools.py \
  --model /path/to/numAnchor_artifact/numAnchor/jtrans/paper_full_baseline/finetune_out_epoch2_probe_10k/finetune_epoch_1 \
  --vocab /path/to/numAnchor/JTrans/pretrain/baseline \
  --pool-dir "${POOL_DIR}" \
  --func-blocks /path/to/numAnchor_artifact/numAnchor/jtrans/paper_full_baseline/test_json/func_blocks_baseline.json \
  --pool-size 10000 \
  --max-len 512 \
  --batch-size 16 \
  --device cuda \
  --fair_token_budget

python /path/to/numAnchor/JTrans/addressaware/summarize_aae_pool_results.py \
  --input "${POOL_DIR}/evaluation_results.json" \
  --output /path/to/numAnchor_artifact/numAnchor/jtrans/paper_full_baseline/summaries/pool_summary_epoch2_probe_10k.json
