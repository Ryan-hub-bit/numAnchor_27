#!/usr/bin/env zsh
set -euo pipefail

source /path/to/conda/etc/profile.d/conda.sh
conda activate jtrans

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

AAE_ROOT=/path/to/numAnchor/JTrans
OUT_ROOT=/path/to/numAnchor_artifact/numAnchor/jtrans/paper_full_addressaware
PRETRAIN_OUT="${OUT_ROOT}/pretrain_out"
FINETUNE_OUT="${OUT_ROOT}/finetune_out_epoch6_probe_10k"
SUMMARY_JSON="${OUT_ROOT}/summaries/pool_summary_epoch6_probe_10k.json"
EVAL_JSON="${OUT_ROOT}/addressaware_eval_results_epoch6_probe_10k.json"

cd "${AAE_ROOT}"

python "${AAE_ROOT}/finetune.py" \
  --model_type addressaware \
  --data_type json \
  --func_blocks "${OUT_ROOT}/train_json/func_blocks_addr.json" \
  --ground_truth "${OUT_ROOT}/train_json/ground_truth.json" \
  --tokenizer "${OUT_ROOT}/tokenizer" \
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
  --no_projection \
  --save_every 1 \
  --data_ratio 1.0

python "${AAE_ROOT}/evaluate_addressaware_pools.py" \
  --vocab_path "${OUT_ROOT}/tokenizer" \
  --checkpoint "${FINETUNE_OUT}/finetune_epoch_1" \
  --func_blocks "${OUT_ROOT}/test_json/func_blocks_addr.json" \
  --ground_truth "${OUT_ROOT}/test_json/ground_truth.json" \
  --pool_dir "${OUT_ROOT}/eval_pools" \
  --pool_size 10000 \
  --max_len 512 \
  --batch_size 16 \
  --device cuda \
  --output "${EVAL_JSON}"

python /path/to/numAnchor/JTrans/addressaware/summarize_aae_pool_results.py \
  --input "${EVAL_JSON}" \
  --output "${SUMMARY_JSON}"
