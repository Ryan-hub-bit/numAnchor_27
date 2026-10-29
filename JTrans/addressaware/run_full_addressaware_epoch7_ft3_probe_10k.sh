#!/usr/bin/env zsh
set -euo pipefail

source /path/to/conda/etc/profile.d/conda.sh
conda activate jtrans

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

AAE_ROOT=/path/to/numAnchor/JTrans
OUT_ROOT=/path/to/numAnchor_artifact/numAnchor/jtrans/paper_full_addressaware
TRAIN_AA_ROOT=/path/to/numAnchor_artifact/numAnchor/jtrans/official_small_train_aa_full_cfgsplit_plt30

PRETRAIN_OUT="${OUT_ROOT}/pretrain_out"
FINETUNE_OUT="${OUT_ROOT}/finetune_out_epoch7_ft3_probe_10k"
SUMMARY_JSON="${OUT_ROOT}/summaries/pool_summary_epoch7_ft3_probe_10k.json"
EVAL_JSON="${OUT_ROOT}/addressaware_eval_results_epoch7_ft3_probe_10k.json"

python "${AAE_ROOT}/pretrain/address_aware/train_addressaware.py" \
  --train_path "${TRAIN_AA_ROOT}/pretrain_train_seq512.txt" \
  --vocab_path "${OUT_ROOT}/vocab_addr.pkl" \
  --output_dir "${PRETRAIN_OUT}" \
  --hidden_size 768 \
  --num_hidden_layers 12 \
  --num_attention_heads 12 \
  --max_len 512 \
  --batch_size 32 \
  --learning_rate 1e-4 \
  --num_epochs 7 \
  --token_mask_prob 0.15 \
  --instruction_mask_prob 0.0 \
  --masking_strategy token \
  --jtp_weight 1.0 \
  --val_split 0.05 \
  --save_every 1 \
  --num_workers 4 \
  --data_ratio 1.0

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
  --data_ratio 1.0

python "${AAE_ROOT}/evaluate_addressaware_pools.py" \
  --vocab_path "${OUT_ROOT}/tokenizer" \
  --checkpoint "${FINETUNE_OUT}/finetune_epoch_3" \
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
