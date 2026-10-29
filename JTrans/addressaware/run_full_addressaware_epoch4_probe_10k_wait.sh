#!/usr/bin/env zsh
set -euo pipefail

while [ ! -d /path/to/numAnchor_artifact/numAnchor/jtrans/paper_full_addressaware/pretrain_out/checkpoint_epoch_4 ]; do
  sleep 300
done

while pgrep -f "finetune_out_epoch2_probe_10k" >/dev/null; do
  sleep 300
done

source /path/to/conda/etc/profile.d/conda.sh
conda activate jtrans

export CUDA_VISIBLE_DEVICES=0

cd /path/to/numAnchor/JTrans

python /path/to/numAnchor/JTrans/finetune.py \
  --model_type addressaware \
  --data_type json \
  --func_blocks /path/to/numAnchor_artifact/numAnchor/jtrans/paper_full_addressaware/train_json/func_blocks_addr.json \
  --ground_truth /path/to/numAnchor_artifact/numAnchor/jtrans/paper_full_addressaware/train_json/ground_truth.json \
  --tokenizer /path/to/numAnchor_artifact/numAnchor/jtrans/paper_full_addressaware/tokenizer \
  --model_path /path/to/numAnchor_artifact/numAnchor/jtrans/paper_full_addressaware/pretrain_out/checkpoint_epoch_4 \
  --output_path /path/to/numAnchor_artifact/numAnchor/jtrans/paper_full_addressaware/finetune_out_epoch4_probe_10k \
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

python /path/to/numAnchor/JTrans/evaluate_addressaware_pools.py \
  --vocab_path /path/to/numAnchor_artifact/numAnchor/jtrans/paper_full_addressaware/tokenizer \
  --checkpoint /path/to/numAnchor_artifact/numAnchor/jtrans/paper_full_addressaware/finetune_out_epoch4_probe_10k/finetune_epoch_1 \
  --func_blocks /path/to/numAnchor_artifact/numAnchor/jtrans/paper_full_addressaware/test_json/func_blocks_addr.json \
  --ground_truth /path/to/numAnchor_artifact/numAnchor/jtrans/paper_full_addressaware/test_json/ground_truth.json \
  --pool_dir /path/to/numAnchor_artifact/numAnchor/jtrans/paper_full_addressaware/eval_pools \
  --pool_size 10000 \
  --max_len 512 \
  --batch_size 16 \
  --device cuda \
  --output /path/to/numAnchor_artifact/numAnchor/jtrans/paper_full_addressaware/addressaware_eval_results_epoch4_probe_10k.json

python /path/to/numAnchor/JTrans/addressaware/summarize_aae_pool_results.py \
  --input /path/to/numAnchor_artifact/numAnchor/jtrans/paper_full_addressaware/addressaware_eval_results_epoch4_probe_10k.json \
  --output /path/to/numAnchor_artifact/numAnchor/jtrans/paper_full_addressaware/summaries/pool_summary_epoch4_probe_10k.json
