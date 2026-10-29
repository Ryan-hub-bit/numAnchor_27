#!/usr/bin/env zsh
set -euo pipefail

source /path/to/conda/etc/profile.d/conda.sh
conda activate jtrans

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

POOL_DIR=/path/to/numAnchor_artifact/numAnchor/jtrans/paper_full_addressaware/eval_pools_epoch5_probe_10k_os
mkdir -p "${POOL_DIR}"

if [[ ! -f "${POOL_DIR}/pool_O0_vs_Os_10000.json" ]]; then
  python /path/to/numAnchor/JTrans/create_filtered_addressaware_pools.py \
    --func_blocks /path/to/numAnchor_artifact/numAnchor/jtrans/paper_full_addressaware/test_json/func_blocks_addr.json \
    --ground_truth /path/to/numAnchor_artifact/numAnchor/jtrans/paper_full_addressaware/test_json/ground_truth.json \
    --output_dir "${POOL_DIR}" \
    --pool_sizes 10000 \
    --query_ratio 0.2 \
    --min_instructions 11 \
    --opt_pairs O0:Os O1:Os O2:Os
fi

cd /path/to/numAnchor/JTrans

python /path/to/numAnchor/JTrans/finetune.py \
  --model_type addressaware \
  --data_type json \
  --func_blocks /path/to/numAnchor_artifact/numAnchor/jtrans/paper_full_addressaware/train_json/func_blocks_addr.json \
  --ground_truth /path/to/numAnchor_artifact/numAnchor/jtrans/paper_full_addressaware/train_json/ground_truth.json \
  --tokenizer /path/to/numAnchor_artifact/numAnchor/jtrans/paper_full_addressaware/tokenizer \
  --model_path /path/to/numAnchor_artifact/numAnchor/jtrans/paper_full_addressaware/pretrain_out/checkpoint_epoch_5 \
  --output_path /path/to/numAnchor_artifact/numAnchor/jtrans/paper_full_addressaware/finetune_out_epoch5_probe_10k_os \
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
  --data_ratio 1.0 \
  --target_opt Os \
  --json_opt_levels O0,O1,O2,Os

python /path/to/numAnchor/JTrans/evaluate_addressaware_pools.py \
  --vocab_path /path/to/numAnchor_artifact/numAnchor/jtrans/paper_full_addressaware/tokenizer \
  --checkpoint /path/to/numAnchor_artifact/numAnchor/jtrans/paper_full_addressaware/finetune_out_epoch5_probe_10k_os/finetune_epoch_1 \
  --func_blocks /path/to/numAnchor_artifact/numAnchor/jtrans/paper_full_addressaware/test_json/func_blocks_addr.json \
  --ground_truth /path/to/numAnchor_artifact/numAnchor/jtrans/paper_full_addressaware/test_json/ground_truth.json \
  --pool_dir "${POOL_DIR}" \
  --pool_size 10000 \
  --max_len 512 \
  --batch_size 16 \
  --device cuda \
  --output /path/to/numAnchor_artifact/numAnchor/jtrans/paper_full_addressaware/addressaware_eval_results_epoch5_probe_10k_os.json

python /path/to/numAnchor/JTrans/addressaware/summarize_aae_pool_results.py \
  --input /path/to/numAnchor_artifact/numAnchor/jtrans/paper_full_addressaware/addressaware_eval_results_epoch5_probe_10k_os.json \
  --output /path/to/numAnchor_artifact/numAnchor/jtrans/paper_full_addressaware/summaries/pool_summary_epoch5_probe_10k_os.json
