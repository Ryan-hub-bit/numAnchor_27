#!/usr/bin/env bash
set -euo pipefail

source /path/to/conda/etc/profile.d/conda.sh
conda activate palmtree

cd /path/to/numAnchor

mkdir -p \
  /path/to/numAnchor_artifact/addressaware/funcsim_runs/addr_ep9resume_full_ep3_logs \
  /path/to/numAnchor_artifact/addressaware/funcsim_runs/addr_ep9resume_full_ep3_eval_logs

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" PYTHONUNBUFFERED=1 \
python PalmTree/addressaware/funcsim/train.py \
  --function_blocks /path/to/numAnchor_artifact/stateformer_x64_dataset/funcsim_block_ctx2/pool_function_blocks_10k.json \
  --funcsim_pairs /path/to/numAnchor_artifact/stateformer_x64_dataset/funcsim_block_ctx2/funcsim_pairs.json \
  --vocab /path/to/numAnchor_artifact/addressaware/stateformer_allbinary_ctx2_sectionaware_balanced_900_100/data/vocab.pkl \
  --pretrained_bert /path/to/numAnchor_artifact/addressaware/pretrain_runs/stateformer_allbinary_balanced_900_100_addr_mlm_nsp_768_ep9_cont2_lr3e5_resume/latest_bert.pt \
  --hidden 768 \
  --n_layers 12 \
  --attn_heads 12 \
  --seq_len 420 \
  --max_instructions 50 \
  --embedding_dim 256 \
  --window_size 32 \
  --window_subbatch 256 \
  --batch_size 4 \
  --epochs 3 \
  --lr 3e-5 \
  --lr_bert 3e-6 \
  --warmup_ratio 0.06 \
  --loss infonce \
  --temperature 0.07 \
  --margin 1.0 \
  --negative_samples 0 \
  --dataset_fraction 1.0 \
  --train_split 0.9 \
  --val_split 0.1 \
  --gradient_checkpointing \
  --output_dir /path/to/numAnchor_artifact/addressaware/funcsim_runs/addr_ep9resume_full_ep3 \
  --log_dir /path/to/numAnchor_artifact/addressaware/funcsim_runs/addr_ep9resume_full_ep3_logs \
  --experiment_name addr_ep9resume_full_ep3 \
  --task_name addr_ep9resume_full_ep3 \
  --device cuda \
  --num_workers 2

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" PYTHONUNBUFFERED=1 \
python PalmTree/addressaware/funcsim/evaluate.py \
  --function_blocks /path/to/numAnchor_artifact/stateformer_x64_dataset/funcsim_block_ctx2/pool_function_blocks_10k.json \
  --funcsim_pairs /path/to/numAnchor_artifact/stateformer_x64_dataset/funcsim_block_ctx2/funcsim_pairs.json \
  --vocab /path/to/numAnchor_artifact/addressaware/stateformer_allbinary_ctx2_sectionaware_balanced_900_100/data/vocab.pkl \
  --checkpoint /path/to/numAnchor_artifact/addressaware/funcsim_runs/addr_ep9resume_full_ep3/best_model.pt \
  --hidden 768 \
  --n_layers 12 \
  --attn_heads 12 \
  --embedding_dim 256 \
  --seq_len 420 \
  --max_instructions 50 \
  --window_size 32 \
  --batch_size 16 \
  --task_name addr_ep9resume_full_ep3_10k_q1000 \
  --output /path/to/numAnchor_artifact/addressaware/funcsim_runs/addr_ep9resume_full_ep3/eval_results_10k_q1000.json \
  --log_dir /path/to/numAnchor_artifact/addressaware/funcsim_runs/addr_ep9resume_full_ep3_eval_logs \
  --device cuda \
  --eval_pool /path/to/numAnchor_artifact/stateformer_x64_dataset/funcsim_block_ctx2/pool_ids_10k.json
