# Address-Aware Function Similarity

This directory ports the function-similarity pipeline from
`PalmTree_vinili_bert/funcsim` into `PalmTree/addressaware/funcsim`, aligned
with the current address-aware pretrain interface.

## What Changed

- Reused the current address-aware parser semantics:
  - code tokens use `binary_pos + function_pos + bb_pos`
  - `daddr_*` tokens use `section_pos + offset_in_section`
  - legacy `daddr(...)` is canonicalized to `daddr_rodata/bss/data/other`
- Function-similarity windows now pass:
  - `section_kind_ids`
  - `addr_type_ids`
- Internal BERT window length is configurable via `--window_size`
  and should match the pretrain checkpoint sequence length.

## Current Data / Checkpoints

- Current pretrain checkpoint:
  `/path/to/numAnchor_artifact/addressaware/pretrain_runs/stateformer_allbinary_balanced_900_100_addr_mlm_nsp_768_ep5/bert_epoch_2.pt`
- Current vocab:
  `/path/to/numAnchor_artifact/addressaware/stateformer_allbinary_ctx2_sectionaware_balanced_900_100/data/vocab.pkl`
- Pilot funcsim training set:
  `/path/to/numAnchor_artifact/stateformer_x64_dataset/funcsim_block_ctx2/pool_function_blocks_10k.json`
- Funcsim labels:
  `/path/to/numAnchor_artifact/stateformer_x64_dataset/funcsim_block_ctx2/funcsim_pairs.json`
- Pilot eval pool:
  `/path/to/numAnchor_artifact/stateformer_x64_dataset/funcsim_block_ctx2/pool_ids_100.json`
  with
  `/path/to/numAnchor_artifact/stateformer_x64_dataset/funcsim_block_ctx2/pool_function_blocks_100.json`

## Pilot Fine-Tune

```bash
source /path/to/conda/etc/profile.d/conda.sh
conda activate palmtree

CUDA_VISIBLE_DEVICES=0 python PalmTree/addressaware/funcsim/train.py \
  --function_blocks /path/to/numAnchor_artifact/stateformer_x64_dataset/funcsim_block_ctx2/pool_function_blocks_10k.json \
  --funcsim_pairs /path/to/numAnchor_artifact/stateformer_x64_dataset/funcsim_block_ctx2/funcsim_pairs.json \
  --vocab /path/to/numAnchor_artifact/addressaware/stateformer_allbinary_ctx2_sectionaware_balanced_900_100/data/vocab.pkl \
  --pretrained_bert /path/to/numAnchor_artifact/addressaware/pretrain_runs/stateformer_allbinary_balanced_900_100_addr_mlm_nsp_768_ep5/bert_epoch_2.pt \
  --hidden 768 \
  --n_layers 12 \
  --attn_heads 12 \
  --embedding_dim 256 \
  --seq_len 420 \
  --max_instructions 50 \
  --window_size 32 \
  --batch_size 4 \
  --epochs 1 \
  --lr 3e-5 \
  --lr_bert 3e-6 \
  --warmup_ratio 0.06 \
  --loss infonce \
  --temperature 0.07 \
  --negative_samples 0 \
  --dataset_fraction 0.05 \
  --train_split 0.9 \
  --val_split 0.1 \
  --output_dir /path/to/numAnchor_artifact/addressaware/funcsim_runs/addr_epoch2_pilot_df005 \
  --log_dir /path/to/numAnchor_artifact/addressaware/funcsim_runs/addr_epoch2_pilot_df005_logs \
  --experiment_name addr_epoch2_pilot_df005 \
  --task_name addr_epoch2_pilot_df005 \
  --device cuda \
  --num_workers 2 \
  --gradient_checkpointing
```

Pilot fine-tune output:

- checkpoint:
  `/path/to/numAnchor_artifact/addressaware/funcsim_runs/addr_epoch2_pilot_df005/best_model.pt`

## Pilot Fine-Tune Baseline

```bash
source /path/to/conda/etc/profile.d/conda.sh
conda activate palmtree

CUDA_VISIBLE_DEVICES=0 python PalmTree/addressaware/funcsim/train.py \
  --function_blocks /path/to/numAnchor_artifact/stateformer_x64_dataset/funcsim_block_ctx2/pool_function_blocks_10k.json \
  --funcsim_pairs /path/to/numAnchor_artifact/stateformer_x64_dataset/funcsim_block_ctx2/funcsim_pairs.json \
  --vocab /path/to/numAnchor_artifact/addressaware/stateformer_allbinary_ctx2_sectionaware_balanced_900_100/data/vocab.pkl \
  --pretrained_bert /path/to/numAnchor_artifact/addressaware/pretrain_runs/stateformer_allbinary_balanced_900_100_baseline_mlm_nsp_768_ep5/bert_epoch_2.pt \
  --hidden 768 \
  --n_layers 12 \
  --attn_heads 12 \
  --embedding_dim 256 \
  --seq_len 420 \
  --max_instructions 50 \
  --window_size 32 \
  --batch_size 4 \
  --epochs 1 \
  --lr 3e-5 \
  --lr_bert 3e-6 \
  --warmup_ratio 0.06 \
  --loss infonce \
  --temperature 0.07 \
  --negative_samples 0 \
  --dataset_fraction 0.05 \
  --train_split 0.9 \
  --val_split 0.1 \
  --output_dir /path/to/numAnchor_artifact/addressaware/funcsim_runs/baseline_epoch2_pilot_df005 \
  --log_dir /path/to/numAnchor_artifact/addressaware/funcsim_runs/baseline_epoch2_pilot_df005_logs \
  --experiment_name baseline_epoch2_pilot_df005 \
  --task_name baseline_epoch2_pilot_df005 \
  --device cuda \
  --num_workers 2 \
  --gradient_checkpointing
```

Pilot baseline fine-tune output:

- checkpoint:
  `/path/to/numAnchor_artifact/addressaware/funcsim_runs/baseline_epoch2_pilot_df005/best_model.pt`

## Pilot Retrieval Eval

```bash
source /path/to/conda/etc/profile.d/conda.sh
conda activate palmtree

CUDA_VISIBLE_DEVICES=0 python PalmTree/addressaware/funcsim/evaluate.py \
  --function_blocks /path/to/numAnchor_artifact/stateformer_x64_dataset/funcsim_block_ctx2/pool_function_blocks_100.json \
  --funcsim_pairs /path/to/numAnchor_artifact/stateformer_x64_dataset/funcsim_block_ctx2/funcsim_pairs.json \
  --vocab /path/to/numAnchor_artifact/addressaware/stateformer_allbinary_ctx2_sectionaware_balanced_900_100/data/vocab.pkl \
  --checkpoint /path/to/numAnchor_artifact/addressaware/funcsim_runs/addr_epoch2_pilot_df005/best_model.pt \
  --hidden 768 \
  --n_layers 12 \
  --attn_heads 12 \
  --embedding_dim 256 \
  --seq_len 420 \
  --max_instructions 50 \
  --window_size 32 \
  --batch_size 16 \
  --task_name addr_epoch2_pilot_df005_100 \
  --output /path/to/numAnchor_artifact/addressaware/funcsim_runs/addr_epoch2_pilot_df005/eval_results_100.json \
  --log_dir /path/to/numAnchor_artifact/addressaware/funcsim_runs/addr_epoch2_pilot_df005_eval_logs \
  --device cuda \
  --eval_pool /path/to/numAnchor_artifact/stateformer_x64_dataset/funcsim_block_ctx2/pool_ids_100.json
```

The command writes retrieval metrics, including `Recall@1/5/10/20` and `MRR`,
to the configured JSON output. Metric values are intentionally not listed here;
they should be generated from the packaged artifact during review.

Result JSON:

- `/path/to/numAnchor_artifact/addressaware/funcsim_runs/addr_epoch2_pilot_df005/eval_results_100.json`

## Pilot Baseline Retrieval Eval

```bash
source /path/to/conda/etc/profile.d/conda.sh
conda activate palmtree

CUDA_VISIBLE_DEVICES=0 python PalmTree/addressaware/funcsim/evaluate.py \
  --function_blocks /path/to/numAnchor_artifact/stateformer_x64_dataset/funcsim_block_ctx2/pool_function_blocks_100.json \
  --funcsim_pairs /path/to/numAnchor_artifact/stateformer_x64_dataset/funcsim_block_ctx2/funcsim_pairs.json \
  --vocab /path/to/numAnchor_artifact/addressaware/stateformer_allbinary_ctx2_sectionaware_balanced_900_100/data/vocab.pkl \
  --checkpoint /path/to/numAnchor_artifact/addressaware/funcsim_runs/baseline_epoch2_pilot_df005/best_model.pt \
  --hidden 768 \
  --n_layers 12 \
  --attn_heads 12 \
  --embedding_dim 256 \
  --seq_len 420 \
  --max_instructions 50 \
  --window_size 32 \
  --batch_size 16 \
  --task_name baseline_epoch2_pilot_df005_100 \
  --output /path/to/numAnchor_artifact/addressaware/funcsim_runs/baseline_epoch2_pilot_df005/eval_results_100.json \
  --log_dir /path/to/numAnchor_artifact/addressaware/funcsim_runs/baseline_epoch2_pilot_df005_eval_logs \
  --device cuda \
  --eval_pool /path/to/numAnchor_artifact/stateformer_x64_dataset/funcsim_block_ctx2/pool_ids_100.json
```

The command writes retrieval metrics, including `Recall@1/5/10/20` and `MRR`,
to the configured JSON output. Metric values are intentionally not listed here;
they should be generated from the packaged artifact during review.

Result JSON:

- `/path/to/numAnchor_artifact/addressaware/funcsim_runs/baseline_epoch2_pilot_df005/eval_results_100.json`
