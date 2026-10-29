#!/usr/bin/env bash
set -euo pipefail

ROOT="/path/to/numAnchor"
TRAIN_PY="$ROOT/PalmTree/addressaware/train.py"
PROBE_DIR="$ROOT/PalmTree/addressaware/probing"

DATA_DIR="/path/to/numAnchor_artifact/addressaware/stateformer_allbinary_ctx2_sectionaware_balanced_900_100/data"
RECORDS_ROOT="/path/to/numAnchor_artifact/stateformer_x64_dataset/funcsim_block_ctx2"
FUNCTION_BLOCKS="$RECORDS_ROOT/pool_function_blocks_1k.json"
FUNCSIM_PAIRS="$RECORDS_ROOT/funcsim_pairs_q30_gt3_showcase_r80.json"
VOCAB="$DATA_DIR/vocab.pkl"

BASE_RUN_OLD="/path/to/numAnchor_artifact/addressaware/pretrain_runs/stateformer_allbinary_balanced_900_100_baseline_mlm_nsp_768_ep5"
AA_RUN_OLD="/path/to/numAnchor_artifact/addressaware/pretrain_runs/stateformer_allbinary_balanced_900_100_addr_mlm_nsp_768_ep5"

BASE_RUN_NEW="/path/to/numAnchor_artifact/addressaware/pretrain_runs/stateformer_allbinary_balanced_900_100_baseline_mlm_nsp_768_ep7_cont2_lr3e5"
AA_RUN_NEW="/path/to/numAnchor_artifact/addressaware/pretrain_runs/stateformer_allbinary_balanced_900_100_addr_mlm_nsp_768_ep7_cont2_lr3e5"

BASE_LOG_NEW="${BASE_RUN_NEW}_logs"
AA_LOG_NEW="${AA_RUN_NEW}_logs"

HIER_OUT="/path/to/numAnchor_artifact/addressaware/pretrain_runs/stateformer_allbinary_balanced_900_100_instr_hierarchy_probe_epoch7_1k_poa_mlp.json"
ANCHOR_OUT="/path/to/numAnchor_artifact/addressaware/pretrain_runs/stateformer_allbinary_balanced_900_100_full_binary_anchor_linear_binary_epoch7.json"
DREF_OUT="/path/to/numAnchor_artifact/addressaware/pretrain_runs/stateformer_allbinary_balanced_900_100_dref_probe_epoch7_1k.json"

GPU_MEM_MAX_MB="${GPU_MEM_MAX_MB:-5000}"
GPU_UTIL_MAX="${GPU_UTIL_MAX:-20}"
GPU_SLEEP_SEC="${GPU_SLEEP_SEC:-120}"
CONT_LR="${CONT_LR:-3e-5}"

log() {
  printf '[%s] %s\n' "$(date -u '+%Y-%m-%d %H:%M:%S UTC')" "$*"
}

wait_for_gpu() {
  while true; do
    while IFS=, read -r idx mem util; do
      idx="$(echo "$idx" | xargs)"
      mem="$(echo "$mem" | xargs)"
      util="$(echo "$util" | xargs)"
      if [[ "$mem" -le "$GPU_MEM_MAX_MB" && "$util" -le "$GPU_UTIL_MAX" ]]; then
        echo "$idx"
        return 0
      fi
    done < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits)
    log "No suitable GPU yet; waiting ${GPU_SLEEP_SEC}s (threshold: mem<=${GPU_MEM_MAX_MB}MB util<=${GPU_UTIL_MAX}%)" >&2
    sleep "$GPU_SLEEP_SEC"
  done
}

run_baseline_continuation() {
  if [[ -f "${BASE_RUN_NEW}/bert_epoch_2.pt" ]]; then
    log "Baseline continuation already finished; skipping."
    return 0
  fi
  local gpu
  gpu="$(wait_for_gpu)"
  log "Launching baseline continuation on GPU ${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" python "$TRAIN_PY" \
    --cfg_train "$DATA_DIR/train_cfg.txt" \
    --cfg_val "$DATA_DIR/val_cfg.txt" \
    --vocab "$VOCAB" \
    --hidden 768 \
    --layers 12 \
    --attn_heads 12 \
    --seq_len 32 \
    --dropout 0.1 \
    --epochs 2 \
    --batch_size 256 \
    --lr "$CONT_LR" \
    --warmup_steps 1000 \
    --num_workers 4 \
    --early_stopping_patience 5 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing \
    --amp \
    --token_mask_prob 0.15 \
    --instruction_mask_prob 0.15 \
    --data_percentage 1.0 \
    --val_percentage 1.0 \
    --enable_nsp_cfg \
    --enable_mlm \
    --disable_address_embedding \
    --disable_var_embedding \
    --output_dir "$BASE_RUN_NEW" \
    --log_dir "$BASE_LOG_NEW" \
    --init_checkpoint "${BASE_RUN_OLD}/checkpoint_epoch_5.pt" \
    --cuda \
    --seed 42 \
    --pair_max_instructions 2
}

run_aa_continuation() {
  if [[ -f "${AA_RUN_NEW}/bert_epoch_2.pt" ]]; then
    log "Address-aware continuation already finished; skipping."
    return 0
  fi
  local gpu
  gpu="$(wait_for_gpu)"
  log "Launching address-aware continuation on GPU ${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" python "$TRAIN_PY" \
    --cfg_train "$DATA_DIR/train_cfg.txt" \
    --cfg_val "$DATA_DIR/val_cfg.txt" \
    --vocab "$VOCAB" \
    --hidden 768 \
    --layers 12 \
    --attn_heads 12 \
    --seq_len 32 \
    --dropout 0.1 \
    --epochs 2 \
    --batch_size 256 \
    --lr "$CONT_LR" \
    --warmup_steps 1000 \
    --num_workers 4 \
    --early_stopping_patience 5 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing \
    --amp \
    --token_mask_prob 0.15 \
    --instruction_mask_prob 0.15 \
    --data_percentage 1.0 \
    --val_percentage 1.0 \
    --enable_nsp_cfg \
    --enable_mlm \
    --use_address_embedding \
    --use_var_embedding \
    --output_dir "$AA_RUN_NEW" \
    --log_dir "$AA_LOG_NEW" \
    --init_checkpoint "${AA_RUN_OLD}/checkpoint_epoch_5.pt" \
    --cuda \
    --seed 42 \
    --pair_max_instructions 2
}

run_hierarchy_probe() {
  if [[ -f "$HIER_OUT" ]]; then
    log "Hierarchy probe output already exists; skipping."
    return 0
  fi
  local gpu
  gpu="$(wait_for_gpu)"
  log "Running instruction-level hierarchy probe on GPU ${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" python "$PROBE_DIR/instruction_level_hierarchy_probe_eval.py" \
    --function_blocks "$FUNCTION_BLOCKS" \
    --function_records_dir "$RECORDS_ROOT" \
    --vocab "$VOCAB" \
    --baseline_checkpoint "${BASE_RUN_NEW}/bert_epoch_2.pt" \
    --aa_checkpoint "${AA_RUN_NEW}/bert_epoch_2.pt" \
    --output "$HIER_OUT" \
    --repr_mode pooled_opcode_address \
    --probe_head mlp \
    --probe_hidden_dim 512 \
    --probe_dropout 0.1 \
    --hidden 768 \
    --n_layers 12 \
    --attn_heads 12 \
    --seq_len 32 \
    --batch_size 128 \
    --device cuda \
    --probe_epochs 200 \
    --probe_lr 0.001 \
    --probe_weight_decay 0.0001 \
    --position_buckets 4 \
    --max_train_units 6000 \
    --max_test_units 3000
}

run_anchor_probe() {
  if [[ -f "$ANCHOR_OUT" ]]; then
    log "Anchor probe output already exists; skipping."
    return 0
  fi
  local gpu
  gpu="$(wait_for_gpu)"
  log "Running full-binary anchor probe on GPU ${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" python "$PROBE_DIR/full_binary_anchor_linear_binary_eval.py" \
    --records_root "$RECORDS_ROOT" \
    --vocab "$VOCAB" \
    --baseline_checkpoint "${BASE_RUN_NEW}/bert_epoch_2.pt" \
    --aa_checkpoint "${AA_RUN_NEW}/bert_epoch_2.pt" \
    --output "$ANCHOR_OUT" \
    --hidden 768 \
    --n_layers 12 \
    --attn_heads 12 \
    --seq_len 32 \
    --max_train_pairs 20000 \
    --max_test_pairs 5000 \
    --probe_epochs 50 \
    --probe_lr 0.001 \
    --probe_weight_decay 0.0001 \
    --batch_size 128 \
    --device cuda
}

run_dref_probe() {
  if [[ -f "$DREF_OUT" ]]; then
    log "DRef probe output already exists; skipping."
    return 0
  fi
  local gpu
  gpu="$(wait_for_gpu)"
  log "Running dref probe on GPU ${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" python "$PROBE_DIR/probe_eval.py" \
    --function_blocks "$FUNCTION_BLOCKS" \
    --function_records_dir "$RECORDS_ROOT" \
    --funcsim_pairs "$FUNCSIM_PAIRS" \
    --vocab "$VOCAB" \
    --baseline_checkpoint "${BASE_RUN_NEW}/bert_epoch_2.pt" \
    --aa_checkpoint "${AA_RUN_NEW}/bert_epoch_2.pt" \
    --output "$DREF_OUT" \
    --tasks same_dref dref_retrieval same_stack_offset \
    --hidden 768 \
    --n_layers 12 \
    --attn_heads 12 \
    --seq_len 32 \
    --max_block_instructions 8 \
    --max_train_pairs 2000 \
    --max_test_pairs 1000 \
    --max_train_queries 500 \
    --max_test_queries 200 \
    --retrieval_pool_size 32 \
    --test_fraction 0.3 \
    --split_seed 42 \
    --few_shots 8 32 128 \
    --few_shot_seeds 0 1 2 \
    --probe_epochs 200 \
    --probe_lr 0.001 \
    --probe_weight_decay 0.0001 \
    --jump_target_opcode mov \
    --batch_size 128 \
    --device cuda
}

main() {
  source /path/to/conda/etc/profile.d/conda.sh
  conda activate palmtree
  export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

  log "Starting epoch-7 continuation and reprobe pipeline"
  run_baseline_continuation
  run_aa_continuation
  run_hierarchy_probe
  run_anchor_probe
  run_dref_probe
  log "Pipeline complete"
}

main "$@"
