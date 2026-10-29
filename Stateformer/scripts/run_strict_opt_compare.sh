#!/usr/bin/env bash
set -euo pipefail

OPT_LEVEL=${OPT_LEVEL:-O0}
MODE=${MODE:-aa}
BASE_ROOT=${BASE_ROOT:-/path/to/numAnchor_artifact/stateformer_aa_pilot/strict_opt_common}
OPT_BASE="$BASE_ROOT/$OPT_LEVEL"
SRC_ROOT=${SRC_ROOT:-$OPT_BASE/src}
PRE_DIR=${PRE_DIR:-$OPT_BASE/pretrain}
DOWNSTREAM_DIR=${DOWNSTREAM_DIR:-$OPT_BASE/downstream_sparse}
EXP_ROOT=${EXP_ROOT:-/path/to/numAnchor_artifact/stateformer_aa_pilot/strict_opt_runs}
MODEL_TAG=${MODEL_TAG:-6x256_u4000_f2400}
EXP=${EXP:-$EXP_ROOT/${OPT_LEVEL,,}_${MODE}_$MODEL_TAG}
GPU=${GPU:-1}
FUNCTIONS_PER_BINARY=${FUNCTIONS_PER_BINARY:-12}
MAX_EMU_INSTRUCTIONS=${MAX_EMU_INSTRUCTIONS:-256}
SKIP_TRAIN=${SKIP_TRAIN:-0}
PYROOT=/path/to/numAnchor/Stateformer

case "$OPT_LEVEL" in
  O0|O1|O2|O3) ;;
  *)
    echo "[error] OPT_LEVEL must be one of O0/O1/O2/O3, got: $OPT_LEVEL" >&2
    exit 1
    ;;
esac

case "$MODE" in
  aa)
    PRE_AA_ARGS=(--use-address-aware-embedding --aa-init-scale 0.25)
    FINETUNE_AA_ARGS=(--use-address-aware-embedding --aa-init-scale 0.25)
    PRE_SAVE_SUBDIR=pretrain_aa_additive
    FINETUNE_SAVE_SUBDIR=finetune_aa_additive
    PRE_LOG_NAME=pretrain_aa_additive.log
    FINETUNE_LOG_NAME=finetune_aa_additive.log
    ;;
  baseline)
    PRE_AA_ARGS=()
    FINETUNE_AA_ARGS=()
    PRE_SAVE_SUBDIR=pretrain_baseline
    FINETUNE_SAVE_SUBDIR=finetune_baseline
    PRE_LOG_NAME=pretrain_baseline.log
    FINETUNE_LOG_NAME=finetune_baseline.log
    ;;
  *)
    echo "[error] MODE must be aa or baseline, got: $MODE" >&2
    exit 1
    ;;
esac

if [ ! -d "$SRC_ROOT" ]; then
  echo "[error] missing source dir: $SRC_ROOT" >&2
  exit 1
fi

if [ -f /path/to/conda/etc/profile.d/conda.sh ]; then
  source /path/to/conda/etc/profile.d/conda.sh
else
  export PATH="/path/to/conda/bin:$PATH"
fi

mkdir -p "$PRE_DIR" "$EXP"

echo "[info] opt level: $OPT_LEVEL"
echo "[info] mode: $MODE"
echo "[info] src root: $SRC_ROOT"
echo "[info] pretrain dir: $PRE_DIR"
echo "[info] downstream dir: $DOWNSTREAM_DIR"
echo "[info] experiment dir: $EXP"
echo "[info] skip train: $SKIP_TRAIN"
date -u '+[info] start %Y-%m-%dT%H:%M:%SZ'

if [ ! -f "$PRE_DIR/pilot_summary.json" ]; then
  PYTHONPATH="$PYROOT" \
  conda run -n stateformer python \
    "$PYROOT/command/pretrain/build_unified_aa_pilot.py" \
    --binary-root "$SRC_ROOT" \
    --output-dir "$PRE_DIR" \
    --max-binaries 100000 \
    --functions-per-binary "$FUNCTIONS_PER_BINARY" \
    --max-emu-instructions "$MAX_EMU_INSTRUCTIONS" || {
      echo "[error] pretrain data build failed for $OPT_LEVEL" >&2
      exit 1
    }
fi

if [ ! -f "$DOWNSTREAM_DIR/manifest.json" ]; then
  PYTHONPATH="$PYROOT" \
  conda run -n stateformer python \
    "$PYROOT/command/finetune/build_type_aa_smoke_dataset.py" \
    --unified-json-dir "$PRE_DIR/unified_json" \
    --source-dict-dir "$PRE_DIR/dataset/dicts" \
    --output-dir "$DOWNSTREAM_DIR" \
    --supervision-mode sparse_register_var \
    --min-positive-instructions 2 \
    --min-tokens 30 \
    --max-tokens 2048 || {
      echo "[error] downstream data build failed for $OPT_LEVEL" >&2
      exit 1
    }
fi

if [ "$SKIP_TRAIN" = "1" ]; then
  date -u '+[info] data-ready %Y-%m-%dT%H:%M:%SZ'
  exit 0
fi

PRE_DATA="$PRE_DIR/dataset"
PRE_SAVE_DIR="$EXP/$PRE_SAVE_SUBDIR"
FINETUNE_SAVE_DIR="$EXP/$FINETUNE_SAVE_SUBDIR"

CUDA_VISIBLE_DEVICES="$GPU" \
PYTHONPATH="$PYROOT" \
conda run -n stateformer python \
  "$PYROOT/train.py" \
  "$PRE_DATA" \
  --task masked_lm_mf_aa_smoke \
  --criterion masked_lm_mf \
  --arch roberta_mf_nau \
  --encoder-layers 6 \
  --encoder-embed-dim 256 \
  --encoder-ffn-embed-dim 1024 \
  --encoder-attention-heads 8 \
  --dropout 0.1 \
  --attention-dropout 0.1 \
  --activation-dropout 0.0 \
  --pooler-dropout 0.0 \
  --tokens-per-sample 2048 \
  --mask-prob 0.15 \
  --max-sentences 2 \
  --max-sentences-valid 2 \
  --num-workers 1 \
  --optimizer adam \
  --adam-betas '(0.9,0.98)' \
  --adam-eps 1e-6 \
  --clip-norm 0.0 \
  --lr 1e-4 \
  --lr-scheduler polynomial_decay \
  --total-num-update 4000 \
  --max-update 4000 \
  --warmup-updates 200 \
  --weight-decay 0.01 \
  --log-format json \
  --log-interval 20 \
  --validate-interval-updates 100 \
  --save-interval-updates 200 \
  "${PRE_AA_ARGS[@]}" \
  --save-dir "$PRE_SAVE_DIR" \
  > "$EXP/$PRE_LOG_NAME" 2>&1

CUDA_VISIBLE_DEVICES="$GPU" \
PYTHONPATH="$PYROOT" \
conda run -n stateformer python \
  "$PYROOT/train.py" \
  "$DOWNSTREAM_DIR" \
  --task data_structure_mf_aa_smoke \
  --criterion data_structure_mf_aa_smoke \
  --arch roberta_mf_nau \
  --classification-head-name data_structure_head \
  --encoder-layers 6 \
  --encoder-embed-dim 256 \
  --encoder-ffn-embed-dim 1024 \
  --encoder-attention-heads 8 \
  --dropout 0.1 \
  --attention-dropout 0.1 \
  --activation-dropout 0.0 \
  --pooler-dropout 0.0 \
  --max-positions 2048 \
  --max-sentences 2 \
  --max-sentences-valid 2 \
  --skip-invalid-size-inputs-valid-test \
  --num-workers 0 \
  --optimizer adam \
  --adam-betas '(0.9,0.98)' \
  --adam-eps 1e-6 \
  --clip-norm 0.0 \
  --lr 1e-4 \
  --lr-scheduler polynomial_decay \
  --total-num-update 2400 \
  --max-update 2400 \
  --warmup-updates 100 \
  --weight-decay 0.01 \
  --log-format json \
  --log-interval 20 \
  --validate-interval-updates 80 \
  --save-interval-updates 80 \
  --best-checkpoint-metric F1 \
  --maximize-best-checkpoint-metric \
  --class-weighting sqrt_inverse \
  --class-weight-max 10.0 \
  "${FINETUNE_AA_ARGS[@]}" \
  --restore-file "$PRE_SAVE_DIR/checkpoint_best.pt" \
  --reset-optimizer \
  --reset-meters \
  --reset-dataloader \
  --seed 7 \
  --save-dir "$FINETUNE_SAVE_DIR" \
  > "$EXP/$FINETUNE_LOG_NAME" 2>&1

date -u '+[info] end %Y-%m-%dT%H:%M:%SZ'
