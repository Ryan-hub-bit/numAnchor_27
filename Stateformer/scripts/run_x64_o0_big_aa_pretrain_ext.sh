#!/usr/bin/env bash
set -euo pipefail

EXP=${EXP:-/path/to/numAnchor_artifact/stateformer_aa_pilot/x64_o0_big_aa_pretrain_ext_live}
BASE=${BASE:-/path/to/numAnchor_artifact/stateformer_aa_pilot/x64_o0_paper_push}
INIT_CKPT=${INIT_CKPT:-/path/to/numAnchor_artifact/stateformer_aa_pilot/x64_o0_big_aa_6x256_u4000_f2400_live/pretrain_aa_big/checkpoint_best.pt}
GPU=${GPU:-0}
LR=${LR:-5e-5}
MAX_UPDATE=${MAX_UPDATE:-4000}
TOTAL_NUM_UPDATE=${TOTAL_NUM_UPDATE:-4000}
WARMUP_UPDATES=${WARMUP_UPDATES:-100}
PYROOT=/path/to/numAnchor/stateformer
PRE_DATA="$BASE/pretrain/dataset"

if [ -f /path/to/conda/etc/profile.d/conda.sh ]; then
  source /path/to/conda/etc/profile.d/conda.sh
else
  export PATH="/path/to/conda/bin:$PATH"
fi

mkdir -p "$EXP"

echo "[info] experiment dir: $EXP"
echo "[info] pretrain data: $PRE_DATA"
echo "[info] init checkpoint: $INIT_CKPT"
echo "[info] gpu: $GPU"
echo "[info] lr: $LR"
echo "[info] max_update: $MAX_UPDATE"
date -u '+[info] start %Y-%m-%dT%H:%M:%SZ'

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
  --lr "$LR" \
  --lr-scheduler polynomial_decay \
  --total-num-update "$TOTAL_NUM_UPDATE" \
  --max-update "$MAX_UPDATE" \
  --warmup-updates "$WARMUP_UPDATES" \
  --weight-decay 0.01 \
  --log-format json \
  --log-interval 20 \
  --validate-interval-updates 100 \
  --save-interval-updates 200 \
  --use-address-aware-embedding \
  --aa-init-scale 0.25 \
  --restore-file "$INIT_CKPT" \
  --reset-optimizer \
  --reset-meters \
  --reset-dataloader \
  --save-dir "$EXP/pretrain_aa_ext" \
  > "$EXP/pretrain_aa_ext.log" 2>&1

date -u '+[info] end %Y-%m-%dT%H:%M:%SZ'
