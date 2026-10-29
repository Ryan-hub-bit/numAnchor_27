#!/usr/bin/env bash
set -euo pipefail

EXP=${EXP:-/path/to/numAnchor_artifact/stateformer_aa_pilot/x64_o0_big_baseline_6x256_u4000_f2400}
BASE=${BASE:-/path/to/numAnchor_artifact/stateformer_aa_pilot/x64_o0_paper_push}
PRE_DATA="$BASE/pretrain/dataset"
DOWNSTREAM_DATA="$BASE/downstream_sparse"
PYROOT=/path/to/numAnchor/stateformer

if [ -f /path/to/conda/etc/profile.d/conda.sh ]; then
  source /path/to/conda/etc/profile.d/conda.sh
else
  export PATH="/path/to/conda/bin:$PATH"
fi

mkdir -p "$EXP"

echo "[info] experiment dir: $EXP"
echo "[info] pretrain data: $PRE_DATA"
echo "[info] downstream data: $DOWNSTREAM_DATA"
date -u '+[info] start %Y-%m-%dT%H:%M:%SZ'

CUDA_VISIBLE_DEVICES=1 \
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
  --save-dir "$EXP/pretrain_baseline_big" \
  > "$EXP/pretrain_baseline_big.log" 2>&1

CUDA_VISIBLE_DEVICES=1 \
PYTHONPATH="$PYROOT" \
conda run -n stateformer python \
  "$PYROOT/train.py" \
  "$DOWNSTREAM_DATA" \
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
  --restore-file "$EXP/pretrain_baseline_big/checkpoint_best.pt" \
  --reset-optimizer \
  --reset-meters \
  --reset-dataloader \
  --seed 7 \
  --save-dir "$EXP/finetune_baseline_big" \
  > "$EXP/finetune_baseline_big.log" 2>&1

date -u '+[info] end %Y-%m-%dT%H:%M:%SZ'
