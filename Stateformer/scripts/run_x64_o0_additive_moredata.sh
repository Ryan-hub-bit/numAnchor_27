#!/usr/bin/env bash
set -euo pipefail

RAW_ROOT=${RAW_ROOT:-/path/to/numAnchor_artifact/stateformer_x64_dataset/raw_input/x86-64}
BASE=${BASE:-/path/to/numAnchor_artifact/stateformer_aa_pilot/x64_o0_additive_moredata}
SRC_ROOT=${SRC_ROOT:-$BASE/src}
EXP=${EXP:-${BASE}_exp_6x256_u4000_f2400}
GPU=${GPU:-1}
FUNCTIONS_PER_BINARY=${FUNCTIONS_PER_BINARY:-12}
MAX_EMU_INSTRUCTIONS=${MAX_EMU_INSTRUCTIONS:-256}
OPT_SUBSTR=${OPT_SUBSTR:--O0}
PYROOT=/path/to/numAnchor/stateformer

if [ -f /path/to/conda/etc/profile.d/conda.sh ]; then
  source /path/to/conda/etc/profile.d/conda.sh
else
  export PATH="/path/to/conda/bin:$PATH"
fi

mkdir -p "$BASE" "$SRC_ROOT" "$EXP"
export RAW_ROOT SRC_ROOT

echo "[info] raw root: $RAW_ROOT"
echo "[info] base dir: $BASE"
echo "[info] src root: $SRC_ROOT"
echo "[info] exp dir: $EXP"
echo "[info] opt filter: ${OPT_SUBSTR:-<all>}"
date -u '+[info] start %Y-%m-%dT%H:%M:%SZ'

python - <<'PY'
from pathlib import Path
import os

raw_root = Path(os.environ["RAW_ROOT"]).resolve()
src_root = Path(os.environ["SRC_ROOT"]).resolve()
opt_filter = os.environ.get("OPT_SUBSTR", "")
src_root.mkdir(parents=True, exist_ok=True)

count = 0
for path in sorted(raw_root.rglob("*")):
    if not path.is_file():
        continue
    if opt_filter and opt_filter not in path.parent.name:
        continue
    try:
        with path.open("rb") as handle:
            if handle.read(4) != b"\x7fELF":
                continue
    except OSError:
        continue

    link_name = f"{path.parent.name}__{path.name}"
    link_path = src_root / link_name
    if link_path.is_symlink() or link_path.exists():
        if link_path.resolve() == path.resolve():
            count += 1
            continue
        link_path.unlink()
    link_path.symlink_to(path.resolve())
    count += 1

print(f"[info] linked_o0_elfs={count}")
PY

PRE_DIR="$BASE/pretrain"
DOWNSTREAM_DIR="$BASE/downstream_sparse"

if [ ! -f "$PRE_DIR/pilot_summary.json" ]; then
  PYTHONPATH="$PYROOT" \
  conda run -n stateformer python \
    "$PYROOT/command/pretrain/build_unified_aa_pilot.py" \
    --binary-root "$SRC_ROOT" \
    --output-dir "$PRE_DIR" \
    --max-binaries 100000 \
    --functions-per-binary "$FUNCTIONS_PER_BINARY" \
    --max-emu-instructions "$MAX_EMU_INSTRUCTIONS"
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
    --max-tokens 2048
fi

PRE_DATA="$PRE_DIR/dataset"

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
  --use-address-aware-embedding \
  --aa-init-scale 0.25 \
  --save-dir "$EXP/pretrain_aa_additive" \
  > "$EXP/pretrain_aa_additive.log" 2>&1

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
  --use-address-aware-embedding \
  --aa-init-scale 0.25 \
  --restore-file "$EXP/pretrain_aa_additive/checkpoint_best.pt" \
  --reset-optimizer \
  --reset-meters \
  --reset-dataloader \
  --seed 7 \
  --save-dir "$EXP/finetune_aa_additive" \
  > "$EXP/finetune_aa_additive.log" 2>&1

date -u '+[info] end %Y-%m-%dT%H:%M:%SZ'
