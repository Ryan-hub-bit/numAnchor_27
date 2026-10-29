# numAnchor StateFormer Artifact

Original base: https://github.com/CUMLSec/stateformer

This component packages the strict optimisation-level StateFormer experiments used in the numAnchor downstream type-recovery results. The artifact provides per-level data and checkpoints for `O0`, `O1`, `O2`, and `O3`:

- `data/stateformer_strict_opt/<OPT>/pretrain_dataset/`
- `data/stateformer_strict_opt/<OPT>/downstream_sparse/`
- `models/stateformer/<OPT>/numanchor_pretrain.pt`
- `models/stateformer/<OPT>/numanchor_finetune.pt`
- `models/stateformer/<OPT>/baseline_pretrain.pt`
- `models/stateformer/<OPT>/baseline_finetune.pt`

Create the environment:

```bash
conda env create -f Stateformer/environment.yml
conda activate stateformer
```

Run one downstream evaluation or continued fine-tuning job with the packaged data:

```bash
export NUMANCHOR_ARTIFACT_ROOT=/path/to/numAnchor_artifact/staging
export OPT_LEVEL=O0

cd Stateformer

python train.py \
  "$NUMANCHOR_ARTIFACT_ROOT/data/stateformer_strict_opt/$OPT_LEVEL/downstream_sparse" \
  --task data_structure_mf_aa_smoke \
  --criterion data_structure_mf_aa_smoke \
  --arch roberta_mf_nau \
  --classification-head-name data_structure_head \
  --encoder-layers 6 \
  --encoder-embed-dim 256 \
  --encoder-ffn-embed-dim 1024 \
  --encoder-attention-heads 8 \
  --max-positions 2048 \
  --max-sentences 2 \
  --max-sentences-valid 2 \
  --skip-invalid-size-inputs-valid-test \
  --use-address-aware-embedding \
  --restore-file "$NUMANCHOR_ARTIFACT_ROOT/models/stateformer/$OPT_LEVEL/numanchor_pretrain.pt" \
  --save-dir "$NUMANCHOR_ARTIFACT_ROOT/work/results_tmp/stateformer_$OPT_LEVEL" \
  --optimizer adam \
  --lr 1e-5 \
  --reset-optimizer \
  --max-update 1
```

For full reproduction, remove the one-update smoke limit and use the matching finetuned checkpoint for validation/reporting. Switch `numanchor_*` to `baseline_*` for baseline comparison.
