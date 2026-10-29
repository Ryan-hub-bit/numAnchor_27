# numAnchor Trex Artifact

Original base: https://github.com/CUMLSec/trex

This component packages the Trex-compatible numAnchor checkpoints and the shared StateFormer x64 retrieval data. The artifact provides:

- `data/shared_stateformer_x64/function_blocks.json`
- `data/shared_stateformer_x64/ground_truth.json`
- `data/trex/numanchor_pretrain_bin/`
- `data/trex/baseline_pretrain_bin/`
- `models/trex/numanchor_pretrain.pt`
- `models/trex/numanchor_finetune.pt`
- `models/trex/baseline_pretrain.pt`
- `models/trex/baseline_finetune.pt`

Create the environment:

```bash
conda env create -f trex/environment.yml
conda activate stateformer
```

Generate a deterministic reviewer pool:

```bash
export NUMANCHOR_ARTIFACT_ROOT=/path/to/numAnchor_artifact/staging

python scripts/artifact/generate_pools.py \
  --ground-truth "$NUMANCHOR_ARTIFACT_ROOT/data/shared_stateformer_x64/ground_truth.json" \
  --output "$NUMANCHOR_ARTIFACT_ROOT/work/pools/trex_pool_10k.json" \
  --pool-size 10000 \
  --query-count 1000 \
  --seed 42
```

Run retrieval evaluation:

```bash
cd trex

python command/stateformer_x64/evaluate_funcsim_retrieval.py \
  --checkpoint "$NUMANCHOR_ARTIFACT_ROOT/models/trex/numanchor_finetune.pt" \
  --pretrained-checkpoint "$NUMANCHOR_ARTIFACT_ROOT/models/trex/numanchor_pretrain.pt" \
  --data-dir "$NUMANCHOR_ARTIFACT_ROOT/data/trex/baseline_pretrain_bin" \
  --function-blocks "$NUMANCHOR_ARTIFACT_ROOT/data/shared_stateformer_x64/function_blocks.json" \
  --funcsim-pairs "$NUMANCHOR_ARTIFACT_ROOT/data/shared_stateformer_x64/ground_truth.json" \
  --pool-file "$NUMANCHOR_ARTIFACT_ROOT/work/pools/trex_pool_10k.json" \
  --output "$NUMANCHOR_ARTIFACT_ROOT/work/results_tmp/trex_eval.json"
```

Use the baseline checkpoint in the same command to reproduce the baseline comparison. The packaged numAnchor Trex checkpoint was trained with the shared baseline fairseq dictionaries plus numAnchor address-aware features, so the retrieval command intentionally points `--data-dir` at `baseline_pretrain_bin`.
