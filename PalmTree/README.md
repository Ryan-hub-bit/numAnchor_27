# numAnchor PalmTree Artifact

Original base: https://github.com/palmtreemodel/PalmTree

This component evaluates the PalmTree-compatible numAnchor model on its paper-aligned x64 function-similarity data. The artifact provides:

- `data/palmtree_x64/function_blocks.json`
- `data/palmtree_x64/ground_truth.json`
- `data/palmtree_x64/vocab.pkl`
- `models/palmtree/numanchor_pretrain.pt`
- `models/palmtree/numanchor_finetune.pt`
- `models/palmtree/baseline_pretrain.pt`
- `models/palmtree/baseline_finetune.pt`

Create the environment:

```bash
conda env create -f PalmTree/environment.yml
conda activate palmtree
```

Generate a deterministic reviewer pool:

```bash
export NUMANCHOR_ARTIFACT_ROOT=/path/to/numAnchor_artifact/staging

python scripts/artifact/generate_pools.py \
  --ground-truth "$NUMANCHOR_ARTIFACT_ROOT/data/palmtree_x64/ground_truth.json" \
  --output "$NUMANCHOR_ARTIFACT_ROOT/work/pools/palmtree_pool_10k.json" \
  --pool-size 10000 \
  --query-count 1000 \
  --seed 42
```

Run retrieval evaluation:

```bash
cd PalmTree/*/funcsim
export PYTHONPATH="$(pwd)/../../..:${PYTHONPATH:-}"

python evaluate.py \
  --function_blocks "$NUMANCHOR_ARTIFACT_ROOT/data/palmtree_x64/function_blocks.json" \
  --funcsim_pairs "$NUMANCHOR_ARTIFACT_ROOT/data/palmtree_x64/ground_truth.json" \
  --vocab "$NUMANCHOR_ARTIFACT_ROOT/data/palmtree_x64/vocab.pkl" \
  --checkpoint "$NUMANCHOR_ARTIFACT_ROOT/models/palmtree/numanchor_finetune.pt" \
  --eval_pool "$NUMANCHOR_ARTIFACT_ROOT/work/pools/palmtree_pool_10k.json" \
  --output "$NUMANCHOR_ARTIFACT_ROOT/work/results_tmp/palmtree_eval.json" \
  --log_dir "$NUMANCHOR_ARTIFACT_ROOT/work/logs/palmtree" \
  --task_name numanchor_palmtree
```

Use the baseline checkpoint paths in the same command to reproduce the baseline comparison.
