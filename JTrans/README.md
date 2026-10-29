# numAnchor JTrans Artifact

Original base: https://github.com/vul337/jTrans

This component packages the JTrans-compatible numAnchor checkpoints and the matching full function-similarity JSON data used for artifact evaluation. The primary reproducible path is `JTrans/addressaware/`.

- `data/jtrans_full/numanchor/train/function_blocks.json`
- `data/jtrans_full/numanchor/train/ground_truth.json`
- `data/jtrans_full/numanchor/test/function_blocks.json`
- `data/jtrans_full/numanchor/test/ground_truth.json`
- `data/jtrans_full/numanchor/tokenizer/`
- `models/jtrans/numanchor_pretrain/`
- `models/jtrans/numanchor_finetune/`

Create the environment:

```bash
conda env create -f JTrans/environment.yml
conda activate jtrans
```

Generate a deterministic reviewer pool from the packaged test ground truth:

```bash
export NUMANCHOR_ARTIFACT_ROOT=/path/to/numAnchor_artifact/staging

python scripts/artifact/generate_pools.py \
  --ground-truth "$NUMANCHOR_ARTIFACT_ROOT/data/jtrans_full/numanchor/test/ground_truth.json" \
  --output "$NUMANCHOR_ARTIFACT_ROOT/work/pools/jtrans_pool_10k.json" \
  --pool-size 10000 \
  --query-count 1000 \
  --seed 42
```

The packaged model directories are HuggingFace-style checkpoints with neutral public names. Use `models/jtrans/numanchor_finetune/` for downstream evaluation and `models/jtrans/numanchor_pretrain/` for fine-tuning reproduction.

## Code Layout

- `addressaware/`: the main numAnchor JTrans implementation and artifact-evaluation path.
- `addressaware_clap/`: the CLAP-style instruction-id integration. It reuses the numAnchor address-aware `WordVocab` pipeline.
