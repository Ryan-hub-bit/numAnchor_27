# Function Similarity Reproduction

This directory contains the current `funcsim` code used to evaluate PalmTree
function-similarity checkpoints.

The recommended reproduction bundle is stored separately at:

`/path/to/numAnchor_artifact/aaedataset/blockonly`

That bundle contains only:

- data files directly used by evaluation
- vocab files
- pretrain checkpoints
- finetuned checkpoints

It does **not** contain code, and it does **not** contain binary-level
generation artifacts.

## What This Bundle Can Reproduce

With the current code in this directory and the files in
`/path/to/numAnchor_artifact/aaedataset/blockonly`, you can reproduce the old `10k pool / 1000
query` evaluation for:

- baseline `mlm`
- address-aware `mlm_addr_var`

This bundle is intended for **evaluation reproduction**.

It is **not sufficient** to rerun:

- pretraining from raw corpus
- function-similarity finetuning from scratch

because the full training corpus is intentionally not included in
`blockonly`.

## Bundle Layout

The current reproduction bundle contains:

### Data

- `/path/to/numAnchor_artifact/aaedataset/blockonly/path/to/funcsim_pairs.json`
- `/path/to/numAnchor_artifact/aaedataset/blockonly/data/pool_ids_10k.json`
- `/path/to/numAnchor_artifact/aaedataset/blockonly/data/pool_function_blocks_10k.json`
- `/path/to/numAnchor_artifact/aaedataset/blockonly/data/vocab.pkl`
- `/path/to/numAnchor_artifact/aaedataset/blockonly/data/vocab_8.pkl`

### Models

- baseline pretrain:
  `/path/to/numAnchor_artifact/aaedataset/blockonly/models/mlm/pretrain/latest_bert.pt`
- baseline finetune:
  `/path/to/numAnchor_artifact/aaedataset/blockonly/models/mlm/finetune/best_model.pt`
- address-aware pretrain:
  `/path/to/numAnchor_artifact/aaedataset/blockonly/models/mlm_addr_var/pretrain/best_bert.pt`
- address-aware finetune:
  `/path/to/numAnchor_artifact/aaedataset/blockonly/models/mlm_addr_var/finetune/best_model.pt`

### Optional Embedding Cache

Evaluation will write reusable embedding caches under:

- `/path/to/numAnchor_artifact/aaedataset/blockonly/models/mlm/finetune/embeddings/`
- `/path/to/numAnchor_artifact/aaedataset/blockonly/models/mlm_addr_var/finetune/embeddings/`

These caches are safe to keep. They speed up repeated evaluation on the same
pool.

## Environment

Use the existing `palmtree` conda environment:

```bash
source /path/to/conda/etc/profile.d/conda.sh
conda activate palmtree
cd /path/to/numAnchor/PalmTree/funcsim
```

## Reproduce Baseline `mlm`

Run:

```bash
python evaluate.py \
  --function_blocks /path/to/numAnchor_artifact/aaedataset/blockonly/data/pool_function_blocks_10k.json \
  --funcsim_pairs /path/to/numAnchor_artifact/aaedataset/blockonly/path/to/funcsim_pairs.json \
  --vocab /path/to/numAnchor_artifact/aaedataset/blockonly/data/vocab.pkl \
  --checkpoint /path/to/numAnchor_artifact/aaedataset/blockonly/models/mlm/finetune/best_model.pt \
  --hidden 768 \
  --n_layers 12 \
  --attn_heads 12 \
  --embedding_dim 256 \
  --seq_len 420 \
  --max_instructions 50 \
  --batch_size 64 \
  --task_name mlm_10k \
  --output /path/to/numAnchor/output/funcsim/blockonly_mlm_eval_10k.json \
  --log_dir /path/to/numAnchor/log/funcsim/blockonly_mlm_eval \
  --device cpu \
  --eval_pool /path/to/numAnchor_artifact/aaedataset/blockonly/data/pool_ids_10k.json
```

The command writes retrieval metrics, including `Recall@1/5/10/20` and `MRR`,
to the configured JSON output. Metric values are intentionally not listed here;
they should be generated from the packaged artifact during review.

Output path:

- `/path/to/numAnchor/output/funcsim/blockonly_mlm_eval_10k.json`

## Reproduce Address-Aware `mlm_addr_var`

Run:

```bash
python evaluate.py \
  --function_blocks /path/to/numAnchor_artifact/aaedataset/blockonly/data/pool_function_blocks_10k.json \
  --funcsim_pairs /path/to/numAnchor_artifact/aaedataset/blockonly/path/to/funcsim_pairs.json \
  --vocab /path/to/numAnchor_artifact/aaedataset/blockonly/data/vocab.pkl \
  --checkpoint /path/to/numAnchor_artifact/aaedataset/blockonly/models/mlm_addr_var/finetune/best_model.pt \
  --hidden 768 \
  --n_layers 12 \
  --attn_heads 12 \
  --embedding_dim 256 \
  --seq_len 420 \
  --max_instructions 50 \
  --batch_size 64 \
  --task_name mlm_addr_var_10k \
  --output /path/to/numAnchor/output/funcsim/blockonly_mlm_addr_var_eval_10k.json \
  --log_dir /path/to/numAnchor/log/funcsim/blockonly_mlm_addr_var_eval \
  --device cpu \
  --eval_pool /path/to/numAnchor_artifact/aaedataset/blockonly/data/pool_ids_10k.json
```

The command writes retrieval metrics, including `Recall@1/5/10/20` and `MRR`,
to the configured JSON output. Metric values are intentionally not listed here;
they should be generated from the packaged artifact during review.

Output path:

- `/path/to/numAnchor/output/funcsim/blockonly_mlm_addr_var_eval_10k.json`

## Notes

1. Use the code in this directory directly.
   Do **not** copy code into `blockonly`.

2. The `blockonly` bundle is intentionally minimal.
   It keeps only the files needed to load checkpoints and evaluate them on the
   stored `10k` pool.

3. The baseline and address-aware models both use:

- hidden size `768`
- `12` layers
- `12` attention heads
- output embedding dim `256`
- sequence length `420`
- max instructions `50`

4. The main difference is checkpoint content:

- baseline `mlm` loads with `use_addr = false`
- `mlm_addr_var` loads with `use_addr = true`

5. If you rerun evaluation on the same pool, the embedding cache under
   `blockonly/models/*/finetune/embeddings/` will be reused automatically.

## Quick Check

Both evaluation commands should complete and write JSON metric files under the
configured output paths.
