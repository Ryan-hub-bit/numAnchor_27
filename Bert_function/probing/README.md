# PalmTree Probe Evaluation

This directory contains a frozen-encoder probing pipeline for comparing:

- baseline non-address-aware PalmTree pretrain checkpoints
- address-aware PalmTree pretrain checkpoints

on three structure-sensitive tasks:

- `same_bb`: do two instructions belong to the same basic block?
- `same_function`: do two basic blocks belong to the same function?
- `jump_target`: does this `j*` instruction point to this candidate instruction?

## What The Script Reports

For each task and each model, `probe_eval.py` reports:

- `zero_shot`
  - cosine-based scoring
  - threshold chosen on the train split
  - test-set `AUC`, `accuracy`, `balanced_accuracy`, `F1`
- `few_shot`
  - frozen encoder + linear probe
  - configurable shots per class such as `8`, `32`, `128`
  - repeated over multiple seeds
  - mean / std in the JSON output

## Recommended Data

Use a sampled function-block file rather than the full raw dump.

Examples:

- `/path/to/numAnchor_artifact/addressaware/funcsim_block/pool_function_blocks_100.json`
- `/path/to/numAnchor_artifact/addressaware/funcsim_block/pool_function_blocks_1k.json`
- `/path/to/numAnchor_artifact/addressaware/funcsim_block/pool_function_blocks_10k.json`

If available, also pass:

- `/path/to/numAnchor_artifact/addressaware/funcsim_block/funcsim_pairs.json`

This lets the script avoid drawing `same source function across opt levels`
as negatives for the `same_function` task.

## Example Run

```bash
source /path/to/conda/etc/profile.d/conda.sh
conda activate palmtree

python PalmTree/probing/probe_eval.py \
  --function_blocks /path/to/numAnchor_artifact/addressaware/funcsim_block/pool_function_blocks_1k.json \
  --funcsim_pairs /path/to/numAnchor_artifact/addressaware/funcsim_block/funcsim_pairs.json \
  --vocab /path/to/numAnchor_artifact/aaedataset/blockonly/data/vocab.pkl \
  --baseline_checkpoint /path/to/numAnchor_artifact/aaedataset/blockonly/models/mlm/pretrain/latest_bert.pt \
  --aa_checkpoint /path/to/numAnchor_artifact/aaedataset/blockonly/models/mlm_addr_var/pretrain/best_bert.pt \
  --output /path/to/numAnchor/output/probing/probe_eval_1k.json \
  --tasks same_bb same_function jump_target \
  --few_shots 8 32 128 \
  --device cpu
```

## Suggested Main Table

Use one row per task / shot setting:

- task
- setup: `zero-shot`, `8-shot`, `32-shot`, `128-shot`
- baseline `AUC`
- address-aware `AUC`
- baseline `accuracy`
- address-aware `accuracy`
- delta

## Metric Notes

For a compact summary, prioritize:

- `AUC` for the main comparison
- `accuracy` and `F1` as supporting numbers

`AUC` is the most stable because it is less sensitive to threshold choice.
