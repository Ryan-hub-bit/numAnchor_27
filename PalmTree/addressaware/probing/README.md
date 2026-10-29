# PalmTree Probe Evaluation

This directory contains a frozen-encoder probing pipeline for comparing:

- baseline non-address-aware PalmTree pretrain checkpoints
- address-aware PalmTree pretrain checkpoints

on instruction-level intrinsic tasks used by the artifact evaluation:

- `same_bb`: do two instructions belong to the same basic block?
- `same_function`: do two basic blocks belong to the same function?
- `jump_target`: does this `j*` instruction point to this candidate instruction?
- `call_target`: does this `call` instruction point to this candidate callee?
- `same_dref`: do two instructions reference the same data address?
- `dref_retrieval`: 32-way data-reference retrieval

The parser in this branch expects:

- code tokens: `opcode(0xADDR:binary_pos:function_pos:bb_pos)`
- data tokens:
  - `daddr_rodata(0xADDR:section_pos:offset_in_section)`
  - `daddr_bss(0xADDR:section_pos:offset_in_section)`
  - `daddr_data(0xADDR:section_pos:offset_in_section)`
  - `daddr_other(0xADDR:section_pos:offset_in_section)`

All four data-address tokens share the same data-address MLP branch; the
section identity is carried by the token embedding itself.

Section mapping:

- `1 = .rodata`
- `2 = .bss`
- `3 = .data`
- `4 = other-data`

For `call_target`, `same_dref`, and `dref_retrieval`, the probe script needs
binary identity so it can enforce same-binary sampling. The recommended setup is:

- `--function_blocks /path/to/numAnchor_artifact/stateformer_x64_dataset/funcsim_block_ctx2/pool_function_blocks_1k.json`
- `--function_records_dir /path/to/numAnchor_artifact/stateformer_x64_dataset/funcsim_block_ctx2`

The script will hash-match pooled functions back to the per-binary `*_functions.json`
records and recover `func_id -> binary_id` automatically when possible.

## What The Script Reports

For pair tasks (`same_bb`, `same_function`, `jump_target`, `call_target`, `same_dref`),
`probe_eval.py` reports:

- `zero_shot`
  - cosine-based scoring
  - threshold chosen on the train split
  - test-set `AUC`, `accuracy`, `balanced_accuracy`, `F1`
- `few_shot`
  - frozen encoder + linear probe
  - configurable shots per class such as `8`, `32`, `128`
  - repeated over multiple seeds
  - mean / std in the JSON output

For `dref_retrieval`, the script reports retrieval metrics on the held-out queries:

- `top1_accuracy`
- `recall_at_5`
- `mrr`

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

python PalmTree/addressaware/probing/probe_eval.py \
  --function_blocks /path/to/numAnchor_artifact/stateformer_x64_dataset/funcsim_block_ctx2/pool_function_blocks_1k.json \
  --function_records_dir /path/to/numAnchor_artifact/stateformer_x64_dataset/funcsim_block_ctx2 \
  --funcsim_pairs /path/to/numAnchor_artifact/stateformer_x64_dataset/funcsim_block_ctx2/funcsim_pairs_q30_gt3_showcase_r80.json \
  --vocab /path/to/numAnchor_artifact/aaedataset/blockonly/data/vocab.pkl \
  --baseline_checkpoint /path/to/numAnchor_artifact/aaedataset/blockonly/models/mlm/pretrain/latest_bert.pt \
  --aa_checkpoint /path/to/numAnchor_artifact/aaedataset/blockonly/models/mlm_addr_var/pretrain/best_bert.pt \
  --output /path/to/numAnchor/output/probing/probe_eval_1k.json \
  --tasks same_bb same_function jump_target call_target same_dref dref_retrieval \
  --retrieval_pool_size 32 \
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

- `AUC` for pair tasks
- `top1_accuracy` or `MRR` for retrieval
- `accuracy` and `F1` as supporting numbers

`AUC` is the most stable because it is less sensitive to threshold choice.
