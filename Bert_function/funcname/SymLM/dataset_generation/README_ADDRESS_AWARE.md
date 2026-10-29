# Address-Aware SymLM Dataset Generation

This repository copy includes an alternative preprocessing path for SymLM that is aligned with the current PalmTree address-aware pretraining pipeline.

## Why this variant exists
The upstream SymLM preprocessing assumes:
- `Ghidra` for binary parsing and calling-context extraction
- `Trex`-style multi-field inputs such as `static`, `byte1-4`, `inst_pos_emb`, and `op_pos_emb`

That is not aligned with this repository's current pretraining setup, where the model learns from:
- `IDA Pro`
- inline assembly with hierarchical address annotations
- tokens such as `address(...)`, `daddr(...)`, and `var(...)`

This address-aware variant changes preprocessing so downstream function-name prediction can consume inputs that stay consistent with the current pretraining distribution.

## New scripts
- `process_address_aware_ida.py`
  - IDA Pro export script
  - emits one JSON per binary with:
    - address-aware inline function bodies
    - caller names
    - internal callee names
    - external callee names
- `prepare_address_aware_dataset.py`
  - converts exported JSON into SymLM-style task shards
  - keeps:
    - self function body
    - caller context
    - internal callee context
    - external callee labels
    - normalized function-name labels
  - but stores function bodies as `input.inline` files instead of upstream Trex multi-field files
- `run_address_aware.sh`
  - example wrapper for the new IDA-based pipeline

## Input format
Each function body is stored in the same inline format used by the current PalmTree pretraining/data-export pipeline, for example:

```text
mov(0x401000:0.12500000:0.30000000:0.00000000) rax daddr(0x404000:0.50000000:0.02000000:0.00000000) \t call(0x401010:0.12500000:0.30000000:0.50000000) address(0x401200:0.12500000:0.90000000:0.00000000)
```

This keeps downstream preprocessing aligned with:
- `PalmTree/data_generator/func_gen.py`
- `PalmTree/data_generator/process_funcsim_ida.py`
- `PalmTree/dataloader.py`

## Output layout
For each exported binary shard, the prepared dataset is written as:

```text
addressaware_dataset/<binary_json_stem>/
  self/
    input.inline
    input.label
  caller1/
    input.inline
  caller2/
    input.inline
  internal_callee1/
    input.inline
  internal_callee2/
    input.inline
  external_callee1/
    input.label
  external_callee2/
    input.label
  manifest.json
```

`input.inline` is the address-aware function body.
`input.label` under `self/` is the normalized function-name token sequence.
`input.label` under `external_callee*/` is the external callee symbol label.

## Recommended workflow
1. Export binaries with `process_address_aware_ida.py`
2. Convert exported JSON with `prepare_address_aware_dataset.py`
3. Build the downstream dataloader on top of `input.inline`
4. Parse `input.inline` with the same token/position rules as the current address-aware pretraining code
5. Initialize the downstream encoder from your address-aware MLM checkpoint instead of Trex

## Important note
This preprocessing update intentionally does not preserve upstream SymLM's Trex multi-field input representation.

That is by design: this variant is meant for your address-aware pretrained backbone, not for the original Trex-based SymLM encoder.
