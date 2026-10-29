# SymLM with Address-Aware PalmTree

This directory keeps a local copy of [SymLM](https://github.com/OSUSecLab/SymLM), but the active workflow in this repo is no longer the original upstream one.

Upstream SymLM is built around:
- `Ghidra`
- Trex-style multi-field inputs such as `static`, `byte1-4`, `inst_pos_emb`, `op_pos_emb`
- the original `fairseq` training stack

This repo adapts the task to **your own address-aware PalmTree pretraining setup**:
- `IDA Pro` instead of Ghidra
- PalmTree-style inline assembly with address information
- your `AddressAwareBERT` checkpoint, especially `output/mlm_addr_var/latest_bert.pt`

So the question this folder is meant to answer is:

> Can we use our address-aware pretrained PalmTree model for SymLM-style function name prediction?

## What Is Implemented Here

The currently supported path is:
1. Export binaries with `IDA Pro`
2. Convert them into address-aware SymLM-style dataset shards
3. Load those shards with our PalmTree-compatible parser
4. Initialize the encoder from `mlm_addr_var`
5. Train a lightweight function-name prediction head
6. Inspect predictions and scale up later

Already implemented:
- local copy of the SymLM repo with upstream `.git` removed
- new `IDA`-based preprocessing path
- new address-aware dataset conversion path
- prototype training script using your PalmTree backbone
- end-to-end smoke test on local binaries

Not fully replaced yet:
- the original upstream `fairseq/roberta_mf` model is still Trex-oriented
- the full SymLM training stack has not yet been rewritten to directly consume `input.inline`

That means the recommended path in this repo is the **address-aware prototype path**, not the original upstream training commands.

## Main Difference From Upstream SymLM

Upstream SymLM expects features like:
- `static`
- `inst_pos_emb`
- `op_pos_emb`
- `arch_emb`
- `byte1`
- `byte2`
- `byte3`
- `byte4`

This repo instead keeps the downstream input format aligned with your pretraining:

```text
opcode(0xADDR:P1:P2:P3) operand ...
address(0xADDR:P1:P2:P3)
daddr(0xADDR:P1:P2:P3)
var(0xOFFSET)
```

This matches the same family of format already used by:
- [func_gen.py](/path/to/numAnchor/PalmTree/data_generator/func_gen.py)
- [process_funcsim_ida.py](/path/to/numAnchor/PalmTree/data_generator/process_funcsim_ida.py)
- [dataloader.py](/path/to/numAnchor/PalmTree/dataloader.py)
- your address-aware MLM checkpoints

## Directory Map

Important files for the address-aware path:

- [dataset_generation/process_address_aware_ida.py](/path/to/numAnchor/PalmTree/FunctionNamePrediction/SymLM/dataset_generation/process_address_aware_ida.py)
  - IDA export script
  - emits JSON records with:
    - address-aware inline function body
    - caller names
    - internal callee names
    - external callee names

- [dataset_generation/prepare_address_aware_dataset.py](/path/to/numAnchor/PalmTree/FunctionNamePrediction/SymLM/dataset_generation/prepare_address_aware_dataset.py)
  - converts exported JSON into SymLM-style task shards
  - writes:
    - `self/input.inline`
    - `self/input.label`
    - `caller*/input.inline`
    - `internal_callee*/input.inline`
    - `external_callee*/input.label`

- [dataset_generation/run_address_aware.sh](/path/to/numAnchor/PalmTree/FunctionNamePrediction/SymLM/dataset_generation/run_address_aware.sh)
  - example wrapper for the new preprocessing path

- [dataset_generation/README_ADDRESS_AWARE.md](/path/to/numAnchor/PalmTree/FunctionNamePrediction/SymLM/dataset_generation/README_ADDRESS_AWARE.md)
  - focused notes for preprocessing only

- [prototype_addr_var_name_pred.py](/path/to/numAnchor/PalmTree/FunctionNamePrediction/prototype_addr_var_name_pred.py)
  - current prototype trainer and predictor using your PalmTree backbone

## Environment

Use the same environment you already use for PalmTree training when possible:

```bash
conda activate palmtree
```

The current address-aware path relies on:
- `torch`
- `nltk`
- `sentencepiece`
- `IDA Pro`

If needed:

```bash
pip install nltk sentencepiece
```

You also need:
- a usable PalmTree vocab file, usually `PalmTree/vocab.pkl`
- a pretrained checkpoint, usually `output/mlm_addr_var/latest_bert.pt`

## End-to-End Workflow

The workflow below is the recommended way to use **our model on SymLM** in this repo.

### Step 1: Export Binaries with IDA Pro

Run the address-aware IDA exporter:

```bash
/path/to/ida-pro-9.0/idat \
  -A \
  -S"/path/to/numAnchor/PalmTree/FunctionNamePrediction/SymLM/dataset_generation/process_address_aware_ida.py /tmp/symlm_addr_test/funcsim_export /path/to/numAnchor_artifact/addressaware/cfg_train/top_symbols_cfg_8inline.txt true" \
  /path/to/numAnchor_artifact/addressaware/funcsim_dataset/bind_O0/arpaname
```

This produces a JSON file like:

```text
/tmp/symlm_addr_test/funcsim_export/arpaname_O0_addressaware_symlm.json
```

Each record stores:
- function name
- address-aware inline function body
- caller list
- internal callee list
- external callee list

Important design choice:
- we use `IDA Pro`, not `Ghidra`
- the instruction/body format is kept consistent with your PalmTree pretraining

### Step 2: Prepare Address-Aware SymLM Dataset Shards

Convert exported JSON into the dataset layout used by the prototype:

```bash
/path/to/conda/envs/palmtree/bin/python \
  PalmTree/FunctionNamePrediction/SymLM/dataset_generation/prepare_address_aware_dataset.py \
  --input-dir /tmp/symlm_addr_test/funcsim_export \
  --output-dir /tmp/symlm_addr_test/funcsim_dataset \
  --topK 2
```

This writes a shard like:

```text
/tmp/symlm_addr_test/funcsim_dataset/arpaname_O0_addressaware_symlm/
```

with files such as:

```text
self/input.inline
self/input.label
caller1/input.inline
caller2/input.inline
internal_callee1/input.inline
internal_callee2/input.inline
external_callee1/input.label
external_callee2/input.label
manifest.json
```

Meaning of the files:
- `self/input.inline`
  - the function body in PalmTree-style address-aware inline format
- `self/input.label`
  - normalized function-name tokens
- `caller*/input.inline`
  - caller function bodies in the same format
- `internal_callee*/input.inline`
  - internal callee function bodies
- `external_callee*/input.label`
  - external callee symbol labels

### Step 3: Train with `mlm_addr_var`

The current supported training path is the prototype script:

- [prototype_addr_var_name_pred.py](/path/to/numAnchor/PalmTree/FunctionNamePrediction/prototype_addr_var_name_pred.py)

Run it like this:

```bash
/path/to/conda/envs/palmtree/bin/python \
  PalmTree/FunctionNamePrediction/prototype_addr_var_name_pred.py \
  --dataset-dir /tmp/symlm_addr_test/addressbin_dataset/4ti2int32_unknown_addressaware_symlm \
  --dataset-dir /tmp/symlm_addr_test/funcsim_dataset/arpaname_O0_addressaware_symlm \
  --checkpoint output/mlm_addr_var/latest_bert.pt \
  --vocab PalmTree/vocab.pkl \
  --output-dir output/function_name_prediction_proto \
  --epochs 30 \
  --batch-size 8 \
  --lr 1e-3
```

What this script does:
- reads `self/input.inline` and `self/input.label`
- parses inline functions with PalmTree-compatible token and position logic
- loads `output/mlm_addr_var/latest_bert.pt`
- builds function embeddings with the PalmTree function encoder pattern
- freezes the encoder by default
- trains a simple multi-label prediction head over function-name subtokens

Current scope of the prototype:
- input:
  - `self/input.inline`
- output:
  - micro `precision`, `recall`, `F1`
  - per-sample top predicted label tokens

### Step 4: Inspect Prediction Output

The prototype writes:

- [prototype_results.json](/path/to/numAnchor/output/function_name_prediction_proto/prototype_results.json)

This file contains:
- dataset size
- train/test split size
- label vocabulary size
- training history
- per-sample predictions
- embedding norm preview

## Training Logic Summary

The current overall training process in this repo is:

1. Use your existing address-aware MLM pretraining as the backbone initialization
2. Keep downstream preprocessing consistent with that same token distribution
3. Turn function-name prediction into a multi-label name-token prediction problem
4. First validate the pipeline with a lightweight frozen-encoder prototype
5. Only after the data path is stable, consider larger-scale fine-tuning

This ordering matters because it avoids two common problems:
- changing the downstream input format away from your pretraining distribution
- mixing debugging of preprocessing, model loading, and fine-tuning all at once

## What Was Already Verified

This address-aware SymLM path has already been smoke-tested on local binaries:

- one `funcsim` binary:
  - `/path/to/numAnchor_artifact/addressaware/funcsim_dataset/bind_O0/arpaname`
- one of your current pretraining binaries:
  - `/path/to/numAnchor_artifact/addressaware/addressbin_train/4ti2int32`

Verified already:
- IDA export works
- address-aware dataset conversion works
- `mlm_addr_var` checkpoint loads correctly
- function embeddings can be produced
- a name-token prediction head can be trained
- predictions can be written out

## How to Choose Data

For function-name prediction, the quality of labels matters a lot more than for MLM.

Prefer binaries with:
- preserved function symbols
- fewer placeholder names like `sub_*`
- less C++ mangled-name noise

Use your current unlabeled corpora mainly for:
- generic MLM pretraining
- optional domain-adaptive MLM

Do not expect stripped pretraining binaries by themselves to provide clean supervision for function-name prediction.

## Recommended Next Steps

The best next scaling path is:

1. Generate more function-name data from non-stripped binaries
2. Keep the same `IDA + address-aware inline` preprocessing path
3. Re-run the prototype on a cleaner and larger dataset
4. Then decide whether to:
   - unfreeze the PalmTree encoder
   - add caller/callee context into training
   - replace more of the original SymLM stack

## What Still Comes From Original SymLM

The original SymLM README is still useful as background for:
- the task definition
- the value of caller/callee context
- the original paper setup and citation

But for this repo, the practical rule is simple:

- keep the original SymLM project as conceptual reference
- use the new address-aware preprocessing path for actual data generation
- use your PalmTree checkpoint as the actual encoder backbone

Do not treat the upstream `Ghidra + Trex multi-field` workflow as the main path here.

## Original Project Reference

Original SymLM paper:
- Xin Jin, Kexin Pei, Jun Yeon Won, Zhiqiang Lin
- *SymLM: Predicting Function Names in Stripped Binaries via Context-Sensitive Execution-Aware Code Embeddings*
- CCS 2022

Original upstream repository:
- https://github.com/OSUSecLab/SymLM

## Citation

If you use the original SymLM idea, please cite:

```text
@inproceedings{jin2022symlm,
  title={SymLM: Predicting Function Names in Stripped Binaries via Context-Sensitive Execution-Aware Code Embeddings},
  author={Jin, Xin and Pei, Kexin and Won, Jun Yeon and Lin, Zhiqiang},
  booktitle={Proceedings of the 2022 ACM SIGSAC Conference on Computer and Communications Security},
  pages={1631--1645},
  year={2022}
}
```
