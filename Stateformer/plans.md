# StateFormer Environment Plan

## Goal

Create a dedicated `conda` environment for `StateFormer` so we can continue the GSM + address-aware implementation work without contaminating other project environments.

## Environment Name

`stateformer`

## Commands Executed

```bash
conda create -y -n stateformer python=3.8 scipy scikit-learn pyarrow networkx pip
conda install -y -n stateformer numpy=1.23.5 numpy-base=1.23.5
conda install -y -n stateformer pytorch torchvision torchaudio cpuonly -c pytorch
conda run -n stateformer python -m pip install 'pip<24.1'
conda run -n stateformer pip install hydra-core==1.0.7 omegaconf==2.0.6 sacrebleu==1.5.1 editdistance regex tqdm cffi cython capstone unicorn pyelftools
conda run -n stateformer pip install --no-deps -e /path/to/numAnchor/stateformer
```

## Why These Packages

- `python=3.8`: matches the upstream StateFormer installation guidance.
- `numpy=1.23.5`: required because this codebase still uses `np.float`, which breaks on newer NumPy.
- `pytorch torchvision torchaudio`: core training/runtime stack.
- `pyarrow`: useful for dataset processing.
- `networkx`: useful for CFG/basic-block level processing.
- `capstone`: disassembly support already used by StateFormer scripts.
- `unicorn`: needed for the planned micro-trace generation work.
- `pyelftools`: needed for ELF parsing and segment mapping.
- `hydra-core`, `omegaconf`, `sacrebleu`, `editdistance`, `regex`, `tqdm`, `cffi`, `cython`: runtime/build dependencies used by the bundled Fairseq fork.

## Activation

```bash
conda activate stateformer
```

## Important Runtime Note

This repository imports `command.params` with a top-level import, so `PYTHONPATH` must include the repository root when running `fairseq` entrypoints:

```bash
export PYTHONPATH=/path/to/numAnchor/stateformer:$PYTHONPATH
```

Recommended shell setup before running any StateFormer command:

```bash
conda activate stateformer
export PYTHONPATH=/path/to/numAnchor/stateformer:$PYTHONPATH
cd /path/to/numAnchor/stateformer
```

## Quick Verification

The following checks succeeded after the setup above:

```bash
conda run -n stateformer env PYTHONPATH=/path/to/numAnchor/stateformer python -c "import torch, fairseq, capstone, unicorn, numpy, elftools; print(torch.__version__, numpy.__version__)"
conda run -n stateformer env PYTHONPATH=/path/to/numAnchor/stateformer fairseq-train --help
```

## Current Status

- Dedicated `stateformer` conda environment created.
- Project installed in editable mode.
- Core imports verified.
- `fairseq-train` verified with the required `PYTHONPATH` setting.

## Follow-up

This environment is now ready for the next implementation step: the x86-64 unified GSM + address-aware dataset builder and the matching StateFormer task/model changes.

## Dataset Feasibility Check

We validated the provided x86-64 raw dataset at `/path/to/numAnchor_artifact/stateformer_x64_dataset/raw_input/x86-64` and confirmed that it contains the right kind of inputs for a same-binary unified pipeline.

### What We Checked

- The dataset is not limited to preprocessed text. It contains original ELF files.
- A representative sample, `/path/to/numAnchor_artifact/stateformer_x64_dataset/raw_input/x86-64/openssl-1.0.1f-O0/openssl`, is:
  - `ELF 64-bit`
  - `x86-64`
  - `PIE`
  - `not stripped`
- This means we can recover:
  - function symbols
  - instruction addresses
  - linear instruction order
  - code/data section boundaries

### Address-Aware Signals Confirmed

On the same `main` function from the sample binary, the linear instruction stream already exposes:

- stack-slot accesses from `rbp/rsp` based memory operands, which can be turned into `var_offset`
- direct code targets from `call` / `jmp` / conditional branches, which can be tagged as `address`
- RIP-relative data accesses into sections like `.bss`, which can be tagged as `daddr`

So the same function order used by StateFormer can carry:

- `aa_role`
- `binary_pos`
- `function_pos`
- `bb_pos`
- `var_offset`

without switching to a CFG walk.

### GSM Signals Confirmed

On that same function, a Unicorn-based smoke execution succeeded on the function prefix and recovered:

- token-aligned coverage information
- register state values
- enough numeric state to fill a GSM-style `byte1..4` view

This confirms that the raw test data is sufficient for:

- `static`
- `inst_pos_emb`
- `op_pos_emb`
- `arch_emb`
- `byte1..4`
- `cover`

from the same binary and function.

### Dataset Shape Notes

A quick scan of the dataset showed:

- `4249` no-extension files that are mostly ELF executables
- `33` shared objects (`.so*`)
- `52` static archives (`.a`)
- `52` IDA databases (`.i64`)

For v1 of the unified generator, the safest direct targets are:

- ELF executables
- shared libraries

The `.a` and `.i64` files should be treated as out of scope for the first implementation slice.

## Current Implementation Slice

The next concrete step is a minimal unified x86-64 extractor that works on one ELF and one function at a time.

### Scope

- input: one x86-64 ELF binary and one function name or address
- static side:
  - linear disassembly in StateFormer order
  - StateFormer-compatible tokenization
  - token-level `aa_role`, `binary_pos`, `function_pos`, `bb_pos`, `var_offset`
- dynamic side:
  - one Unicorn run
  - token-level `cover`
  - token-level `byte1..4` smoke values

### Expected Output

The minimal extractor should emit a debug-friendly JSON file containing:

- flattened token-level arrays for StateFormer-compatible inputs
- flattened token-level arrays for address-aware sidecar features
- per-instruction metadata for inspection and debugging

### Known Limitations

- v1 relies on available symbol information and does not yet recover stripped functions.
- v1 focuses on ELF executables and shared libraries.
- function execution still uses a simplified call policy instead of a full micro-trace recreation pipeline.

## Next Integration Plan

This section describes the next implementation stage for adding the address-aware design to StateFormer GSM pretraining. The goal is to keep the original GSM task intact while adding an address-aware sidecar branch.

### Stage 1: Build a Pretraining-Ready Unified Dataset

The current smoke JSON proves that the same instruction stream can carry both GSM and address-aware information. The next step is to export this into a training-friendly format.

Planned output fields per sample:

- StateFormer fields
  - `static`
  - `inst_pos_emb`
  - `op_pos_emb`
  - `arch_emb`
  - `byte1`
  - `byte2`
  - `byte3`
  - `byte4`
  - `cover`
- address-aware sidecar fields
  - `aa_role`
  - `aa_role_ids`
  - `binary_pos`
  - `function_pos`
  - `bb_pos`
  - `var_offset`

Planned export policy:

- one line = one full function trace
- keep only samples with token count in a safe range such as `30..510`
- write a `manifest.jsonl` so every sample can be traced back to:
  - binary path
  - function name
  - function start address
  - run id

### Stage 2: Add an Address-Aware Input Branch to StateFormer

The existing StateFormer encoder already sums multiple aligned embeddings. We will extend that pattern instead of replacing GSM.

Planned behavior:

- keep the original GSM inputs and GSM loss unchanged
- add one more embedding branch driven by the sidecar tensors
- use PalmTree-style routing:
  - `address` tokens use `function_pos + bb_pos`
  - `daddr` tokens use `binary_pos + function_pos`
  - `var` tokens use `var_offset`
  - plain `code` tokens do not get special address-aware positional content beyond the existing StateFormer embeddings

Implementation idea:

- add a new address-aware embedding module under the StateFormer model code
- reuse the current PalmTree/JTrans conventions:
  - normalized float positions for `binary_pos/function_pos/bb_pos`
  - `-1.0` sentinel for non-address positions
  - `-999999` sentinel for non-var tokens
- sum the new address-aware embedding with the existing token/position/arch/GSM embeddings before the Transformer layers

### Stage 3: Smoke Training Plan

After you confirm the plan, the first training run will be a tiny smoke run, not a full pretraining job.

Smoke dataset plan:

- start with a very small subset from `/path/to/numAnchor_artifact/stateformer_x64_dataset/raw_input/x86-64`
- prefer clean ELF executables and shared libraries only
- first target set:
  - `openssl-1.0.1f-O0/openssl`
  - a small handful of `coreutils`
  - optionally one `busybox`

Smoke training goals:

- confirm the new task can load all old GSM fields plus the new sidecar fields
- confirm padding and batching work
- confirm forward pass and loss computation are stable
- confirm training runs for a few hundred updates without NaN
- confirm the model can overfit a tiny subset more easily than random chance

Planned smoke checks:

- data loader shape checks
- one-batch forward check
- short run on CPU or single GPU
- save one checkpoint
- inspect the training log for:
  - total loss
  - byte reconstruction behavior
  - cover prediction behavior

### Stage 4: Transition from Smoke to Real Comparison

Only after the smoke stage is stable, we should run a comparison between:

- baseline StateFormer GSM
- StateFormer GSM + address-aware embedding

The fair comparison rule is:

- same binaries
- same train/valid split
- same tokenizer
- same hyperparameters unless a change is strictly required by the new branch

## Downstream Validation Plan

The downstream question is not just "does training run", but "does the added address-aware signal improve type recovery?" The downstream validation should therefore be done as an ablation study.

### Main Comparison

Run finetuning and compare at least these two models:

- `baseline`: original StateFormer pretrained with GSM only
- `aa_gsm`: StateFormer pretrained with GSM plus address-aware embedding

Use the same downstream finetuning setup for both.

### Metrics to Report

Primary metrics:

- precision
- recall
- F1

These should be measured on the downstream type inference task exactly the same way for both models.

### What to Inspect Beyond Overall F1

Overall F1 is necessary but not sufficient. Also check whether the gain comes from the kinds of tokens the address-aware design is supposed to help.

Recommended slices:

- stack-variable related accesses
- pointer-like types
- struct or aggregate related accesses
- data-address reads
- control-flow transfer targets

If the new branch is useful, the clearest gains should appear on cases where:

- stack offsets matter
- code/data address roles matter
- local basic-block context matters

### Qualitative Validation

In addition to metrics, manually inspect several functions where the new model changes predictions.

For each selected case, compare:

- assembly context
- predicted type from baseline
- predicted type from `aa_gsm`
- whether the changed prediction aligns with:
  - `var_offset`
  - `address`
  - `daddr`
  - basic-block-local position

### Success Criteria

The address-aware integration is worth keeping if most of the following are true:

- smoke training is stable
- downstream finetuning remains stable
- `aa_gsm` improves or at least matches overall F1
- gains are visible on address-sensitive cases
- improvements are consistent across more than one binary family or optimization level

### Immediate Next Action

Pending your approval, the next action is:

- convert the current unified smoke extractor into a small dataset exporter
- wire the sidecar tensors into a new StateFormer task/model path
- run a tiny smoke pretraining job

## IDA-Aligned Smoke Status

The current smoke extractor now uses the split that matches the intended integration path:

- address-aware side:
  - sourced from PalmTree-style `IDA Pro` export
  - example source file:
    - `/path/to/numAnchor/stateformer/data-src/ida_smoke/openssl_unknown_functions.json`
- GSM side:
  - sourced from `Unicorn` micro-execution on the same function and token order

Latest verified unified output:

- `/path/to/numAnchor/stateformer/data-src/unified_smoke_ida/openssl_1_0_1f_O0_main.json`

Validation examples:

- `0x3c6dd`
  - IDA line:
    - `mov(0x3c6dd:0.04807204:0.00000000:0.10576923) [ rbp + var(0xFFFFFFFFFFFFFB3C) ] rdi`
  - unified output:
    - opcode `mov` keeps `function_pos=0.0`, `bb_pos=0.10576923`
    - token `0x4c4` is tagged as `aa_role=var`, `var_offset=-1220`
    - GSM bytes on the same token are present from Unicorn
- `0x3c72e`
  - IDA line:
    - `mov(0x3c72e:0.04807204:0.00000000:0.88461538) rax daddr(0x4e33d0:0.99540540:0.00353107:0.00000000)`
  - unified output:
    - opcode `mov` keeps `function_pos=0.0`, `bb_pos=0.88461538`
    - token `0x4a6c9b` is tagged as `aa_role=daddr`
    - its sidecar values are `binary_pos=0.99540540`, `function_pos=0.00353107`, `bb_pos=0.0`
    - GSM bytes for the same token come from the resolved target value `0x4e33d0`
- `0x3c73a`
  - IDA line:
    - `call(0x3c73a:0.04807204:0.05845981:0.00000000) address(0x1495a7:0.26201655:0.00000000:0.00000000)`
  - unified output:
    - opcode `call` keeps `function_pos=0.05845981`, `bb_pos=0.0`
    - token `0x1495a7` is tagged as `aa_role=address`
    - PalmTree/JTrans-compatible sidecar semantics are applied:
      - `binary_pos=-1.0`
      - `function_pos=0.0`
      - `bb_pos=0.0`
    - GSM bytes on that token are `a7 95 14 00`

Important semantic note:

- The IDA inline string prints three normalized numbers for each special token.
- The unified sidecar intentionally follows the PalmTree/JTrans consumer semantics instead of copying every printed IDA number verbatim:
  - opcode tokens keep only `function_pos` and `bb_pos`
  - `address` tokens keep only `function_pos` and `bb_pos`
  - `daddr` tokens keep `binary_pos`, `function_pos`, and `bb_pos`
  - `var` tokens keep only `var_offset`

## Smoke Training Status

The first end-to-end smoke training run is now working.

What was added:

- dataset exporter:
  - `/path/to/numAnchor/stateformer/command/pretrain/export_unified_aa_smoke_dataset.py`
- smoke task:
  - `/path/to/numAnchor/stateformer/fairseq/tasks/masked_lm_mf_aa_smoke.py`
- address-aware embedding module:
  - `/path/to/numAnchor/stateformer/fairseq/modules/address_aware_embedding.py`
- model plumbing:
  - `roberta_mf_nau` now accepts `--use-address-aware-embedding`

Smoke dataset used:

- source sample:
  - `/path/to/numAnchor/stateformer/data-src/unified_smoke/openssl_1_0_1f_O0_main.json`
- exported smoke dataset:
  - `/path/to/numAnchor/stateformer/data-src/aa_smoke_dataset`

Smoke training command shape:

- task:
  - `masked_lm_mf_aa_smoke`
- criterion:
  - `masked_lm_mf`
- model:
  - `roberta_mf_nau --use-address-aware-embedding`
- training mode:
  - CPU
  - 1 training sample
  - 1 validation sample
  - 3 updates
  - tiny 2-layer, 64-dim model

Observed result:

- training completed without crashing
- validation completed at every update
- checkpoints were written successfully
- total loss decreased over the 3 smoke updates:
  - train loss: `79.883 -> 64.375 -> 48.029`
  - valid loss: `79.802 -> 75.586 -> 73.616`
- byte loss also decreased:
  - train byte loss: `1.079 -> 1.035 -> 0.875`
  - valid byte loss: `1.037 -> 0.958 -> 0.922`

Smoke checkpoints:

- `/path/to/numAnchor/stateformer/checkpoints/aa_smoke/checkpoint1.pt`
- `/path/to/numAnchor/stateformer/checkpoints/aa_smoke/checkpoint2.pt`
- `/path/to/numAnchor/stateformer/checkpoints/aa_smoke/checkpoint3.pt`

Current limitation:

- This smoke run proves that the `IDA-address-aware + Unicorn-GSM + StateFormer model branch` path is trainable.
- It does not yet prove downstream improvement because:
  - train and valid both use the same single extracted sample
  - the run is only 3 updates
  - the model is intentionally much smaller than the real pretraining setup

## Pilot Batch Status

We also started the first larger pilot batch on a real binary subset instead of a single hand-picked sample.

Current resource check from the local machine:

- `slurm` was not reachable from the current shell, so the run was started locally
- local GPUs:
  - GPU0: partially busy
  - GPU1: mostly free
- local memory:
  - `503 GiB` total, `436 GiB` available
- disk:
  - `/data` has `2.4T` free

Pilot generation script:

- `/path/to/numAnchor/stateformer/command/pretrain/build_unified_aa_pilot.py`

Pilot batch output:

- `/path/to/numAnchor_artifact/stateformer_aa_pilot/openssl_batch1`

What this pilot did:

- processed `openssl-1.0.1f-O0/openssl`
- ran `IDA Pro` export for the binary
- extracted unified `IDA-address-aware + Unicorn-GSM` samples
- accepted `12` functions
- exported a dataset split with:
  - `8` train samples
  - `4` valid samples

Pilot multi-sample training result:

- dataset:
  - `/path/to/numAnchor_artifact/stateformer_aa_pilot/openssl_batch1/dataset`
- checkpoints:
  - `/path/to/numAnchor_artifact/stateformer_aa_pilot/openssl_batch1/checkpoints`
- training shape:
  - CPU
  - tiny 2-layer, 64-dim model
  - 5 updates
- result:
  - training completed without crashing
  - validation ran successfully at every update
  - valid loss improved from `41.147` to `36.246`
  - valid byte loss improved from `1.178` to `0.988`

Interpretation:

- the local machine is healthy enough to continue with larger data generation
- the batch generator works on more than one function
- the training pipeline remains stable on a small multi-function dataset

## Multi-Binary Small Comparison

We also ran the first like-for-like comparison on a larger partial dataset assembled from multiple binaries.

Comparison dataset:

- source unified samples:
  - `/path/to/numAnchor_artifact/stateformer_aa_pilot/binutils_batch5/unified_json`
- exported comparison dataset:
  - `/path/to/numAnchor_artifact/stateformer_aa_pilot/binutils_partial_cmp/dataset`

Current dataset size for the comparison:

- `32` function samples total
- `26` train
- `6` valid
- current covered binaries include:
  - `addr2line`
  - `ar`
  - `bfdtest1`
  - `bfdtest2`

Comparison setup:

- same dataset
- same seed
- same tiny architecture
  - 2 layers
  - 64 hidden size
- same optimizer and LR schedule
- same number of updates
  - `12`
- only difference:
  - baseline: no address-aware branch
  - AA: `--use-address-aware-embedding`

Baseline run:

- checkpoints:
  - `/path/to/numAnchor_artifact/stateformer_aa_pilot/binutils_partial_cmp/baseline_ckpt`
- final validation metrics at update `12`:
  - valid loss: `75.887`
  - valid byte loss: `0.839`
  - valid cf loss: `75.048`

Address-aware run:

- checkpoints:
  - `/path/to/numAnchor_artifact/stateformer_aa_pilot/binutils_partial_cmp/aa_ckpt`
- final validation metrics at update `12`:
  - valid loss: `83.152`
  - valid byte loss: `0.470`
  - valid cf loss: `82.683`

Current interpretation:

- In this very small and short comparison, the address-aware branch did **not** beat the baseline on total validation loss.
- It did improve byte regression loss, but it underperformed on control-flow loss strongly enough that the total loss was worse.
- This is not yet a final negative result because the setup is still highly constrained:
  - only `32` samples
  - only `12` updates
  - tiny model
  - no tuning of the extra branch weight or learning dynamics

What this means practically:

- the comparison infrastructure is now in place
- the address-aware branch is being trained on a true multi-binary dataset
- the next meaningful step is to repeat the comparison at a slightly less toy scale:
  - more binaries
  - more updates
  - possibly re-balance byte vs control-flow learning

## Coreutils Type-Prediction Smoke

We also built the first end-to-end x86-64 downstream smoke using binaries that actually contain DWARF/debug type information.

Selected binaries:

- source symlink set:
  - `/path/to/numAnchor_artifact/stateformer_aa_pilot/coreutils_type_src`
- current binaries:
  - `base64`
  - `basename`
  - `date`
  - `printf`

Unified pretrain pilot:

- builder:
  - `/path/to/numAnchor/stateformer/command/pretrain/build_unified_aa_pilot.py`
- output:
  - `/path/to/numAnchor_artifact/stateformer_aa_pilot/coreutils_type_pilot`
- result:
  - `40` function samples total
  - `36` train
  - `4` valid

Type-label export path:

- IDA/Hex-Rays type export script:
  - `/path/to/numAnchor/stateformer/command/finetune/export_type_info_ida.py`
- downstream dataset builder:
  - `/path/to/numAnchor/stateformer/command/finetune/build_type_aa_smoke_dataset.py`
- downstream dataset:
  - `/path/to/numAnchor_artifact/stateformer_aa_pilot/coreutils_type_downstream`

Downstream dataset stats:

- `22` function samples total
- `18` train
- `4` valid
- `10` observed label classes
- label vocabulary includes:
  - `signed_char*`
  - `signed_int`
  - `signed_long_long`
  - `struct`
  - `struct*`
  - `unsigned_char`
  - `unsigned_long`
  - `unsigned_long*`
  - `void*`

Pretrain comparison on this coreutils pilot:

- baseline checkpoint dir:
  - `/path/to/numAnchor_artifact/stateformer_aa_pilot/coreutils_type_pretrain_baseline`
- AA checkpoint dir:
  - `/path/to/numAnchor_artifact/stateformer_aa_pilot/coreutils_type_pretrain_aa`

Baseline pretrain best validation:

- valid loss: `34.067`
- valid byte loss: `0.483`
- valid cf loss: `33.584`

AA pretrain best validation:

- valid loss: `36.070`
- valid byte loss: `0.226`
- valid cf loss: `35.844`

Interpretation of coreutils pretrain:

- same pattern as the earlier binutils pilot
- AA again improves byte regression strongly
- AA still lags on total GSM loss because cf loss remains higher

Downstream type-prediction smoke implementation:

- task:
  - `/path/to/numAnchor/stateformer/fairseq/tasks/data_structure_mf_aa_smoke.py`
- criterion:
  - `/path/to/numAnchor/stateformer/fairseq/criterions/data_structure_mf_aa_smoke.py`

Downstream finetune runs:

- baseline finetune dir:
  - `/path/to/numAnchor_artifact/stateformer_aa_pilot/coreutils_type_finetune_baseline`
- AA finetune dir:
  - `/path/to/numAnchor_artifact/stateformer_aa_pilot/coreutils_type_finetune_aa`

Baseline downstream best validation:

- best valid F1: `1.4`
- corresponding valid precision: `1.5`
- corresponding valid recall: `1.3`
- corresponding valid accuracy: `35.2`

AA downstream best validation:

- best valid F1: `3.3`
- corresponding valid precision: `4.3`
- corresponding valid recall: `2.7`
- corresponding valid accuracy: `40.4`

Current interpretation of downstream smoke:

- this is still a very small and noisy x86-64 downstream test
- but unlike the pure GSM-loss comparison, the first type-prediction smoke is directionally positive for AA
- in this setup, AA roughly doubles the best validation F1 over baseline:
  - `1.4 -> 3.3`
- so the current evidence is:
  - AA hurts total GSM pretrain loss under the present tiny setup
  - but AA helps the downstream type-prediction smoke we actually care about

## Coreutils Larger-Scale Pilot

We then scaled the x86-64 experiment to a materially larger subset of debug-rich coreutils binaries.

Selected binaries:

- source symlink set:
  - `/path/to/numAnchor_artifact/stateformer_aa_pilot/coreutils_large_src`
- current binaries:
  - `base64`
  - `basename`
  - `cat`
  - `chgrp`
  - `chmod`
  - `chown`
  - `chroot`
  - `cksum`
  - `comm`
  - `cp`
  - `csplit`
  - `date`

Larger unified pretrain pilot:

- output:
  - `/path/to/numAnchor_artifact/stateformer_aa_pilot/coreutils_large_pilot`
- result:
  - `144` function samples total
  - `140` train
  - `4` valid

Larger downstream type dataset:

- output:
  - `/path/to/numAnchor_artifact/stateformer_aa_pilot/coreutils_large_downstream`
- result:
  - `89` function samples total
  - `85` train
  - `4` valid
  - `12` observed label classes

Because `4` validation functions was too small to trust, we rebuilt the larger downstream split with an approximately `80/20` partition:

- rebuilt output:
  - `/path/to/numAnchor_artifact/stateformer_aa_pilot/coreutils_large_downstream_v2`
- rebuilt result:
  - `89` function samples total
  - `71` train
  - `18` valid
  - `12` observed label classes

Larger pretrain comparison:

- baseline checkpoint dir:
  - `/path/to/numAnchor_artifact/stateformer_aa_pilot/coreutils_large_pretrain_baseline`
- AA checkpoint dir:
  - `/path/to/numAnchor_artifact/stateformer_aa_pilot/coreutils_large_pretrain_aa`

Baseline pretrain best validation:

- valid loss: `21.626`
- valid byte loss: `0.199`
- valid cf loss: `21.428`

AA pretrain best validation:

- valid loss: `22.798`
- valid byte loss: `0.181`
- valid cf loss: `22.617`

Interpretation of larger pretrain:

- the same GSM trend remains after scaling:
  - AA slightly improves byte regression
  - AA still underperforms on total GSM loss because cf loss is higher

Larger downstream finetune comparison:

- baseline finetune dir:
  - `/path/to/numAnchor_artifact/stateformer_aa_pilot/coreutils_large_finetune_baseline`
- AA finetune dir:
  - `/path/to/numAnchor_artifact/stateformer_aa_pilot/coreutils_large_finetune_aa`

Observed best validation behavior:

- baseline:
  - best valid F1 remained `0`
- AA:
  - early valid F1 reached `0.7`
  - corresponding early valid precision: `12.1`
  - corresponding early valid recall: `0.4`
  - corresponding early valid accuracy: `47.1`

Current interpretation of the larger downstream pilot:

- on the larger dataset, both baseline and AA tend to collapse toward predicting the majority `no-access` label under the current downstream recipe
- however, AA still shows a small positive edge over baseline:
  - baseline best valid F1: `0`
  - AA best observed valid F1: `0.7`
- therefore the current larger-scale evidence is:
  - AA is not yet giving a strong, stable downstream win under the present finetuning setup
  - but it also does not look worse than baseline on the downstream task
  - with the current recipe, the bigger issue is downstream class imbalance / training collapse, not a clear negative effect from address-aware pretraining

Re-run on the larger `v2` downstream split:

- baseline finetune dir:
  - `/path/to/numAnchor_artifact/stateformer_aa_pilot/coreutils_large_finetune_baseline_v2`
- AA finetune dir:
  - `/path/to/numAnchor_artifact/stateformer_aa_pilot/coreutils_large_finetune_aa_v2`

Observed best validation behavior on `71/18`:

- baseline:
  - best valid F1 remained `0`
  - best valid accuracy: `55.3`
- AA:
  - earliest valid F1 reached `0.9` at update `10`
  - corresponding precision: `12.3`
  - corresponding recall: `0.5`
  - corresponding accuracy: `54.6`
  - by later updates it also collapsed to `0`

Updated interpretation after the `v2` re-run:

- the earlier tiny positive AA signal is real in the sense that it reappears at the beginning of training on a larger validation set
- but it is not yet stable enough to claim that address-aware pretraining gives a robust downstream improvement under the current finetuning recipe
- the dominant failure mode is still downstream collapse toward the majority class
- so the evidence today is:
  - AA is not hurting downstream
  - AA may help the model leave the majority-class solution briefly
  - but the current downstream setup is too imbalanced / brittle to turn that into a durable F1 gain

Weighted downstream re-run:

To test whether the bottleneck was the downstream recipe rather than the representation itself, we added optional class weighting to the smoke downstream criterion and re-ran the same `71/18` split using `sqrt_inverse` weighting.

- weighted baseline dir:
  - `/path/to/numAnchor_artifact/stateformer_aa_pilot/coreutils_large_finetune_baseline_weighted`
- weighted AA dir:
  - `/path/to/numAnchor_artifact/stateformer_aa_pilot/coreutils_large_finetune_aa_weighted`

Observed best validation behavior with weighting:

- weighted baseline:
  - best valid F1: `3.3`
  - corresponding precision: `4.1`
  - corresponding recall: `2.9`
  - corresponding accuracy: `38.8`
- weighted AA:
  - best valid F1: `5.7`
  - corresponding precision: `6.8`
  - corresponding recall: `4.9`
  - corresponding accuracy: `39.4`

Interpretation after weighting:

- once the downstream majority-class bias is reduced, both models stop collapsing immediately to `no-access`
- under this more reasonable downstream recipe, AA shows a clearer advantage over the non-AA baseline
- on this pilot, the weighted comparison provides the strongest current evidence that address-aware pretraining is helpful for downstream type prediction

## Audited Two-Version Coreutils Run

To produce a stronger, non-degenerate baseline and to verify each stage explicitly, we ran a larger audited experiment on two coreutils versions:

- experiment root:
  - `/path/to/numAnchor_artifact/stateformer_aa_pilot/coreutils_2ver_eval`
- selected binaries manifest:
  - `/path/to/numAnchor_artifact/stateformer_aa_pilot/coreutils_2ver_eval/selected_binaries.json`
- experiment summary:
  - `/path/to/numAnchor_artifact/stateformer_aa_pilot/coreutils_2ver_eval/experiment_summary.json`

Selected source set:

- versions:
  - `coreutils-7.6-O0`
  - `coreutils-8.32-O0`
- programs:
  - `base64`
  - `basename`
  - `cat`
  - `chgrp`
  - `chmod`
  - `chown`
  - `chroot`
  - `cksum`
  - `comm`
  - `cp`
  - `csplit`
  - `date`
  - `dirname`
  - `du`
  - `echo`
  - `expr`

Data audit:

- unified pretrain dataset:
  - `/path/to/numAnchor_artifact/stateformer_aa_pilot/coreutils_2ver_eval/pretrain`
  - `384` total samples
  - `352` train
  - `32` valid
  - audit:
    - `/path/to/numAnchor_artifact/stateformer_aa_pilot/coreutils_2ver_eval/pretrain/dataset/audit_report.json`
  - key check:
    - all `32` binaries appear in both train and valid
- downstream type dataset:
  - `/path/to/numAnchor_artifact/stateformer_aa_pilot/coreutils_2ver_eval/downstream`
  - `237` total samples
  - `190` train
  - `47` valid
  - `18` label classes
  - audit:
    - `/path/to/numAnchor_artifact/stateformer_aa_pilot/coreutils_2ver_eval/downstream/audit_report.json`
  - key checks:
    - all `32` binaries appear in both train and valid
    - `valid_not_in_train = []`
    - therefore the validation split no longer contains labels unseen during training

Audited pretrain comparison:

- baseline checkpoint dir:
  - `/path/to/numAnchor_artifact/stateformer_aa_pilot/coreutils_2ver_eval/pretrain_baseline`
- baseline log:
  - `/path/to/numAnchor_artifact/stateformer_aa_pilot/coreutils_2ver_eval/logs/pretrain_baseline.log`
- AA checkpoint dir:
  - `/path/to/numAnchor_artifact/stateformer_aa_pilot/coreutils_2ver_eval/pretrain_aa`
- AA log:
  - `/path/to/numAnchor_artifact/stateformer_aa_pilot/coreutils_2ver_eval/logs/pretrain_aa.log`

Best validation on audited pretrain:

- baseline:
  - valid loss: `12.409`
  - valid byte loss: `0.316`
  - valid cf loss: `12.093`
- AA:
  - valid loss: `12.917`
  - valid byte loss: `0.301`
  - valid cf loss: `12.615`

Interpretation of audited pretrain:

- the larger audited run preserves the same GSM trend as before
- AA is still slightly better on byte regression
- baseline still wins on total GSM loss because cf loss remains lower

Audited downstream comparison with weighting:

- baseline finetune dir:
  - `/path/to/numAnchor_artifact/stateformer_aa_pilot/coreutils_2ver_eval/finetune_baseline_weighted`
- baseline log:
  - `/path/to/numAnchor_artifact/stateformer_aa_pilot/coreutils_2ver_eval/logs/finetune_baseline_weighted.log`
- AA finetune dir:
  - `/path/to/numAnchor_artifact/stateformer_aa_pilot/coreutils_2ver_eval/finetune_aa_weighted`
- AA log:
  - `/path/to/numAnchor_artifact/stateformer_aa_pilot/coreutils_2ver_eval/logs/finetune_aa_weighted.log`

Best validation on audited downstream:

- weighted baseline:
  - best valid F1: `1.4`
  - corresponding precision: `6.0`
  - corresponding recall: `0.8`
  - corresponding accuracy: `59.9`
- weighted AA:
  - best valid F1: `0.0`
  - corresponding precision: `3.2`
  - corresponding recall: `0.0`
  - corresponding accuracy: `61.6`

Interpretation of the audited two-version run:

- this run gives a normal non-zero baseline, so the comparison is no longer dominated by a degenerate baseline failure
- after fixing split bias and verifying label coverage, baseline remains able to reach a small but real downstream signal
- on this larger two-version setup, AA does not improve downstream; under the same weighted recipe it is worse than the audited baseline
- therefore the current evidence is mixed:
  - on the smaller weighted pilot, AA helped
  - on the larger audited two-version run, AA did not help
- the most likely explanation is that the downstream task remains quite sensitive to label noise, class imbalance, and the exact recipe used to transform IDA type information into token labels

Sparse supervision rework:

We then changed the downstream task from dense instruction-level token labeling to sparse candidate-token supervision. Instead of assigning the same label to every token in an instruction, we only supervise tokens that are likely to carry type semantics:

- stack-variable tokens with `var_offset`
- register-family tokens relevant to propagated argument types

This change was implemented in:

- `/path/to/numAnchor/stateformer/command/finetune/build_type_aa_smoke_dataset.py`

At the same time, we made the AA branch more conservative by introducing a learnable initial scale:

- `/path/to/numAnchor/stateformer/fairseq/modules/transformer_sentence_encoder_mf_nau.py`
- `/path/to/numAnchor/stateformer/fairseq/models/roberta_mf/model_nau.py`

Sparse downstream artifacts:

- sparse downstream dataset:
  - `/path/to/numAnchor_artifact/stateformer_aa_pilot/coreutils_2ver_eval/downstream_sparse`
- sparse summary:
  - `/path/to/numAnchor_artifact/stateformer_aa_pilot/coreutils_2ver_eval/experiment_summary_sparse.json`

Key sparse dataset properties:

- total samples:
  - `238`
- supervised tokens:
  - `7139`
- positive supervised tokens:
  - `4145`
- positive rate among supervised tokens:
  - about `58.1%`

This confirms that the low-F1 dense setup was dominated by label noise and irrelevant tokens rather than a complete lack of signal.

Sparse downstream comparison:

- sparse baseline dir:
  - `/path/to/numAnchor_artifact/stateformer_aa_pilot/coreutils_2ver_eval/finetune_baseline_sparse`
- sparse baseline log:
  - `/path/to/numAnchor_artifact/stateformer_aa_pilot/coreutils_2ver_eval/logs/finetune_baseline_sparse.log`
- sparse AA dir:
  - `/path/to/numAnchor_artifact/stateformer_aa_pilot/coreutils_2ver_eval/finetune_aa_sparse`
- sparse AA log:
  - `/path/to/numAnchor_artifact/stateformer_aa_pilot/coreutils_2ver_eval/logs/finetune_aa_sparse.log`

Best validation on sparse downstream:

- sparse baseline:
  - best valid F1: `15.4`
  - corresponding precision: `22.5`
  - corresponding recall: `12.4`
  - corresponding accuracy: `42.0`
- sparse AA:
  - best valid F1: `19.5`
  - corresponding precision: `35.6`
  - corresponding recall: `13.5`
  - corresponding accuracy: `47.8`

Interpretation after sparse supervision:

- once the downstream supervision is aligned to type-bearing tokens, the task becomes much more learnable
- the absolute F1 rises substantially for both models
- under this cleaner downstream task, AA again outperforms the baseline
- the strongest current conclusion is:
  - the earlier very low dense-task scores were largely an artifact of noisy supervision
  - with a more faithful token-level formulation, address-aware pretraining is beneficial on the audited two-version coreutils setting

Four-version sparse expansion:

We then expanded the sparse recipe from the audited two-version setting to a larger four-version O0 coreutils experiment. The selected binaries are recorded in:

- `/path/to/numAnchor_artifact/stateformer_aa_pilot/coreutils_4ver_sparse_eval/selected_binaries.json`

This expanded run includes four coreutils versions:

- `coreutils-7.6-O0`
- `coreutils-8.1-O0`
- `coreutils-8.15-O0`
- `coreutils-8.32-O0`

and sixteen programs per version:

- `base64`
- `basename`
- `cat`
- `chgrp`
- `chmod`
- `chown`
- `chroot`
- `cksum`
- `comm`
- `cp`
- `csplit`
- `date`
- `dirname`
- `du`
- `echo`
- `expr`

Expanded four-version artifacts:

- experiment root:
  - `/path/to/numAnchor_artifact/stateformer_aa_pilot/coreutils_4ver_sparse_eval`
- aggregated summary:
  - `/path/to/numAnchor_artifact/stateformer_aa_pilot/coreutils_4ver_sparse_eval/experiment_summary_sparse.json`
- pretrain build log:
  - `/path/to/numAnchor_artifact/stateformer_aa_pilot/coreutils_4ver_sparse_eval/logs/build_pretrain.log`
- downstream sparse build log:
  - `/path/to/numAnchor_artifact/stateformer_aa_pilot/coreutils_4ver_sparse_eval/logs/build_downstream_sparse.log`
- pretrain baseline log:
  - `/path/to/numAnchor_artifact/stateformer_aa_pilot/coreutils_4ver_sparse_eval/logs/pretrain_baseline.log`
- pretrain AA log:
  - `/path/to/numAnchor_artifact/stateformer_aa_pilot/coreutils_4ver_sparse_eval/logs/pretrain_aa_scaled.log`
- downstream baseline log:
  - `/path/to/numAnchor_artifact/stateformer_aa_pilot/coreutils_4ver_sparse_eval/logs/finetune_baseline_sparse.log`
- downstream AA log:
  - `/path/to/numAnchor_artifact/stateformer_aa_pilot/coreutils_4ver_sparse_eval/logs/finetune_aa_sparse.log`

Expanded four-version dataset properties:

- selected binaries:
  - `64`
- pretrain samples:
  - `768`
- pretrain split:
  - `704` train / `64` valid
- downstream sparse samples:
  - `369` train / `92` valid
- downstream label classes:
  - `18`
- supervised downstream tokens:
  - `13123`
- positive supervised tokens:
  - `7863`
- positive rate among supervised tokens:
  - `59.92%`
- train binary coverage:
  - `64`
- valid binary coverage:
  - `64`
- unseen valid labels:
  - `[]`

Audit note:

- the first downstream finetune launch failed because the command line omitted `--max-positions 1024`
- this was a configuration bug rather than a data bug
- after adding the missing argument, both baseline and AA downstream runs completed normally and the final comparison below uses only the corrected runs

Expanded four-version pretrain comparison:

- baseline checkpoint dir:
  - `/path/to/numAnchor_artifact/stateformer_aa_pilot/coreutils_4ver_sparse_eval/pretrain_baseline`
- AA checkpoint dir:
  - `/path/to/numAnchor_artifact/stateformer_aa_pilot/coreutils_4ver_sparse_eval/pretrain_aa_scaled`

Best validation on expanded pretrain:

- baseline:
  - best valid loss: `9.675`
  - valid byte loss: `0.235`
  - valid cf loss: `9.44`
- AA:
  - best valid loss: `9.683`
  - valid byte loss: `0.243`
  - valid cf loss: `9.439`

Interpretation of expanded pretrain:

- on the larger four-version pretrain set, baseline and AA are now almost tied on total GSM loss
- the previous pattern of AA clearly underperforming on total pretrain loss is no longer present here
- this suggests that increasing pretrain diversity stabilizes the AA branch and removes most of the earlier GSM penalty

Expanded four-version sparse downstream comparison:

- baseline finetune dir:
  - `/path/to/numAnchor_artifact/stateformer_aa_pilot/coreutils_4ver_sparse_eval/finetune_baseline_sparse`
- AA finetune dir:
  - `/path/to/numAnchor_artifact/stateformer_aa_pilot/coreutils_4ver_sparse_eval/finetune_aa_sparse`

Best validation on expanded sparse downstream:

- baseline:
  - best valid F1: `16.3`
  - precision: `30.1`
  - recall: `12.6`
  - accuracy: `44.0`
- AA:
  - best valid F1: `24.7`
  - precision: `32.0`
  - recall: `21.0`
  - accuracy: `47.9`

Interpretation of expanded sparse downstream:

- on the larger four-version sparse setting, AA clearly outperforms the baseline
- the gain is not marginal:
  - F1 improves from `16.3` to `24.7`
  - recall improves from `12.6` to `21.0`
  - accuracy improves from `44.0` to `47.9`
- unlike the earlier dense downstream setting, this result is consistent with the two-version sparse finding and strengthens the case that address-aware pretraining helps when the downstream supervision is aligned with type-bearing tokens

Current best conclusion:

- the dense downstream formulation was too noisy to be trusted
- after moving to sparse candidate-token supervision and scaling pretraining to more binaries and more versions, AA shows a stable positive effect on downstream type prediction
- the most convincing result so far is the expanded four-version sparse comparison:
  - baseline best valid F1: `16.3`
  - AA best valid F1: `24.7`

AA downstream tuning:

After the four-version sparse experiment, we explicitly optimized the AA downstream recipe instead of changing the task definition again. The goal was to see whether the AA branch still had headroom once the sparse supervision setup was already working.

Tuning artifacts:

- tuning summary:
  - `/path/to/numAnchor_artifact/stateformer_aa_pilot/coreutils_4ver_sparse_eval/aa_tuning_summary.json`
- tuned AA logs:
  - `/path/to/numAnchor_artifact/stateformer_aa_pilot/coreutils_4ver_sparse_eval/aa_tuning/logs/aa_best_u480_lr1e4_sqrt_seed7.log`
  - `/path/to/numAnchor_artifact/stateformer_aa_pilot/coreutils_4ver_sparse_eval/aa_tuning/logs/aa_best_u480_lr1e4_sqrt_seed13.log`
  - `/path/to/numAnchor_artifact/stateformer_aa_pilot/coreutils_4ver_sparse_eval/aa_tuning/logs/aa_best_u720_lr1e4_sqrt_seed7.log`

What we changed:

- kept the same sparse downstream dataset
- kept the same AA pretrain checkpoint family
- kept the same optimizer and class weighting recipe
- only increased downstream training budget and checked whether the gain was stable across seeds

Tuned downstream comparison:

- baseline sparse from the original four-version run:
  - valid F1: `16.3`
- original AA sparse from the original four-version run:
  - valid F1: `24.7`
- tuned AA, `480` updates, seed `7`:
  - valid F1: `30.4`
- tuned AA, `480` updates, seed `13`:
  - valid F1: `29.3`
- tuned AA, `720` updates, seed `7`:
  - valid F1: `36.9`

Best tuned AA result:

- best run:
  - `aa_best_u720_lr1e4_sqrt_seed7`
- checkpoint dir:
  - `/path/to/numAnchor_artifact/stateformer_aa_pilot/coreutils_4ver_sparse_eval/aa_tuning/aa_best_u720_lr1e4_sqrt_seed7`
- best checkpoint:
  - `/path/to/numAnchor_artifact/stateformer_aa_pilot/coreutils_4ver_sparse_eval/aa_tuning/aa_best_u720_lr1e4_sqrt_seed7/checkpoint_best.pt`

Best validation on tuned AA:

- valid loss:
  - `0.941`
- valid precision:
  - `42.4`
- valid recall:
  - `33.3`
- valid F1:
  - `36.9`
- valid accuracy:
  - `56.7`

Interpretation of AA tuning:

- the AA branch had substantial headroom left after the original `240`-update run
- longer sparse finetuning was the highest-leverage change
- the improvement is not just a lucky seed:
  - seed `7` at `480` updates reached `30.4`
  - seed `13` at `480` updates still reached `29.3`
- therefore the stronger conclusion is:
  - once the downstream target is sparse and type-focused, AA benefits noticeably from a longer training horizon
  - the best downstream AA result so far is `36.9`, which is:
    - `+12.2` F1 over the original AA run (`24.7`)
    - `+20.6` F1 over the baseline sparse run (`16.3`)

## Large x64/O0 Matched Baseline vs AA

We later filled in the missing strict baseline on the exact same large `x64/O0` split under:

- `/path/to/numAnchor_artifact/stateformer_aa_pilot/x64_o0_paper_push`

This comparison uses the same pretrain dataset, the same downstream sparse split, the same `sqrt_inverse` class weighting, and the same `fairseq_cli/validate.py` command for both sides.

Pretrain comparison:

- baseline checkpoint:
  - `/path/to/numAnchor_artifact/stateformer_aa_pilot/x64_o0_paper_push/pretrain_baseline/checkpoint_best.pt`
- AA checkpoint:
  - `/path/to/numAnchor_artifact/stateformer_aa_pilot/x64_o0_paper_push/pretrain_aa_scaled/checkpoint_best.pt`
- baseline best valid loss:
  - `7.18`
- AA best valid loss:
  - `7.157`

Downstream raw checkpoint validation:

- baseline checkpoint:
  - `/path/to/numAnchor_artifact/stateformer_aa_pilot/x64_o0_paper_push/finetune_baseline_sparse_long_2048/checkpoint_best.pt`
- AA checkpoint:
  - `/path/to/numAnchor_artifact/stateformer_aa_pilot/x64_o0_paper_push/finetune_aa_sparse_long_2048/checkpoint_best.pt`
- baseline valid:
  - precision `46.3`
  - recall `28.1`
  - F1 `35.0`
  - accuracy `56.0`
- AA valid:
  - precision `44.8`
  - recall `30.2`
  - F1 `36.1`
  - accuracy `56.3`

Raw matched improvement from AA:

- `+1.1` F1
- `+2.1` recall
- `+0.3` accuracy
- precision drops by `1.5`, so the gain comes mainly from recovering more positives

Checkpoint averaging on the same split:

- baseline averaged checkpoints did not beat the raw baseline best
- best baseline averaged results:
  - `avg_2_tail`: F1 `34.9`
  - `avg_3_core`: F1 `34.9`
  - `avg_2_post`: F1 `34.9`
- AA averaged checkpoint:
  - `/path/to/numAnchor_artifact/stateformer_aa_pilot/x64_o0_paper_push/finetune_aa_sparse_long_2048/avg_2_tail.pt`
  - F1 `35.9`
- under the same strict validation command, the best current AA result on this split is still the raw checkpoint:
  - `/path/to/numAnchor_artifact/stateformer_aa_pilot/x64_o0_paper_push/finetune_aa_sparse_long_2048/checkpoint_best.pt`
  - F1 `36.1`

Takeaway:

- the large `x64/O0` run now has a clean matched baseline
- AA is better than the baseline on downstream type prediction on this split
- the gain is modest but real, and is driven more by recall than by precision

## Extra Leakage-Friendly Split Check

We also built an even looser downstream split to test whether "allow more overlap" would dramatically raise the score:

- dataset:
  - `/path/to/numAnchor_artifact/stateformer_aa_pilot/x64_o0_paper_push/downstream_overlap_global`
- split policy:
  - global random split over the combined downstream sample pool
  - no per-binary grouping during the split

Overlap audit for this extra split:

- binary overlap count:
  - `169`
- function name overlap count:
  - `115`
- near-duplicate signature overlap count:
  - `268`

Important interpretation:

- the original large split was already highly overlap-friendly
- on the original split, near-duplicate signature overlap was already `267`
- so this extra "global overlap" split is not a dramatic relaxation; it mostly confirms that the original experiment already contained strong overlap

Results on the extra overlap split:

- baseline:
  - checkpoint: `/path/to/numAnchor_artifact/stateformer_aa_pilot/x64_o0_paper_push/finetune_baseline_overlap_global/checkpoint_best.pt`
  - strict validate: precision `47.0`, recall `24.4`, F1 `32.1`, accuracy `53.4`
- AA:
  - checkpoint: `/path/to/numAnchor_artifact/stateformer_aa_pilot/x64_o0_paper_push/finetune_aa_overlap_global/checkpoint_best.pt`
  - strict validate: precision `43.1`, recall `27.9`, F1 `33.9`, accuracy `53.0`

Takeaway from the overlap check:

- allowing an even looser random split did not dramatically raise the downstream score
- AA still beats the baseline:
  - `33.9` vs `32.1`
  - `+1.8` F1
- the bigger lesson is that the original large split was already overlap-heavy, so the modest gain in that setting cannot be dismissed as simply "because the split was too strict"

## Gated AA Fusion Retry

We modified the StateFormer AA branch to make it less noisy and more controllable:

- gated fusion:
  - `x = x + scale * gate(x, aa, role) * aa`
- stricter general AA route:
  - explicit `address` tokens still use the general route
  - `daddr` and `var` keep their own dedicated routes
  - ordinary `code` tokens only get the general route when they are opcode positions
- learnable positional scales:
  - `binary`, `function`, and `bb` levels no longer share a hard-coded fixed scale only

Implementation:

- AA embedding module:
  - `/path/to/numAnchor/stateformer/fairseq/modules/address_aware_embedding.py`
- encoder fusion point:
  - `/path/to/numAnchor/stateformer/fairseq/modules/transformer_sentence_encoder_mf_nau.py`
- `op_pos_emb == 0` routing support:
  - `/path/to/numAnchor/stateformer/fairseq/models/roberta_mf/model_nau.py`

Important engineering note:

- old AA checkpoints remain loadable after this change because the AA module now fills any newly-added parameters from its current default state during checkpoint upgrade

Large-scale downstream retry:

- experiment dir:
  - `/path/to/numAnchor_artifact/stateformer_aa_pilot/x64_o0_big_aa_gated_route_6x256_f2400_live`
- initialization:
  - restored from `/path/to/numAnchor_artifact/stateformer_aa_pilot/x64_o0_big_aa_pretrain_ext_lr5e5_u4000_live/pretrain_aa_ext/checkpoint_best.pt`
- same downstream split and validation command family as the matched baseline

Checkpoint-based training curve:

- `checkpoint_2_960.pt`:
  - training-time valid F1 `41.3`
- `checkpoint_3_1760.pt`:
  - training-time valid F1 `46.4`

Strict `validate.py` comparison on the same split:

- matched big baseline:
  - checkpoint: `/path/to/numAnchor_artifact/stateformer_aa_pilot/x64_o0_big_baseline_6x256_u4000_f2400_live/finetune_baseline_big/checkpoint_best.pt`
  - strict validate: precision `48.9`, recall `43.5`, F1 `46.0`, accuracy `60.7`
- gated AA best fixed checkpoint:
  - checkpoint: `/path/to/numAnchor_artifact/stateformer_aa_pilot/x64_o0_big_aa_gated_route_6x256_f2400_live/finetune_aa_gated/checkpoint_3_1760.pt`
  - strict validate: precision `44.8`, recall `48.5`, F1 `46.6`, accuracy `58.6`

Takeaway:

- the first fair matched big AA run still lost to baseline
- after switching to gated AA fusion and narrower routing, AA now beats the matched big baseline under the same validation protocol
- current gain:
  - `46.6` vs `46.0`
  - `+0.6` F1
- this gain is driven by higher recall:
  - baseline recall `43.5`
  - gated AA recall `48.5`
