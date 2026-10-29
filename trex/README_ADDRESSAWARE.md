# Trex AddressAware

## 1. What This README Covers

This README describes the Trex route as it is used in this repository now.
It does not try to reproduce the exact upstream Trex paper pipeline.

In the current `addressaware` project, Trex is a local comparison path built on
top of the prepared `Stateformer x86-64` dataset already stored on this
machine. The maintained use case is:

- pretrain Trex on Stateformer inline CFG text
- compare `baseline` and address-aware variants
- evaluate retrieval on fixed function-similarity pools

In the current project, Trex is mainly a pretraining plus retrieval-evaluation
route. It is not the main downstream function-similarity fine-tuning branch.

## 2. Current Modes

The current Trex adapter exposes four names in code, but only two are the main
project path:

- `baseline`
  - no address-aware side channel
  - `byte1..byte4` are filled with the blank token `##`
  - no AA sidecar is used
- `addressaware`
  - legacy compatibility mode
  - numeric values extracted from inline `opcode(...)`, `address(...)`,
    `daddr(...)`, and `var(...)` tokens are written into `byte1..byte4`
  - this mode is kept for comparison compatibility, but it is not the current
    primary address-aware route
- `addressaware_aa`
  - current primary address-aware Trex mode in this repository
  - the main eight Trex text fields stay compatible with the current baseline
    token/bin layout
  - address information is supplied through aligned AA sidecar files and
    `--use-address-aware-embedding`
- `native`
  - optional reference path closer to the original Trex byte and micro-trace
    style
  - separate from the current inline Stateformer workflow

For the current local project, the main Trex comparison is:

- `baseline` vs `addressaware_aa`

## 3. Important Files

- `command/stateformer_x64/prepare_stateformer_trex.py`
  - converts Stateformer inline CFG corpora into Trex multi-field text
  - supports `baseline`, `addressaware`, and `addressaware_aa`
- `command/stateformer_x64/train_baseline.sh`
  - launcher for baseline Trex pretraining
- `command/stateformer_x64/train_addressaware_aa.sh`
  - launcher for address-aware pretraining with AA sidecar inputs
- `command/stateformer_x64/evaluate_pool.py`
  - evaluates a pretrained Trex checkpoint on fixed `100`, `1k`, and `10k`
    retrieval pools
- `command/stateformer_x64/evaluate_pool_opt_splits.py`
  - optional evaluation split by optimization level
- `command/stateformer_x64/prepare_trex_native_pretrain.py`
  - helper for the separate `native` route
- `fairseq/tasks/trex.py`
  - loads aligned AA sidecar inputs when
    `--use-address-aware-embedding --aa-data-dir` is enabled
- `fairseq/modules/address_aware_embedding.py`
  - current address-aware embedding implementation used by
    `addressaware_aa`

## 4. Current Local Data and Artifact Layout

The current Trex route uses these local datasets:

- pretraining corpus
  - `/path/to/numAnchor_artifact/stateformer_x64_dataset/pretrain_ctx2/splits/train_cfg.txt`
  - `/path/to/numAnchor_artifact/stateformer_x64_dataset/pretrain_ctx2/splits/val_cfg.txt`
- retrieval evaluation
  - `/path/to/numAnchor_artifact/stateformer_x64_dataset/funcsim_block_ctx2/function_blocks.json`
  - `/path/to/numAnchor_artifact/stateformer_x64_dataset/funcsim_block_ctx2/funcsim_pairs.json`
  - `/path/to/numAnchor_artifact/stateformer_x64_dataset/funcsim_block_ctx2/pool_ids_100.json`
  - `/path/to/numAnchor_artifact/stateformer_x64_dataset/funcsim_block_ctx2/pool_ids_1k.json`
  - `/path/to/numAnchor_artifact/stateformer_x64_dataset/funcsim_block_ctx2/pool_ids_10k.json`

The project currently reuses these prepared Trex inputs:

- baseline text and binarized data
  - `/path/to/numAnchor_artifact/addressaware_runs/trex_stateformer_x64_20260414/baseline_text`
  - `/path/to/numAnchor_artifact/addressaware_runs/trex_stateformer_x64_20260414/baseline_bin`
- AA sidecar text for `addressaware_aa`
  - `/path/to/numAnchor_artifact/addressaware_runs/trex_stateformer_x64_20260414/addressaware_aa_text`

The preferred current pretrained checkpoints are:

- baseline
  - `/path/to/numAnchor_artifact/addressaware_runs/trex_stateformer_x64_768_pretrain3q_10h_20260422/baseline_pretrain_ckpt/checkpoint_best.pt`
  - `/path/to/numAnchor_artifact/addressaware_runs/trex_stateformer_x64_768_pretrain3q_10h_20260422/baseline_pretrain_ckpt/checkpoint_last.pt`
- `addressaware_aa`
  - `/path/to/numAnchor_artifact/addressaware_runs/trex_stateformer_x64_768_pretrain3q_10h_20260422/addressaware_aa_pretrain_ckpt/checkpoint_best.pt`
  - `/path/to/numAnchor_artifact/addressaware_runs/trex_stateformer_x64_768_pretrain3q_10h_20260422/addressaware_aa_pretrain_ckpt/checkpoint_last.pt`

The current evaluation outputs are written under:

- `/path/to/numAnchor/trex/output/stateformer_x64_baseline_pretrain3q10h_20260422`
- `/path/to/numAnchor/trex/output/stateformer_x64_addressaware_aa_pretrain3q10h_20260422`

For retrieval evaluation, use `checkpoint_best.pt`.
For resuming pretraining, use `checkpoint_last.pt`.

## 5. Environment

The current launchers in `command/stateformer_x64/` assume:

```bash
source /path/to/conda/etc/profile.d/conda.sh
conda activate jtrans
cd /path/to/numAnchor/trex
```

If you call Python modules directly instead of the shell wrappers, staying in
the `trex/` directory is usually enough for imports.

## 6. Preparing Trex Inputs

### 6.1 Generate text fields

Baseline:

```bash
python command/stateformer_x64/prepare_stateformer_trex.py \
  --mode baseline \
  --train-corpus /path/to/numAnchor_artifact/stateformer_x64_dataset/pretrain_ctx2/splits/train_cfg.txt \
  --valid-corpus /path/to/numAnchor_artifact/stateformer_x64_dataset/pretrain_ctx2/splits/val_cfg.txt \
  --output-dir /path/to/numAnchor_artifact/addressaware_runs/trex_stateformer_x64_current/baseline_text
```

Legacy byte-side-channel mode:

```bash
python command/stateformer_x64/prepare_stateformer_trex.py \
  --mode addressaware \
  --train-corpus /path/to/numAnchor_artifact/stateformer_x64_dataset/pretrain_ctx2/splits/train_cfg.txt \
  --valid-corpus /path/to/numAnchor_artifact/stateformer_x64_dataset/pretrain_ctx2/splits/val_cfg.txt \
  --output-dir /path/to/numAnchor_artifact/addressaware_runs/trex_stateformer_x64_current/addressaware_text
```

Current project address-aware mode:

```bash
python command/stateformer_x64/prepare_stateformer_trex.py \
  --mode addressaware_aa \
  --train-corpus /path/to/numAnchor_artifact/stateformer_x64_dataset/pretrain_ctx2/splits/train_cfg.txt \
  --valid-corpus /path/to/numAnchor_artifact/stateformer_x64_dataset/pretrain_ctx2/splits/val_cfg.txt \
  --output-dir /path/to/numAnchor_artifact/addressaware_runs/trex_stateformer_x64_current/addressaware_aa_text
```

All three modes write the eight standard Trex fields:

- `static`
- `inst_pos_emb`
- `op_pos_emb`
- `arch_emb`
- `byte1`
- `byte2`
- `byte3`
- `byte4`

`addressaware_aa` additionally writes aligned AA sidecar files:

- `train.aa_role_ids`
- `train.binary_pos`
- `train.function_pos`
- `train.bb_pos`
- `train.var_offset`
- `valid.*` with the same field names

### 6.2 Binarize the eight standard fields

```bash
TEXT_DIR=/path/to/numAnchor_artifact/addressaware_runs/trex_stateformer_x64_current/baseline_text
BIN_DIR=/path/to/numAnchor_artifact/addressaware_runs/trex_stateformer_x64_current/baseline_bin

for field in static inst_pos_emb op_pos_emb arch_emb byte1 byte2 byte3 byte4; do
  mkdir -p "${BIN_DIR}/${field}"
  python -m fairseq_cli.preprocess \
    --only-source \
    --srcdict "${TEXT_DIR}/dicts/${field}/dict.txt" \
    --trainpref "${TEXT_DIR}/train.${field}" \
    --validpref "${TEXT_DIR}/valid.${field}" \
    --destdir "${BIN_DIR}/${field}" \
    --workers 8
done
```

`addressaware_aa` still uses the same kind of binarized main fields. The AA
information is not binarized into a separate fairseq dictionary. It stays in
the sidecar text directory and is loaded by `--aa-data-dir`.

## 7. Current Recommended Pretraining Commands

The shell launchers are generic and default to short local settings. For the
current full project runs, override them explicitly.

Baseline `3q_10h` style run:

```bash
CUDA_VISIBLE_DEVICES=0 \
SAVE_DIR=/path/to/numAnchor_artifact/addressaware_runs/trex_stateformer_x64_768_pretrain3q_10h_20260422/baseline_pretrain_ckpt \
LOG_PATH=/path/to/numAnchor_artifact/addressaware_runs/trex_stateformer_x64_768_pretrain3q_10h_20260422/logs/baseline_pretrain.log \
TOTAL_UPDATES=16500 \
WARMUP_UPDATES=750 \
MAX_SENTENCES=3072 \
BATCH_SIZE_VALID=512 \
VALIDATE_INTERVAL_UPDATES=3000 \
SAVE_INTERVAL_UPDATES=3000 \
LOG_INTERVAL=100 \
NUM_WORKERS=8 \
bash command/stateformer_x64/train_baseline.sh
```

`addressaware_aa` `3q_10h` style run:

```bash
CUDA_VISIBLE_DEVICES=1 \
AA_DATA_DIR=/path/to/numAnchor_artifact/addressaware_runs/trex_stateformer_x64_20260414/addressaware_aa_text \
SAVE_DIR=/path/to/numAnchor_artifact/addressaware_runs/trex_stateformer_x64_768_pretrain3q_10h_20260422/addressaware_aa_pretrain_ckpt \
LOG_PATH=/path/to/numAnchor_artifact/addressaware_runs/trex_stateformer_x64_768_pretrain3q_10h_20260422/logs/addressaware_aa_pretrain.log \
TOTAL_UPDATES=16500 \
WARMUP_UPDATES=750 \
MAX_SENTENCES=3072 \
BATCH_SIZE_VALID=512 \
VALIDATE_INTERVAL_UPDATES=3000 \
SAVE_INTERVAL_UPDATES=3000 \
LOG_INTERVAL=100 \
NUM_WORKERS=8 \
bash command/stateformer_x64/train_addressaware_aa.sh
```

Notes:

- both launchers currently read the same main token/bin data from
  `/path/to/numAnchor_artifact/addressaware_runs/trex_stateformer_x64_20260414/baseline_bin`
- `addressaware_aa` adds `--use-address-aware-embedding` and reads aligned
  sidecar files from `AA_DATA_DIR`
- to resume a run, keep the same `SAVE_DIR` and continue from
  `checkpoint_last.pt`

## 8. Retrieval Evaluation

The current Trex evaluation path is direct retrieval from pretrained
checkpoints using `evaluate_pool.py`.

Baseline example:

```bash
python command/stateformer_x64/evaluate_pool.py \
  --checkpoint /path/to/numAnchor_artifact/addressaware_runs/trex_stateformer_x64_768_pretrain3q_10h_20260422/baseline_pretrain_ckpt/checkpoint_best.pt \
  --data-dir /path/to/numAnchor_artifact/addressaware_runs/trex_stateformer_x64_20260414/baseline_bin \
  --function-blocks /path/to/numAnchor_artifact/stateformer_x64_dataset/funcsim_block_ctx2/function_blocks.json \
  --funcsim-pairs /path/to/numAnchor_artifact/stateformer_x64_dataset/funcsim_block_ctx2/funcsim_pairs.json \
  --pool-file /path/to/numAnchor_artifact/stateformer_x64_dataset/funcsim_block_ctx2/pool_ids_10k.json \
  --mode baseline \
  --max-positions 128 \
  --batch-size 64 \
  --device cuda \
  --output /path/to/numAnchor/trex/output/stateformer_x64_baseline_pretrain3q10h_20260422/eval_pool_10k.json
```

`addressaware_aa` example:

```bash
python command/stateformer_x64/evaluate_pool.py \
  --checkpoint /path/to/numAnchor_artifact/addressaware_runs/trex_stateformer_x64_768_pretrain3q_10h_20260422/addressaware_aa_pretrain_ckpt/checkpoint_best.pt \
  --data-dir /path/to/numAnchor_artifact/addressaware_runs/trex_stateformer_x64_20260414/baseline_bin \
  --function-blocks /path/to/numAnchor_artifact/stateformer_x64_dataset/funcsim_block_ctx2/function_blocks.json \
  --funcsim-pairs /path/to/numAnchor_artifact/stateformer_x64_dataset/funcsim_block_ctx2/funcsim_pairs.json \
  --pool-file /path/to/numAnchor_artifact/stateformer_x64_dataset/funcsim_block_ctx2/pool_ids_10k.json \
  --mode addressaware_aa \
  --max-positions 128 \
  --batch-size 64 \
  --device cuda \
  --output /path/to/numAnchor/trex/output/stateformer_x64_addressaware_aa_pretrain3q10h_20260422/eval_pool_10k.json
```

## 9. Current Project Boundaries

- Trex is maintained here as a compatible pretraining and retrieval-evaluation
  route, not as the main downstream function-similarity training branch
- `trex/command/stateformer_x64/finetune_similarity_*.sh` still exist, but
  they are not the preferred route for this project
- the current primary Trex comparison is `baseline` vs `addressaware_aa`, not
  `baseline` vs legacy `addressaware`
- `native` remains a separate reference path and depends on raw ELF or
  micro-trace style assets that are outside the main inline Stateformer
  workflow
- most real outputs now live under `/path/to/numAnchor_artifact/addressaware_runs/...` and
  `/path/to/numAnchor/trex/output/...`, not under
  `trex/Trexmodel/`
