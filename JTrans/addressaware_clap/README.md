# numAnchor JTrans CLAP-Style Address-Aware Variant

This directory keeps the CLAP-style jTrans instruction-id variant inside the
main `JTrans` component.

The variant keeps the numAnchor address-aware pipeline:

- `WordVocab`-based tokenization from `vocab.pkl`
- code/data/stack address-aware tokens and structural descriptors
- `MLM + JTP` pretraining
- the same function-similarity training code path as address-aware jTrans

The local CLAP-style change is instruction-id conditioning: `segment` /
`token_type_ids` carry per-instruction ids instead of staying all-zero, so the
model can preserve instruction-level structure after flattening the function
sequence.

## Entry Points

- `run_full_cfgsplit_addraware_instruction_id.sh`: full local launcher for the instruction-id variant.
- `run_pretrain_full_cfgsplit_addraware_instruction_id.sh`: pretraining-only launcher.
- `pretrain.py`, `train_funcsim.py`, `evaluate_opt_pairs.py`: script-level entry points shared with the address-aware jTrans workflow.

Use `JTrans/addressaware/README.md` for the main packaged JTrans workflow.
