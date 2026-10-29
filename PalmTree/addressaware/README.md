# PalmTree AddressAware

This directory contains an instruction-level, address-aware PalmTree variant
ported from `PalmTree_vinilia_bert` into an isolated `PalmTree/addressaware`
branch. It keeps the Transformer attention kernel unchanged and injects
address/data-structure information in the embedding frontend.

## Scope

- Included:
  - pretraining model and dataloader
  - probing / instruction-level evaluation
  - IDA / inline data generation utilities
- Intentionally excluded:
  - `funcsim/` fine-tuning and evaluation pipeline

## Token Formats

- Code tokens:
  - `opcode(0xADDR:binary_pos:function_pos:bb_pos)`
  - `address(0xADDR:binary_pos:function_pos:bb_pos)`
- Data tokens:
  - `daddr_rodata(0xADDR:section_pos:offset_in_section)`
  - `daddr_bss(0xADDR:section_pos:offset_in_section)`
  - `daddr_data(0xADDR:section_pos:offset_in_section)`
  - `daddr_other(0xADDR:section_pos:offset_in_section)`

These section-specific tokens share one common data-address MLP branch in the
embedding frontend.

Legacy corpora using `daddr(0xADDR:section_pos:offset_in_section:section_kind_id)`
are still parsed and canonicalized to the section-specific tokens above.

Section mapping:

- `1 = .rodata`
- `2 = .bss`
- `3 = .data`
- `4 = other-data`

## Directory Layout

- `train.py`: pretraining entrypoint
- `dataloader.py`: instruction-masking dataset
- `model.py`, `address_embedding.py`, `transformer_components.py`: backbone
- `probing/`: frozen-encoder probing and pair finetuning
- `data_generator/`: IDA / inline corpus generation scripts
- `tests/`: parser and smoke tests for the new address-aware path
