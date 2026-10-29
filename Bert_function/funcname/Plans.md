# Function Name Prediction Plan

## Goal
Reproduce the paper's `Function Name Prediction` downstream task using the current address-aware pretraining setup.

This task should be treated after `Algorithm Classification` because it needs:
- new labeled data handling
- new target vocabulary construction
- a multi-label decoder head on top of function embeddings

## Recommended Backbone
- Start from the same address-aware `MLM-only` checkpoint used for Algorithm Classification.
- Default choice in this repo: `output/mlm_addr_var/latest_bert.pt`
- If your best address-aware MLM-only run is stored elsewhere, point the downstream loader to that checkpoint instead.

## What To Reuse From The Current Repo
- `data_generator/func_gen.py`
  - for whole-function address-aware export
- `data_generator/process_funcsim_ida.py`
  - as the template for function-level JSON export
- `model.py`
  - for the address-aware encoder
- `funcsim/model.py`
  - for the function-level windowing and pooling pattern

## Task Definition For This Repo
- One function is one downstream sample.
- Input is the stripped function body in address-aware inline format.
- Target is a set of normalized function-name subtokens.
- This is a multi-label classification task over name subtokens, not full sequence generation.
- Evaluate with token-set:
  - `precision`
  - `recall`
  - `F1`

## Data Pipeline
### 1. Dataset source
- Use the SymLM-style x86 dataset split mentioned in the paper when available.
- Keep the released train/val/test boundaries fixed.
- Do not replace this with current `funcsim` exports; they do not preserve useful ground-truth names.

### 2. Ground-truth preservation
- Preserve source/linked symbol names before stripping.
- The downstream converter must carry both:
  - stripped function body for input
  - original function name for label creation

### 3. Address-aware function export
- Export each function body using the same formatting used in current address-aware pretraining:
  - keep instruction addresses
  - keep `address(...)`
  - keep `daddr(...)`
  - keep `var(...)`
- Do not simplify bodies into plain text assembly if the model will be initialized from address-aware pretraining.

### 4. Name normalization
- Split function names into subtokens using:
  - snake_case
  - camelCase
  - digits
  - punctuation boundaries
- Lowercase all subtokens.
- Drop empty tokens.
- Keep common abbreviations when they are meaningful.
- Build a stable name-token vocabulary from the training split only.

## Preprocess Advice Specific To Address-Aware Pretraining
- Keep downstream assembly input format identical to pretraining input format wherever possible.
- Keep the assembly vocabulary separate from the function-name label vocabulary.
- Do not add function-name tokens into the assembly vocab.
- Keep external symbol handling and symbol merging consistent with your current export pipeline.
- If the SymLM data contains raw constants or symbol patterns unseen during pretraining, normalize them into your existing address-aware token scheme instead of introducing many new assembly tokens.

## How To Use Your Current Data
Your current local unlabeled data is still useful, but only for backbone quality, not for final supervision.

### A. Generic address-aware pretraining
Keep using your current address-aware corpora as the main source of binary-language knowledge:
- `cfg_train/cfg_2inline_pretrain.txt`
- `cfg_train/cfg_8inline_pretrain.txt`
- `addressbin_train/`, `addressbin_val/`, `addressbin_test/`

### B. Optional SymLM-domain adaptation
After the SymLM train/val/test split is frozen, run a short additional MLM stage on function bodies from the training split only.

Do not use:
- validation functions
- test functions
- any ground-truth name tokens during MLM adaptation

Recommended order:
- generic address-aware MLM pretraining
- optional SymLM-train-only MLM adaptation
- function-name fine-tuning

## Model Design
### Encoder
- Reuse the function-level encoding strategy from `funcsim/model.py`.
- Encode each function body into one pooled embedding.

### Decoder head
- Add a multi-label classification head over the name-token vocabulary.
- A simple starting point is:
  - pooled function embedding
  - linear or 2-layer MLP
  - sigmoid outputs over name-token labels

### Targets
- Represent ground truth as a multi-hot vector over training-vocabulary name subtokens.
- Ignore exact subtoken order in the first version to stay aligned with the paper summary you provided.

## Metrics
- Primary metrics:
  - `precision`
  - `recall`
  - `F1`
- Compute them on predicted token sets vs ground-truth token sets.

## Sanity Checks Before Full Runs
- verify stripped input bodies no longer leak the original function name
- verify each function keeps the correct original label before stripping
- verify training-vocabulary-only construction for name subtokens
- verify multi-hot target generation
- verify empty or rare-name edge cases are handled consistently
- verify no train/val/test leakage in optional MLM adaptation

## Implementation Order
1. Build a SymLM converter that emits:
   - address-aware inline function body
   - normalized function-name subtoken labels
2. Reuse the `funcsim` encoder path for one-function embedding extraction.
3. Add a multi-label name-token head.
4. Implement set-based `precision`, `recall`, and `F1`.
5. Add optional train-only domain-adaptive MLM.

## Practical Default
If you want the fastest first result:
- start with a frozen encoder
- add only a linear multi-label head
- build the name-token vocabulary from the top training subtokens first

Then expand to encoder fine-tuning after the preprocessing and metric pipeline is stable.
