# Plan: Implement Our Address-Aware Design on Top of JTrans

## Objective

Implement a new `address-aware` JTrans track that keeps the strong parts of jTrans:

- function-level representation,
- function similarity downstream training,
- large-pool retrieval evaluation,

while replacing its original `jump-aware` input design with our current `PalmTree`-style address-aware design:

- hierarchical address positions,
- separate handling for code addresses and data addresses,
- stack-variable offset modeling,
- pretraining that directly learns these signals.

The original jTrans path should remain available as a baseline.

## What I Re-read from `jtrans.txt`

After re-reading `jtrans.txt`, the core jTrans design is:

1. `function-level sequence input`
   - one function is flattened into one token sequence.

2. `jump-aware representation`
   - jump operands are rewritten as `JUMP_xxx`,
   - for each jump pair, the source token embedding shares parameters with the target position embedding.

3. `pretraining = MLM + JTP`
   - MLM: standard masked language modeling.
   - JTP: replace jump source token with `<LOC>` and predict the jump target index.

4. `downstream similarity`
   - use the function embedding for contrastive / triplet-style binary similarity training.

So if we want to put our current address-aware design onto jTrans, the main question is:

`what do we keep from jTrans, and what do we replace with PalmTree-style address-aware design?`

My answer is:

- keep `function as the training / retrieval unit`,
- keep `contrastive function similarity finetuning`,
- replace `jump-aware token/position sharing` with `address-aware hierarchical embeddings`,
- keep `MLM + JTP` as the pretraining task structure,
- but make both tasks operate on our address-aware representation.

## High-Level Design Decision

I do **not** think the right path is to lightly patch the current `JUMP_xxx` preprocessing.

That would leave the whole representation centered around jump indices, while our current design is centered around:

- where code lives in the binary,
- where an instruction lives inside a function / basic block,
- where a data reference lives inside a section,
- where a stack variable lives by offset.

So the clean plan is:

1. keep jTrans's `function-level task framing`,
2. replace its `input representation`,
3. reuse as much of PalmTree's address-aware stack as possible,
4. adapt PalmTree's pretraining / funcsim to fit jTrans's function-level setting.

## Mapping: JTrans Design -> Our Address-Aware Design

### 1. Input unit

`jTrans`
- one function -> one flattened token sequence.

`our implementation plan`
- still one function -> one flattened token sequence or function windows.

Reason:
- this preserves jTrans's downstream retrieval setting,
- and lets us compare fairly against original jTrans.

### 2. Control-flow representation

`jTrans`
- represents control flow through `JUMP_xxx` and parameter sharing between jump source token embedding and target position embedding.

`our implementation plan`
- do not use `JUMP_xxx` as the main representation.
- instead, represent control-flow and structural context through PalmTree-style annotations:
  - `opcode(0xADDR:b:f:bb)`
  - `address(0xADDR:b:f:bb)`
  - `daddr(0xADDR:b:f:bb)`
  - `var(0xOFFSET)`

and feed side channels:

- `binary_pos`
- `function_pos`
- `bb_pos`
- `var_offsets`

into an address-aware embedding module.

### 3. Embedding mechanism

`jTrans`
- token embedding + position embedding
- special jump-aware parameter sharing

`our implementation plan`
- token embedding
- sequence positional embedding
- address positional embedding
- var-offset embedding
- segment embedding if needed by the pretraining format

This means the architectural center moves from:

- `jump-index coupling`

to:

- `hierarchical location awareness`

### 4. Pretraining objective

`jTrans`
- MLM + JTP

`our implementation plan`
- keep the same two-task structure:
  - MLM
  - JTP

but replace the underlying representation with our address-aware input design.

So the rule is:

- `do not use IMC here`
- `do not change the pretraining family away from MLM + JTP`

What changes is not the task family, but the input representation and model embedding stack.

### 5. Downstream similarity

`jTrans`
- function embedding + contrastive/triplet training.

`our implementation plan`
- keep function similarity as the downstream task,
- likely use PalmTree's current `funcsim` pipeline as the base implementation,
- adapt it so the sample unit and evaluation style remain compatible with jTrans's function-level benchmarking.

## Concrete Implementation Plan

## Phase 1: Build an address-aware extraction format for JTrans

### Why this is first

The current public JTrans extraction format is too thin for our design.

Right now it mainly stores:

- function entry,
- asm list,
- raw bytes,
- CFG with BB-level asm/raw.

That is not enough to faithfully rebuild our address-aware annotations, because we also need:

- per-instruction address,
- function range / basic-block range,
- binary-wide address range,
- section info for data addresses,
- enough metadata to distinguish code addresses vs data addresses.

### Planned work

Add a richer extraction path, likely:

- `JTrans/datautils/process_addraware.py`

This extractor should save:

- function entry address,
- per-BB start/end,
- per-instruction addresses and disassembly,
- CFG edges,
- binary section ranges,
- binary min/max address range,
- optional symbol map if available.

### Deliverable

A new extracted dataset format that is sufficient for PalmTree-style address-aware preprocessing.

## Phase 2: Replace jTrans preprocessing with address-aware preprocessing

### jTrans original preprocessing

The paper's original design normalizes:

- constants -> `<const>`
- strings -> `<str>`
- internal calls -> `<function>`
- jump operands -> `JUMP_xxx`

### Our new preprocessing

We will build a new preprocessing path that does **not** use `JUMP_xxx` as the primary representation.

Instead, for each function we will emit a PalmTree-style address-aware linearization:

- instruction opcode carries code-position annotation,
- code targets become `address(...)`,
- data references become `daddr(...)`,
- stack variables become `var(...)`.

### Planned files

- `JTrans/addressaware_preprocess.py`
- or `JTrans/datautils/build_addraware_corpus.py`

### Design choice for linearization

Because jTrans is function-level, I plan to produce:

1. one canonical instruction order per function,
2. optional function windows if the function is longer than the model limit.

The likely initial order is:

- deterministic CFG-aware order close to the current jTrans flattening,

so that:

- we preserve the function-level setup,
- but inject our address-aware semantics.

## Phase 3: Port PalmTree's address-aware embedding/model stack into JTrans

### Goal

Use our current address-aware design directly instead of trying to retrofit it into the original HuggingFace `BinBertModel`.

### Planned work

Add or port:

- `JTrans/vocab.py` or reuse `PalmTree/vocab.py`
- `JTrans/create_vocab_addraware.py`
- `JTrans/dataloader_addraware.py`
- `JTrans/address_embedding.py`
- `JTrans/model_addraware.py`
- `JTrans/pretrain_addraware.py`

### Why I prefer porting over patching

PalmTree already has the exact mechanisms we need:

- address-aware embedding routing,
- separate `address` vs `daddr`,
- `var_offsets` modeling,
- MLM-compatible token prediction head,
- data parsing logic for the inline annotation format.

So the lowest-risk path is:

- transplant the proven address-aware core,
- adapt the batching and sequence construction to jTrans's function-level setting.

## Phase 4: Adapt pretraining to the jTrans function-level regime

### Main decision

PalmTree's original pretraining data is random-walk / inline-window oriented.

jTrans is function-level.

So the adaptation is:

- keep PalmTree's token semantics and embeddings,
- change the sample construction to operate on full functions or function windows.

### Pretraining setup

The pretraining setup should stay aligned with original jTrans:

- `MLM`
- `JTP`

No `IMC`.

### Address-aware MLM

This stays conceptually standard:

- mask tokens in the function sequence,
- predict the original token,
- but now the model also sees:
  - `binary_pos`
  - `function_pos`
  - `bb_pos`
  - `var_offsets`

through the address-aware embedding stack.

### Address-aware JTP

This is where we preserve jTrans's control-flow supervision, but redefine it over our new representation.

The most likely implementation is:

1. identify jump/call target tokens represented as `address(...)`,
2. replace the source-side target token with a special placeholder such as `<loc>` or `<mask>`,
3. predict the corresponding target location within the function.

There are two implementation options:

1. `index classification`
   - predict the target instruction slot / token slot in the current function.
   - this is conceptually closest to original jTrans's `JUMP_xxx`.

2. `token reconstruction`
   - predict the masked target token itself, e.g. the correct `address(...)`-derived token identity.

My recommendation is to use `index classification` first, because it matches the original JTP objective more closely.

## Phase 5: Bring function similarity into JTrans using the address-aware checkpoint

### Goal

Evaluate whether address-aware pretraining helps the final BCSD task in the jTrans setting.

### Planned work

Add:

- `JTrans/funcsim/`

with adapted versions of PalmTree's:

- dataloader,
- function encoder,
- train/eval scripts.

### Core idea

The downstream encoder should consume the same address-aware function representation produced by the new preprocessing pipeline.

We can keep the downstream objective close to jTrans:

- contrastive / triplet function similarity,

while using our address-aware backbone as the feature extractor.

## Phase 6: Compare fairly against original jTrans

### Baselines

We should keep at least:

1. original jTrans
   - jump-aware preprocessing
   - original finetune path

2. address-aware JTrans
   - our new preprocessing
   - our address-aware pretraining
   - address-aware funcsim downstream

3. address-aware ablations
   - no address embedding
   - no var embedding
   - maybe no auxiliary JTP-style task

### Target question

This comparison answers the real research question:

`if we replace jump-aware control-flow encoding with our address-aware design, does function-level BCSD improve?`

## Recommended File Layout

### Likely new files

- `JTrans/Plan.md`
- `JTrans/datautils/process_addraware.py`
- `JTrans/addressaware_preprocess.py`
- `JTrans/create_vocab_addraware.py`
- `JTrans/dataloader_addraware.py`
- `JTrans/address_embedding.py`
- `JTrans/model_addraware.py`
- `JTrans/pretrain_addraware.py`
- `JTrans/funcsim/dataloader.py`
- `JTrans/funcsim/model.py`
- `JTrans/funcsim/train.py`
- `JTrans/funcsim/evaluate.py`

### Existing files we may touch

- `JTrans/datautils/process.py`
- `JTrans/data.py`
- `JTrans/finetune.py`

## Risks

### 1. Old extracted data may be insufficient

This is the biggest practical risk.

If we only have the old JTrans pickles, they likely do not preserve enough metadata for full address-aware reconstruction.

That means we may need to regenerate extracted data from binaries.

### 2. Sequence length mismatch

jTrans uses function-level sequences up to model limits like 512.

PalmTree often works on shorter windows.

So we will need a clear policy for:

- long functions,
- truncation vs sliding windows,
- how to pool window embeddings back to function embeddings.

### 3. Address-aware JTP needs a precise label definition

We are keeping JTP, so the main risk is not whether to include it, but how to define the target labels cleanly.

The safest first version is:

- intra-function target index prediction,

because it is the closest analogue to original jTrans's jump-position prediction.

## Milestones

## Milestone 1

Address-aware extraction and corpus generation for JTrans.

Success signal:

- one function can be turned into a correct PalmTree-style annotated sequence.

## Milestone 2

Address-aware pretraining runs end-to-end in JTrans.

Success signal:

- train/val loop works,
- checkpoint saves and reloads,
- tensors include `binary_pos/function_pos/bb_pos/var_offsets`.

## Milestone 3

Address-aware function similarity training works.

Success signal:

- pretrain checkpoint initializes downstream model,
- retrieval metrics are reproducible.

## Milestone 4

Fair comparison against original jTrans.

Success signal:

- we can report whether address-aware design helps, hurts, or changes the tradeoff.

## Immediate Next Step

The next concrete step should be:

1. implement richer extraction for JTrans,
2. generate a tiny address-aware corpus from a few functions,
3. verify the parser and embeddings on that corpus,
4. only then start porting the pretraining runner.

That is the safest path to put our current address-aware design onto jTrans without losing jTrans's function-level evaluation setup.
