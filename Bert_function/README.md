# numAnchor BERT-Function Artifact

Original bases: https://github.com/codertimo/BERT-pytorch and https://github.com/palmtreemodel/transformer-evaluation

This component evaluates the function-level BERT variant used in the numAnchor retrieval results. The artifact provides:

- `data/bert_function_x64/function_blocks.json`
- `data/bert_function_x64/ground_truth.json`
- `data/bert_function_x64/vocab.pkl`
- `models/bert_function/numanchor_pretrain.pt`
- `models/bert_function/numanchor_finetune.pt`

Create the environment:

```bash
conda env create -f Bert_function/environment.yml
conda activate palmtree
```

Generate a deterministic reviewer pool:

```bash
export NUMANCHOR_ARTIFACT_ROOT=/path/to/numAnchor_artifact/staging

python scripts/artifact/generate_pools.py \
  --ground-truth "$NUMANCHOR_ARTIFACT_ROOT/data/bert_function_x64/ground_truth.json" \
  --output "$NUMANCHOR_ARTIFACT_ROOT/work/pools/bert_function_pool_10k.json" \
  --pool-size 10000 \
  --query-count 1000 \
  --seed 42
```

Run retrieval evaluation:

```bash
cd Bert_function/funcsim

python evaluate.py \
  --function_blocks "$NUMANCHOR_ARTIFACT_ROOT/data/bert_function_x64/function_blocks.json" \
  --funcsim_pairs "$NUMANCHOR_ARTIFACT_ROOT/data/bert_function_x64/ground_truth.json" \
  --vocab "$NUMANCHOR_ARTIFACT_ROOT/data/bert_function_x64/vocab.pkl" \
  --checkpoint "$NUMANCHOR_ARTIFACT_ROOT/models/bert_function/numanchor_finetune.pt" \
  --eval_pool "$NUMANCHOR_ARTIFACT_ROOT/work/pools/bert_function_pool_10k.json" \
  --output "$NUMANCHOR_ARTIFACT_ROOT/work/results_tmp/bert_function_eval.json" \
  --log_dir "$NUMANCHOR_ARTIFACT_ROOT/work/logs/bert_function" \
  --seq_len 512 \
  --task_name numanchor_bert_function
```

The pretrain checkpoint is included for fine-tuning reproduction; the reported downstream evaluation uses the packaged finetuned checkpoint.
