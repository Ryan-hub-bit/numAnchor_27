# Address-Aware jTrans

This directory contains the address-aware adaptation of jTrans that keeps the
original function-level setting and uses `MLM + JTP` for pretraining.

## Data Root

All datasets, vocabularies, checkpoints, logs, and evaluation outputs must live
under:

```bash
/path/to/numAnchor_artifact/numAnchor/
```

The helper scripts in this directory enforce that rule and will reject paths
outside this root.

## Environment

Use the Python that already has PyTorch installed on this machine:

```bash
/path/to/conda/bin/python
```

## Official jTrans Workflow

The paper evaluates on BinaryCorp-3M / BinaryCorp-26M. For the public jTrans
release, the easiest official entry point is `small_train.tar.gz` and
`small_test.tar.gz`, which match the released extracted-pickle format used by
jTrans.

### 1. Download the official extracted data

The public files are hosted on the jTrans Nextcloud share:

```bash
mkdir -p /path/to/numAnchor_artifact/numAnchor/jtrans/official
cd /path/to/numAnchor_artifact/numAnchor/jtrans/official

wget --user='cxnH8DfZTADLKCs' --password='' \
  https://cloud.vul337.team:8443/public.php/webdav/small_train.tar.gz
wget --user='cxnH8DfZTADLKCs' --password='' \
  https://cloud.vul337.team:8443/public.php/webdav/small_test.tar.gz

tar -xzf small_train.tar.gz
tar -xzf small_test.tar.gz
```

### 2. Convert official jTrans pickles into address-aware data

For a full 3M-style run, remove `--max_projects`. The command below shows the
same shape used for the current `p100` sanity run.

```bash
cd /path/to/numAnchor/JTrans/addressaware
/path/to/conda/bin/python build_official_dataset.py \
  --extract_root /path/to/numAnchor_artifact/numAnchor/jtrans/official/small_train \
  --output_dir /path/to/numAnchor_artifact/numAnchor/jtrans/official_small_train_aa_p100 \
  --max_projects 100 \
  --write_corpus

/path/to/conda/bin/python build_official_dataset.py \
  --extract_root /path/to/numAnchor_artifact/numAnchor/jtrans/official/small_test \
  --output_dir /path/to/numAnchor_artifact/numAnchor/jtrans/official_small_test_aa
```

### 3. Create a shuffled train/val split with JTP coverage

Do not use `head`/`tail` to split the corpus. That can create validation batches
with no valid JTP targets and make `val_jtp_loss` become `NaN`.

```bash
cd /path/to/numAnchor/JTrans/addressaware
/path/to/conda/bin/python split_pretrain_corpus.py \
  --corpus /path/to/numAnchor_artifact/numAnchor/jtrans/official_small_train_aa_p100/pretrain_corpus.txt \
  --train_output /path/to/numAnchor_artifact/numAnchor/jtrans/official_small_train_aa_p100/pretrain_train_v2.txt \
  --val_output /path/to/numAnchor_artifact/numAnchor/jtrans/official_small_train_aa_p100/pretrain_val_v2.txt \
  --stats_output /path/to/numAnchor_artifact/numAnchor/jtrans/official_small_train_aa_p100/pretrain_split_stats_v2.json \
  --seq_len 128 \
  --val_size 5000 \
  --min_val_jtp 0.35 \
  --seed 42
```

### 4. Create the vocabulary

```bash
cd /path/to/numAnchor/JTrans/addressaware
/path/to/conda/bin/python create_vocab.py \
  --corpus /path/to/numAnchor_artifact/numAnchor/jtrans/official_small_train_aa_p100/pretrain_train_v2.txt \
  --output /path/to/numAnchor_artifact/numAnchor/jtrans/official_small_train_aa_p100/vocab_v2.pkl \
  --max_size 20000 \
  --min_freq 2
```

### 5. Pretrain with address-aware `MLM + JTP`

```bash
cd /path/to/numAnchor/JTrans/addressaware
/path/to/conda/bin/python pretrain.py \
  --train_corpus /path/to/numAnchor_artifact/numAnchor/jtrans/official_small_train_aa_p100/pretrain_train_v2.txt \
  --val_corpus /path/to/numAnchor_artifact/numAnchor/jtrans/official_small_train_aa_p100/pretrain_val_v2.txt \
  --vocab /path/to/numAnchor_artifact/numAnchor/jtrans/official_small_train_aa_p100/vocab_v2.pkl \
  --output_dir /path/to/numAnchor_artifact/numAnchor/jtrans/official_small_train_aa_p100/pretrain_out_v2 \
  --log_dir /path/to/numAnchor_artifact/numAnchor/jtrans/official_small_train_aa_p100/logs_v2 \
  --seq_len 128 \
  --hidden 128 \
  --n_layers 2 \
  --attn_heads 4 \
  --batch_size 32 \
  --epochs 4 \
  --lr 1e-4 \
  --device cuda
```

### 6. Fine-tune funcsim

```bash
cd /path/to/numAnchor/JTrans/addressaware
/path/to/conda/bin/python train_funcsim.py \
  --function_blocks /path/to/numAnchor_artifact/numAnchor/jtrans/official_small_train_aa_p100/funcsim_function_blocks.json \
  --funcsim_pairs /path/to/numAnchor_artifact/numAnchor/jtrans/official_small_train_aa_p100/funcsim_pairs.json \
  --vocab /path/to/numAnchor_artifact/numAnchor/jtrans/official_small_train_aa_p100/vocab_v2.pkl \
  --pretrained_bert /path/to/numAnchor_artifact/numAnchor/jtrans/official_small_train_aa_p100/pretrain_out_v2/best_model.pt \
  --output_dir /path/to/numAnchor_artifact/numAnchor/jtrans/official_small_train_aa_p100/funcsim_out_v2 \
  --log_dir /path/to/numAnchor_artifact/numAnchor/jtrans/official_small_train_aa_p100/funcsim_logs_v2 \
  --hidden 128 \
  --n_layers 2 \
  --attn_heads 4 \
  --seq_len 128 \
  --max_instructions 64 \
  --embedding_dim 64 \
  --batch_size 16 \
  --epochs 1 \
  --lr 1e-4 \
  --num_workers 0 \
  --device cuda \
  --train_split 0.9 \
  --val_split 0.1 \
  --negative_samples 1
```

### 7. Evaluate paper-style optimisation pairs

```bash
cd /path/to/numAnchor/JTrans/addressaware
/path/to/conda/bin/python evaluate_opt_pairs.py \
  --function_blocks /path/to/numAnchor_artifact/numAnchor/jtrans/official_small_test_aa/funcsim_function_blocks.json \
  --funcsim_pairs /path/to/numAnchor_artifact/numAnchor/jtrans/official_small_test_aa/funcsim_pairs.json \
  --vocab /path/to/numAnchor_artifact/numAnchor/jtrans/official_small_train_aa_p100/vocab_v2.pkl \
  --checkpoint /path/to/numAnchor_artifact/numAnchor/jtrans/official_small_train_aa_p100/funcsim_out_v2/best_model.pt \
  --output /path/to/numAnchor_artifact/numAnchor/jtrans/official_small_train_aa_p100/funcsim_out_v2/official_small_test_opt_eval.json \
  --log_dir /path/to/numAnchor_artifact/numAnchor/jtrans/official_small_train_aa_p100/funcsim_logs_v2 \
  --hidden 128 \
  --n_layers 2 \
  --attn_heads 4 \
  --seq_len 128 \
  --max_instructions 64 \
  --embedding_dim 64 \
  --batch_size 64 \
  --poolsizes 32,10000 \
  --device cuda
```

`evaluate_opt_pairs.py` reports the same optimisation pairs used in the paper:

- `O0-O3`
- `O0-Os`
- `O1-O3`
- `O1-Os`
- `O2-O3`
- `O2-Os`

## Smaller Smoke Workflow

The commands below are still useful when we only want a very small pipeline
sanity check.

### 1. Download a jTrans-compatible dataset

```bash
mkdir -p /path/to/numAnchor_artifact/numAnchor/jtrans
cd /path/to/numAnchor_artifact/numAnchor/jtrans
wget -O BinaryCorp_small_train.zip \
  "https://zenodo.org/records/18072785/files/BinaryCorp_small_train.zip?download=1"
```

### 2. Build address-aware pretraining and funcsim files

```bash
cd /path/to/numAnchor/JTrans/addressaware
/path/to/conda/bin/python build_dataset.py \
  --zip /path/to/numAnchor_artifact/numAnchor/jtrans/BinaryCorp_small_train.zip \
  --output_dir /path/to/numAnchor_artifact/numAnchor/jtrans/aa_binarycorp_small_smoke \
  --max_files 12 \
  --max_functions 512
```

### 3. Create the vocabulary

```bash
cd /path/to/numAnchor/JTrans/addressaware
/path/to/conda/bin/python create_vocab.py \
  --corpus /path/to/numAnchor_artifact/numAnchor/jtrans/aa_binarycorp_small_smoke/pretrain_train.txt \
  --output /path/to/numAnchor_artifact/numAnchor/jtrans/aa_binarycorp_small_smoke/vocab.pkl \
  --max_size 12000 \
  --min_freq 1
```

### 4. Pretrain with MLM + JTP

```bash
cd /path/to/numAnchor/JTrans/addressaware
/path/to/conda/bin/python pretrain.py \
  --train_corpus /path/to/numAnchor_artifact/numAnchor/jtrans/aa_binarycorp_small_smoke/pretrain_train.txt \
  --val_corpus /path/to/numAnchor_artifact/numAnchor/jtrans/aa_binarycorp_small_smoke/pretrain_val.txt \
  --vocab /path/to/numAnchor_artifact/numAnchor/jtrans/aa_binarycorp_small_smoke/vocab.pkl \
  --output_dir /path/to/numAnchor_artifact/numAnchor/jtrans/aa_binarycorp_small_smoke/pretrain_out \
  --log_dir /path/to/numAnchor_artifact/numAnchor/jtrans/aa_binarycorp_small_smoke/logs \
  --seq_len 128 \
  --hidden 128 \
  --n_layers 2 \
  --attn_heads 4 \
  --batch_size 16 \
  --epochs 1 \
  --lr 1e-4 \
  --device cuda
```

### 5. Fine-tune the funcsim model

```bash
cd /path/to/numAnchor/JTrans/addressaware
/path/to/conda/bin/python train_funcsim.py \
  --function_blocks /path/to/numAnchor_artifact/numAnchor/jtrans/aa_binarycorp_small_smoke/funcsim_function_blocks.json \
  --funcsim_pairs /path/to/numAnchor_artifact/numAnchor/jtrans/aa_binarycorp_small_smoke/funcsim_pairs.json \
  --vocab /path/to/numAnchor_artifact/numAnchor/jtrans/aa_binarycorp_small_smoke/vocab.pkl \
  --pretrained_bert /path/to/numAnchor_artifact/numAnchor/jtrans/aa_binarycorp_small_smoke/pretrain_out/best_model.pt \
  --output_dir /path/to/numAnchor_artifact/numAnchor/jtrans/aa_binarycorp_small_smoke/funcsim_out \
  --log_dir /path/to/numAnchor_artifact/numAnchor/jtrans/aa_binarycorp_small_smoke/funcsim_logs \
  --hidden 128 \
  --n_layers 2 \
  --attn_heads 4 \
  --seq_len 128 \
  --max_instructions 64 \
  --embedding_dim 64 \
  --batch_size 8 \
  --epochs 1 \
  --lr 1e-4 \
  --num_workers 0 \
  --device cuda \
  --train_split 0.8 \
  --val_split 0.2 \
  --negative_samples 2
```

### 6. Evaluate funcsim retrieval

```bash
cd /path/to/numAnchor/JTrans/addressaware
/path/to/conda/bin/python evaluate_funcsim.py \
  --function_blocks /path/to/numAnchor_artifact/numAnchor/jtrans/aa_binarycorp_small_smoke/funcsim_function_blocks.json \
  --funcsim_pairs /path/to/numAnchor_artifact/numAnchor/jtrans/aa_binarycorp_small_smoke/funcsim_pairs.json \
  --vocab /path/to/numAnchor_artifact/numAnchor/jtrans/aa_binarycorp_small_smoke/vocab.pkl \
  --checkpoint /path/to/numAnchor_artifact/numAnchor/jtrans/aa_binarycorp_small_smoke/funcsim_out/best_model.pt \
  --output /path/to/numAnchor_artifact/numAnchor/jtrans/aa_binarycorp_small_smoke/funcsim_out/eval_results.json \
  --log_dir /path/to/numAnchor_artifact/numAnchor/jtrans/aa_binarycorp_small_smoke/funcsim_logs \
  --hidden 128 \
  --n_layers 2 \
  --attn_heads 4 \
  --seq_len 128 \
  --max_instructions 64 \
  --embedding_dim 64 \
  --batch_size 32 \
  --num_workers 0 \
  --device cuda
```

## Notes

- All data, logs, checkpoints, and evaluation files must stay under
  `/path/to/numAnchor_artifact/numAnchor/`.
- The current official workflow has been exercised locally on the released
  `small_train` / `small_test` data, first with a `p100` subset and then with
  the new shuffled split that avoids `NaN` validation JTP loss.
