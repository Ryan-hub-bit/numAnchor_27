#!/usr/bin/env python3
"""
Small prototype for function-name prediction using the current address-aware
PalmTree backbone and the address-aware SymLM-style dataset shards generated in
this repository.

This is intentionally lightweight:
- uses only self function bodies (`self/input.inline`)
- uses the current address-aware pretrained checkpoint
- freezes the encoder by default
- trains a simple multi-label head over function-name subtokens

The goal is to quickly verify whether:
  address-aware preprocessing -> mlm_addr_var backbone -> function embedding ->
  name-token prediction
can run end-to-end on small local datasets.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


THIS_DIR = Path(__file__).resolve().parent
PALMTREE_ROOT = THIS_DIR.parent
if str(PALMTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(PALMTREE_ROOT))
if str(PALMTREE_ROOT / "funcsim") not in sys.path:
    sys.path.insert(0, str(PALMTREE_ROOT / "funcsim"))

from vocab import WordVocab
from funcsim.model import FunctionSimilarityModel


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class Sample:
    sample_id: str
    dataset_dir: str
    function_inline: str
    label_text: str


class InlineFunctionParser:
    def __init__(self, vocab: WordVocab, seq_len: int = 420, max_instructions: int = 50):
        self.vocab = vocab
        self.seq_len = seq_len
        self.max_instructions = max_instructions

        self.pad_idx = vocab.stoi.get("<pad>", 0)
        self.unk_idx = vocab.stoi.get("<unk>", 1)
        self.eos_idx = vocab.stoi.get("<eos>", 2)
        self.sos_idx = vocab.stoi.get("<sos>", 3)

        self.addr_pattern = re.compile(r"(\w+)\((0x[0-9a-fA-F]+):([0-9.]+):([0-9.]+):([0-9.]+)\)")
        self.nested_addr_pattern = re.compile(r"address\((0x[0-9a-fA-F]+):([0-9.]+):([0-9.]+):([0-9.]+)\)")
        self.daddr_pattern = re.compile(r"daddr\((0x[0-9a-fA-F]+):([0-9.]+):([0-9.]+):([0-9.]+)\)")
        self.var_pattern = re.compile(r"var\((0x[0-9a-fA-F]+)\)")

    def _parse_instruction(self, inst_text: str):
        tokens = []
        binary_pos = []
        func_pos = []
        bb_pos_list = []
        var_offsets = []

        m = self.addr_pattern.match(inst_text)
        if m:
            opcode = m.group(1)
            fp_val = float(m.group(4))
            bb_val = float(m.group(5))
            operands_text = inst_text[m.end():].strip()
        else:
            parts = inst_text.split()
            opcode = parts[0] if parts else inst_text
            fp_val = bb_val = -1.0
            operands_text = " ".join(parts[1:]) if len(parts) > 1 else ""

        tokens.append(opcode)
        binary_pos.append(-1.0)
        func_pos.append(fp_val)
        bb_pos_list.append(bb_val)
        var_offsets.append(-999999)

        pos = 0
        while pos < len(operands_text):
            nm = self.nested_addr_pattern.match(operands_text[pos:])
            if nm:
                tokens.append("address")
                binary_pos.append(-1.0)
                func_pos.append(float(nm.group(3)))
                bb_pos_list.append(float(nm.group(4)))
                var_offsets.append(-999999)
                pos += nm.end()
                continue

            dm = self.daddr_pattern.match(operands_text[pos:])
            if dm:
                tokens.append("daddr")
                binary_pos.append(float(dm.group(2)))
                func_pos.append(float(dm.group(3)))
                bb_pos_list.append(-1.0)
                var_offsets.append(-999999)
                pos += dm.end()
                continue

            vm = self.var_pattern.match(operands_text[pos:])
            if vm:
                raw = int(vm.group(1), 16)
                if raw >= (1 << 31):
                    raw -= (1 << 32)
                raw = max(-(1 << 20), min((1 << 20), raw))
                tokens.append("var")
                binary_pos.append(-1.0)
                func_pos.append(-1.0)
                bb_pos_list.append(-1.0)
                var_offsets.append(raw)
                pos += vm.end()
                continue

            ch = operands_text[pos]
            if ch.isspace():
                pos += 1
                continue
            if ch in "[]+-*,":
                tokens.append(ch)
                binary_pos.append(-1.0)
                func_pos.append(-1.0)
                bb_pos_list.append(-1.0)
                var_offsets.append(-999999)
                pos += 1
                continue

            end = pos
            while end < len(operands_text) and operands_text[end] not in " []+-*,":
                end += 1
            tokens.append(operands_text[pos:end])
            binary_pos.append(-1.0)
            func_pos.append(-1.0)
            bb_pos_list.append(-1.0)
            var_offsets.append(-999999)
            pos = end

        return tokens, binary_pos, func_pos, bb_pos_list, var_offsets

    def encode_inline(self, inline_text: str) -> dict:
        instructions = [s for s in inline_text.split("\t") if s.strip()]
        instructions = instructions[: self.max_instructions]

        raw_ids = [self.sos_idx]
        raw_bp = [-1.0]
        raw_fp = [-1.0]
        raw_bb = [-1.0]
        raw_var = [-999999]
        boundaries = []

        for inst in instructions:
            boundaries.append(len(raw_ids))
            toks, bp, fp, bb, vo = self._parse_instruction(inst)
            for t in toks:
                raw_ids.append(self.vocab.stoi.get(t, self.unk_idx))
            raw_bp.extend(bp)
            raw_fp.extend(fp)
            raw_bb.extend(bb)
            raw_var.extend(vo)

        raw_ids.append(self.eos_idx)
        raw_bp.append(-1.0)
        raw_fp.append(-1.0)
        raw_bb.append(-1.0)
        raw_var.append(-999999)

        if len(raw_ids) > self.seq_len:
            raw_ids = raw_ids[: self.seq_len]
            raw_bp = raw_bp[: self.seq_len]
            raw_fp = raw_fp[: self.seq_len]
            raw_bb = raw_bb[: self.seq_len]
            raw_var = raw_var[: self.seq_len]

        boundaries = [b for b in boundaries if b < len(raw_ids) - 1]
        if not boundaries:
            boundaries = [1]

        pad = self.seq_len - len(raw_ids)
        raw_ids += [self.pad_idx] * pad
        raw_bp += [-1.0] * pad
        raw_fp += [-1.0] * pad
        raw_bb += [-1.0] * pad
        raw_var += [-999999] * pad

        return {
            "input_ids": torch.tensor(raw_ids, dtype=torch.long),
            "segment": torch.zeros(self.seq_len, dtype=torch.long),
            "binary_pos": torch.tensor(raw_bp, dtype=torch.float),
            "function_pos": torch.tensor(raw_fp, dtype=torch.float),
            "bb_pos": torch.tensor(raw_bb, dtype=torch.float),
            "var_offsets": torch.tensor(raw_var, dtype=torch.long),
            "num_instructions": len(boundaries),
            "instruction_boundaries": boundaries,
        }


class FunctionNamePrototypeDataset(Dataset):
    def __init__(self, samples, parser: InlineFunctionParser, label_to_idx: dict[str, int]):
        self.samples = samples
        self.parser = parser
        self.label_to_idx = label_to_idx

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        enc = self.parser.encode_inline(sample.function_inline)
        target = torch.zeros(len(self.label_to_idx), dtype=torch.float)
        for tok in sample.label_text.split():
            if tok in self.label_to_idx:
                target[self.label_to_idx[tok]] = 1.0
        enc["target"] = target
        enc["label_text"] = sample.label_text
        enc["sample_id"] = sample.sample_id
        return enc


def collate_fn(batch):
    keys = ["input_ids", "segment", "binary_pos", "function_pos", "bb_pos", "var_offsets", "target"]
    out = {k: torch.stack([x[k] for x in batch], dim=0) for k in keys}
    out["num_instructions"] = [x["num_instructions"] for x in batch]
    out["instruction_boundaries"] = [x["instruction_boundaries"] for x in batch]
    out["label_text"] = [x["label_text"] for x in batch]
    out["sample_id"] = [x["sample_id"] for x in batch]
    return out


def load_samples(dataset_dirs: list[str]) -> list[Sample]:
    samples = []
    for dataset_dir in dataset_dirs:
        base = Path(dataset_dir)
        inline_path = base / "self" / "input.inline"
        label_path = base / "self" / "input.label"
        if not inline_path.exists() or not label_path.exists():
            continue

        with inline_path.open() as f_inline, label_path.open() as f_label:
            for idx, (inline, label) in enumerate(zip(f_inline, f_label)):
                inline = inline.strip()
                label = label.strip()
                if not inline or not label:
                    continue
                samples.append(
                    Sample(
                        sample_id=f"{base.name}:{idx}",
                        dataset_dir=base.name,
                        function_inline=inline,
                        label_text=label,
                    )
                )
    return samples


def split_samples(samples: list[Sample], train_ratio: float, seed: int):
    rng = random.Random(seed)
    indices = list(range(len(samples)))
    rng.shuffle(indices)
    split = max(1, int(len(indices) * train_ratio))
    train_idx = set(indices[:split])
    train_samples = [samples[i] for i in range(len(samples)) if i in train_idx]
    test_samples = [samples[i] for i in range(len(samples)) if i not in train_idx]
    return train_samples, test_samples


def build_label_vocab(samples: list[Sample]) -> dict[str, int]:
    toks = sorted({tok for s in samples for tok in s.label_text.split()})
    return {tok: idx for idx, tok in enumerate(toks)}


def compute_micro_metrics(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5):
    probs = torch.sigmoid(logits)
    preds = probs >= threshold
    gold = targets > 0.5
    tp = (preds & gold).sum().item()
    fp = (preds & ~gold).sum().item()
    fn = (~preds & gold).sum().item()
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def topk_tokens(logits: torch.Tensor, idx_to_label: list[str], k: int = 10):
    probs = torch.sigmoid(logits)
    values, indices = torch.topk(probs, k=min(k, probs.numel()))
    return [(idx_to_label[i], float(v)) for v, i in zip(values.tolist(), indices.tolist())]


def run_epoch(encoder, head, loader, optimizer, device, train: bool):
    head.train(train)
    encoder.train(False)
    criterion = nn.BCEWithLogitsLoss()

    all_logits = []
    all_targets = []
    total_loss = 0.0
    total_samples = 0

    for batch in loader:
        inputs = {
            "token_ids": batch["input_ids"].to(device),
            "segment": batch["segment"].to(device),
            "binary_pos": batch["binary_pos"].to(device),
            "function_pos": batch["function_pos"].to(device),
            "bb_pos": batch["bb_pos"].to(device),
            "var_offsets": batch["var_offsets"].to(device),
            "num_instructions": batch["num_instructions"],
            "instruction_boundaries": batch["instruction_boundaries"],
        }
        targets = batch["target"].to(device)

        with torch.set_grad_enabled(train):
            embeddings = encoder(
                inputs["token_ids"],
                inputs["segment"],
                inputs["binary_pos"],
                inputs["function_pos"],
                inputs["bb_pos"],
                inputs["var_offsets"],
                num_instructions=inputs["num_instructions"],
                instruction_boundaries=inputs["instruction_boundaries"],
            )
            logits = head(embeddings)
            loss = criterion(logits, targets)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        all_logits.append(logits.detach().cpu())
        all_targets.append(targets.detach().cpu())
        total_loss += loss.item() * targets.size(0)
        total_samples += targets.size(0)

    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    metrics = compute_micro_metrics(all_logits, all_targets)
    metrics["loss"] = total_loss / max(1, total_samples)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Prototype address-aware function-name prediction with mlm_addr_var.")
    parser.add_argument(
        "--dataset-dir",
        action="append",
        required=True,
        help="Prepared dataset shard directory containing self/input.inline and self/input.label.",
    )
    parser.add_argument("--checkpoint", default="output/mlm_addr_var/latest_bert.pt")
    parser.add_argument("--vocab", default="PalmTree/vocab.pkl")
    parser.add_argument("--output-dir", default="output/function_name_prediction_proto")
    parser.add_argument("--seq-len", type=int, default=420)
    parser.add_argument("--max-instructions", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze-encoder", action="store_true", default=True)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    samples = load_samples(args.dataset_dir)
    if len(samples) < 2:
        raise RuntimeError("Need at least 2 labeled samples to run the prototype.")

    train_samples, test_samples = split_samples(samples, args.train_ratio, args.seed)
    label_to_idx = build_label_vocab(samples)
    idx_to_label = [tok for tok, _ in sorted(label_to_idx.items(), key=lambda x: x[1])]

    vocab = WordVocab.load_vocab(args.vocab)
    parser_obj = InlineFunctionParser(vocab, seq_len=args.seq_len, max_instructions=args.max_instructions)
    train_ds = FunctionNamePrototypeDataset(train_samples, parser_obj, label_to_idx)
    test_ds = FunctionNamePrototypeDataset(test_samples, parser_obj, label_to_idx)
    train_loader = DataLoader(train_ds, batch_size=min(args.batch_size, len(train_ds)), shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=min(args.batch_size, len(test_ds)) if len(test_ds) else 1, shuffle=False, collate_fn=collate_fn)

    encoder = FunctionSimilarityModel(
        vocab_size=len(vocab),
        hidden=768,
        n_layers=12,
        attn_heads=12,
        dropout=0.1,
        max_len=20,
        window_size=20,
        instrs_per_window=2,
        use_address_embedding=True,
        use_var_embedding=True,
        embedding_dim=768,
    )
    encoder.projection = nn.Identity()
    encoder.load_pretrained_bert(args.checkpoint)
    daddr_id = vocab.stoi.get("daddr")
    if daddr_id is not None:
        encoder.bert.embedding.set_special_token_ids(daddr_id)
    encoder.to(device)

    if args.freeze_encoder:
        for p in encoder.parameters():
            p.requires_grad = False

    head = nn.Sequential(
        nn.Linear(768, 768),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(768, len(label_to_idx)),
    ).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=args.lr)

    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(encoder, head, train_loader, optimizer, device, train=True)
        eval_metrics = run_epoch(encoder, head, test_loader if len(test_ds) else train_loader, optimizer, device, train=False)
        history.append({"epoch": epoch, "train": train_metrics, "eval": eval_metrics})
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(
                f"[epoch {epoch:03d}] "
                f"train loss={train_metrics['loss']:.4f} f1={train_metrics['f1']:.4f} | "
                f"eval loss={eval_metrics['loss']:.4f} f1={eval_metrics['f1']:.4f}"
            )

    os.makedirs(args.output_dir, exist_ok=True)

    prediction_rows = []
    head.eval()
    encoder.eval()
    for sample in samples:
        enc = parser_obj.encode_inline(sample.function_inline)
        with torch.no_grad():
            emb = encoder(
                enc["input_ids"].unsqueeze(0).to(device),
                enc["segment"].unsqueeze(0).to(device),
                enc["binary_pos"].unsqueeze(0).to(device),
                enc["function_pos"].unsqueeze(0).to(device),
                enc["bb_pos"].unsqueeze(0).to(device),
                enc["var_offsets"].unsqueeze(0).to(device),
                num_instructions=[enc["num_instructions"]],
                instruction_boundaries=[enc["instruction_boundaries"]],
            )
            logits = head(emb)[0].cpu()
            top_preds = topk_tokens(logits, idx_to_label, k=10)
            prediction_rows.append(
                {
                    "sample_id": sample.sample_id,
                    "dataset_dir": sample.dataset_dir,
                    "gold": sample.label_text,
                    "top_predictions": top_preds,
                    "embedding_norm": float(torch.norm(emb[0]).item()),
                    "embedding_preview": [float(x) for x in emb[0][:8].detach().cpu().tolist()],
                }
            )

    summary = {
        "checkpoint": args.checkpoint,
        "num_samples": len(samples),
        "num_train": len(train_samples),
        "num_test": len(test_samples),
        "num_label_tokens": len(label_to_idx),
        "device": str(device),
        "freeze_encoder": args.freeze_encoder,
        "history_tail": history[-5:],
        "predictions": prediction_rows,
    }

    out_path = Path(args.output_dir) / "prototype_results.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[done] wrote results to {out_path}")
    for row in prediction_rows[:5]:
        preds = ", ".join(f"{tok}:{score:.3f}" for tok, score in row["top_predictions"][:5])
        print(f"{row['sample_id']} | gold={row['gold']} | pred={preds}")


if __name__ == "__main__":
    main()
