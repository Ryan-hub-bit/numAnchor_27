#!/usr/bin/env python3
"""
Supervised finetuning for instruction-level pair classification on top of
PalmTree pretrain BERT checkpoints.

Current tasks:
  - same_bb
  - jump_target
  - call_target
  - same_dref

The script keeps the unit construction consistent with probe_eval.py but
finetunes the encoder end-to-end with a task-specific pair head.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if CUR_DIR not in sys.path:
    sys.path.insert(0, CUR_DIR)

from model import AddressAwareBERT  # noqa: E402
from vocab import WordVocab  # noqa: E402
from probe_eval import (  # noqa: E402
    HEADER_RE,
    NESTED_ADDR_RE,
    DADDR_RE,
    VAR_RE,
    ProbePair,
    ProbeUnit,
    build_call_target_pairs,
    build_equivalence_groups,
    build_jump_target_pairs,
    build_same_bb_pairs,
    build_same_dref_pairs,
    canonicalize_daddr_token,
    choose_function_subset,
    ensure_balanced_pairs,
    infer_binary_metadata_from_records_dir,
    load_json,
    load_function_blocks_with_metadata,
    load_function_metadata,
    reconstruct_blocks,
    split_function_ids,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tie_aware_ranks(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(scores, kind="mergesort")
    ranks = np.zeros_like(scores, dtype=np.float64)
    i = 0
    while i < len(scores):
        j = i + 1
        while j < len(scores) and scores[order[j]] == scores[order[i]]:
            j += 1
        avg_rank = (i + j - 1) / 2.0 + 1.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def binary_auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    pos = y == 1
    neg = y == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = tie_aware_ranks(s)
    pos_rank_sum = ranks[pos].sum()
    auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def f1_score(pred: Sequence[int], labels: Sequence[int]) -> float:
    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(pred, dtype=np.int64)
    tp = int(((p == 1) & (y == 1)).sum())
    fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum())
    denom = 2 * tp + fp + fn
    return 0.0 if denom == 0 else (2.0 * tp) / denom


def balanced_accuracy(pred: Sequence[int], labels: Sequence[int]) -> float:
    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(pred, dtype=np.int64)
    pos = y == 1
    neg = y == 0
    tpr = float((p[pos] == 1).mean()) if pos.any() else 0.0
    tnr = float((p[neg] == 0).mean()) if neg.any() else 0.0
    return 0.5 * (tpr + tnr)


def compute_metrics_from_scores(scores: Sequence[float], labels: Sequence[int], threshold: float = 0.5) -> Dict[str, float]:
    pred = [1 if s >= threshold else 0 for s in scores]
    acc = float(np.mean(np.asarray(pred) == np.asarray(labels)))
    return {
        "auc": binary_auc(scores, labels),
        "accuracy": acc,
        "balanced_accuracy": balanced_accuracy(pred, labels),
        "f1": f1_score(pred, labels),
    }


def choose_threshold(scores: Sequence[float], labels: Sequence[int]) -> float:
    unique = sorted(set(float(x) for x in scores))
    if not unique:
        return 0.5
    if len(unique) == 1:
        return unique[0]

    candidates = [unique[0] - 1e-6]
    candidates.extend((unique[i] + unique[i + 1]) / 2.0 for i in range(len(unique) - 1))
    candidates.append(unique[-1] + 1e-6)

    best_thr = candidates[0]
    best_score = -1.0
    for thr in candidates:
        pred = [1 if s >= thr else 0 for s in scores]
        score = balanced_accuracy(pred, labels)
        if score > best_score:
            best_score = score
            best_thr = thr
    return float(best_thr)


class UnitSequenceBuilder:
    def __init__(self, vocab: WordVocab, seq_len: int, max_instructions: int):
        self.vocab = vocab
        self.seq_len = seq_len
        self.max_instructions = max_instructions
        self.pad_idx = vocab.stoi.get("<pad>", 0)
        self.unk_idx = vocab.stoi.get("<unk>", 1)
        self.eos_idx = vocab.stoi.get("<eos>", 2)
        self.sos_idx = vocab.stoi.get("<sos>", 3)

    def _parse_instruction(self, inst_text: str):
        tokens: List[str] = []
        binary_pos: List[float] = []
        func_pos: List[float] = []
        bb_pos: List[float] = []
        var_offsets: List[int] = []
        section_kind_ids: List[int] = []
        addr_type_ids: List[int] = []

        m = HEADER_RE.match(inst_text)
        if m:
            opcode = m.group(1)
            bp_val = float(m.group(3))
            fp_val = float(m.group(4))
            bb_val = float(m.group(5))
            operands_text = inst_text[m.end():].strip()
        else:
            parts = inst_text.split()
            opcode = parts[0] if parts else inst_text
            bp_val = -1.0
            fp_val = -1.0
            bb_val = -1.0
            operands_text = " ".join(parts[1:]) if len(parts) > 1 else ""

        tokens.append(opcode)
        binary_pos.append(bp_val)
        func_pos.append(fp_val)
        bb_pos.append(bb_val)
        var_offsets.append(-999999)
        section_kind_ids.append(0)
        addr_type_ids.append(1 if bp_val >= 0 or fp_val >= 0 or bb_val >= 0 else 0)

        pos = 0
        while pos < len(operands_text):
            nm = NESTED_ADDR_RE.match(operands_text[pos:])
            if nm:
                tokens.append("address")
                binary_pos.append(float(nm.group(2)))
                func_pos.append(float(nm.group(3)))
                bb_pos.append(float(nm.group(4)))
                var_offsets.append(-999999)
                section_kind_ids.append(0)
                addr_type_ids.append(1)
                pos += nm.end()
                continue

            dm = DADDR_RE.match(operands_text[pos:])
            if dm:
                daddr_token, daddr_kind = canonicalize_daddr_token(dm.group(1), dm.group(5))
                tokens.append(daddr_token)
                binary_pos.append(float(dm.group(3)))
                func_pos.append(float(dm.group(4)))
                bb_pos.append(-1.0)
                var_offsets.append(-999999)
                section_kind_ids.append(daddr_kind)
                addr_type_ids.append(2)
                pos += dm.end()
                continue

            vm = VAR_RE.match(operands_text[pos:])
            if vm:
                raw = int(vm.group(1), 16)
                if raw >= (1 << 31):
                    raw -= (1 << 32)
                raw = max(-(1 << 20), min((1 << 20), raw))
                tokens.append("var")
                binary_pos.append(-1.0)
                func_pos.append(-1.0)
                bb_pos.append(-1.0)
                var_offsets.append(raw)
                section_kind_ids.append(0)
                addr_type_ids.append(3)
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
                bb_pos.append(-1.0)
                var_offsets.append(-999999)
                section_kind_ids.append(0)
                addr_type_ids.append(0)
                pos += 1
                continue
            end = pos
            while end < len(operands_text) and operands_text[end] not in " []+-*,":
                end += 1
            tokens.append(operands_text[pos:end])
            binary_pos.append(-1.0)
            func_pos.append(-1.0)
            bb_pos.append(-1.0)
            var_offsets.append(-999999)
            section_kind_ids.append(0)
            addr_type_ids.append(0)
            pos = end

        return tokens, binary_pos, func_pos, bb_pos, var_offsets, section_kind_ids, addr_type_ids

    def build(self, instructions: Sequence[str]) -> Dict[str, torch.Tensor]:
        token_ids = [self.sos_idx]
        segment_labels = [1]
        binary_pos = [-1.0]
        function_pos = [-1.0]
        bb_pos = [-1.0]
        var_offsets = [-999999]
        section_kind_ids = [0]
        addr_type_ids = [0]

        opcode_index: Optional[int] = None
        address_index: Optional[int] = None

        for inst_idx, inst in enumerate(instructions[: self.max_instructions]):
            seg_id = min(inst_idx + 1, 15)
            toks, bp, fp, bbp, vo, sk, at = self._parse_instruction(inst)
            for local_idx, tok in enumerate(toks):
                if opcode_index is None:
                    opcode_index = len(token_ids)
                if tok == "address" and address_index is None:
                    address_index = len(token_ids)
                token_ids.append(self.vocab.stoi.get(tok, self.unk_idx))
                segment_labels.append(seg_id)
                binary_pos.append(bp[local_idx])
                function_pos.append(fp[local_idx])
                bb_pos.append(bbp[local_idx])
                var_offsets.append(vo[local_idx])
                section_kind_ids.append(sk[local_idx])
                addr_type_ids.append(at[local_idx])
            token_ids.append(self.eos_idx)
            segment_labels.append(seg_id)
            binary_pos.append(-1.0)
            function_pos.append(-1.0)
            bb_pos.append(-1.0)
            var_offsets.append(-999999)
            section_kind_ids.append(0)
            addr_type_ids.append(0)

        token_ids = token_ids[: self.seq_len]
        segment_labels = segment_labels[: self.seq_len]
        binary_pos = binary_pos[: self.seq_len]
        function_pos = function_pos[: self.seq_len]
        bb_pos = bb_pos[: self.seq_len]
        var_offsets = var_offsets[: self.seq_len]
        section_kind_ids = section_kind_ids[: self.seq_len]
        addr_type_ids = addr_type_ids[: self.seq_len]

        pad = self.seq_len - len(token_ids)
        if pad > 0:
            token_ids += [self.pad_idx] * pad
            segment_labels += [0] * pad
            binary_pos += [-1.0] * pad
            function_pos += [-1.0] * pad
            bb_pos += [-1.0] * pad
            var_offsets += [-999999] * pad
            section_kind_ids += [0] * pad
            addr_type_ids += [0] * pad

        if opcode_index is None or opcode_index >= self.seq_len:
            opcode_index = 1
        if address_index is None or address_index >= self.seq_len:
            address_index = -1

        return {
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
            "segment_labels": torch.tensor(segment_labels, dtype=torch.long),
            "binary_pos": torch.tensor(binary_pos, dtype=torch.float),
            "function_pos": torch.tensor(function_pos, dtype=torch.float),
            "bb_pos": torch.tensor(bb_pos, dtype=torch.float),
            "var_offsets": torch.tensor(var_offsets, dtype=torch.long),
            "section_kind_ids": torch.tensor(section_kind_ids, dtype=torch.long),
            "addr_type_ids": torch.tensor(addr_type_ids, dtype=torch.long),
            "opcode_index": torch.tensor(opcode_index, dtype=torch.long),
            "address_index": torch.tensor(address_index, dtype=torch.long),
        }


class PairDataset(Dataset):
    def __init__(self, pairs: Sequence[ProbePair], units: Dict[str, ProbeUnit], builder: UnitSequenceBuilder):
        self.pairs = list(pairs)
        self.unit_tensors = {uid: builder.build(unit.instructions) for uid, unit in units.items()}

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        pair = self.pairs[idx]
        left = self.unit_tensors[pair.left_id]
        right = self.unit_tensors[pair.right_id]
        out = {
            "label": torch.tensor(pair.label, dtype=torch.float32),
        }
        for prefix, item in [("left", left), ("right", right)]:
            for key, value in item.items():
                out[f"{prefix}_{key}"] = value
        return out


def collate_batch(batch: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    keys = batch[0].keys()
    return {key: torch.stack([item[key] for item in batch], dim=0) for key in keys}


class PairFinetuneModel(nn.Module):
    def __init__(
        self,
        checkpoint_path: str,
        vocab: WordVocab,
        hidden: int,
        n_layers: int,
        attn_heads: int,
        seq_len: int,
        task: str,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.task = task
        self.pad_idx = vocab.stoi.get("<pad>", 0)
        self.sos_idx = vocab.stoi.get("<sos>", 3)
        self.eos_idx = vocab.stoi.get("<eos>", 2)

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict):
            sd = ckpt.get("bert_state_dict", ckpt.get("model_state_dict", ckpt))
        else:
            sd = ckpt.state_dict()
        if any(k.startswith("module.") for k in sd):
            sd = {k[len("module."):]: v for k, v in sd.items()}

        has_address = any("address_pos" in k or "address_position" in k for k in sd)
        has_var = any("var_offset" in k or "var_position" in k for k in sd)
        self.use_addr = has_address and has_var

        self.bert = AddressAwareBERT(
            vocab_size=len(vocab),
            hidden=hidden,
            n_layers=n_layers,
            attn_heads=attn_heads,
            max_len=seq_len,
            use_address_embedding=self.use_addr,
            use_var_embedding=self.use_addr,
        )
        missing, unexpected = self.bert.load_state_dict(sd, strict=False)
        if missing:
            print(f"[Finetune] Warning: missing keys when loading {checkpoint_path}: {missing[:5]}")
        if unexpected:
            print(f"[Finetune] Warning: unexpected keys when loading {checkpoint_path}: {unexpected[:5]}")

        daddr_token_ids = sorted(
            token_id
            for token, token_id in vocab.stoi.items()
            if token == "daddr" or token.startswith("daddr_")
        )
        if daddr_token_ids and self.use_addr:
            self.bert.embedding.set_special_token_ids(daddr_token_ids)

        repr_dim = hidden * 2
        self.classifier = nn.Sequential(
            nn.Linear(repr_dim * 4, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def _mean_pool(self, hidden: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        valid_mask = (token_ids != self.pad_idx) & (token_ids != self.sos_idx) & (token_ids != self.eos_idx)
        if not bool(valid_mask.any()):
            valid_mask = token_ids != self.pad_idx
        valid_mask = valid_mask.float().unsqueeze(-1)
        denom = valid_mask.sum(dim=1).clamp_min(1.0)
        return (hidden * valid_mask).sum(dim=1) / denom

    def _gather_index(self, hidden: torch.Tensor, index_tensor: torch.Tensor) -> torch.Tensor:
        batch = torch.arange(hidden.size(0), device=hidden.device)
        return hidden[batch, index_tensor]

    def _encode_side(self, batch: Dict[str, torch.Tensor], prefix: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ids = batch[f"{prefix}_input_ids"]
        seg = batch[f"{prefix}_segment_labels"]
        bp = batch[f"{prefix}_binary_pos"]
        fp = batch[f"{prefix}_function_pos"]
        bb = batch[f"{prefix}_bb_pos"]
        vo = batch[f"{prefix}_var_offsets"]
        sk = batch[f"{prefix}_section_kind_ids"]
        at = batch[f"{prefix}_addr_type_ids"]
        opcode_idx = batch[f"{prefix}_opcode_index"]
        address_idx = batch[f"{prefix}_address_index"]

        hidden = self.bert(
            ids, seg, bp, fp, bb, vo,
            section_kind_ids=sk,
            addr_type_ids=at,
        )
        pooled = self._mean_pool(hidden, ids)
        opcode = self._gather_index(hidden, opcode_idx)

        address = pooled.clone()
        valid_addr = address_idx >= 0
        if bool(valid_addr.any()):
            batch_ids = torch.arange(hidden.size(0), device=hidden.device)
            address[valid_addr] = hidden[batch_ids[valid_addr], address_idx[valid_addr]]

        return pooled, opcode, address

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        left_pooled, left_opcode, left_address = self._encode_side(batch, "left")
        right_pooled, right_opcode, right_address = self._encode_side(batch, "right")

        if self.task == "jump_target":
            left_repr = torch.cat([left_pooled, left_address], dim=-1)
            right_repr = torch.cat([right_pooled, right_opcode], dim=-1)
        elif self.task == "call_target":
            left_repr = torch.cat([left_pooled, left_address], dim=-1)
            right_repr = torch.cat([right_pooled, right_opcode], dim=-1)
        elif self.task in {"same_bb", "same_dref"}:
            left_repr = torch.cat([left_pooled, left_opcode], dim=-1)
            right_repr = torch.cat([right_pooled, right_opcode], dim=-1)
        else:
            raise ValueError(f"Unsupported task: {self.task}")

        feat = torch.cat([left_repr, right_repr, torch.abs(left_repr - right_repr), left_repr * right_repr], dim=-1)
        return self.classifier(feat).squeeze(-1)


@torch.no_grad()
def collect_scores(model: PairFinetuneModel, loader: DataLoader, device: torch.device) -> Tuple[List[float], List[int], float]:
    model.eval()
    logits_all: List[float] = []
    labels_all: List[int] = []
    total_loss = 0.0
    n_batches = 0
    criterion = nn.BCEWithLogitsLoss()

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch)
        loss = criterion(logits, batch["label"])
        probs = torch.sigmoid(logits).detach().cpu().numpy().tolist()
        labels = batch["label"].detach().cpu().numpy().astype(int).tolist()
        logits_all.extend(probs)
        labels_all.extend(labels)
        total_loss += float(loss.item())
        n_batches += 1
    return logits_all, labels_all, total_loss / max(1, n_batches)


@torch.no_grad()
def evaluate(model: PairFinetuneModel, loader: DataLoader, device: torch.device, threshold: float = 0.5) -> Dict[str, float]:
    scores, labels, avg_loss = collect_scores(model, loader, device)
    metrics = compute_metrics_from_scores(scores, labels, threshold=threshold)
    metrics["loss"] = avg_loss
    return metrics


def train_one_model(
    args: argparse.Namespace,
    checkpoint_path: str,
    train_pairs: Sequence[ProbePair],
    val_pairs: Sequence[ProbePair],
    test_pairs: Sequence[ProbePair],
    units: Dict[str, ProbeUnit],
    vocab: WordVocab,
    device: torch.device,
) -> Dict[str, object]:
    builder = UnitSequenceBuilder(vocab=vocab, seq_len=args.seq_len, max_instructions=args.max_instructions)
    train_ds = PairDataset(train_pairs, units, builder)
    val_ds = PairDataset(val_pairs, units, builder)
    test_ds = PairDataset(test_pairs, units, builder)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_batch)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_batch)

    model = PairFinetuneModel(
        checkpoint_path=checkpoint_path,
        vocab=vocab,
        hidden=args.hidden,
        n_layers=args.n_layers,
        attn_heads=args.attn_heads,
        seq_len=args.seq_len,
        task=args.task,
        dropout=args.dropout,
    ).to(device)

    enc_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("bert."):
            enc_params.append(param)
        else:
            head_params.append(param)

    optimizer = AdamW(
        [
            {"params": enc_params, "lr": args.encoder_lr},
            {"params": head_params, "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
    )
    criterion = nn.BCEWithLogitsLoss()

    best_state = None
    best_epoch = -1
    best_val_auc = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            loss = criterion(logits, batch["label"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.item())
            n_batches += 1

        train_metrics = evaluate(model, train_loader, device, threshold=0.5)
        val_metrics = evaluate(model, val_loader, device, threshold=0.5)
        record = {
            "epoch": epoch,
            "train_loss_epoch_avg": total_loss / max(1, n_batches),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        }
        history.append(record)
        print(
            f"[Finetune] epoch={epoch} "
            f"train_auc={train_metrics['auc']:.4f} "
            f"val_auc={val_metrics['auc']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f}"
        )

        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
        if args.save_model:
            os.makedirs(os.path.dirname(os.path.abspath(args.save_model)), exist_ok=True)
            payload = {
                "model_state_dict": best_state,
                "checkpoint": checkpoint_path,
                "task": args.task,
                "config": vars(args),
            }
            torch.save(payload, args.save_model)
            print(f"[Finetune] Saved best model to {args.save_model}")

    val_scores, val_labels, val_loss = collect_scores(model, val_loader, device)
    best_threshold = choose_threshold(val_scores, val_labels)
    val_metrics = compute_metrics_from_scores(val_scores, val_labels, threshold=best_threshold)
    val_metrics["loss"] = val_loss

    test_scores, test_labels, test_loss = collect_scores(model, test_loader, device)
    test_metrics = compute_metrics_from_scores(test_scores, test_labels, threshold=best_threshold)
    test_metrics["loss"] = test_loss
    test_metrics["threshold"] = best_threshold
    return {
        "checkpoint": checkpoint_path,
        "use_addr": model.use_addr,
        "best_epoch": best_epoch,
        "best_val_auc": best_val_auc,
        "best_threshold": best_threshold,
        "val_metrics_best_threshold": val_metrics,
        "test_metrics": test_metrics,
        "history": history,
    }


def split_train_val(function_ids: Sequence[str], val_fraction: float, seed: int) -> Tuple[List[str], List[str]]:
    ids = list(function_ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    val_n = max(1, int(len(ids) * val_fraction))
    val_ids = sorted(ids[:val_n])
    train_ids = sorted(ids[val_n:])
    return train_ids, val_ids


def build_task_splits(
    args: argparse.Namespace,
    function_blocks: Dict[str, List[str]],
    func_groups: Dict[str, int],
    binary_by_func: Dict[str, str],
) -> Tuple[List[ProbePair], List[ProbePair], List[ProbePair], Dict[str, ProbeUnit]]:
    instructions_by_func, blocks_by_func = reconstruct_blocks(function_blocks)
    train_plus_val_ids, test_ids = split_function_ids(list(function_blocks.keys()), args.test_fraction, args.split_seed)
    train_ids, val_ids = split_train_val(train_plus_val_ids, args.val_fraction, args.split_seed + 1)

    if args.task == "same_bb":
        train_pairs, train_units = build_same_bb_pairs(blocks_by_func, train_ids, args.max_train_pairs, args.split_seed + 11)
        val_pairs, val_units = build_same_bb_pairs(blocks_by_func, val_ids, args.max_val_pairs, args.split_seed + 12)
        test_pairs, test_units = build_same_bb_pairs(blocks_by_func, test_ids, args.max_test_pairs, args.split_seed + 13)
    elif args.task == "jump_target":
        train_pairs, train_units = build_jump_target_pairs(
            instructions_by_func, train_ids, args.max_train_pairs, args.split_seed + 21, args.jump_target_opcode
        )
        val_pairs, val_units = build_jump_target_pairs(
            instructions_by_func, val_ids, args.max_val_pairs, args.split_seed + 22, args.jump_target_opcode
        )
        test_pairs, test_units = build_jump_target_pairs(
            instructions_by_func, test_ids, args.max_test_pairs, args.split_seed + 23, args.jump_target_opcode
        )
    elif args.task == "call_target":
        train_pairs, train_units = build_call_target_pairs(
            instructions_by_func, blocks_by_func, train_ids, binary_by_func, args.max_train_pairs, args.split_seed + 31
        )
        val_pairs, val_units = build_call_target_pairs(
            instructions_by_func, blocks_by_func, val_ids, binary_by_func, args.max_val_pairs, args.split_seed + 32
        )
        test_pairs, test_units = build_call_target_pairs(
            instructions_by_func, blocks_by_func, test_ids, binary_by_func, args.max_test_pairs, args.split_seed + 33
        )
    elif args.task == "same_dref":
        train_pairs, train_units = build_same_dref_pairs(
            instructions_by_func, train_ids, binary_by_func, args.max_train_pairs, args.split_seed + 41
        )
        val_pairs, val_units = build_same_dref_pairs(
            instructions_by_func, val_ids, binary_by_func, args.max_val_pairs, args.split_seed + 42
        )
        test_pairs, test_units = build_same_dref_pairs(
            instructions_by_func, test_ids, binary_by_func, args.max_test_pairs, args.split_seed + 43
        )
    else:
        raise ValueError(f"Unsupported task: {args.task}")

    train_pairs = ensure_balanced_pairs(train_pairs)
    val_pairs = ensure_balanced_pairs(val_pairs)
    test_pairs = ensure_balanced_pairs(test_pairs)

    units: Dict[str, ProbeUnit] = {}
    for pool in [train_units, val_units, test_units]:
        units.update(pool)

    if not train_pairs or not val_pairs or not test_pairs:
        raise RuntimeError(
            f"Insufficient pairs for task={args.task}: "
            f"train={len(train_pairs)} val={len(val_pairs)} test={len(test_pairs)}"
        )

    print(
        f"[Finetune] task={args.task} "
        f"train_pairs={len(train_pairs)} val_pairs={len(val_pairs)} test_pairs={len(test_pairs)} "
        f"units={len(units)}"
    )
    return train_pairs, val_pairs, test_pairs, units


def main():
    parser = argparse.ArgumentParser(description="Supervised finetuning for PalmTree pair tasks")
    parser.add_argument("--task", choices=["same_bb", "jump_target", "call_target", "same_dref"], required=True)
    parser.add_argument("--function_blocks", required=True)
    parser.add_argument("--function_metadata", default=None)
    parser.add_argument("--function_records_dir", default=None)
    parser.add_argument("--funcsim_pairs", default=None)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--save_model", default=None)
    parser.add_argument("--hidden", type=int, default=768)
    parser.add_argument("--n_layers", type=int, default=12)
    parser.add_argument("--attn_heads", type=int, default=12)
    parser.add_argument("--seq_len", type=int, default=20)
    parser.add_argument("--max_instructions", type=int, default=8)
    parser.add_argument("--max_functions", type=int, default=None)
    parser.add_argument("--max_train_pairs", type=int, default=2000)
    parser.add_argument("--max_val_pairs", type=int, default=600)
    parser.add_argument("--max_test_pairs", type=int, default=1000)
    parser.add_argument("--test_fraction", type=float, default=0.3)
    parser.add_argument("--val_fraction", type=float, default=0.15)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--jump_target_opcode", default="mov")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--encoder_lr", type=float, default=2e-5)
    parser.add_argument("--head_lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    set_seed(args.seed)
    device = torch.device(args.device)

    function_blocks, inline_binary_by_func = load_function_blocks_with_metadata(args.function_blocks)
    selected_func_ids = choose_function_subset(list(function_blocks.keys()), args.max_functions, args.split_seed)
    function_blocks = {fid: function_blocks[fid] for fid in selected_func_ids}
    binary_by_func = {fid: bid for fid, bid in inline_binary_by_func.items() if fid in function_blocks}
    explicit_binary_by_func = load_function_metadata(args.function_metadata)
    for func_id, binary_id in explicit_binary_by_func.items():
        if func_id in function_blocks:
            binary_by_func[func_id] = binary_id
    inferred_binary_by_func = infer_binary_metadata_from_records_dir(function_blocks, args.function_records_dir)
    for func_id, binary_id in inferred_binary_by_func.items():
        binary_by_func.setdefault(func_id, binary_id)
    func_groups = build_equivalence_groups(args.funcsim_pairs)
    vocab = WordVocab.load_vocab(args.vocab)

    train_pairs, val_pairs, test_pairs, units = build_task_splits(args, function_blocks, func_groups, binary_by_func)
    result = train_one_model(
        args=args,
        checkpoint_path=args.checkpoint,
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        test_pairs=test_pairs,
        units=units,
        vocab=vocab,
        device=device,
    )

    report = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "config": vars(args),
        "num_functions": len(selected_func_ids),
        "num_functions_with_binary_metadata": len(binary_by_func),
        "num_train_pairs": len(train_pairs),
        "num_val_pairs": len(val_pairs),
        "num_test_pairs": len(test_pairs),
        "result": result,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[Finetune] Wrote report to {args.output}")


if __name__ == "__main__":
    main()
