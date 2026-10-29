#!/usr/bin/env python3
"""
Compare pretrained PalmTree checkpoints on a lightweight SymLM-style
function-name prediction probe that uses:

- self function body
- caller1 / caller2 function bodies
- internal_callee1 / internal_callee2 function bodies
- external callee labels

No new pretraining is performed. The pretrained checkpoint is frozen and only a
small classifier head is trained.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


THIS_DIR = Path(__file__).resolve().parent
PALMTREE_ROOT = THIS_DIR.parent
FUNCSIM_DIR = PALMTREE_ROOT / "funcsim"
if str(PALMTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(PALMTREE_ROOT))
if str(FUNCSIM_DIR) not in sys.path:
    sys.path.insert(0, str(FUNCSIM_DIR))

from vocab import WordVocab
from funcsim.model import FunctionSimilarityModel


PLACEHOLDER_PREFIXES = (
    "sub_",
    "nullsub_",
    "loc_",
    "j_",
    "unknown_",
    "__x86.",
)
PLACEHOLDER_EXACT = {
    "_init",
    "_fini",
    "frame_dummy",
    "register_tm_clones",
    "deregister_tm_clones",
    "__do_global_dtors_aux",
}


@dataclass
class ContextSample:
    sample_id: str
    binary_name: str
    self_inline: str
    caller_inlines: list[str]
    callee_inlines: list[str]
    external_tokens: list[str]
    label_text: str


class InlineFunctionParser:
    def __init__(self, vocab: WordVocab, seq_len: int = 512, max_instructions: int = 80):
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


class EmbeddingDataset(Dataset):
    def __init__(self, embeddings: torch.Tensor, targets: torch.Tensor):
        self.embeddings = embeddings
        self.targets = targets

    def __len__(self):
        return self.embeddings.size(0)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.targets[idx]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_placeholder_name(name: str) -> bool:
    lowered = name.lower()
    if lowered.startswith(PLACEHOLDER_PREFIXES):
        return True
    return lowered in PLACEHOLDER_EXACT


def normalize_function_name(name: str) -> str | None:
    if not name:
        return None
    if is_placeholder_name(name):
        return None
    if name.startswith("_Z") or name.startswith("__Z") or name.startswith("?"):
        return None

    name = name.split("::")[-1].strip("_")
    if not name:
        return None

    pieces = []
    chunks = re.split(r"[_\W]+", name)
    for chunk in chunks:
        if not chunk:
            continue
        parts = re.findall(r"[A-Z]+(?=[A-Z][a-z]|$)|[A-Z]?[a-z]+|\d+", chunk)
        for part in parts:
            part = part.lower()
            if not part or part.isdigit():
                continue
            if len(part) <= 1:
                continue
            pieces.append(part)

    if not pieces:
        return None
    if len(pieces) > 6:
        return None
    return " ".join(pieces)


def normalize_external_name(name: str) -> str | None:
    if not name:
        return None
    name = name.strip()
    if not name:
        return None
    if name.startswith("."):
        name = name[1:]
    if "@plt" in name:
        name = name.split("@plt", 1)[0]
    if "::" in name:
        name = name.split("::", 1)[-1]
    name = name.strip("_")
    if not name:
        return None
    return name


def tokenize_external_name(name: str) -> list[str]:
    normalized = normalize_external_name(name)
    if not normalized:
        return []
    pieces = []
    for chunk in re.split(r"[_\W]+", normalized):
        if not chunk:
            continue
        parts = re.findall(r"[A-Z]+(?=[A-Z][a-z]|$)|[A-Z]?[a-z]+|\d+", chunk)
        for part in parts:
            part = part.lower()
            if not part or part.isdigit() or len(part) <= 1:
                continue
            pieces.append(part)
    if not pieces and normalized.isalpha() and len(normalized) > 1:
        pieces.append(normalized.lower())
    return pieces


def parse_model_specs(model_specs: list[str]) -> dict[str, str]:
    out = {}
    for spec in model_specs:
        name, checkpoint = spec.split("=", 1)
        out[name] = checkpoint
    return out


def infer_checkpoint_features(checkpoint_path: str) -> tuple[bool, bool]:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    has_addr = any("address_pos" in k or "address_position" in k for k in sd)
    has_var = any("var_pos" in k or "var_position" in k for k in sd)
    return has_addr, has_var


def load_encoder(vocab: WordVocab, checkpoint_path: str, device: torch.device) -> tuple[nn.Module, bool]:
    has_addr, has_var = infer_checkpoint_features(checkpoint_path)
    model = FunctionSimilarityModel(
        vocab_size=len(vocab),
        hidden=768,
        n_layers=12,
        attn_heads=12,
        dropout=0.1,
        max_len=20,
        window_size=20,
        instrs_per_window=2,
        use_address_embedding=has_addr,
        use_var_embedding=has_var,
        embedding_dim=768,
        freeze_bert=False,
    )
    model.projection = nn.Identity()
    model.load_pretrained_bert(checkpoint_path)
    daddr_id = vocab.stoi.get("daddr")
    if daddr_id is not None:
        model.bert.embedding.set_special_token_ids(daddr_id)
    model.to(device)
    model.eval()
    return model, (has_addr or has_var)


def load_samples_from_export(input_json_dir: Path, topk: int) -> list[ContextSample]:
    samples = []
    for json_path in sorted(input_json_dir.glob("*.json")):
        with json_path.open("r", encoding="utf-8") as f:
            records = json.load(f)
        sequence_by_name = {}
        for record in records:
            sequence_by_name.setdefault(record["function"], record["funcins"])

        for idx, record in enumerate(records):
            label_text = normalize_function_name(record["function"])
            if label_text is None:
                continue

            caller_inlines = []
            for name in record.get("callers", []):
                seq = sequence_by_name.get(name)
                if seq:
                    caller_inlines.append(seq)
                if len(caller_inlines) >= topk:
                    break

            callee_inlines = []
            for name in record.get("internal_callees", []):
                seq = sequence_by_name.get(name)
                if seq:
                    callee_inlines.append(seq)
                if len(callee_inlines) >= topk:
                    break

            external_tokens = []
            for name in record.get("external_callees", [])[:topk]:
                external_tokens.extend(tokenize_external_name(name))

            samples.append(
                ContextSample(
                    sample_id=f"{json_path.stem}:{idx}",
                    binary_name=json_path.stem,
                    self_inline=record["funcins"],
                    caller_inlines=caller_inlines,
                    callee_inlines=callee_inlines,
                    external_tokens=external_tokens,
                    label_text=label_text,
                )
            )
    return samples


def split_samples(samples: list[ContextSample], seed: int) -> dict[str, list[ContextSample]]:
    rng = random.Random(seed)
    items = list(samples)
    rng.shuffle(items)
    n_total = len(items)
    n_train = max(1, int(n_total * 0.8))
    n_val = max(1, int(n_total * 0.1))
    n_test = max(1, n_total - n_train - n_val)
    if n_train + n_val + n_test > n_total:
        n_test = n_total - n_train - n_val
    return {
        "train": items[:n_train],
        "val": items[n_train:n_train + n_val],
        "test": items[n_train + n_val:n_train + n_val + n_test],
    }


def build_label_vocab(train_samples: list[ContextSample], min_freq: int, max_tokens: int) -> dict[str, int]:
    counter = Counter()
    for sample in train_samples:
        counter.update(sample.label_text.split())
    items = [(tok, freq) for tok, freq in counter.items() if freq >= min_freq]
    items.sort(key=lambda x: (-x[1], x[0]))
    if max_tokens > 0:
        items = items[:max_tokens]
    return {tok: idx for idx, (tok, _) in enumerate(items)}


def build_external_vocab(train_samples: list[ContextSample], min_freq: int, max_tokens: int) -> dict[str, int]:
    counter = Counter()
    for sample in train_samples:
        counter.update(sample.external_tokens)
    items = [(tok, freq) for tok, freq in counter.items() if freq >= min_freq]
    items.sort(key=lambda x: (-x[1], x[0]))
    if max_tokens > 0:
        items = items[:max_tokens]
    return {tok: idx for idx, (tok, _) in enumerate(items)}


def filter_samples_with_vocab(samples: list[ContextSample], label_to_idx: dict[str, int]) -> list[ContextSample]:
    out = []
    for sample in samples:
        active = [tok for tok in sample.label_text.split() if tok in label_to_idx]
        if not active:
            continue
        sample = ContextSample(
            sample_id=sample.sample_id,
            binary_name=sample.binary_name,
            self_inline=sample.self_inline,
            caller_inlines=sample.caller_inlines,
            callee_inlines=sample.callee_inlines,
            external_tokens=sample.external_tokens,
            label_text=" ".join(active),
        )
        out.append(sample)
    return out


def export_symlm_style_shards(split_samples: dict[str, list[ContextSample]], output_dir: Path, topk: int):
    symlm_root = output_dir / "symlm"
    for split_name, samples in split_samples.items():
        split_dir = symlm_root / split_name
        paths = [split_dir / "self"]
        for i in range(topk):
            paths.extend(
                [
                    split_dir / f"caller{i + 1}",
                    split_dir / f"internal_callee{i + 1}",
                    split_dir / f"external_callee{i + 1}",
                ]
            )
        for path in paths:
            path.mkdir(parents=True, exist_ok=True)

        handles = {
            "self_inline": (split_dir / "self" / "input.inline").open("w", encoding="utf-8"),
            "self_label": (split_dir / "self" / "input.label").open("w", encoding="utf-8"),
            "caller_inline": [
                (split_dir / f"caller{i + 1}" / "input.inline").open("w", encoding="utf-8")
                for i in range(topk)
            ],
            "callee_inline": [
                (split_dir / f"internal_callee{i + 1}" / "input.inline").open("w", encoding="utf-8")
                for i in range(topk)
            ],
            "external_label": [
                (split_dir / f"external_callee{i + 1}" / "input.label").open("w", encoding="utf-8")
                for i in range(topk)
            ],
        }

        for sample in samples:
            handles["self_inline"].write(sample.self_inline + "\n")
            handles["self_label"].write(sample.label_text + "\n")
            for idx in range(topk):
                caller = sample.caller_inlines[idx] if idx < len(sample.caller_inlines) else ""
                callee = sample.callee_inlines[idx] if idx < len(sample.callee_inlines) else ""
                ext = " ".join(sample.external_tokens[idx:idx + 1]) if idx < len(sample.external_tokens) else "##"
                handles["caller_inline"][idx].write(caller + "\n")
                handles["callee_inline"][idx].write(callee + "\n")
                handles["external_label"][idx].write((ext or "##") + "\n")

        handles["self_inline"].close()
        handles["self_label"].close()
        for stream in handles["caller_inline"] + handles["callee_inline"] + handles["external_label"]:
            stream.close()


@torch.no_grad()
def compute_inline_embeddings(
    encoder: nn.Module,
    parser: InlineFunctionParser,
    inline_texts: list[str],
    device: torch.device,
    batch_size: int,
    cache_path: Path | None = None,
) -> dict[str, torch.Tensor]:
    if cache_path is not None and cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu")
        cached = payload.get("embeddings", {})
        missing = [txt for txt in inline_texts if txt not in cached]
        if not missing:
            return cached

    embeddings = {}
    all_texts = inline_texts
    for start in tqdm(range(0, len(all_texts), batch_size), desc="Encoding", leave=False):
        batch_texts = all_texts[start:start + batch_size]
        encoded = [parser.encode_inline(txt) for txt in batch_texts]
        ids_t = torch.stack([e["input_ids"] for e in encoded]).to(device)
        seg_t = torch.stack([e["segment"] for e in encoded]).to(device)
        bp_t = torch.stack([e["binary_pos"] for e in encoded]).to(device)
        fp_t = torch.stack([e["function_pos"] for e in encoded]).to(device)
        bb_t = torch.stack([e["bb_pos"] for e in encoded]).to(device)
        vo_t = torch.stack([e["var_offsets"] for e in encoded]).to(device)
        ni = [e["num_instructions"] for e in encoded]
        bd = [e["instruction_boundaries"] for e in encoded]
        out = encoder(ids_t, seg_t, bp_t, fp_t, bb_t, vo_t, num_instructions=ni, instruction_boundaries=bd)
        for txt, emb in zip(batch_texts, out.cpu()):
            embeddings[txt] = emb

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"embeddings": embeddings}, cache_path)
    return embeddings


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
    exact = (preds == gold).all(dim=1).float().mean().item()
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_match": exact,
    }


def compute_topk_metrics(logits: torch.Tensor, targets: torch.Tensor):
    probs = torch.sigmoid(logits)
    tp = fp = fn = 0
    exact = 0
    n = targets.size(0)
    for i in range(n):
        gold_idx = torch.nonzero(targets[i] > 0.5, as_tuple=False).view(-1)
        k = int(gold_idx.numel())
        if k == 0:
            continue
        pred_idx = torch.topk(probs[i], k=min(k, probs.size(1))).indices
        gold_set = set(gold_idx.tolist())
        pred_set = set(pred_idx.tolist())
        hit = len(gold_set & pred_set)
        tp += hit
        fp += len(pred_set - gold_set)
        fn += len(gold_set - pred_set)
        if pred_set == gold_set:
            exact += 1
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_match": exact / max(n, 1),
    }


def find_best_threshold(logits: torch.Tensor, targets: torch.Tensor):
    candidates = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    best_threshold = 0.5
    best_metrics = compute_micro_metrics(logits, targets, threshold=best_threshold)
    for threshold in candidates:
        metrics = compute_micro_metrics(logits, targets, threshold=threshold)
        if metrics["f1"] > best_metrics["f1"]:
            best_threshold = threshold
            best_metrics = metrics
    return best_threshold, best_metrics


def run_epoch(model, loader, optimizer, device, train: bool):
    criterion = nn.BCEWithLogitsLoss()
    model.train(train)
    total_loss = 0.0
    total = 0
    logits_all = []
    targets_all = []
    for emb, target in loader:
        emb = emb.to(device)
        target = target.to(device)
        with torch.set_grad_enabled(train):
            logits = model(emb)
            loss = criterion(logits, target)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * target.size(0)
        total += target.size(0)
        logits_all.append(logits.detach().cpu())
        targets_all.append(target.detach().cpu())
    logits_all = torch.cat(logits_all, dim=0)
    targets_all = torch.cat(targets_all, dim=0)
    loss = total_loss / max(total, 1)
    return loss, logits_all, targets_all


def train_head(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    batch_size: int,
    epochs: int,
    lr: float,
    device: torch.device,
):
    train_loader = DataLoader(EmbeddingDataset(train_x, train_y), batch_size=min(batch_size, len(train_x)), shuffle=True)
    val_loader = DataLoader(EmbeddingDataset(val_x, val_y), batch_size=min(batch_size, len(val_x)), shuffle=False)
    test_loader = DataLoader(EmbeddingDataset(test_x, test_y), batch_size=min(batch_size, len(test_x)), shuffle=False)

    hidden_dim = min(1024, max(256, train_x.size(1) // 2))
    model = nn.Sequential(
        nn.Linear(train_x.size(1), hidden_dim),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(hidden_dim, train_y.size(1)),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_state = None
    best_val = -1.0
    best_threshold = 0.5
    history = []
    for epoch in range(1, epochs + 1):
        train_loss, train_logits, train_targets = run_epoch(model, train_loader, optimizer, device, train=True)
        val_loss, val_logits, val_targets = run_epoch(model, val_loader, optimizer, device, train=False)
        val_threshold, val_threshold_metrics = find_best_threshold(val_logits, val_targets)
        train_threshold_metrics = compute_micro_metrics(train_logits, train_targets, threshold=val_threshold)
        train_topk_metrics = compute_topk_metrics(train_logits, train_targets)
        val_topk_metrics = compute_topk_metrics(val_logits, val_targets)

        history.append({
            "epoch": epoch,
            "threshold": val_threshold,
            "train": {**train_threshold_metrics, "loss": train_loss},
            "val": {**val_threshold_metrics, "loss": val_loss},
            "train_topk": train_topk_metrics,
            "val_topk": val_topk_metrics,
        })
        if val_topk_metrics["f1"] > best_val:
            best_val = val_topk_metrics["f1"]
            best_threshold = val_threshold
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    test_loss, test_logits, test_targets = run_epoch(model, test_loader, optimizer, device, train=False)
    test_threshold_metrics = compute_micro_metrics(test_logits, test_targets, threshold=best_threshold)
    test_threshold_metrics["loss"] = test_loss
    test_topk_metrics = compute_topk_metrics(test_logits, test_targets)
    return history, best_threshold, test_threshold_metrics, test_topk_metrics


def build_targets(samples: list[ContextSample], label_to_idx: dict[str, int]) -> torch.Tensor:
    ys = []
    for sample in samples:
        y = torch.zeros(len(label_to_idx), dtype=torch.float)
        for tok in sample.label_text.split():
            if tok in label_to_idx:
                y[label_to_idx[tok]] = 1.0
        ys.append(y)
    return torch.stack(ys)


def build_features(
    samples: list[ContextSample],
    embeddings: dict[str, torch.Tensor],
    external_to_idx: dict[str, int],
) -> dict[str, torch.Tensor]:
    if not samples:
        raise RuntimeError("No samples available for feature construction.")

    zero = torch.zeros_like(next(iter(embeddings.values())))
    ext_dim = len(external_to_idx)
    self_rows = []
    ctx_rows = []
    ctx_no_ext_rows = []
    graph_rows = []
    for sample in samples:
        self_emb = embeddings[sample.self_inline]

        caller_rows = []
        for idx in range(2):
            if idx < len(sample.caller_inlines):
                caller_rows.append(embeddings[sample.caller_inlines[idx]])
            else:
                caller_rows.append(zero)

        callee_rows = []
        for idx in range(2):
            if idx < len(sample.callee_inlines):
                callee_rows.append(embeddings[sample.callee_inlines[idx]])
            else:
                callee_rows.append(zero)

        ctx_no_ext = torch.cat([self_emb] + caller_rows + callee_rows, dim=0)
        graph_only = torch.cat(caller_rows + callee_rows, dim=0)

        if ext_dim > 0:
            ext_vec = torch.zeros(ext_dim, dtype=torch.float)
            for tok in sample.external_tokens:
                if tok in external_to_idx:
                    ext_vec[external_to_idx[tok]] = 1.0
            ctx = torch.cat([ctx_no_ext, ext_vec], dim=0)
        else:
            ctx = ctx_no_ext

        self_rows.append(self_emb)
        ctx_rows.append(ctx)
        ctx_no_ext_rows.append(ctx_no_ext)
        graph_rows.append(graph_only)

    return {
        "self_only": torch.stack(self_rows),
        "context_concat": torch.stack(ctx_rows),
        "context_no_external": torch.stack(ctx_no_ext_rows),
        "graph_only": torch.stack(graph_rows),
    }


def main():
    parser = argparse.ArgumentParser(description="Compare pretrained PalmTree checkpoints on SymLM-style context probes.")
    parser.add_argument("--input-json-dir", required=True)
    parser.add_argument("--vocab", default="/path/to/numAnchor_artifact/aaedataset/blockonly/data/vocab.pkl")
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="Model spec in the form name=/path/to/checkpoint.pt . Repeat for multiple models.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--max-instructions", type=int, default=80)
    parser.add_argument("--min-label-freq", type=int, default=10)
    parser.add_argument("--max-label-tokens", type=int, default=1024)
    parser.add_argument("--min-external-freq", type=int, default=5)
    parser.add_argument("--max-external-tokens", type=int, default=512)
    parser.add_argument("--encode-batch-size", type=int, default=32)
    parser.add_argument("--classifier-batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    input_json_dir = Path(args.input_json_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_samples = load_samples_from_export(input_json_dir, topk=args.topk)
    if not all_samples:
        raise RuntimeError(f"No usable samples found under {input_json_dir}")

    split_samples_dict = split_samples(all_samples, seed=args.seed)
    label_to_idx = build_label_vocab(split_samples_dict["train"], args.min_label_freq, args.max_label_tokens)
    split_samples_dict = {k: filter_samples_with_vocab(v, label_to_idx) for k, v in split_samples_dict.items()}
    external_to_idx = build_external_vocab(split_samples_dict["train"], args.min_external_freq, args.max_external_tokens)

    export_symlm_style_shards(split_samples_dict, output_dir, topk=args.topk)

    manifest = {
        "input_json_dir": str(input_json_dir),
        "num_raw_samples": len(all_samples),
        "split_sizes": {k: len(v) for k, v in split_samples_dict.items()},
        "num_label_tokens": len(label_to_idx),
        "num_external_tokens": len(external_to_idx),
        "topk": args.topk,
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    unique_inline_texts = sorted(
        {
            sample.self_inline
            for split_name in ("train", "val", "test")
            for sample in split_samples_dict[split_name]
        }
        |
        {
            txt
            for split_name in ("train", "val", "test")
            for sample in split_samples_dict[split_name]
            for txt in (sample.caller_inlines + sample.callee_inlines)
            if txt
        }
    )

    vocab = WordVocab.load_vocab(args.vocab)
    parser_obj = InlineFunctionParser(vocab, seq_len=args.seq_len, max_instructions=args.max_instructions)

    summary = {
        "manifest": manifest,
        "models": {},
    }

    model_specs = parse_model_specs(args.model)
    for name, checkpoint in model_specs.items():
        print(f"[symlm-context] Evaluating model {name}: {checkpoint}")
        encoder, use_addr_var = load_encoder(vocab, checkpoint, device)
        embeddings = compute_inline_embeddings(
            encoder=encoder,
            parser=parser_obj,
            inline_texts=unique_inline_texts,
            device=device,
            batch_size=args.encode_batch_size,
            cache_path=output_dir / "embedding_cache" / f"{name}.pt",
        )

        feature_by_split = {}
        target_by_split = {}
        for split_name in ("train", "val", "test"):
            feature_by_split[split_name] = build_features(split_samples_dict[split_name], embeddings, external_to_idx)
            target_by_split[split_name] = build_targets(split_samples_dict[split_name], label_to_idx)

        summary["models"][name] = {
            "checkpoint": checkpoint,
            "use_addr_var": use_addr_var,
            "modes": {},
        }

        for mode in ("self_only", "context_no_external", "context_concat", "graph_only"):
            train_x = feature_by_split["train"][mode]
            val_x = feature_by_split["val"][mode]
            test_x = feature_by_split["test"][mode]
            train_y = target_by_split["train"]
            val_y = target_by_split["val"]
            test_y = target_by_split["test"]
            history, best_threshold, test_threshold_metrics, test_topk_metrics = train_head(
                train_x=train_x,
                train_y=train_y,
                val_x=val_x,
                val_y=val_y,
                test_x=test_x,
                test_y=test_y,
                batch_size=args.classifier_batch_size,
                epochs=args.epochs,
                lr=args.lr,
                device=device,
            )
            summary["models"][name]["modes"][mode] = {
                "feature_dim": int(train_x.size(1)),
                "best_threshold": best_threshold,
                "history_tail": history[-5:],
                "test_threshold_metrics": test_threshold_metrics,
                "test_topk_metrics": test_topk_metrics,
            }
            print(
                f"[symlm-context] {name} {mode} | "
                f"topk_f1={test_topk_metrics['f1']:.4f} "
                f"thr_f1={test_threshold_metrics['f1']:.4f} "
                f"topk_exact={test_topk_metrics['exact_match']:.4f}"
            )

    results_path = output_dir / "compare_results_context.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[done] wrote comparison results to {results_path}")


if __name__ == "__main__":
    main()
