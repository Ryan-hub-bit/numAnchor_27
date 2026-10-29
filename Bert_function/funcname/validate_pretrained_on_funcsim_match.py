#!/usr/bin/env python3
"""
Validate pretrained PalmTree checkpoints on function-name prediction using the
existing funcsim_match corpus.

This script does four things:
1. Loads funcsim_match function blocks and metadata.
2. Builds a clean, group-aware subset for function-name prediction.
3. Exports that subset into SymLM-style self/input.inline + self/input.label
   shards under train/val/test.
4. Compares one or more pretrained checkpoints with the same split and the same
   lightweight multi-label classifier.

The comparison target is the pretrained backbone only:
- baseline MLM checkpoint
- address-aware MLM_ADDR_VAR checkpoint

No additional pretraining is performed.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
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
from funcsim.dataloader import FunctionSimilarityDataset
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
class Sample:
    func_id: str
    group_key: str
    label_text: str
    opt_level: str


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
    """
    Lightweight, reproducible normalization for funcname prediction.

    The goal here is not to perfectly replicate SymLM's tokenizer, but to build
    a clean comparison set where both checkpoints see the same labels.
    """
    if not name:
        return None
    if is_placeholder_name(name):
        return None
    if name.startswith("_Z") or name.startswith("__Z") or name.startswith("?"):
        return None

    name = name.split("::")[-1]
    name = name.strip("_")
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


def load_or_build_subset(
    function_blocks_path: str,
    funcsim_pairs_path: str,
    output_dir: Path,
    max_groups: int,
    seed: int,
) -> dict:
    manifest_path = output_dir / "manifest.json"
    subset_blocks_path = output_dir / "subset_function_blocks.json"
    subset_pairs_path = output_dir / "subset_funcsim_pairs.json"
    if manifest_path.exists() and subset_blocks_path.exists() and subset_pairs_path.exists():
        with manifest_path.open() as f:
            return json.load(f)

    print(f"[funcname] Loading function blocks from {function_blocks_path}")
    with open(function_blocks_path, "r", encoding="utf-8") as f:
        function_blocks = json.load(f)
    print(f"[funcname] Loaded {len(function_blocks):,} function blocks")

    print(f"[funcname] Loading funcsim pairs from {funcsim_pairs_path}")
    with open(funcsim_pairs_path, "r", encoding="utf-8") as f:
        funcsim_pairs = json.load(f)
    print(f"[funcname] Loaded {len(funcsim_pairs):,} func metadata entries")

    groups = defaultdict(list)
    group_labels = {}
    for func_id, meta in funcsim_pairs.items():
        if func_id not in function_blocks:
            continue
        label_text = normalize_function_name(meta.get("function_name", ""))
        if label_text is None:
            continue
        group_key = f"{meta.get('project')}::{meta.get('binary')}::{meta.get('index')}"
        sample = Sample(
            func_id=str(func_id),
            group_key=group_key,
            label_text=label_text,
            opt_level=str(meta.get("opt_level", "")),
        )
        groups[group_key].append(sample)
        group_labels.setdefault(group_key, set()).add(label_text)

    eligible = []
    for group_key, samples in groups.items():
        if len(samples) != 4:
            continue
        if len(group_labels[group_key]) != 1:
            continue
        eligible.append((group_key, sorted(samples, key=lambda s: s.opt_level)))

    rng = random.Random(seed)
    rng.shuffle(eligible)
    if max_groups > 0:
        eligible = eligible[:max_groups]

    n_total = len(eligible)
    n_train = max(1, int(n_total * 0.8))
    n_val = max(1, int(n_total * 0.1))
    n_test = max(1, n_total - n_train - n_val)
    if n_train + n_val + n_test > n_total:
        n_test = n_total - n_train - n_val

    split_groups = {
        "train": eligible[:n_train],
        "val": eligible[n_train:n_train + n_val],
        "test": eligible[n_train + n_val:n_train + n_val + n_test],
    }

    selected_ids = []
    subset_pairs = {}
    subset_blocks = {}
    split_counts = {}

    for split_name, items in split_groups.items():
        split_dir = output_dir / "symlm" / split_name / "self"
        split_dir.mkdir(parents=True, exist_ok=True)
        inline_fp = (split_dir / "input.inline").open("w", encoding="utf-8")
        label_fp = (split_dir / "input.label").open("w", encoding="utf-8")

        sample_count = 0
        for group_key, samples in items:
            for sample in samples:
                meta = dict(funcsim_pairs[sample.func_id])
                meta["normalized_label"] = sample.label_text
                meta["split"] = split_name
                meta["group_key"] = group_key
                subset_pairs[sample.func_id] = meta
                subset_blocks[sample.func_id] = function_blocks[sample.func_id]
                selected_ids.append(sample.func_id)
                inline_fp.write("\t".join(function_blocks[sample.func_id]) + "\n")
                label_fp.write(sample.label_text + "\n")
                sample_count += 1

        inline_fp.close()
        label_fp.close()
        split_counts[split_name] = sample_count

    output_dir.mkdir(parents=True, exist_ok=True)
    with subset_blocks_path.open("w", encoding="utf-8") as f:
        json.dump(subset_blocks, f)
    with subset_pairs_path.open("w", encoding="utf-8") as f:
        json.dump(subset_pairs, f)

    manifest = {
        "function_blocks_path": function_blocks_path,
        "funcsim_pairs_path": funcsim_pairs_path,
        "num_selected_groups": n_total,
        "num_selected_functions": len(selected_ids),
        "split_function_counts": split_counts,
        "subset_function_blocks": str(subset_blocks_path),
        "subset_funcsim_pairs": str(subset_pairs_path),
    }
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest


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


@torch.no_grad()
def compute_embeddings(
    encoder: nn.Module,
    dataset: FunctionSimilarityDataset,
    func_ids: list[str],
    device: torch.device,
    batch_size: int,
    cache_path: Path | None = None,
) -> dict[str, torch.Tensor]:
    if cache_path is not None and cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu")
        cached = payload.get("embeddings", {})
        missing = [fid for fid in func_ids if fid not in cached]
        if not missing:
            return cached

    embeddings = {}
    for start in tqdm(range(0, len(func_ids), batch_size), desc="Encoding", leave=False):
        batch_ids = func_ids[start:start + batch_size]
        encoded = [dataset.encode_function(fid) for fid in batch_ids]
        ids_t = torch.stack([e["input_ids"] for e in encoded]).to(device)
        seg_t = torch.stack([e["segment"] for e in encoded]).to(device)
        bp_t = torch.stack([e["binary_pos"] for e in encoded]).to(device)
        fp_t = torch.stack([e["function_pos"] for e in encoded]).to(device)
        bb_t = torch.stack([e["bb_pos"] for e in encoded]).to(device)
        vo_t = torch.stack([e["var_offsets"] for e in encoded]).to(device)
        ni = [e["num_instructions"] for e in encoded]
        bd = [e["instruction_boundaries"] for e in encoded]
        out = encoder(ids_t, seg_t, bp_t, fp_t, bb_t, vo_t,
                      num_instructions=ni, instruction_boundaries=bd)
        for fid, emb in zip(batch_ids, out.cpu()):
            embeddings[fid] = emb
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"embeddings": embeddings}, cache_path)
    return embeddings


def build_label_vocab(train_samples: list[dict], min_freq: int, max_tokens: int) -> dict[str, int]:
    counter = Counter()
    for sample in train_samples:
        counter.update(sample["label_text"].split())
    items = [(tok, freq) for tok, freq in counter.items() if freq >= min_freq]
    items.sort(key=lambda x: (-x[1], x[0]))
    if max_tokens > 0:
        items = items[:max_tokens]
    return {tok: idx for idx, (tok, _) in enumerate(items)}


def filter_samples_with_vocab(samples: list[dict], label_to_idx: dict[str, int]) -> list[dict]:
    out = []
    for sample in samples:
        active = [tok for tok in sample["label_text"].split() if tok in label_to_idx]
        if active:
            sample = dict(sample)
            sample["active_tokens"] = active
            out.append(sample)
    return out


def build_split_samples(subset_pairs_path: Path) -> dict[str, list[dict]]:
    with subset_pairs_path.open() as f:
        pairs = json.load(f)
    splits = {"train": [], "val": [], "test": []}
    for func_id, meta in pairs.items():
        split = meta["split"]
        splits[split].append({
            "func_id": func_id,
            "label_text": meta["normalized_label"],
            "group_key": meta["group_key"],
            "opt_level": meta.get("opt_level", ""),
        })
    return splits


def samples_to_tensors(samples: list[dict], embeddings: dict[str, torch.Tensor], label_to_idx: dict[str, int]):
    xs = []
    ys = []
    for sample in samples:
        xs.append(embeddings[sample["func_id"]])
        y = torch.zeros(len(label_to_idx), dtype=torch.float)
        for tok in sample["active_tokens"]:
            y[label_to_idx[tok]] = 1.0
        ys.append(y)
    return torch.stack(xs), torch.stack(ys)


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

    model = nn.Sequential(
        nn.Linear(train_x.size(1), train_x.size(1)),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(train_x.size(1), train_y.size(1)),
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


def main():
    parser = argparse.ArgumentParser(description="Compare pretrained PalmTree checkpoints on funcname prediction.")
    parser.add_argument("--function-blocks", default="/path/to/numAnchor_artifact/funcsim_match/function_blocks.json")
    parser.add_argument("--funcsim-pairs", default="/path/to/numAnchor_artifact/funcsim_match/funcsim_pairs.json")
    parser.add_argument("--vocab", default="/path/to/numAnchor_artifact/aaedataset/blockonly/data/vocab.pkl")
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="Model spec in the form name=/path/to/checkpoint.pt . Repeat for multiple models.",
    )
    parser.add_argument("--output-dir", default="/path/to/numAnchor_artifact/aaedataset/funcname_symlm_validation")
    parser.add_argument("--max-groups", type=int, default=800)
    parser.add_argument("--seq-len", type=int, default=420)
    parser.add_argument("--max-instructions", type=int, default=50)
    parser.add_argument("--min-label-freq", type=int, default=10)
    parser.add_argument("--max-label-tokens", type=int, default=1024)
    parser.add_argument("--encode-batch-size", type=int, default=32)
    parser.add_argument("--classifier-batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    output_dir = Path(args.output_dir)
    dataset_dir = output_dir / f"funcsim_match_clean_g{args.max_groups}_seed{args.seed}"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_or_build_subset(
        function_blocks_path=args.function_blocks,
        funcsim_pairs_path=args.funcsim_pairs,
        output_dir=dataset_dir,
        max_groups=args.max_groups,
        seed=args.seed,
    )
    print(f"[funcname] Dataset manifest: {dataset_dir / 'manifest.json'}")

    split_samples = build_split_samples(dataset_dir / "subset_funcsim_pairs.json")
    label_to_idx = build_label_vocab(split_samples["train"], args.min_label_freq, args.max_label_tokens)
    split_samples = {k: filter_samples_with_vocab(v, label_to_idx) for k, v in split_samples.items()}

    summary = {
        "dataset_manifest": str(dataset_dir / "manifest.json"),
        "num_label_tokens": len(label_to_idx),
        "split_sizes": {k: len(v) for k, v in split_samples.items()},
        "models": {},
    }

    func_ids = []
    for split_name in ("train", "val", "test"):
        func_ids.extend([s["func_id"] for s in split_samples[split_name]])
    func_ids = sorted(set(func_ids), key=lambda x: int(x))

    vocab = WordVocab.load_vocab(args.vocab)
    subset_dataset = FunctionSimilarityDataset(
        str(dataset_dir / "subset_function_blocks.json"),
        str(dataset_dir / "subset_funcsim_pairs.json"),
        vocab=vocab,
        seq_len=args.seq_len,
        max_instructions=args.max_instructions,
        negative_samples=0,
    )

    for spec in args.model:
        name, checkpoint = spec.split("=", 1)
        print(f"[funcname] Evaluating model {name}: {checkpoint}")
        encoder, use_addr_var = load_encoder(vocab, checkpoint, device)
        subset_dataset.use_address_var = use_addr_var
        embeddings = compute_embeddings(
            encoder=encoder,
            dataset=subset_dataset,
            func_ids=func_ids,
            device=device,
            batch_size=args.encode_batch_size,
            cache_path=dataset_dir / "embedding_cache" / f"{name}.pt",
        )

        train_x, train_y = samples_to_tensors(split_samples["train"], embeddings, label_to_idx)
        val_x, val_y = samples_to_tensors(split_samples["val"], embeddings, label_to_idx)
        test_x, test_y = samples_to_tensors(split_samples["test"], embeddings, label_to_idx)
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
        summary["models"][name] = {
            "checkpoint": checkpoint,
            "use_addr_var": use_addr_var,
            "best_threshold": best_threshold,
            "history_tail": history[-5:],
            "test_threshold_metrics": test_threshold_metrics,
            "test_topk_metrics": test_topk_metrics,
        }
        print(
            f"[funcname] {name} | "
            f"topk_f1={test_topk_metrics['f1']:.4f} "
            f"thr_f1={test_threshold_metrics['f1']:.4f} "
            f"topk_exact={test_topk_metrics['exact_match']:.4f}"
        )

    results_path = dataset_dir / "compare_results.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[done] wrote comparison results to {results_path}")


if __name__ == "__main__":
    main()
