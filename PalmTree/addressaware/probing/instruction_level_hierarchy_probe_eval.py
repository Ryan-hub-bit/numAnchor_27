#!/usr/bin/env python3
"""
Instruction-level hierarchy probes on frozen PalmTree checkpoints.

This variant focuses only on single-instruction position decoding tasks and
lets us vary the frozen feature view and the lightweight probe head:

- instruction_in_binary
- instruction_in_function
- instruction_in_block

Compared with the older hierarchy probe, this script:
- keeps every task strictly instruction-level
- supports opcode / pooled / address-token feature views
- supports a linear or shallow MLP probe
- standardizes frozen features using train-set statistics
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from hierarchy_probe_eval import (
    LabeledUnit,
    balanced_accuracy_multiclass,
    build_position_decoding_dataset,
    macro_f1,
)
from probe_eval import (
    EncodedUnit,
    FrozenBackboneEncoder,
    ProbeUnit,
    choose_function_subset,
    infer_binary_metadata_from_records_dir,
    load_function_blocks_with_metadata,
    load_function_metadata,
    reconstruct_blocks,
    split_function_ids,
)
from vocab import WordVocab


class MultiClassMLPProbe(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, num_classes: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiClassLinearProbe(nn.Module):
    def __init__(self, dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def build_feature(encoded: EncodedUnit, repr_mode: str) -> torch.Tensor:
    zero_addr = torch.zeros_like(encoded.opcode)
    address = encoded.address_token if encoded.address_token is not None else zero_addr

    if repr_mode == "pooled":
        return encoded.pooled
    if repr_mode == "opcode":
        return encoded.opcode
    if repr_mode == "address":
        return address
    if repr_mode == "pooled_opcode":
        return torch.cat([encoded.pooled, encoded.opcode], dim=0)
    if repr_mode == "opcode_address":
        return torch.cat([encoded.opcode, address], dim=0)
    if repr_mode == "pooled_opcode_address":
        return torch.cat([encoded.pooled, encoded.opcode, address], dim=0)
    raise ValueError(f"Unsupported repr_mode: {repr_mode}")


def build_feature_tensor(
    embeddings: Dict[str, EncodedUnit],
    items: Sequence[LabeledUnit],
    repr_mode: str,
) -> torch.Tensor:
    return torch.stack([build_feature(embeddings[item.unit_id], repr_mode) for item in items], dim=0)


def standardize_train_test(
    x_train: torch.Tensor,
    x_test: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    mean = x_train.mean(dim=0, keepdim=True)
    std = x_train.std(dim=0, keepdim=True, unbiased=False)
    std = torch.where(std < 1e-6, torch.ones_like(std), std)
    return (x_train - mean) / std, (x_test - mean) / std


def train_probe(
    embeddings: Dict[str, EncodedUnit],
    train_units: Sequence[LabeledUnit],
    test_units: Sequence[LabeledUnit],
    repr_mode: str,
    probe_head: str,
    probe_hidden_dim: int,
    probe_dropout: float,
    num_classes: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
) -> Dict[str, float]:
    x_train = build_feature_tensor(embeddings, train_units, repr_mode).to(device)
    y_train = torch.tensor([item.label for item in train_units], dtype=torch.long, device=device)
    x_test = build_feature_tensor(embeddings, test_units, repr_mode).to(device)
    y_test = torch.tensor([item.label for item in test_units], dtype=torch.long, device=device)

    x_train, x_test = standardize_train_test(x_train, x_test)

    if probe_head == "linear":
        model: nn.Module = MultiClassLinearProbe(x_train.size(1), num_classes).to(device)
    elif probe_head == "mlp":
        model = MultiClassMLPProbe(
            dim=x_train.size(1),
            hidden_dim=probe_hidden_dim,
            num_classes=num_classes,
            dropout=probe_dropout,
        ).to(device)
    else:
        raise ValueError(f"Unsupported probe_head: {probe_head}")

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        logits = model(x_train)
        loss = criterion(logits, y_train)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        logits = model(x_test)
        pred = torch.argmax(logits, dim=-1).cpu().tolist()
        labels = y_test.cpu().tolist()

    chance = 1.0 / float(num_classes) if num_classes > 0 else 0.0
    return {
        "accuracy": float(np.mean(np.asarray(pred) == np.asarray(labels))),
        "balanced_accuracy": balanced_accuracy_multiclass(pred, labels, num_classes),
        "macro_f1": macro_f1(pred, labels, num_classes),
        "num_classes": int(num_classes),
        "num_train_units": int(len(train_units)),
        "num_test_units": int(len(test_units)),
        "chance_accuracy": chance,
    }


def run_task_for_model(
    model_name: str,
    encoder: FrozenBackboneEncoder,
    train_items: Sequence[LabeledUnit],
    test_items: Sequence[LabeledUnit],
    units: Dict[str, ProbeUnit],
    repr_mode: str,
    probe_head: str,
    probe_hidden_dim: int,
    probe_dropout: float,
    num_classes: int,
    batch_size: int,
    probe_epochs: int,
    probe_lr: float,
    probe_weight_decay: float,
    device: torch.device,
) -> Dict[str, object]:
    embeddings = encoder.encode_units(units, batch_size=batch_size)
    metrics = train_probe(
        embeddings=embeddings,
        train_units=train_items,
        test_units=test_items,
        repr_mode=repr_mode,
        probe_head=probe_head,
        probe_hidden_dim=probe_hidden_dim,
        probe_dropout=probe_dropout,
        num_classes=num_classes,
        epochs=probe_epochs,
        lr=probe_lr,
        weight_decay=probe_weight_decay,
        device=device,
    )
    return {
        "model_name": model_name,
        "use_addr": encoder.use_addr,
        "repr_mode": repr_mode,
        "probe_head": probe_head,
        "probe_metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Instruction-level hierarchy probes")
    parser.add_argument("--function_blocks", required=True)
    parser.add_argument("--function_metadata", default=None)
    parser.add_argument("--function_records_dir", default=None)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--baseline_checkpoint", required=True)
    parser.add_argument("--aa_checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=[
            "instruction_in_binary",
            "instruction_in_function",
            "instruction_in_block",
        ],
    )
    parser.add_argument("--repr_mode", default="pooled_opcode_address")
    parser.add_argument("--probe_head", choices=["linear", "mlp"], default="mlp")
    parser.add_argument("--probe_hidden_dim", type=int, default=256)
    parser.add_argument("--probe_dropout", type=float, default=0.1)
    parser.add_argument("--hidden", type=int, default=768)
    parser.add_argument("--n_layers", type=int, default=12)
    parser.add_argument("--attn_heads", type=int, default=12)
    parser.add_argument("--seq_len", type=int, default=32)
    parser.add_argument("--max_functions", type=int, default=None)
    parser.add_argument("--test_fraction", type=float, default=0.3)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--probe_epochs", type=int, default=200)
    parser.add_argument("--probe_lr", type=float, default=1e-3)
    parser.add_argument("--probe_weight_decay", type=float, default=1e-4)
    parser.add_argument("--position_buckets", type=int, default=4)
    parser.add_argument("--max_train_units", type=int, default=6000)
    parser.add_argument("--max_test_units", type=int, default=3000)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    device = torch.device(args.device)

    print("[InstrHierarchyProbe] Loading data ...")
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

    instructions_by_func, _ = reconstruct_blocks(function_blocks)
    train_func_ids, test_func_ids = split_function_ids(selected_func_ids, args.test_fraction, args.split_seed)
    vocab = WordVocab.load_vocab(args.vocab)

    model_specs = [
        ("baseline", args.baseline_checkpoint),
        ("address_aware", args.aa_checkpoint),
    ]
    coord_map = {
        "instruction_in_binary": "binary_pos",
        "instruction_in_function": "block_pos",
        "instruction_in_block": "inst_pos",
    }

    report: Dict[str, object] = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "config": vars(args),
        "data_summary": {
            "num_functions": len(selected_func_ids),
            "num_train_functions": len(train_func_ids),
            "num_test_functions": len(test_func_ids),
            "num_functions_with_binary_metadata": len(binary_by_func),
            "binary_metadata_coverage": (
                float(len(binary_by_func)) / float(len(selected_func_ids))
                if selected_func_ids else 0.0
            ),
        },
        "tasks": {},
    }

    for task in args.tasks:
        coord_name = coord_map[task]
        print(f"[InstrHierarchyProbe] Building task: {task}")
        train_items, train_units = build_position_decoding_dataset(
            instructions_by_func,
            train_func_ids,
            coord_name=coord_name,
            num_buckets=args.position_buckets,
            max_units=args.max_train_units,
            seed=args.split_seed + 100 + len(report["tasks"]),
        )
        test_items, test_units = build_position_decoding_dataset(
            instructions_by_func,
            test_func_ids,
            coord_name=coord_name,
            num_buckets=args.position_buckets,
            max_units=args.max_test_units,
            seed=args.split_seed + 200 + len(report["tasks"]),
        )
        units: Dict[str, ProbeUnit] = {}
        units.update(train_units)
        units.update(test_units)
        task_report = {
            "num_units": len(units),
            "num_train_units": len(train_items),
            "num_test_units": len(test_items),
            "num_buckets": args.position_buckets,
            "models": {},
        }
        for model_name, checkpoint_path in model_specs:
            print(f"[InstrHierarchyProbe] Encoding/evaluating {task} with {model_name}: {checkpoint_path}")
            encoder = FrozenBackboneEncoder(
                checkpoint_path=checkpoint_path,
                vocab=vocab,
                hidden=args.hidden,
                n_layers=args.n_layers,
                attn_heads=args.attn_heads,
                seq_len=args.seq_len,
                max_instructions=1,
                device=device,
            )
            task_report["models"][model_name] = run_task_for_model(
                model_name=model_name,
                encoder=encoder,
                train_items=train_items,
                test_items=test_items,
                units=units,
                repr_mode=args.repr_mode,
                probe_head=args.probe_head,
                probe_hidden_dim=args.probe_hidden_dim,
                probe_dropout=args.probe_dropout,
                num_classes=args.position_buckets,
                batch_size=args.batch_size,
                probe_epochs=args.probe_epochs,
                probe_lr=args.probe_lr,
                probe_weight_decay=args.probe_weight_decay,
                device=device,
            )
        report["tasks"][task] = task_report

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[InstrHierarchyProbe] Wrote report to {args.output}")


if __name__ == "__main__":
    main()
