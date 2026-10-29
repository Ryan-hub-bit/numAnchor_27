#!/usr/bin/env python3
"""
Hierarchy-focused frozen-encoder probing for PalmTree checkpoints.

Tasks:
  - decode_binary_pos: bucket-decode function position in binary from an instruction embedding
  - decode_block_pos:  bucket-decode block position in function from an instruction embedding
  - decode_inst_pos:   bucket-decode instruction position in block from an instruction embedding
  - same_bb_retrieval: retrieve an instruction from the same basic block
  - parent_block:      retrieve the parent basic block for an instruction
  - parent_function:   retrieve the parent function for a basic block
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from probe_eval import (
    BasicBlockRecord,
    FrozenBackboneEncoder,
    InstructionRecord,
    ProbeUnit,
    RetrievalQuery,
    choose_function_subset,
    compute_retrieval_metrics,
    infer_binary_metadata_from_records_dir,
    load_function_blocks_with_metadata,
    load_function_metadata,
    reconstruct_blocks,
    split_function_ids,
)
from vocab import WordVocab


@dataclass
class LabeledUnit:
    unit_id: str
    label: int


@dataclass
class ParentFunctionQuery:
    query_block_id: str
    candidate_func_ids: List[str]
    positive_func_id: str
    group_id: str
    metadata: Dict[str, object]


class MultiClassLinearProbe(nn.Module):
    def __init__(self, dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def bucketize_position(value: float, num_buckets: int) -> int:
    if math.isnan(value) or value < 0:
        return -1
    if value >= 1.0:
        return num_buckets - 1
    return min(num_buckets - 1, max(0, int(value * num_buckets)))


def macro_f1(pred: Sequence[int], labels: Sequence[int], num_classes: int) -> float:
    pred_arr = np.asarray(pred, dtype=np.int64)
    label_arr = np.asarray(labels, dtype=np.int64)
    scores = []
    for cls in range(num_classes):
        tp = int(((pred_arr == cls) & (label_arr == cls)).sum())
        fp = int(((pred_arr == cls) & (label_arr != cls)).sum())
        fn = int(((pred_arr != cls) & (label_arr == cls)).sum())
        denom = 2 * tp + fp + fn
        scores.append(0.0 if denom == 0 else (2.0 * tp) / denom)
    return float(np.mean(scores)) if scores else 0.0


def balanced_accuracy_multiclass(pred: Sequence[int], labels: Sequence[int], num_classes: int) -> float:
    pred_arr = np.asarray(pred, dtype=np.int64)
    label_arr = np.asarray(labels, dtype=np.int64)
    recalls = []
    for cls in range(num_classes):
        mask = label_arr == cls
        if not mask.any():
            continue
        recalls.append(float((pred_arr[mask] == cls).mean()))
    return float(np.mean(recalls)) if recalls else 0.0


def sample_balanced_labeled_units(
    bucket_to_units: Dict[int, List[LabeledUnit]],
    max_total_units: int,
    seed: int,
) -> List[LabeledUnit]:
    rng = random.Random(seed)
    buckets = [bucket for bucket, units in bucket_to_units.items() if units]
    if not buckets:
        return []
    per_bucket = max(1, max_total_units // len(buckets))
    sampled: List[LabeledUnit] = []
    for bucket in sorted(buckets):
        units = list(bucket_to_units[bucket])
        rng.shuffle(units)
        sampled.extend(units[:per_bucket])
    rng.shuffle(sampled)
    if len(sampled) > max_total_units:
        sampled = sampled[:max_total_units]
    return sampled


def build_position_decoding_dataset(
    instructions_by_func: Dict[str, List[InstructionRecord]],
    func_ids: Sequence[str],
    coord_name: str,
    num_buckets: int,
    max_units: int,
    seed: int,
) -> Tuple[List[LabeledUnit], Dict[str, ProbeUnit]]:
    bucket_to_units: Dict[int, List[LabeledUnit]] = defaultdict(list)
    units: Dict[str, ProbeUnit] = {}

    for func_id in func_ids:
        for inst in instructions_by_func.get(func_id, []):
            if coord_name == "binary_pos":
                value = inst.binary_pos
            elif coord_name == "block_pos":
                value = inst.function_pos
            elif coord_name == "inst_pos":
                value = inst.bb_pos
            else:
                raise ValueError(f"Unsupported coord_name: {coord_name}")
            bucket = bucketize_position(value, num_buckets)
            if bucket < 0:
                continue
            units[inst.unit_id] = ProbeUnit(inst.unit_id, "instruction", [inst.text])
            bucket_to_units[bucket].append(LabeledUnit(inst.unit_id, bucket))

    sampled = sample_balanced_labeled_units(bucket_to_units, max_units, seed)
    sampled_ids = {item.unit_id for item in sampled}
    filtered_units = {uid: unit for uid, unit in units.items() if uid in sampled_ids}
    return sampled, filtered_units


def train_multiclass_probe(
    embeddings: Dict[str, object],
    train_units: Sequence[LabeledUnit],
    test_units: Sequence[LabeledUnit],
    num_classes: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
) -> Dict[str, float]:
    x_train = torch.stack([embeddings[item.unit_id].pooled for item in train_units], dim=0).to(device)
    y_train = torch.tensor([item.label for item in train_units], dtype=torch.long, device=device)
    x_test = torch.stack([embeddings[item.unit_id].pooled for item in test_units], dim=0).to(device)
    y_test = torch.tensor([item.label for item in test_units], dtype=torch.long, device=device)

    model = MultiClassLinearProbe(x_train.size(1), num_classes).to(device)
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


def run_decoding_task_for_model(
    model_name: str,
    encoder: FrozenBackboneEncoder,
    train_items: Sequence[LabeledUnit],
    test_items: Sequence[LabeledUnit],
    units: Dict[str, ProbeUnit],
    num_classes: int,
    batch_size: int,
    probe_epochs: int,
    probe_lr: float,
    probe_weight_decay: float,
    device: torch.device,
) -> Dict[str, object]:
    embeddings = encoder.encode_units(units, batch_size=batch_size)
    metrics = train_multiclass_probe(
        embeddings=embeddings,
        train_units=train_items,
        test_units=test_items,
        num_classes=num_classes,
        epochs=probe_epochs,
        lr=probe_lr,
        weight_decay=probe_weight_decay,
        device=device,
    )
    return {
        "model_name": model_name,
        "use_addr": encoder.use_addr,
        "linear_probe": metrics,
    }


def build_same_bb_retrieval_queries(
    blocks_by_func: Dict[str, List[BasicBlockRecord]],
    func_ids: Sequence[str],
    max_queries: int,
    pool_size: int,
    seed: int,
) -> Tuple[List[RetrievalQuery], Dict[str, ProbeUnit]]:
    rng = random.Random(seed)
    queries: List[RetrievalQuery] = []
    units: Dict[str, ProbeUnit] = {}

    shuffled_funcs = list(func_ids)
    rng.shuffle(shuffled_funcs)
    for func_id in shuffled_funcs:
        blocks = [block for block in blocks_by_func.get(func_id, []) if block.instructions]
        for block in blocks:
            if len(block.instructions) < 2:
                continue
            negatives = [
                inst
                for other in blocks
                if other.unit_id != block.unit_id
                for inst in other.instructions
            ]
            if len(negatives) < max(1, pool_size - 1):
                continue

            query_inst, positive_inst = rng.sample(block.instructions, 2)
            matched_opcode = [cand for cand in negatives if cand.opcode == positive_inst.opcode]
            if len(matched_opcode) >= pool_size - 1:
                negative_sample = rng.sample(matched_opcode, pool_size - 1)
            else:
                negative_sample = rng.sample(negatives, pool_size - 1)

            units[query_inst.unit_id] = ProbeUnit(query_inst.unit_id, "instruction", [query_inst.text])
            units[positive_inst.unit_id] = ProbeUnit(positive_inst.unit_id, "instruction", [positive_inst.text])
            candidate_ids = [positive_inst.unit_id]
            for negative in negative_sample:
                units[negative.unit_id] = ProbeUnit(negative.unit_id, "instruction", [negative.text])
                candidate_ids.append(negative.unit_id)
            rng.shuffle(candidate_ids)
            queries.append(
                RetrievalQuery(
                    query_id=query_inst.unit_id,
                    candidate_ids=candidate_ids,
                    positive_id=positive_inst.unit_id,
                    group_id=f"same_bb_retrieval:{block.unit_id}:{query_inst.unit_id}",
                    metadata={
                        "task": "same_bb_retrieval",
                        "func_id": func_id,
                        "block_id": block.unit_id,
                    },
                )
            )
            if len(queries) >= max_queries:
                return queries[:max_queries], units

    return queries[:max_queries], units


def build_parent_block_queries(
    blocks_by_func: Dict[str, List[BasicBlockRecord]],
    func_ids: Sequence[str],
    max_queries: int,
    pool_size: int,
    seed: int,
) -> Tuple[List[RetrievalQuery], Dict[str, ProbeUnit]]:
    rng = random.Random(seed)
    queries: List[RetrievalQuery] = []
    units: Dict[str, ProbeUnit] = {}

    shuffled_funcs = list(func_ids)
    rng.shuffle(shuffled_funcs)
    for func_id in shuffled_funcs:
        blocks = [block for block in blocks_by_func.get(func_id, []) if block.instructions]
        if len(blocks) < 2:
            continue

        for block in blocks:
            query_inst = rng.choice(block.instructions)
            other_blocks = [candidate for candidate in blocks if candidate.unit_id != block.unit_id]
            if not other_blocks:
                continue
            if len(other_blocks) >= pool_size - 1:
                negative_blocks = rng.sample(other_blocks, pool_size - 1)
            else:
                negative_blocks = list(other_blocks)

            units[query_inst.unit_id] = ProbeUnit(query_inst.unit_id, "instruction", [query_inst.text])
            units[block.unit_id] = ProbeUnit(block.unit_id, "basic_block", [inst.text for inst in block.instructions])
            candidate_ids = [block.unit_id]
            for negative in negative_blocks:
                units[negative.unit_id] = ProbeUnit(
                    negative.unit_id,
                    "basic_block",
                    [inst.text for inst in negative.instructions],
                )
                candidate_ids.append(negative.unit_id)
            rng.shuffle(candidate_ids)
            queries.append(
                RetrievalQuery(
                    query_id=query_inst.unit_id,
                    candidate_ids=candidate_ids,
                    positive_id=block.unit_id,
                    group_id=f"parent_block:{func_id}:{query_inst.unit_id}",
                    metadata={
                        "task": "parent_block",
                        "func_id": func_id,
                        "block_id": block.unit_id,
                    },
                )
            )
            if len(queries) >= max_queries:
                return queries[:max_queries], units

    return queries[:max_queries], units


def build_parent_function_queries(
    blocks_by_func: Dict[str, List[BasicBlockRecord]],
    func_ids: Sequence[str],
    binary_by_func: Dict[str, str],
    max_queries: int,
    pool_size: int,
    seed: int,
) -> Tuple[List[ParentFunctionQuery], Dict[str, ProbeUnit], Dict[str, List[str]]]:
    rng = random.Random(seed)
    queries: List[ParentFunctionQuery] = []
    units: Dict[str, ProbeUnit] = {}
    func_to_block_ids: Dict[str, List[str]] = defaultdict(list)

    funcs_by_binary: Dict[str, List[str]] = defaultdict(list)
    for func_id in func_ids:
        if blocks_by_func.get(func_id):
            binary_id = binary_by_func.get(func_id)
            if binary_id:
                funcs_by_binary[binary_id].append(func_id)

    shuffled_funcs = list(func_ids)
    rng.shuffle(shuffled_funcs)
    for func_id in shuffled_funcs:
        blocks = [block for block in blocks_by_func.get(func_id, []) if block.instructions]
        if len(blocks) < 2:
            continue

        for block in blocks:
            units[block.unit_id] = ProbeUnit(block.unit_id, "basic_block", [inst.text for inst in block.instructions])
            func_to_block_ids[func_id].append(block.unit_id)

        binary_id = binary_by_func.get(func_id)
        if binary_id:
            negatives = [candidate for candidate in funcs_by_binary.get(binary_id, []) if candidate != func_id]
        else:
            negatives = [candidate for candidate in func_ids if candidate != func_id and blocks_by_func.get(candidate)]
        negatives = [candidate for candidate in negatives if len(blocks_by_func.get(candidate, [])) >= 1]
        if not negatives:
            continue

        for query_block in blocks:
            if len(negatives) >= pool_size - 1:
                negative_funcs = rng.sample(negatives, pool_size - 1)
            else:
                negative_funcs = list(negatives)
            candidate_func_ids = [func_id] + negative_funcs
            rng.shuffle(candidate_func_ids)

            for candidate_func in candidate_func_ids:
                for candidate_block in blocks_by_func.get(candidate_func, []):
                    if candidate_block.instructions:
                        units[candidate_block.unit_id] = ProbeUnit(
                            candidate_block.unit_id,
                            "basic_block",
                            [inst.text for inst in candidate_block.instructions],
                        )
                        func_to_block_ids[candidate_func].append(candidate_block.unit_id)

            queries.append(
                ParentFunctionQuery(
                    query_block_id=query_block.unit_id,
                    candidate_func_ids=candidate_func_ids,
                    positive_func_id=func_id,
                    group_id=f"parent_function:{func_id}:{query_block.unit_id}",
                    metadata={
                        "task": "parent_function",
                        "func_id": func_id,
                        "query_block_id": query_block.unit_id,
                        "binary_id": binary_id,
                    },
                )
            )
            if len(queries) >= max_queries:
                deduped = {func: sorted(set(block_ids)) for func, block_ids in func_to_block_ids.items()}
                return queries[:max_queries], units, deduped

    deduped = {func: sorted(set(block_ids)) for func, block_ids in func_to_block_ids.items()}
    return queries[:max_queries], units, deduped


def compute_parent_function_metrics(
    queries: Sequence[ParentFunctionQuery],
    embeddings: Dict[str, object],
    func_to_block_ids: Dict[str, List[str]],
) -> Dict[str, float]:
    if not queries:
        return {}

    def centroid_for_function(func_id: str, exclude_block_id: Optional[str] = None) -> Optional[torch.Tensor]:
        block_ids = [block_id for block_id in func_to_block_ids.get(func_id, []) if block_id in embeddings]
        if exclude_block_id is not None:
            filtered = [block_id for block_id in block_ids if block_id != exclude_block_id]
            if filtered:
                block_ids = filtered
        if not block_ids:
            return None
        vecs = [embeddings[block_id].pooled for block_id in block_ids]
        return torch.stack(vecs, dim=0).mean(dim=0)

    top1_hits = 0
    recall5_hits = 0
    mrr_total = 0.0
    kept = 0

    for query in queries:
        if query.query_block_id not in embeddings:
            continue
        query_vec = embeddings[query.query_block_id].pooled
        scored = []
        for func_id in query.candidate_func_ids:
            centroid = centroid_for_function(
                func_id,
                exclude_block_id=query.query_block_id if func_id == query.positive_func_id else None,
            )
            if centroid is None:
                continue
            score = float(torch.nn.functional.cosine_similarity(query_vec.unsqueeze(0), centroid.unsqueeze(0)).item())
            scored.append((func_id, score))

        if not scored:
            continue
        scored.sort(key=lambda item: item[1], reverse=True)
        ranked_ids = [func_id for func_id, _ in scored]
        if query.positive_func_id not in ranked_ids:
            continue
        rank = ranked_ids.index(query.positive_func_id) + 1
        kept += 1
        top1_hits += int(rank == 1)
        recall5_hits += int(rank <= 5)
        mrr_total += 1.0 / rank

    if kept == 0:
        return {}
    return {
        "top1_accuracy": top1_hits / kept,
        "recall_at_5": recall5_hits / kept,
        "mrr": mrr_total / kept,
        "num_queries": float(kept),
    }


def run_parent_function_task_for_model(
    model_name: str,
    encoder: FrozenBackboneEncoder,
    queries: Sequence[ParentFunctionQuery],
    units: Dict[str, ProbeUnit],
    func_to_block_ids: Dict[str, List[str]],
    batch_size: int,
) -> Dict[str, object]:
    embeddings = encoder.encode_units(units, batch_size=batch_size)
    return {
        "model_name": model_name,
        "use_addr": encoder.use_addr,
        "retrieval": compute_parent_function_metrics(queries, embeddings, func_to_block_ids),
    }


def main():
    parser = argparse.ArgumentParser(description="Hierarchy-focused frozen-encoder probing")
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
            "decode_binary_pos",
            "decode_block_pos",
            "decode_inst_pos",
            "same_bb_retrieval",
            "parent_block",
            "parent_function",
        ],
    )
    parser.add_argument("--hidden", type=int, default=768)
    parser.add_argument("--n_layers", type=int, default=12)
    parser.add_argument("--attn_heads", type=int, default=12)
    parser.add_argument("--seq_len", type=int, default=20)
    parser.add_argument("--max_block_instructions", type=int, default=8)
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
    parser.add_argument("--max_train_queries", type=int, default=500)
    parser.add_argument("--max_test_queries", type=int, default=200)
    parser.add_argument("--retrieval_pool_size", type=int, default=32)
    parser.add_argument("--parent_block_pool_size", type=int, default=8)
    parser.add_argument("--parent_function_pool_size", type=int, default=8)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    device = torch.device(args.device)

    print("[HierarchyProbe] Loading data ...")
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

    instructions_by_func, blocks_by_func = reconstruct_blocks(function_blocks)
    train_func_ids, test_func_ids = split_function_ids(selected_func_ids, args.test_fraction, args.split_seed)
    vocab = WordVocab.load_vocab(args.vocab)

    model_specs = [
        ("baseline", args.baseline_checkpoint),
        ("address_aware", args.aa_checkpoint),
    ]

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

    decoding_coord_map = {
        "decode_binary_pos": "binary_pos",
        "decode_block_pos": "block_pos",
        "decode_inst_pos": "inst_pos",
    }

    for task in args.tasks:
        print(f"[HierarchyProbe] Building task: {task}")
        if task in decoding_coord_map:
            train_items, train_units = build_position_decoding_dataset(
                instructions_by_func,
                train_func_ids,
                coord_name=decoding_coord_map[task],
                num_buckets=args.position_buckets,
                max_units=args.max_train_units,
                seed=args.split_seed + 100 + len(report["tasks"]),
            )
            test_items, test_units = build_position_decoding_dataset(
                instructions_by_func,
                test_func_ids,
                coord_name=decoding_coord_map[task],
                num_buckets=args.position_buckets,
                max_units=args.max_test_units,
                seed=args.split_seed + 200 + len(report["tasks"]),
            )
            units = {}
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
                print(f"[HierarchyProbe] Encoding/evaluating {task} with {model_name}: {checkpoint_path}")
                encoder = FrozenBackboneEncoder(
                    checkpoint_path=checkpoint_path,
                    vocab=vocab,
                    hidden=args.hidden,
                    n_layers=args.n_layers,
                    attn_heads=args.attn_heads,
                    seq_len=args.seq_len,
                    max_instructions=args.max_block_instructions,
                    device=device,
                )
                task_report["models"][model_name] = run_decoding_task_for_model(
                    model_name=model_name,
                    encoder=encoder,
                    train_items=train_items,
                    test_items=test_items,
                    units=units,
                    num_classes=args.position_buckets,
                    batch_size=args.batch_size,
                    probe_epochs=args.probe_epochs,
                    probe_lr=args.probe_lr,
                    probe_weight_decay=args.probe_weight_decay,
                    device=device,
                )
            report["tasks"][task] = task_report
            continue

        if task == "same_bb_retrieval":
            train_queries, train_units = build_same_bb_retrieval_queries(
                blocks_by_func, train_func_ids, args.max_train_queries, args.retrieval_pool_size, args.split_seed + 301
            )
            test_queries, test_units = build_same_bb_retrieval_queries(
                blocks_by_func, test_func_ids, args.max_test_queries, args.retrieval_pool_size, args.split_seed + 302
            )
            units = {}
            units.update(train_units)
            units.update(test_units)
            task_report = {
                "num_units": len(units),
                "num_train_queries": len(train_queries),
                "num_test_queries": len(test_queries),
                "pool_size": args.retrieval_pool_size,
                "models": {},
            }
            for model_name, checkpoint_path in model_specs:
                print(f"[HierarchyProbe] Encoding/evaluating {task} with {model_name}: {checkpoint_path}")
                encoder = FrozenBackboneEncoder(
                    checkpoint_path=checkpoint_path,
                    vocab=vocab,
                    hidden=args.hidden,
                    n_layers=args.n_layers,
                    attn_heads=args.attn_heads,
                    seq_len=args.seq_len,
                    max_instructions=args.max_block_instructions,
                    device=device,
                )
                task_report["models"][model_name] = {
                    "model_name": model_name,
                    "use_addr": encoder.use_addr,
                    "retrieval": compute_retrieval_metrics(test_queries, encoder.encode_units(units, batch_size=args.batch_size)),
                }
            report["tasks"][task] = task_report
            continue

        if task == "parent_block":
            train_queries, train_units = build_parent_block_queries(
                blocks_by_func, train_func_ids, args.max_train_queries, args.parent_block_pool_size, args.split_seed + 401
            )
            test_queries, test_units = build_parent_block_queries(
                blocks_by_func, test_func_ids, args.max_test_queries, args.parent_block_pool_size, args.split_seed + 402
            )
            units = {}
            units.update(train_units)
            units.update(test_units)
            task_report = {
                "num_units": len(units),
                "num_train_queries": len(train_queries),
                "num_test_queries": len(test_queries),
                "pool_size": args.parent_block_pool_size,
                "models": {},
            }
            for model_name, checkpoint_path in model_specs:
                print(f"[HierarchyProbe] Encoding/evaluating {task} with {model_name}: {checkpoint_path}")
                encoder = FrozenBackboneEncoder(
                    checkpoint_path=checkpoint_path,
                    vocab=vocab,
                    hidden=args.hidden,
                    n_layers=args.n_layers,
                    attn_heads=args.attn_heads,
                    seq_len=args.seq_len,
                    max_instructions=args.max_block_instructions,
                    device=device,
                )
                task_report["models"][model_name] = {
                    "model_name": model_name,
                    "use_addr": encoder.use_addr,
                    "retrieval": compute_retrieval_metrics(test_queries, encoder.encode_units(units, batch_size=args.batch_size)),
                }
            report["tasks"][task] = task_report
            continue

        if task == "parent_function":
            train_queries, train_units, train_func_to_block_ids = build_parent_function_queries(
                blocks_by_func, train_func_ids, binary_by_func, args.max_train_queries, args.parent_function_pool_size, args.split_seed + 501
            )
            test_queries, test_units, test_func_to_block_ids = build_parent_function_queries(
                blocks_by_func, test_func_ids, binary_by_func, args.max_test_queries, args.parent_function_pool_size, args.split_seed + 502
            )
            units = {}
            units.update(train_units)
            units.update(test_units)
            func_to_block_ids = dict(train_func_to_block_ids)
            for func_id, block_ids in test_func_to_block_ids.items():
                func_to_block_ids.setdefault(func_id, [])
                func_to_block_ids[func_id].extend(block_ids)
            func_to_block_ids = {func_id: sorted(set(block_ids)) for func_id, block_ids in func_to_block_ids.items()}

            task_report = {
                "num_units": len(units),
                "num_train_queries": len(train_queries),
                "num_test_queries": len(test_queries),
                "pool_size": args.parent_function_pool_size,
                "requires_binary_metadata": True,
                "models": {},
            }
            for model_name, checkpoint_path in model_specs:
                print(f"[HierarchyProbe] Encoding/evaluating {task} with {model_name}: {checkpoint_path}")
                encoder = FrozenBackboneEncoder(
                    checkpoint_path=checkpoint_path,
                    vocab=vocab,
                    hidden=args.hidden,
                    n_layers=args.n_layers,
                    attn_heads=args.attn_heads,
                    seq_len=args.seq_len,
                    max_instructions=args.max_block_instructions,
                    device=device,
                )
                task_report["models"][model_name] = run_parent_function_task_for_model(
                    model_name=model_name,
                    encoder=encoder,
                    queries=test_queries,
                    units=units,
                    func_to_block_ids=func_to_block_ids,
                    batch_size=args.batch_size,
                )
            report["tasks"][task] = task_report
            continue

        raise ValueError(f"Unsupported task: {task}")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[HierarchyProbe] Wrote report to {args.output}")


if __name__ == "__main__":
    main()
