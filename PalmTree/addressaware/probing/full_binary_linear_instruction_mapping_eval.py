#!/usr/bin/env python3
"""
Full-binary linear source->target instruction mapping for jump and call.

For each task we freeze the backbone, encode every instruction with the pooled
instruction representation, then train a single linear map:

    target_hat = W(source_pooled) + b

The learned target_hat is used for retrieval against task-specific candidate
pools:

- jump_target_mapping:
  query      = jump instruction
  positive   = true jump target instruction
  candidates = instructions from the same function

- call_target_mapping:
  query      = call instruction
  positive   = callee entry's first instruction
  candidates = entry-first instructions from all functions in the same binary
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
PROBE_DIR = os.path.dirname(os.path.abspath(__file__))
if PROBE_DIR not in sys.path:
    sys.path.insert(0, PROBE_DIR)

from probe_eval import (  # noqa: E402
    BasicBlockRecord,
    FrozenBackboneEncoder,
    InstructionRecord,
    JUMP_TARGET_RE,
    ProbeUnit,
    WordVocab,
    extract_call_target_address,
    reconstruct_blocks,
)


@dataclass
class MappingExample:
    source_id: str
    target_id: str
    metadata: Dict[str, object]


@dataclass
class MappingQuery:
    query_id: str
    positive_id: str
    candidate_ids: List[str]
    metadata: Dict[str, object]


class LinearMap(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def load_records_paths(records_root: str, max_binary_units: Optional[int], seed: int) -> List[Path]:
    root = Path(records_root)
    if root.is_file():
        return [root]

    paths = sorted(root.glob("*_functions.json"))
    if max_binary_units is None or max_binary_units >= len(paths):
        return paths

    rng = random.Random(seed)
    shuffled = list(paths)
    rng.shuffle(shuffled)
    return sorted(shuffled[:max_binary_units])


def load_full_binary_function_blocks(
    records_root: str,
    max_binary_units: Optional[int],
    seed: int,
) -> Tuple[Dict[str, List[str]], Dict[str, str], List[str]]:
    function_blocks: Dict[str, List[str]] = {}
    binary_by_func: Dict[str, str] = {}
    binary_unit_ids: List[str] = []

    for path in load_records_paths(records_root, max_binary_units, seed):
        binary_unit_id = (
            path.name[: -len("_functions.json")]
            if path.name.endswith("_functions.json")
            else path.stem
        )
        binary_unit_ids.append(binary_unit_id)

        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        if not isinstance(payload, list):
            continue

        for idx, record in enumerate(payload):
            if not isinstance(record, dict):
                continue

            func_key = record.get("func_id", idx)
            funcins = record.get("funcins")
            if isinstance(funcins, str):
                instructions = [inst for inst in funcins.split("\t") if inst]
            else:
                instructions = list(record.get("instructions", []))

            if not instructions:
                continue

            global_func_id = f"{binary_unit_id}::func::{func_key}"
            function_blocks[global_func_id] = instructions
            binary_by_func[global_func_id] = binary_unit_id

    return function_blocks, binary_by_func, sorted(set(binary_unit_ids))


def split_binary_unit_ids(
    binary_unit_ids: Sequence[str],
    test_fraction: float,
    seed: int,
) -> Tuple[List[str], List[str]]:
    ids = list(binary_unit_ids)
    if len(ids) <= 1:
        return [], sorted(ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    test_n = max(1, int(len(ids) * test_fraction))
    test_ids = sorted(ids[:test_n])
    train_ids = sorted(ids[test_n:])
    return train_ids, test_ids


def summarize_pool_sizes(pool_sizes: Sequence[int]) -> Dict[str, float]:
    if not pool_sizes:
        return {}
    arr = np.asarray(pool_sizes, dtype=np.float64)
    return {
        "mean_pool_size": float(arr.mean()),
        "median_pool_size": float(np.median(arr)),
        "min_pool_size": float(arr.min()),
        "max_pool_size": float(arr.max()),
        "random_top1_mean": float(np.mean(1.0 / arr)),
        "random_recall_at_5_mean": float(np.mean(np.minimum(5.0, arr) / arr)),
    }


def build_jump_mapping_data(
    instructions_by_func: Dict[str, List[InstructionRecord]],
    func_ids: Sequence[str],
    max_examples: Optional[int],
    seed: int,
    target_opcode: str,
) -> Tuple[List[MappingExample], List[MappingQuery], Dict[str, ProbeUnit], Dict[str, float]]:
    rng = random.Random(seed)
    examples: List[MappingExample] = []
    queries: List[MappingQuery] = []
    units: Dict[str, ProbeUnit] = {}
    pool_sizes: List[int] = []

    shuffled_funcs = list(func_ids)
    rng.shuffle(shuffled_funcs)
    for func_id in shuffled_funcs:
        instructions = instructions_by_func.get(func_id, [])
        if len(instructions) < 3:
            continue
        addr_map = {inst.address: inst for inst in instructions}
        candidate_ids = [inst.unit_id for inst in instructions]
        for inst in instructions:
            units[inst.unit_id] = ProbeUnit(inst.unit_id, "instruction", [inst.text])

        jumps = list(instructions)
        rng.shuffle(jumps)
        for jump_inst in jumps:
            jm = JUMP_TARGET_RE.match(jump_inst.text)
            if not jm:
                continue
            target_addr = jm.group(2)
            target_inst = addr_map.get(target_addr)
            if target_inst is None:
                continue
            if target_opcode and target_inst.opcode != target_opcode:
                continue

            candidates = [cand_id for cand_id in candidate_ids if cand_id != jump_inst.unit_id]
            if len(candidates) < 2 or target_inst.unit_id not in candidates:
                continue

            examples.append(
                MappingExample(
                    source_id=jump_inst.unit_id,
                    target_id=target_inst.unit_id,
                    metadata={
                        "task": "jump_target_mapping",
                        "func_id": func_id,
                        "jump_opcode": jump_inst.opcode,
                        "target_opcode": target_inst.opcode,
                        "target_addr": target_addr,
                    },
                )
            )
            queries.append(
                MappingQuery(
                    query_id=jump_inst.unit_id,
                    positive_id=target_inst.unit_id,
                    candidate_ids=list(candidates),
                    metadata={
                        "task": "jump_target_mapping",
                        "func_id": func_id,
                        "jump_opcode": jump_inst.opcode,
                        "target_opcode": target_inst.opcode,
                        "target_addr": target_addr,
                        "pool_size": len(candidates),
                    },
                )
            )
            pool_sizes.append(len(candidates))

            if max_examples is not None and len(examples) >= max_examples:
                return examples[:max_examples], queries[:max_examples], units, summarize_pool_sizes(pool_sizes)

    return examples, queries, units, summarize_pool_sizes(pool_sizes)


def build_call_mapping_data(
    instructions_by_func: Dict[str, List[InstructionRecord]],
    blocks_by_func: Dict[str, List[BasicBlockRecord]],
    func_ids: Sequence[str],
    binary_by_func: Dict[str, str],
    max_examples: Optional[int],
    seed: int,
) -> Tuple[List[MappingExample], List[MappingQuery], Dict[str, ProbeUnit], Dict[str, float]]:
    rng = random.Random(seed)
    examples: List[MappingExample] = []
    queries: List[MappingQuery] = []
    units: Dict[str, ProbeUnit] = {}
    pool_sizes: List[int] = []

    entry_inst_by_binary_addr: Dict[Tuple[str, str], InstructionRecord] = {}
    entry_ids_by_binary: Dict[str, List[str]] = defaultdict(list)

    for func_id in func_ids:
        blocks = [block for block in blocks_by_func.get(func_id, []) if block.instructions]
        if not blocks:
            continue
        binary_id = binary_by_func.get(func_id)
        if not binary_id:
            continue
        entry_block = min(blocks, key=lambda block: block.block_index)
        entry_inst = entry_block.instructions[0]
        entry_inst_by_binary_addr[(binary_id, entry_block.entry_address)] = entry_inst
        entry_ids_by_binary[binary_id].append(entry_inst.unit_id)
        units[entry_inst.unit_id] = ProbeUnit(entry_inst.unit_id, "instruction", [entry_inst.text])

    shuffled_funcs = list(func_ids)
    rng.shuffle(shuffled_funcs)
    for func_id in shuffled_funcs:
        binary_id = binary_by_func.get(func_id)
        if not binary_id:
            continue
        candidate_ids = sorted(set(entry_ids_by_binary.get(binary_id, [])))
        if len(candidate_ids) < 2:
            continue

        calls = list(instructions_by_func.get(func_id, []))
        rng.shuffle(calls)
        for call_inst in calls:
            target_addr = extract_call_target_address(call_inst.text)
            if not target_addr:
                continue
            target_inst = entry_inst_by_binary_addr.get((binary_id, target_addr))
            if target_inst is None:
                continue
            if target_inst.unit_id not in candidate_ids:
                continue

            units[call_inst.unit_id] = ProbeUnit(call_inst.unit_id, "instruction", [call_inst.text])
            examples.append(
                MappingExample(
                    source_id=call_inst.unit_id,
                    target_id=target_inst.unit_id,
                    metadata={
                        "task": "call_target_mapping",
                        "caller_func_id": func_id,
                        "binary_unit_id": binary_id,
                        "call_opcode": call_inst.opcode,
                        "callee_func_id": target_inst.func_id,
                        "callee_entry_addr": target_addr,
                    },
                )
            )
            queries.append(
                MappingQuery(
                    query_id=call_inst.unit_id,
                    positive_id=target_inst.unit_id,
                    candidate_ids=list(candidate_ids),
                    metadata={
                        "task": "call_target_mapping",
                        "caller_func_id": func_id,
                        "binary_unit_id": binary_id,
                        "call_opcode": call_inst.opcode,
                        "callee_func_id": target_inst.func_id,
                        "callee_entry_addr": target_addr,
                        "pool_size": len(candidate_ids),
                    },
                )
            )
            pool_sizes.append(len(candidate_ids))

            if max_examples is not None and len(examples) >= max_examples:
                return examples[:max_examples], queries[:max_examples], units, summarize_pool_sizes(pool_sizes)

    return examples, queries, units, summarize_pool_sizes(pool_sizes)


def train_linear_map(
    examples: Sequence[MappingExample],
    embeddings: Dict[str, object],
    dim: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> LinearMap:
    model = LinearMap(dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    rng = random.Random(seed)

    source_vecs = torch.stack([embeddings[item.source_id].pooled for item in examples], dim=0).to(device)
    target_vecs = torch.stack([embeddings[item.target_id].pooled for item in examples], dim=0).to(device)
    target_vecs = F.normalize(target_vecs, dim=-1)

    indices = list(range(len(examples)))
    for _ in range(epochs):
        rng.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start : start + batch_size]
            src = source_vecs[batch_idx]
            tgt = target_vecs[batch_idx]
            pred = F.normalize(model(src), dim=-1)
            logits = torch.matmul(pred, tgt.transpose(0, 1)) / 0.07
            labels = torch.arange(len(batch_idx), device=device)
            loss = F.cross_entropy(logits, labels)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    return model.eval()


def evaluate_mapping(
    model: LinearMap,
    queries: Sequence[MappingQuery],
    embeddings: Dict[str, object],
    device: torch.device,
) -> Dict[str, float]:
    if not queries:
        return {}

    top1_hits = 0
    recall5_hits = 0
    mrr_total = 0.0
    kept = 0

    with torch.no_grad():
        for query in queries:
            if query.query_id not in embeddings or query.positive_id not in embeddings:
                continue

            src = embeddings[query.query_id].pooled.to(device)
            pred = F.normalize(model(src.unsqueeze(0)).squeeze(0), dim=-1)

            candidate_tensors = []
            candidate_ids = []
            for candidate_id in query.candidate_ids:
                if candidate_id not in embeddings:
                    continue
                candidate_ids.append(candidate_id)
                candidate_tensors.append(embeddings[candidate_id].pooled)
            if not candidate_tensors or query.positive_id not in candidate_ids:
                continue

            cands = F.normalize(torch.stack(candidate_tensors, dim=0).to(device), dim=-1)
            scores = torch.matmul(cands, pred)
            ranked = torch.argsort(scores, descending=True).tolist()
            ranked_ids = [candidate_ids[idx] for idx in ranked]
            rank = ranked_ids.index(query.positive_id) + 1

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


def run_mapping_task(
    task_name: str,
    train_examples: Sequence[MappingExample],
    test_queries: Sequence[MappingQuery],
    units: Dict[str, ProbeUnit],
    vocab: WordVocab,
    model_name: str,
    checkpoint_path: str,
    hidden: int,
    n_layers: int,
    attn_heads: int,
    seq_len: int,
    batch_size: int,
    device: torch.device,
    train_epochs: int,
    train_lr: float,
    train_weight_decay: float,
    train_batch_size: int,
    train_seed: int,
) -> Dict[str, object]:
    encoder = FrozenBackboneEncoder(
        checkpoint_path=checkpoint_path,
        vocab=vocab,
        hidden=hidden,
        n_layers=n_layers,
        attn_heads=attn_heads,
        seq_len=seq_len,
        max_instructions=1,
        device=device,
    )
    embeddings = encoder.encode_units(units, batch_size=batch_size)
    linear_map = train_linear_map(
        examples=train_examples,
        embeddings=embeddings,
        dim=hidden,
        epochs=train_epochs,
        lr=train_lr,
        weight_decay=train_weight_decay,
        batch_size=train_batch_size,
        device=device,
        seed=train_seed,
    )
    metrics = evaluate_mapping(linear_map, test_queries, embeddings, device)
    return {
        "model_name": model_name,
        "use_addr": encoder.use_addr,
        "mapping": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Full-binary linear instruction mapping for jump and call")
    parser.add_argument("--records_root", required=True, help="Directory containing *_functions.json files")
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--baseline_checkpoint", required=True)
    parser.add_argument("--aa_checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--hidden", type=int, default=768)
    parser.add_argument("--n_layers", type=int, default=12)
    parser.add_argument("--attn_heads", type=int, default=12)
    parser.add_argument("--seq_len", type=int, default=32)
    parser.add_argument("--max_binary_units", type=int, default=None)
    parser.add_argument("--max_train_examples", type=int, default=20000)
    parser.add_argument("--max_test_queries", type=int, default=5000)
    parser.add_argument("--test_fraction", type=float, default=0.3)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--jump_target_opcode", default="mov")
    parser.add_argument("--encode_batch_size", type=int, default=128)
    parser.add_argument("--train_epochs", type=int, default=50)
    parser.add_argument("--train_lr", type=float, default=1e-3)
    parser.add_argument("--train_weight_decay", type=float, default=1e-4)
    parser.add_argument("--train_batch_size", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    device = torch.device(args.device)

    print("[LinearMapProbe] Loading full-binary function records ...")
    function_blocks, binary_by_func, binary_unit_ids = load_full_binary_function_blocks(
        args.records_root,
        args.max_binary_units,
        args.split_seed,
    )
    instructions_by_func, blocks_by_func = reconstruct_blocks(function_blocks)
    train_binary_ids, test_binary_ids = split_binary_unit_ids(
        binary_unit_ids,
        args.test_fraction,
        args.split_seed,
    )
    train_func_ids = sorted(
        func_id for func_id, binary_id in binary_by_func.items() if binary_id in train_binary_ids
    )
    test_func_ids = sorted(
        func_id for func_id, binary_id in binary_by_func.items() if binary_id in test_binary_ids
    )

    print("[LinearMapProbe] Building jump mapping data ...")
    train_jump_examples, _, train_jump_units, jump_train_pool = build_jump_mapping_data(
        instructions_by_func=instructions_by_func,
        func_ids=train_func_ids,
        max_examples=args.max_train_examples,
        seed=args.split_seed + 11,
        target_opcode=args.jump_target_opcode,
    )
    _, test_jump_queries, test_jump_units, jump_test_pool = build_jump_mapping_data(
        instructions_by_func=instructions_by_func,
        func_ids=test_func_ids,
        max_examples=args.max_test_queries,
        seed=args.split_seed + 12,
        target_opcode=args.jump_target_opcode,
    )

    print("[LinearMapProbe] Building call mapping data ...")
    train_call_examples, _, train_call_units, call_train_pool = build_call_mapping_data(
        instructions_by_func=instructions_by_func,
        blocks_by_func=blocks_by_func,
        func_ids=train_func_ids,
        binary_by_func=binary_by_func,
        max_examples=args.max_train_examples,
        seed=args.split_seed + 21,
    )
    _, test_call_queries, test_call_units, call_test_pool = build_call_mapping_data(
        instructions_by_func=instructions_by_func,
        blocks_by_func=blocks_by_func,
        func_ids=test_func_ids,
        binary_by_func=binary_by_func,
        max_examples=args.max_test_queries,
        seed=args.split_seed + 22,
    )

    report: Dict[str, object] = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "config": vars(args),
        "data_summary": {
            "num_binary_units": len(binary_unit_ids),
            "num_train_binary_units": len(train_binary_ids),
            "num_test_binary_units": len(test_binary_ids),
            "num_functions": len(function_blocks),
            "num_train_functions": len(train_func_ids),
            "num_test_functions": len(test_func_ids),
        },
        "tasks": {},
    }

    vocab = WordVocab.load_vocab(args.vocab)
    model_specs = [
        ("baseline", args.baseline_checkpoint),
        ("address_aware", args.aa_checkpoint),
    ]

    task_specs = [
        (
            "jump_target_mapping",
            train_jump_examples,
            test_jump_queries,
            {**train_jump_units, **test_jump_units},
            jump_train_pool,
            jump_test_pool,
            args.split_seed + 101,
        ),
        (
            "call_target_mapping",
            train_call_examples,
            test_call_queries,
            {**train_call_units, **test_call_units},
            call_train_pool,
            call_test_pool,
            args.split_seed + 201,
        ),
    ]

    for task_name, train_examples, test_queries, units, train_pool, test_pool, train_seed in task_specs:
        if not train_examples or not test_queries:
            report["tasks"][task_name] = {
                "error": "Insufficient mapping data after construction.",
                "num_train_examples": len(train_examples),
                "num_test_queries": len(test_queries),
            }
            continue

        task_report: Dict[str, object] = {
            "num_units": len(units),
            "num_train_examples": len(train_examples),
            "num_test_queries": len(test_queries),
            "train_pool_stats": train_pool,
            "test_pool_stats": test_pool,
            "models": {},
        }
        for model_name, checkpoint_path in model_specs:
            print(f"[LinearMapProbe] Training/evaluating {task_name} with {model_name}: {checkpoint_path}")
            task_report["models"][model_name] = run_mapping_task(
                task_name=task_name,
                train_examples=train_examples,
                test_queries=test_queries,
                units=units,
                vocab=vocab,
                model_name=model_name,
                checkpoint_path=checkpoint_path,
                hidden=args.hidden,
                n_layers=args.n_layers,
                attn_heads=args.attn_heads,
                seq_len=args.seq_len,
                batch_size=args.encode_batch_size,
                device=device,
                train_epochs=args.train_epochs,
                train_lr=args.train_lr,
                train_weight_decay=args.train_weight_decay,
                train_batch_size=args.train_batch_size,
                train_seed=train_seed,
            )

        report["tasks"][task_name] = task_report

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[LinearMapProbe] Wrote report to {args.output}")


if __name__ == "__main__":
    main()
