#!/usr/bin/env python3
"""
Within-binary call-target retrieval over function entry blocks.

For each direct `call sub_xxx`, the query is the call instruction itself.
Candidates are all function entry basic blocks from the same compiled binary
unit. The task succeeds when the correct callee entry block is ranked first.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
PROBE_DIR = os.path.dirname(os.path.abspath(__file__))
if PROBE_DIR not in sys.path:
    sys.path.insert(0, PROBE_DIR)

from probe_eval import (  # noqa: E402
    BasicBlockRecord,
    EncodedUnit,
    FrozenBackboneEncoder,
    InstructionRecord,
    ProbeUnit,
    RetrievalQuery,
    WordVocab,
    extract_call_target_address,
    reconstruct_blocks,
)


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


def build_call_target_entry_retrieval_queries(
    instructions_by_func: Dict[str, List[InstructionRecord]],
    blocks_by_func: Dict[str, List[BasicBlockRecord]],
    func_ids: Sequence[str],
    binary_by_func: Dict[str, str],
    max_queries: Optional[int],
    seed: int,
    candidate_mode: str,
) -> Tuple[List[RetrievalQuery], Dict[str, ProbeUnit], Dict[str, float]]:
    rng = random.Random(seed)
    queries: List[RetrievalQuery] = []
    units: Dict[str, ProbeUnit] = {}

    entry_block_by_binary_addr: Dict[Tuple[str, str], BasicBlockRecord] = {}
    entry_block_by_func: Dict[str, BasicBlockRecord] = {}
    entry_block_ids_by_binary: Dict[str, List[str]] = defaultdict(list)
    candidate_sizes: List[int] = []

    for func_id in func_ids:
        blocks = [block for block in blocks_by_func.get(func_id, []) if block.instructions]
        if not blocks:
            continue
        binary_id = binary_by_func.get(func_id)
        if not binary_id:
            continue

        entry_block = min(blocks, key=lambda block: block.block_index)
        entry_block_by_func[func_id] = entry_block
        entry_block_by_binary_addr[(binary_id, entry_block.entry_address)] = entry_block
        entry_block_ids_by_binary[binary_id].append(entry_block.unit_id)
        if candidate_mode == "entry_instruction":
            inst = entry_block.instructions[0]
            units[entry_block.unit_id] = ProbeUnit(
                entry_block.unit_id,
                "instruction",
                [inst.text],
            )
        else:
            units[entry_block.unit_id] = ProbeUnit(
                entry_block.unit_id,
                "basic_block",
                [inst.text for inst in entry_block.instructions],
            )

    shuffled_funcs = list(func_ids)
    rng.shuffle(shuffled_funcs)

    for func_id in shuffled_funcs:
        binary_id = binary_by_func.get(func_id)
        if not binary_id:
            continue

        candidate_ids = sorted(set(entry_block_ids_by_binary.get(binary_id, [])))
        if len(candidate_ids) < 2:
            continue

        call_insts = list(instructions_by_func.get(func_id, []))
        rng.shuffle(call_insts)
        for call_inst in call_insts:
            target_addr = extract_call_target_address(call_inst.text)
            if not target_addr:
                continue

            callee_block = entry_block_by_binary_addr.get((binary_id, target_addr))
            if callee_block is None:
                continue
            if callee_block.unit_id not in candidate_ids:
                continue

            units[call_inst.unit_id] = ProbeUnit(call_inst.unit_id, "instruction", [call_inst.text])
            queries.append(
                RetrievalQuery(
                    query_id=call_inst.unit_id,
                    candidate_ids=candidate_ids,
                    positive_id=callee_block.unit_id,
                    group_id=f"call_target_entry_retrieval:{call_inst.unit_id}",
                    metadata={
                        "task": "call_target_entry_retrieval",
                        "caller_func_id": func_id,
                        "binary_unit_id": binary_id,
                        "call_opcode": call_inst.opcode,
                        "callee_func_id": callee_block.func_id,
                        "callee_entry_addr": target_addr,
                        "pool_size": len(candidate_ids),
                    },
                )
            )
            candidate_sizes.append(len(candidate_ids))

            if max_queries is not None and len(queries) >= max_queries:
                return queries, units, summarize_candidate_sizes(candidate_sizes)

    return queries, units, summarize_candidate_sizes(candidate_sizes)


def summarize_candidate_sizes(candidate_sizes: Sequence[int]) -> Dict[str, float]:
    if not candidate_sizes:
        return {}
    arr = np.asarray(candidate_sizes, dtype=np.float64)
    return {
        "mean_pool_size": float(arr.mean()),
        "median_pool_size": float(np.median(arr)),
        "min_pool_size": float(arr.min()),
        "max_pool_size": float(arr.max()),
        "random_top1_mean": float(np.mean(1.0 / arr)),
        "random_recall_at_5_mean": float(np.mean(np.minimum(5.0, arr) / arr)),
    }


def cosine_score(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())


def compute_call_target_retrieval_metrics(
    queries: Sequence[RetrievalQuery],
    embeddings: Dict[str, EncodedUnit],
) -> Dict[str, float]:
    if not queries:
        return {}

    top1_hits = 0
    recall5_hits = 0
    mrr_total = 0.0
    kept = 0

    for query in queries:
        if query.query_id not in embeddings or query.positive_id not in embeddings:
            continue

        query_unit = embeddings[query.query_id]
        query_vec = query_unit.address_token if query_unit.address_token is not None else query_unit.pooled

        scored = []
        for candidate_id in query.candidate_ids:
            if candidate_id not in embeddings:
                continue
            candidate_vec = embeddings[candidate_id].pooled
            scored.append((candidate_id, cosine_score(query_vec, candidate_vec)))

        if not scored:
            continue

        scored.sort(key=lambda item: item[1], reverse=True)
        ranked_ids = [candidate_id for candidate_id, _ in scored]
        if query.positive_id not in ranked_ids:
            continue

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


def run_call_target_retrieval_for_model(
    model_name: str,
    encoder: FrozenBackboneEncoder,
    queries: Sequence[RetrievalQuery],
    units: Dict[str, ProbeUnit],
    batch_size: int,
) -> Dict[str, object]:
    embeddings = encoder.encode_units(units, batch_size=batch_size)
    return {
        "model_name": model_name,
        "use_addr": encoder.use_addr,
        "retrieval": compute_call_target_retrieval_metrics(queries, embeddings),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Within-binary call-target retrieval over entry blocks")
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
    parser.add_argument("--max_test_queries", type=int, default=5000)
    parser.add_argument("--test_fraction", type=float, default=0.3)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument(
        "--candidate_mode",
        choices=["entry_block", "entry_instruction"],
        default="entry_block",
        help="Use the whole entry block or only its first instruction as the candidate representation.",
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    device = torch.device(args.device)

    print("[BinaryEntryCallRetrieval] Loading full-binary function records ...")
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
    test_func_ids = sorted(
        func_id for func_id, binary_id in binary_by_func.items() if binary_id in test_binary_ids
    )

    print("[BinaryEntryCallRetrieval] Building retrieval queries ...")
    queries, units, pool_stats = build_call_target_entry_retrieval_queries(
        instructions_by_func=instructions_by_func,
        blocks_by_func=blocks_by_func,
        func_ids=test_func_ids,
        binary_by_func=binary_by_func,
        max_queries=args.max_test_queries,
        seed=args.split_seed + 91,
        candidate_mode=args.candidate_mode,
    )

    report: Dict[str, object] = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "config": vars(args),
        "data_summary": {
            "num_binary_units": len(binary_unit_ids),
            "num_train_binary_units": len(train_binary_ids),
            "num_test_binary_units": len(test_binary_ids),
            "num_functions": len(function_blocks),
            "num_test_functions": len(test_func_ids),
        },
        "task": {
            "name": "call_target_entry_retrieval",
            "candidate_kind": args.candidate_mode,
            "num_queries": len(queries),
            "num_units": len(units),
            "pool_stats": pool_stats,
            "models": {},
        },
    }

    if not queries:
        report["task"]["error"] = "No eligible call-target retrieval queries were constructed."
    else:
        vocab = WordVocab.load_vocab(args.vocab)
        model_specs = [
            ("baseline", args.baseline_checkpoint),
            ("address_aware", args.aa_checkpoint),
        ]
        for model_name, checkpoint_path in model_specs:
            print(
                f"[BinaryEntryCallRetrieval] Encoding/evaluating with {model_name}: {checkpoint_path}"
            )
            encoder = FrozenBackboneEncoder(
                checkpoint_path=checkpoint_path,
                vocab=vocab,
                hidden=args.hidden,
                n_layers=args.n_layers,
                attn_heads=args.attn_heads,
                seq_len=args.seq_len,
                max_instructions=32,
                device=device,
            )
            report["task"]["models"][model_name] = run_call_target_retrieval_for_model(
                model_name=model_name,
                encoder=encoder,
                queries=queries,
                units=units,
                batch_size=args.batch_size,
            )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[BinaryEntryCallRetrieval] Wrote report to {args.output}")


if __name__ == "__main__":
    main()
