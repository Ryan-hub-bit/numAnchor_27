#!/usr/bin/env python3
"""
Full-binary call-target probing.

This evaluation differs from the pool-based `call_target` probe in `probe_eval.py`:

- positives are built from direct call instructions whose target matches a
  function entry inside the same compiled binary unit (`*_functions.json`);
- negatives are incorrect functions from the same binary unit;
- the candidate side is encoded as a function unit, not an entry basic block.

This makes the task a true "correct function vs wrong function" probe and avoids
the severe sample-loss caused by requiring the callee to also exist in a sampled
function pool.
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

import torch


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
    ProbePair,
    ProbeUnit,
    WordVocab,
    ensure_balanced_pairs,
    extract_call_target_address,
    reconstruct_blocks,
    run_task_for_model,
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
    rng = random.Random(seed)
    rng.shuffle(ids)
    test_n = max(1, int(len(ids) * test_fraction)) if len(ids) > 1 else 0
    test_ids = sorted(ids[:test_n])
    train_ids = sorted(ids[test_n:])
    return train_ids, test_ids


def build_call_target_function_pairs(
    instructions_by_func: Dict[str, List[InstructionRecord]],
    function_blocks: Dict[str, List[str]],
    blocks_by_func: Dict[str, List[BasicBlockRecord]],
    func_ids: Sequence[str],
    binary_by_func: Dict[str, str],
    max_pairs: int,
    seed: int,
) -> Tuple[List[ProbePair], Dict[str, ProbeUnit]]:
    rng = random.Random(seed)
    pairs: List[ProbePair] = []
    units: Dict[str, ProbeUnit] = {}

    entry_func_by_binary_addr: Dict[Tuple[str, str], str] = {}
    entry_opcode_by_func: Dict[str, str] = {}
    funcs_by_binary: Dict[str, List[str]] = defaultdict(list)
    func_unit_id_by_func: Dict[str, str] = {}

    for func_id in func_ids:
        blocks = [block for block in blocks_by_func.get(func_id, []) if block.instructions]
        if not blocks:
            continue
        binary_id = binary_by_func.get(func_id)
        if not binary_id:
            continue

        entry_block = min(blocks, key=lambda block: block.block_index)
        entry_func_by_binary_addr[(binary_id, entry_block.entry_address)] = func_id
        entry_opcode_by_func[func_id] = entry_block.entry_opcode
        funcs_by_binary[binary_id].append(func_id)
        func_unit_id_by_func[func_id] = f"func:{func_id}"

    shuffled_funcs = list(func_ids)
    rng.shuffle(shuffled_funcs)
    for func_id in shuffled_funcs:
        binary_id = binary_by_func.get(func_id)
        if not binary_id:
            continue

        for call_inst in instructions_by_func.get(func_id, []):
            target_addr = extract_call_target_address(call_inst.text)
            if not target_addr:
                continue

            callee_func_id = entry_func_by_binary_addr.get((binary_id, target_addr))
            if callee_func_id is None:
                continue

            negatives = [
                cand_func_id
                for cand_func_id in funcs_by_binary.get(binary_id, [])
                if cand_func_id != callee_func_id
            ]
            if not negatives:
                continue

            matched_opcode = [
                cand_func_id
                for cand_func_id in negatives
                if entry_opcode_by_func.get(cand_func_id) == entry_opcode_by_func.get(callee_func_id)
            ]
            negative_func_id = rng.choice(matched_opcode or negatives)

            callee_unit_id = func_unit_id_by_func[callee_func_id]
            negative_unit_id = func_unit_id_by_func[negative_func_id]

            units[call_inst.unit_id] = ProbeUnit(call_inst.unit_id, "instruction", [call_inst.text])
            units[callee_unit_id] = ProbeUnit(
                callee_unit_id,
                "function",
                function_blocks[callee_func_id],
            )
            units[negative_unit_id] = ProbeUnit(
                negative_unit_id,
                "function",
                function_blocks[negative_func_id],
            )

            pairs.append(
                ProbePair(
                    left_id=call_inst.unit_id,
                    right_id=callee_unit_id,
                    label=1,
                    group_id=f"call_target:{call_inst.unit_id}",
                    metadata={
                        "task": "call_target",
                        "caller_func_id": func_id,
                        "binary_unit_id": binary_id,
                        "call_opcode": call_inst.opcode,
                        "callee_func_id": callee_func_id,
                        "callee_entry_addr": target_addr,
                        "candidate_kind": "function",
                    },
                )
            )
            pairs.append(
                ProbePair(
                    left_id=call_inst.unit_id,
                    right_id=negative_unit_id,
                    label=0,
                    group_id=f"call_target_neg:{call_inst.unit_id}",
                    metadata={
                        "task": "call_target",
                        "caller_func_id": func_id,
                        "binary_unit_id": binary_id,
                        "call_opcode": call_inst.opcode,
                        "callee_func_id": callee_func_id,
                        "negative_func_id": negative_func_id,
                        "callee_entry_addr": target_addr,
                        "candidate_kind": "function",
                    },
                )
            )

            if len(pairs) >= max_pairs:
                return pairs[:max_pairs], units

    return pairs[:max_pairs], units


def main() -> None:
    parser = argparse.ArgumentParser(description="Full-binary function-level call-target probing")
    parser.add_argument("--records_root", required=True, help="Directory containing *_functions.json files")
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--baseline_checkpoint", required=True)
    parser.add_argument("--aa_checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--hidden", type=int, default=768)
    parser.add_argument("--n_layers", type=int, default=12)
    parser.add_argument("--attn_heads", type=int, default=12)
    parser.add_argument("--seq_len", type=int, default=32)
    parser.add_argument(
        "--max_candidate_instructions",
        type=int,
        default=8,
        help="Maximum number of function instructions kept by the frozen encoder",
    )
    parser.add_argument("--max_binary_units", type=int, default=None)
    parser.add_argument("--max_train_pairs", type=int, default=20000)
    parser.add_argument("--max_test_pairs", type=int, default=5000)
    parser.add_argument("--test_fraction", type=float, default=0.3)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--few_shots", nargs="+", type=int, default=[8, 32, 128])
    parser.add_argument("--few_shot_seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--probe_epochs", type=int, default=200)
    parser.add_argument("--probe_lr", type=float, default=1e-3)
    parser.add_argument("--probe_weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    device = torch.device(args.device)

    print("[FullCallProbe] Loading full-binary function records ...")
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

    print("[FullCallProbe] Building call-target pairs ...")
    train_pairs, train_units = build_call_target_function_pairs(
        instructions_by_func=instructions_by_func,
        function_blocks=function_blocks,
        blocks_by_func=blocks_by_func,
        func_ids=train_func_ids,
        binary_by_func=binary_by_func,
        max_pairs=args.max_train_pairs,
        seed=args.split_seed + 41,
    )
    test_pairs, test_units = build_call_target_function_pairs(
        instructions_by_func=instructions_by_func,
        function_blocks=function_blocks,
        blocks_by_func=blocks_by_func,
        func_ids=test_func_ids,
        binary_by_func=binary_by_func,
        max_pairs=args.max_test_pairs,
        seed=args.split_seed + 42,
    )

    train_pairs = ensure_balanced_pairs(train_pairs)
    test_pairs = ensure_balanced_pairs(test_pairs)

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

    if not train_pairs or not test_pairs:
        report["tasks"]["call_target"] = {
            "error": "Insufficient full-binary call-target pairs after construction.",
            "num_train_pairs": len(train_pairs),
            "num_test_pairs": len(test_pairs),
        }
    else:
        units = {}
        units.update(train_units)
        units.update(test_units)

        vocab = WordVocab.load_vocab(args.vocab)
        model_specs = [
            ("baseline", args.baseline_checkpoint),
            ("address_aware", args.aa_checkpoint),
        ]
        task_report = {
            "candidate_kind": "function",
            "num_units": len(units),
            "num_train_pairs": len(train_pairs),
            "num_test_pairs": len(test_pairs),
            "models": {},
        }

        for model_name, checkpoint_path in model_specs:
            print(f"[FullCallProbe] Encoding/evaluating call_target with {model_name}: {checkpoint_path}")
            encoder = FrozenBackboneEncoder(
                checkpoint_path=checkpoint_path,
                vocab=vocab,
                hidden=args.hidden,
                n_layers=args.n_layers,
                attn_heads=args.attn_heads,
                seq_len=args.seq_len,
                max_instructions=args.max_candidate_instructions,
                device=device,
            )
            task_report["models"][model_name] = run_task_for_model(
                model_name=model_name,
                encoder=encoder,
                task="call_target",
                train_pairs=train_pairs,
                test_pairs=test_pairs,
                units=units,
                few_shots=args.few_shots,
                few_shot_seeds=args.few_shot_seeds,
                batch_size=args.batch_size,
                probe_epochs=args.probe_epochs,
                probe_lr=args.probe_lr,
                probe_weight_decay=args.probe_weight_decay,
                device=device,
            )

        report["tasks"]["call_target"] = task_report

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[FullCallProbe] Wrote report to {args.output}")


if __name__ == "__main__":
    main()
