#!/usr/bin/env python3
"""
Build same_bb / same_function pair corpora from *_functions.json records.

The generated JSONL files are intended for auxiliary pretraining, not final
evaluation. Each line has the form:

  {
    "task": "same_bb" | "same_function",
    "label": 0 | 1,
    "left": ["inst ...", ...],
    "right": ["inst ...", ...],
    "binary": "...",
    "func_id": "...",
    ...
  }

The builder supports excluding functions that appear in a probing benchmark
such as pool_function_blocks_1k.json, preventing exact train/eval overlap.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import re
from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


HEADER_RE = re.compile(r"^(\w+)\((0x[0-9a-fA-F]+):([0-9.]+):([0-9.]+):([0-9.]+)\)")


@dataclass
class InstructionRecord:
    text: str
    address: str
    opcode: str
    binary_pos: float
    block_pos: float
    inst_pos: float


@dataclass
class BasicBlockRecord:
    block_id: str
    func_uid: str
    entry_address: str
    entry_opcode: str
    instructions: List[InstructionRecord]


@dataclass
class FunctionRecord:
    func_uid: str
    binary: str
    opt_level: str
    function_name: str
    instructions: List[InstructionRecord]
    blocks: List[BasicBlockRecord]
    signature: str


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def make_function_uid(binary: str, opt_level: str, func_id: object) -> str:
    return f"{binary}::{opt_level}::{func_id}"


def parse_instruction(inst_text: str) -> Optional[InstructionRecord]:
    m = HEADER_RE.match(inst_text)
    if not m:
        return None
    return InstructionRecord(
        text=inst_text,
        address=m.group(2),
        opcode=m.group(1),
        binary_pos=float(m.group(3)),
        block_pos=float(m.group(4)),
        inst_pos=float(m.group(5)),
    )


def reconstruct_blocks(func_uid: str, instructions: Sequence[InstructionRecord]) -> List[BasicBlockRecord]:
    blocks: List[BasicBlockRecord] = []
    current: List[InstructionRecord] = []
    current_block_pos: Optional[float] = None
    block_idx = -1

    for inst in instructions:
        if current_block_pos is None or abs(inst.block_pos - current_block_pos) > 1e-8:
            if current:
                entry = current[0]
                blocks.append(
                    BasicBlockRecord(
                        block_id=f"bb:{func_uid}:{block_idx}",
                        func_uid=func_uid,
                        entry_address=entry.address,
                        entry_opcode=entry.opcode,
                        instructions=list(current),
                    )
                )
            block_idx += 1
            current = [inst]
            current_block_pos = inst.block_pos
        else:
            current.append(inst)

    if current:
        entry = current[0]
        blocks.append(
            BasicBlockRecord(
                block_id=f"bb:{func_uid}:{block_idx}",
                func_uid=func_uid,
                entry_address=entry.address,
                entry_opcode=entry.opcode,
                instructions=list(current),
            )
        )

    return blocks


def load_excluded_signatures(function_blocks_path: Optional[str]) -> set[str]:
    if not function_blocks_path:
        return set()
    payload = load_json(function_blocks_path)
    excluded = set()
    if isinstance(payload, dict):
        for instructions in payload.values():
            if isinstance(instructions, list):
                excluded.add(sha1("\t".join(instructions).encode("utf-8")).hexdigest())
    elif isinstance(payload, list):
        for record in payload:
            if not isinstance(record, dict):
                continue
            funcins = record.get("funcins")
            if isinstance(funcins, str):
                excluded.add(sha1(funcins.encode("utf-8")).hexdigest())
            else:
                insts = record.get("instructions")
                if isinstance(insts, list):
                    excluded.add(sha1("\t".join(insts).encode("utf-8")).hexdigest())
    return excluded


def iter_function_jsons(root: str) -> Iterable[str]:
    path = Path(root)
    if path.is_file():
        yield str(path)
        return
    for item in sorted(path.glob("*_functions.json")):
        if item.is_file():
            yield str(item)


def load_functions(records_root: str, excluded_signatures: set[str]) -> List[FunctionRecord]:
    functions: List[FunctionRecord] = []
    for path in iter_function_jsons(records_root):
        try:
            payload = load_json(path)
        except Exception:
            continue
        if not isinstance(payload, list):
            continue
        for record in payload:
            if not isinstance(record, dict):
                continue
            funcins = record.get("funcins")
            if not isinstance(funcins, str) or not funcins.strip():
                continue
            signature = sha1(funcins.encode("utf-8")).hexdigest()
            if signature in excluded_signatures:
                continue
            binary = str(record.get("binary", "unknown_binary"))
            opt_level = str(record.get("opt_level", "unknown_opt"))
            func_uid = make_function_uid(binary, opt_level, record.get("func_id", len(functions)))
            function_name = str(record.get("function", func_uid))
            parsed = [
                inst for inst in
                (parse_instruction(text) for text in funcins.split("\t") if text.strip())
                if inst is not None
            ]
            if len(parsed) < 2:
                continue
            blocks = reconstruct_blocks(func_uid, parsed)
            if not blocks:
                continue
            functions.append(
                FunctionRecord(
                    func_uid=func_uid,
                    binary=binary,
                    opt_level=opt_level,
                    function_name=function_name,
                    instructions=parsed,
                    blocks=blocks,
                    signature=signature,
                )
            )
    return functions


def split_functions(functions: Sequence[FunctionRecord], val_fraction: float, seed: int) -> Tuple[List[FunctionRecord], List[FunctionRecord]]:
    items = list(functions)
    rng = random.Random(seed)
    rng.shuffle(items)
    if not items:
        return [], []
    val_n = max(1, int(len(items) * val_fraction)) if val_fraction > 0 else 0
    val = items[:val_n]
    train = items[val_n:]
    return train, val


def sample_unique_pairs(index_pairs: Sequence[Tuple[int, int]], sample_size: int, rng: random.Random) -> List[Tuple[int, int]]:
    pairs = list(index_pairs)
    if len(pairs) <= sample_size:
        return pairs
    return rng.sample(pairs, sample_size)


def build_same_bb_pairs(
    functions: Sequence[FunctionRecord],
    rng: random.Random,
    max_pairs: int,
    positives_per_block: int,
) -> List[dict]:
    pairs: List[dict] = []
    shuffled = list(functions)
    rng.shuffle(shuffled)
    for function in shuffled:
        blocks = [block for block in function.blocks if block.instructions]
        positive_blocks = [block for block in blocks if len(block.instructions) >= 2]
        if not positive_blocks:
            continue
        for block in positive_blocks:
            candidates = list(itertools.combinations(range(len(block.instructions)), 2))
            for left_idx, right_idx in sample_unique_pairs(candidates, positives_per_block, rng):
                left = block.instructions[left_idx]
                right = block.instructions[right_idx]
                pairs.append(
                    {
                        "task": "same_bb",
                        "label": 1,
                        "left": [left.text],
                        "right": [right.text],
                        "binary": function.binary,
                        "opt_level": function.opt_level,
                        "func_uid": function.func_uid,
                        "block_id": block.block_id,
                    }
                )

                other_blocks = [candidate for candidate in blocks if candidate.block_id != block.block_id and candidate.instructions]
                if not other_blocks:
                    continue
                neg_block = rng.choice(other_blocks)
                neg_inst = rng.choice(neg_block.instructions)
                pairs.append(
                    {
                        "task": "same_bb",
                        "label": 0,
                        "left": [left.text],
                        "right": [neg_inst.text],
                        "binary": function.binary,
                        "opt_level": function.opt_level,
                        "func_uid": function.func_uid,
                        "anchor_block_id": block.block_id,
                        "negative_block_id": neg_block.block_id,
                    }
                )
                if len(pairs) >= max_pairs:
                    return pairs[:max_pairs]
    return pairs[:max_pairs]


def build_same_function_pairs(
    functions: Sequence[FunctionRecord],
    rng: random.Random,
    max_pairs: int,
    positives_per_function: int,
    max_block_instructions: int,
) -> List[dict]:
    pairs: List[dict] = []
    eligible = [fn for fn in functions if len(fn.blocks) >= 2]
    if len(eligible) < 2:
        return pairs

    block_pool = [block for fn in eligible for block in fn.blocks if block.instructions]
    by_entry_opcode: Dict[str, List[BasicBlockRecord]] = {}
    for block in block_pool:
        by_entry_opcode.setdefault(block.entry_opcode, []).append(block)

    shuffled = list(eligible)
    rng.shuffle(shuffled)
    for function in shuffled:
        blocks = [block for block in function.blocks if block.instructions]
        block_pairs = list(itertools.combinations(range(len(blocks)), 2))
        for left_idx, right_idx in sample_unique_pairs(block_pairs, positives_per_function, rng):
            left = blocks[left_idx]
            right = blocks[right_idx]
            pairs.append(
                {
                    "task": "same_function",
                    "label": 1,
                    "left": [inst.text for inst in left.instructions[:max_block_instructions]],
                    "right": [inst.text for inst in right.instructions[:max_block_instructions]],
                    "binary": function.binary,
                    "opt_level": function.opt_level,
                    "func_uid": function.func_uid,
                    "left_block_id": left.block_id,
                    "right_block_id": right.block_id,
                }
            )

            negatives = [
                candidate
                for candidate in by_entry_opcode.get(left.entry_opcode, [])
                if candidate.func_uid != function.func_uid
            ]
            if not negatives:
                negatives = [candidate for candidate in block_pool if candidate.func_uid != function.func_uid]
            if not negatives:
                continue
            neg = rng.choice(negatives)
            pairs.append(
                {
                    "task": "same_function",
                    "label": 0,
                    "left": [inst.text for inst in left.instructions[:max_block_instructions]],
                    "right": [inst.text for inst in neg.instructions[:max_block_instructions]],
                    "binary": function.binary,
                    "opt_level": function.opt_level,
                    "func_uid": function.func_uid,
                    "left_block_id": left.block_id,
                    "negative_func_uid": neg.func_uid,
                    "negative_block_id": neg.block_id,
                }
            )
            if len(pairs) >= max_pairs:
                return pairs[:max_pairs]
    return pairs[:max_pairs]


def write_jsonl(path: str, rows: Sequence[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True))
            f.write("\n")


def main():
    parser = argparse.ArgumentParser(description="Build same_bb / same_function pair corpora")
    parser.add_argument("--records_root", required=True, help="Directory containing *_functions.json or a single *_functions.json file")
    parser.add_argument("--output_dir", required=True, help="Output directory for JSONL corpora")
    parser.add_argument("--exclude_function_blocks", default=None, help="Optional pool_function_blocks JSON to exclude from auxiliary training")
    parser.add_argument("--val_fraction", type=float, default=0.05, help="Fraction of functions to reserve for auxiliary validation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max_same_bb_train_pairs", type=int, default=200000)
    parser.add_argument("--max_same_bb_val_pairs", type=int, default=10000)
    parser.add_argument("--max_same_function_train_pairs", type=int, default=200000)
    parser.add_argument("--max_same_function_val_pairs", type=int, default=10000)
    parser.add_argument("--positives_per_block", type=int, default=4)
    parser.add_argument("--positives_per_function", type=int, default=4)
    parser.add_argument("--max_block_instructions", type=int, default=8)
    args = parser.parse_args()

    excluded_signatures = load_excluded_signatures(args.exclude_function_blocks)
    functions = load_functions(args.records_root, excluded_signatures)
    train_functions, val_functions = split_functions(functions, args.val_fraction, args.seed)

    rng_train = random.Random(args.seed + 101)
    rng_val = random.Random(args.seed + 202)

    same_bb_train = build_same_bb_pairs(
        train_functions,
        rng=rng_train,
        max_pairs=args.max_same_bb_train_pairs,
        positives_per_block=args.positives_per_block,
    )
    same_bb_val = build_same_bb_pairs(
        val_functions,
        rng=rng_val,
        max_pairs=args.max_same_bb_val_pairs,
        positives_per_block=args.positives_per_block,
    )
    same_function_train = build_same_function_pairs(
        train_functions,
        rng=rng_train,
        max_pairs=args.max_same_function_train_pairs,
        positives_per_function=args.positives_per_function,
        max_block_instructions=args.max_block_instructions,
    )
    same_function_val = build_same_function_pairs(
        val_functions,
        rng=rng_val,
        max_pairs=args.max_same_function_val_pairs,
        positives_per_function=args.positives_per_function,
        max_block_instructions=args.max_block_instructions,
    )

    out_dir = Path(args.output_dir)
    write_jsonl(str(out_dir / "same_bb_train.jsonl"), same_bb_train)
    write_jsonl(str(out_dir / "same_bb_val.jsonl"), same_bb_val)
    write_jsonl(str(out_dir / "same_function_train.jsonl"), same_function_train)
    write_jsonl(str(out_dir / "same_function_val.jsonl"), same_function_val)

    summary = {
        "records_root": args.records_root,
        "exclude_function_blocks": args.exclude_function_blocks,
        "excluded_signatures": len(excluded_signatures),
        "num_functions_loaded": len(functions),
        "num_train_functions": len(train_functions),
        "num_val_functions": len(val_functions),
        "same_bb_train_pairs": len(same_bb_train),
        "same_bb_val_pairs": len(same_bb_val),
        "same_function_train_pairs": len(same_function_train),
        "same_function_val_pairs": len(same_function_val),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
