#!/usr/bin/env python3
"""
Frozen-encoder probing for PalmTree pretrain checkpoints.

Supported tasks:
  - same_bb:          do two instructions belong to the same basic block?
  - same_function:    do two basic blocks belong to the same function?
  - jump_target:      does this jump point to this candidate instruction?
  - call_target:      does this call point to this candidate callee?
  - same_dref:        do two instructions reference the same data address?
  - same_stack_offset: do two instructions reference the same stack offset?
  - dref_retrieval:   can the model retrieve the matching data reference?

The script compares a baseline non-address-aware checkpoint against an
address-aware checkpoint using:
  - zero-shot cosine scoring
  - few-shot linear probes on frozen embeddings

Notes:
  - Use a sampled function-block file such as pool_function_blocks_100.json,
    pool_function_blocks_1k.json, or pool_function_blocks_10k.json.
  - The checkpoint must be a pretrain BERT backbone checkpoint such as
    best_bert.pt / latest_bert.pt.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
import re
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from hashlib import sha1
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from model import AddressAwareBERT  # noqa: E402
from vocab import WordVocab  # noqa: E402


HEADER_RE = re.compile(r"^(\w+)\((0x[0-9a-fA-F]+):([0-9.]+):([0-9.]+):([0-9.]+)\)")
NESTED_ADDR_RE = re.compile(r"address\((0x[0-9a-fA-F]+):([0-9.]+):([0-9.]+):([0-9.]+)\)")
DADDR_RE = re.compile(r"((?:daddr(?:_[a-z]+)?))\((0x[0-9a-fA-F]+):([0-9.]+):([0-9.]+)(?::([0-9.]+))?\)")
VAR_RE = re.compile(r"var\((0x[0-9a-fA-F]+)\)")
JUMP_TARGET_RE = re.compile(r"^(j\w+)\([^)]*\)\s+address\((0x[0-9a-fA-F]+):")
CALL_TARGET_RE = re.compile(r"^(callq?|bl)\([^)]*\)\s+address\((0x[0-9a-fA-F]+):")


def parse_section_kind(raw_value: str) -> int:
    try:
        parsed = int(float(raw_value))
    except (TypeError, ValueError):
        return 4
    return parsed if parsed in (1, 2, 3, 4) else 4


def normalize_var_offset(raw_hex: str) -> int:
    raw = int(raw_hex, 16)
    if raw >= (1 << 31):
        raw -= (1 << 32)
    return max(-(1 << 20), min((1 << 20), raw))


def canonicalize_daddr_token(token_name: str, legacy_kind: Optional[str] = None) -> Tuple[str, int]:
    mapping = {
        "daddr_rodata": 1,
        "daddr_bss": 2,
        "daddr_data": 3,
        "daddr_other": 4,
    }
    if token_name in mapping:
        return token_name, mapping[token_name]
    kind_id = parse_section_kind(legacy_kind)
    canonical = {
        1: "daddr_rodata",
        2: "daddr_bss",
        3: "daddr_data",
        4: "daddr_other",
    }[kind_id]
    return canonical, kind_id


@dataclass
class InstructionRecord:
    unit_id: str
    func_id: str
    block_id: str
    instruction_index: int
    address: str
    opcode: str
    binary_pos: float
    function_pos: float
    bb_pos: float
    text: str


@dataclass
class BasicBlockRecord:
    unit_id: str
    func_id: str
    block_index: int
    function_pos: float
    entry_address: str
    entry_opcode: str
    instructions: List[InstructionRecord]


@dataclass
class ProbeUnit:
    unit_id: str
    kind: str
    instructions: List[str]


@dataclass
class ProbePair:
    left_id: str
    right_id: str
    label: int
    group_id: str
    metadata: Dict[str, object]


@dataclass
class RetrievalQuery:
    query_id: str
    candidate_ids: List[str]
    positive_id: str
    group_id: str
    metadata: Dict[str, object]


@dataclass
class EncodedUnit:
    pooled: torch.Tensor
    opcode: torch.Tensor
    address_token: Optional[torch.Tensor]


class UnionFind:
    def __init__(self):
        self.parent: Dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def make_binary_unit_id(binary_name: Optional[str], opt_level: Optional[str], fallback: str) -> str:
    if binary_name and opt_level:
        return f"{binary_name}::{opt_level}"
    if binary_name:
        return str(binary_name)
    return fallback


def load_function_blocks_with_metadata(path: str) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    payload = load_json(path)
    if isinstance(payload, dict):
        return {str(k): list(v) for k, v in payload.items()}, {}

    if isinstance(payload, list):
        function_blocks: Dict[str, List[str]] = {}
        binary_by_func: Dict[str, str] = {}
        for idx, record in enumerate(payload):
            if not isinstance(record, dict):
                continue
            func_id = str(record.get("func_id", idx))
            funcins = record.get("funcins")
            if isinstance(funcins, str):
                instructions = [inst for inst in funcins.split("\t") if inst]
            else:
                instructions = list(record.get("instructions", []))
            function_blocks[func_id] = instructions
            binary_by_func[func_id] = make_binary_unit_id(
                record.get("binary"),
                record.get("opt_level"),
                fallback=f"inline::{os.path.basename(path)}",
            )
        return function_blocks, binary_by_func

    raise ValueError(f"Unsupported function_blocks format in {path}: {type(payload).__name__}")


def load_function_metadata(path: Optional[str]) -> Dict[str, str]:
    if not path:
        return {}

    payload = load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported function metadata format in {path}: {type(payload).__name__}")

    metadata: Dict[str, str] = {}
    for key, value in payload.items():
        if isinstance(value, dict):
            binary_id = value.get("binary_id") or value.get("binary") or value.get("binary_uid")
        else:
            binary_id = value
        if binary_id:
            metadata[str(key)] = str(binary_id)
    return metadata


def infer_binary_metadata_from_records_dir(
    function_blocks: Dict[str, List[str]],
    records_dir: Optional[str],
) -> Dict[str, str]:
    if not records_dir:
        return {}

    root = Path(records_dir)
    if not root.exists():
        return {}

    sig_to_func_ids: Dict[str, List[str]] = defaultdict(list)
    for func_id, instructions in function_blocks.items():
        signature = sha1("\t".join(instructions).encode("utf-8")).hexdigest()
        sig_to_func_ids[signature].append(func_id)

    matches: Dict[str, set] = defaultdict(set)
    for path in root.glob("*_functions.json"):
        try:
            payload = load_json(str(path))
        except Exception:
            continue
        if not isinstance(payload, list):
            continue

        fallback_binary = path.name[: -len("_functions.json")] if path.name.endswith("_functions.json") else path.stem
        for record in payload:
            if not isinstance(record, dict):
                continue
            funcins = record.get("funcins")
            if not isinstance(funcins, str):
                continue
            signature = sha1(funcins.encode("utf-8")).hexdigest()
            if signature not in sig_to_func_ids:
                continue
            binary_id = make_binary_unit_id(
                record.get("binary"),
                record.get("opt_level"),
                fallback=fallback_binary,
            )
            for func_id in sig_to_func_ids[signature]:
                matches[func_id].add(binary_id)

    binary_by_func: Dict[str, str] = {}
    for func_id, candidates in matches.items():
        if len(candidates) == 1:
            binary_by_func[func_id] = next(iter(candidates))
    return binary_by_func


def parse_instruction_header(inst_text: str) -> Optional[Tuple[str, str, float, float, float]]:
    m = HEADER_RE.match(inst_text)
    if not m:
        return None
    return (
        m.group(1),
        m.group(2),
        float(m.group(3)),
        float(m.group(4)),
        float(m.group(5)),
    )


def extract_single_daddr_address(inst_text: str) -> Optional[str]:
    matches = [m.group(2) for m in DADDR_RE.finditer(inst_text)]
    if len(matches) != 1:
        return None
    return matches[0]


def extract_single_var_offset(inst_text: str) -> Optional[int]:
    matches = [normalize_var_offset(m.group(1)) for m in VAR_RE.finditer(inst_text)]
    if len(matches) != 1:
        return None
    return matches[0]


def extract_call_target_address(inst_text: str) -> Optional[str]:
    m = CALL_TARGET_RE.match(inst_text)
    if not m:
        return None
    return m.group(2)


def build_equivalence_groups(funcsim_pairs_path: Optional[str]) -> Dict[str, int]:
    if not funcsim_pairs_path or not os.path.exists(funcsim_pairs_path):
        return {}
    pairs = load_json(funcsim_pairs_path)
    uf = UnionFind()
    for query_id, payload in pairs.items():
        for gt in payload.get("ground_truth", []):
            uf.union(str(query_id), str(gt))

    roots: Dict[str, int] = {}
    mapping: Dict[str, int] = {}
    for func_id in pairs.keys():
        root = uf.find(str(func_id))
        if root not in roots:
            roots[root] = len(roots)
        mapping[str(func_id)] = roots[root]
    return mapping


def reconstruct_blocks(function_blocks: Dict[str, List[str]]) -> Tuple[Dict[str, List[InstructionRecord]], Dict[str, List[BasicBlockRecord]]]:
    instructions_by_func: Dict[str, List[InstructionRecord]] = {}
    blocks_by_func: Dict[str, List[BasicBlockRecord]] = {}

    for func_id, insts in function_blocks.items():
        parsed_insts: List[InstructionRecord] = []
        for idx, inst_text in enumerate(insts):
            parsed = parse_instruction_header(inst_text)
            if not parsed:
                continue
            opcode, addr, bp, fp, bb = parsed
            parsed_insts.append(
                InstructionRecord(
                    unit_id=f"inst:{func_id}:{idx}",
                    func_id=func_id,
                    block_id="",
                    instruction_index=idx,
                    address=addr,
                    opcode=opcode,
                    binary_pos=bp,
                    function_pos=fp,
                    bb_pos=bb,
                    text=inst_text,
                )
            )

        instructions_by_func[func_id] = parsed_insts
        blocks: List[BasicBlockRecord] = []
        current: List[InstructionRecord] = []
        current_fp: Optional[float] = None
        block_idx = -1

        for inst in parsed_insts:
            if current_fp is None or not math.isclose(inst.function_pos, current_fp, abs_tol=1e-7):
                if current:
                    entry = current[0]
                    block = BasicBlockRecord(
                        unit_id=f"bb:{func_id}:{block_idx}",
                        func_id=func_id,
                        block_index=block_idx,
                        function_pos=current_fp if current_fp is not None else 0.0,
                        entry_address=entry.address,
                        entry_opcode=entry.opcode,
                        instructions=current,
                    )
                    for item in current:
                        item.block_id = block.unit_id
                    blocks.append(block)
                block_idx += 1
                current = [inst]
                current_fp = inst.function_pos
            else:
                current.append(inst)

        if current:
            entry = current[0]
            block = BasicBlockRecord(
                unit_id=f"bb:{func_id}:{block_idx}",
                func_id=func_id,
                block_index=block_idx,
                function_pos=current_fp if current_fp is not None else 0.0,
                entry_address=entry.address,
                entry_opcode=entry.opcode,
                instructions=current,
            )
            for item in current:
                item.block_id = block.unit_id
            blocks.append(block)

        blocks_by_func[func_id] = blocks

    return instructions_by_func, blocks_by_func


def choose_function_subset(function_ids: List[str], max_functions: Optional[int], seed: int) -> List[str]:
    ids = list(function_ids)
    ids.sort()
    if max_functions is None or max_functions >= len(ids):
        return ids
    rng = random.Random(seed)
    rng.shuffle(ids)
    return sorted(ids[:max_functions])


def split_function_ids(function_ids: Sequence[str], test_fraction: float, seed: int) -> Tuple[List[str], List[str]]:
    ids = list(function_ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    test_n = max(1, int(len(ids) * test_fraction)) if len(ids) > 1 else 0
    test_ids = sorted(ids[:test_n])
    train_ids = sorted(ids[test_n:])
    return train_ids, test_ids


def sample_unique_pairs(index_pairs: Sequence[Tuple[int, int]], sample_size: int, rng: random.Random) -> List[Tuple[int, int]]:
    pairs = list(index_pairs)
    if len(pairs) <= sample_size:
        return pairs
    return rng.sample(pairs, sample_size)


def build_same_bb_pairs(
    blocks_by_func: Dict[str, List[BasicBlockRecord]],
    func_ids: Sequence[str],
    max_pairs: int,
    seed: int,
) -> Tuple[List[ProbePair], Dict[str, ProbeUnit]]:
    rng = random.Random(seed)
    pairs: List[ProbePair] = []
    units: Dict[str, ProbeUnit] = {}

    eligible = []
    for func_id in func_ids:
        blocks = blocks_by_func.get(func_id, [])
        if sum(len(block.instructions) for block in blocks) >= 3 and any(len(block.instructions) >= 2 for block in blocks):
            eligible.append(func_id)

    rng.shuffle(eligible)
    for func_id in eligible:
        blocks = [b for b in blocks_by_func[func_id] if len(b.instructions) >= 1]
        positive_blocks = [b for b in blocks if len(b.instructions) >= 2]
        if not positive_blocks:
            continue

        for block in positive_blocks:
            candidates = list(itertools.combinations(range(len(block.instructions)), 2))
            for left_idx, right_idx in sample_unique_pairs(candidates, 4, rng):
                left = block.instructions[left_idx]
                right = block.instructions[right_idx]
                units[left.unit_id] = ProbeUnit(left.unit_id, "instruction", [left.text])
                units[right.unit_id] = ProbeUnit(right.unit_id, "instruction", [right.text])
                pairs.append(
                    ProbePair(
                        left_id=left.unit_id,
                        right_id=right.unit_id,
                        label=1,
                        group_id=f"same_bb:{block.unit_id}",
                        metadata={
                            "task": "same_bb",
                            "func_id": func_id,
                            "block_id": block.unit_id,
                            "left_opcode": left.opcode,
                            "right_opcode": right.opcode,
                        },
                    )
                )

                other_blocks = [b for b in blocks if b.unit_id != block.unit_id and b.instructions]
                if not other_blocks:
                    continue
                neg_block = rng.choice(other_blocks)
                neg_inst = rng.choice(neg_block.instructions)
                units[neg_inst.unit_id] = ProbeUnit(neg_inst.unit_id, "instruction", [neg_inst.text])
                pairs.append(
                    ProbePair(
                        left_id=left.unit_id,
                        right_id=neg_inst.unit_id,
                        label=0,
                        group_id=f"diff_bb:{func_id}:{left.unit_id}",
                        metadata={
                            "task": "same_bb",
                            "func_id": func_id,
                            "anchor_block_id": block.unit_id,
                            "negative_block_id": neg_block.unit_id,
                            "left_opcode": left.opcode,
                            "right_opcode": neg_inst.opcode,
                        },
                    )
                )

                if len(pairs) >= max_pairs:
                    return pairs[:max_pairs], units

    return pairs[:max_pairs], units


def build_same_function_pairs(
    blocks_by_func: Dict[str, List[BasicBlockRecord]],
    func_ids: Sequence[str],
    func_groups: Dict[str, int],
    max_pairs: int,
    seed: int,
) -> Tuple[List[ProbePair], Dict[str, ProbeUnit]]:
    rng = random.Random(seed)
    pairs: List[ProbePair] = []
    units: Dict[str, ProbeUnit] = {}

    eligible_funcs = [fid for fid in func_ids if len(blocks_by_func.get(fid, [])) >= 2]
    if len(eligible_funcs) < 2:
        return pairs, units

    block_pool = [block for fid in eligible_funcs for block in blocks_by_func[fid] if block.instructions]
    by_entry_opcode: Dict[str, List[BasicBlockRecord]] = {}
    for block in block_pool:
        by_entry_opcode.setdefault(block.entry_opcode, []).append(block)

    rng.shuffle(eligible_funcs)
    for func_id in eligible_funcs:
        blocks = [b for b in blocks_by_func[func_id] if b.instructions]
        if len(blocks) < 2:
            continue
        block_pairs = list(itertools.combinations(range(len(blocks)), 2))
        for left_idx, right_idx in sample_unique_pairs(block_pairs, 4, rng):
            left = blocks[left_idx]
            right = blocks[right_idx]
            units[left.unit_id] = ProbeUnit(left.unit_id, "basic_block", [x.text for x in left.instructions])
            units[right.unit_id] = ProbeUnit(right.unit_id, "basic_block", [x.text for x in right.instructions])
            pairs.append(
                ProbePair(
                    left_id=left.unit_id,
                    right_id=right.unit_id,
                    label=1,
                    group_id=f"same_function:{func_id}",
                    metadata={
                        "task": "same_function",
                        "func_id": func_id,
                        "left_entry_opcode": left.entry_opcode,
                        "right_entry_opcode": right.entry_opcode,
                    },
                )
            )

            negatives = [
                cand
                for cand in by_entry_opcode.get(left.entry_opcode, [])
                if cand.func_id != func_id
                and func_groups.get(cand.func_id, -1) != func_groups.get(func_id, -2)
            ]
            if not negatives:
                negatives = [
                    cand
                    for cand in block_pool
                    if cand.func_id != func_id
                    and func_groups.get(cand.func_id, -1) != func_groups.get(func_id, -2)
                ]
            if not negatives:
                continue
            neg = rng.choice(negatives)
            units[neg.unit_id] = ProbeUnit(neg.unit_id, "basic_block", [x.text for x in neg.instructions])
            pairs.append(
                ProbePair(
                    left_id=left.unit_id,
                    right_id=neg.unit_id,
                    label=0,
                    group_id=f"diff_function:{func_id}:{left.unit_id}",
                    metadata={
                        "task": "same_function",
                        "func_id": func_id,
                        "negative_func_id": neg.func_id,
                        "left_entry_opcode": left.entry_opcode,
                        "right_entry_opcode": neg.entry_opcode,
                    },
                )
            )

            if len(pairs) >= max_pairs:
                return pairs[:max_pairs], units

    return pairs[:max_pairs], units


def build_jump_target_pairs(
    instructions_by_func: Dict[str, List[InstructionRecord]],
    func_ids: Sequence[str],
    max_pairs: int,
    seed: int,
    target_opcode: str,
) -> Tuple[List[ProbePair], Dict[str, ProbeUnit]]:
    rng = random.Random(seed)
    pairs: List[ProbePair] = []
    units: Dict[str, ProbeUnit] = {}

    for func_id in func_ids:
        instructions = instructions_by_func.get(func_id, [])
        if len(instructions) < 3:
            continue
        addr_map = {inst.address: inst for inst in instructions}

        for jump_inst in instructions:
            jm = JUMP_TARGET_RE.match(jump_inst.text)
            if not jm:
                continue
            target_addr = jm.group(2)
            target_inst = addr_map.get(target_addr)
            if target_inst is None:
                continue
            if target_opcode and target_inst.opcode != target_opcode:
                continue

            candidates = [
                inst
                for inst in instructions
                if inst.address != target_addr and inst.opcode == target_inst.opcode
            ]
            if not candidates:
                candidates = [inst for inst in instructions if inst.address != target_addr]
            if not candidates:
                continue
            negative = rng.choice(candidates)

            units[jump_inst.unit_id] = ProbeUnit(jump_inst.unit_id, "instruction", [jump_inst.text])
            units[target_inst.unit_id] = ProbeUnit(target_inst.unit_id, "instruction", [target_inst.text])
            units[negative.unit_id] = ProbeUnit(negative.unit_id, "instruction", [negative.text])

            pairs.append(
                ProbePair(
                    left_id=jump_inst.unit_id,
                    right_id=target_inst.unit_id,
                    label=1,
                    group_id=f"jump_target:{jump_inst.unit_id}",
                    metadata={
                        "task": "jump_target",
                        "func_id": func_id,
                        "jump_opcode": jump_inst.opcode,
                        "target_opcode": target_inst.opcode,
                        "target_addr": target_addr,
                    },
                )
            )
            pairs.append(
                ProbePair(
                    left_id=jump_inst.unit_id,
                    right_id=negative.unit_id,
                    label=0,
                    group_id=f"jump_target_neg:{jump_inst.unit_id}",
                    metadata={
                        "task": "jump_target",
                        "func_id": func_id,
                        "jump_opcode": jump_inst.opcode,
                        "target_opcode": target_inst.opcode,
                        "negative_addr": negative.address,
                    },
                )
            )

            if len(pairs) >= max_pairs:
                return pairs[:max_pairs], units

    return pairs[:max_pairs], units


def build_call_target_pairs(
    instructions_by_func: Dict[str, List[InstructionRecord]],
    blocks_by_func: Dict[str, List[BasicBlockRecord]],
    func_ids: Sequence[str],
    binary_by_func: Dict[str, str],
    max_pairs: int,
    seed: int,
) -> Tuple[List[ProbePair], Dict[str, ProbeUnit]]:
    rng = random.Random(seed)
    pairs: List[ProbePair] = []
    units: Dict[str, ProbeUnit] = {}

    entry_block_by_func: Dict[str, BasicBlockRecord] = {}
    entry_block_by_addr: Dict[str, BasicBlockRecord] = {}
    funcs_by_binary: Dict[str, List[str]] = defaultdict(list)

    for func_id in func_ids:
        blocks = [block for block in blocks_by_func.get(func_id, []) if block.instructions]
        if not blocks:
            continue
        entry = min(blocks, key=lambda block: block.block_index)
        entry_block_by_func[func_id] = entry
        entry_block_by_addr[entry.entry_address] = entry
        binary_id = binary_by_func.get(func_id)
        if binary_id:
            funcs_by_binary[binary_id].append(func_id)

    for func_id in func_ids:
        binary_id = binary_by_func.get(func_id)
        if not binary_id:
            continue

        for call_inst in instructions_by_func.get(func_id, []):
            target_addr = extract_call_target_address(call_inst.text)
            if not target_addr:
                continue

            callee_block = entry_block_by_addr.get(target_addr)
            if callee_block is None:
                continue
            if binary_by_func.get(callee_block.func_id) != binary_id:
                continue

            negatives = [
                entry_block_by_func[cand_func_id]
                for cand_func_id in funcs_by_binary.get(binary_id, [])
                if cand_func_id in entry_block_by_func
                and cand_func_id != callee_block.func_id
            ]
            if not negatives:
                continue
            matched_opcode = [block for block in negatives if block.entry_opcode == callee_block.entry_opcode]
            negative = rng.choice(matched_opcode or negatives)

            units[call_inst.unit_id] = ProbeUnit(call_inst.unit_id, "instruction", [call_inst.text])
            units[callee_block.unit_id] = ProbeUnit(
                callee_block.unit_id,
                "basic_block",
                [item.text for item in callee_block.instructions],
            )
            units[negative.unit_id] = ProbeUnit(
                negative.unit_id,
                "basic_block",
                [item.text for item in negative.instructions],
            )

            pairs.append(
                ProbePair(
                    left_id=call_inst.unit_id,
                    right_id=callee_block.unit_id,
                    label=1,
                    group_id=f"call_target:{call_inst.unit_id}",
                    metadata={
                        "task": "call_target",
                        "func_id": func_id,
                        "binary_id": binary_id,
                        "call_opcode": call_inst.opcode,
                        "callee_func_id": callee_block.func_id,
                        "callee_entry_addr": callee_block.entry_address,
                    },
                )
            )
            pairs.append(
                ProbePair(
                    left_id=call_inst.unit_id,
                    right_id=negative.unit_id,
                    label=0,
                    group_id=f"call_target_neg:{call_inst.unit_id}",
                    metadata={
                        "task": "call_target",
                        "func_id": func_id,
                        "binary_id": binary_id,
                        "call_opcode": call_inst.opcode,
                        "negative_func_id": negative.func_id,
                        "negative_entry_addr": negative.entry_address,
                    },
                )
            )

            if len(pairs) >= max_pairs:
                return pairs[:max_pairs], units

    return pairs[:max_pairs], units


def collect_data_reference_instructions(
    instructions_by_func: Dict[str, List[InstructionRecord]],
    func_ids: Sequence[str],
    binary_by_func: Dict[str, str],
):
    by_binary: Dict[str, List[Tuple[InstructionRecord, str]]] = defaultdict(list)
    by_binary_addr: Dict[Tuple[str, str], List[InstructionRecord]] = defaultdict(list)

    for func_id in func_ids:
        binary_id = binary_by_func.get(func_id)
        if not binary_id:
            continue
        for inst in instructions_by_func.get(func_id, []):
            daddr = extract_single_daddr_address(inst.text)
            if not daddr:
                continue
            by_binary[binary_id].append((inst, daddr))
            by_binary_addr[(binary_id, daddr)].append(inst)

    return by_binary, by_binary_addr


def build_same_dref_pairs(
    instructions_by_func: Dict[str, List[InstructionRecord]],
    func_ids: Sequence[str],
    binary_by_func: Dict[str, str],
    max_pairs: int,
    seed: int,
) -> Tuple[List[ProbePair], Dict[str, ProbeUnit]]:
    rng = random.Random(seed)
    pairs: List[ProbePair] = []
    units: Dict[str, ProbeUnit] = {}

    by_binary, by_binary_addr = collect_data_reference_instructions(
        instructions_by_func,
        func_ids,
        binary_by_func,
    )

    groups = list(by_binary_addr.items())
    rng.shuffle(groups)
    for (binary_id, daddr), insts in groups:
        if len(insts) < 2:
            continue

        combos = list(itertools.combinations(range(len(insts)), 2))
        for left_idx, right_idx in sample_unique_pairs(combos, 4, rng):
            left = insts[left_idx]
            right = insts[right_idx]
            negatives = [
                cand
                for cand, cand_daddr in by_binary.get(binary_id, [])
                if cand_daddr != daddr and cand.unit_id not in {left.unit_id, right.unit_id}
            ]
            if not negatives:
                continue
            matched_opcode = [cand for cand in negatives if cand.opcode == right.opcode]
            negative = rng.choice(matched_opcode or negatives)

            units[left.unit_id] = ProbeUnit(left.unit_id, "instruction", [left.text])
            units[right.unit_id] = ProbeUnit(right.unit_id, "instruction", [right.text])
            units[negative.unit_id] = ProbeUnit(negative.unit_id, "instruction", [negative.text])

            pairs.append(
                ProbePair(
                    left_id=left.unit_id,
                    right_id=right.unit_id,
                    label=1,
                    group_id=f"same_dref:{binary_id}:{daddr}",
                    metadata={
                        "task": "same_dref",
                        "binary_id": binary_id,
                        "data_addr": daddr,
                        "left_opcode": left.opcode,
                        "right_opcode": right.opcode,
                    },
                )
            )
            pairs.append(
                ProbePair(
                    left_id=left.unit_id,
                    right_id=negative.unit_id,
                    label=0,
                    group_id=f"diff_dref:{binary_id}:{left.unit_id}",
                    metadata={
                        "task": "same_dref",
                        "binary_id": binary_id,
                        "anchor_data_addr": daddr,
                        "negative_data_addr": extract_single_daddr_address(negative.text),
                        "left_opcode": left.opcode,
                        "right_opcode": negative.opcode,
                    },
                )
            )

            if len(pairs) >= max_pairs:
                return pairs[:max_pairs], units

    return pairs[:max_pairs], units


def collect_stack_reference_instructions(
    instructions_by_func: Dict[str, List[InstructionRecord]],
    func_ids: Sequence[str],
):
    by_func: Dict[str, List[Tuple[InstructionRecord, int]]] = defaultdict(list)
    by_func_offset: Dict[Tuple[str, int], List[InstructionRecord]] = defaultdict(list)

    for func_id in func_ids:
        for inst in instructions_by_func.get(func_id, []):
            offset = extract_single_var_offset(inst.text)
            if offset is None:
                continue
            by_func[func_id].append((inst, offset))
            by_func_offset[(func_id, offset)].append(inst)

    return by_func, by_func_offset


def build_same_stack_offset_pairs(
    instructions_by_func: Dict[str, List[InstructionRecord]],
    func_ids: Sequence[str],
    max_pairs: int,
    seed: int,
) -> Tuple[List[ProbePair], Dict[str, ProbeUnit]]:
    rng = random.Random(seed)
    pairs: List[ProbePair] = []
    units: Dict[str, ProbeUnit] = {}

    by_func, by_func_offset = collect_stack_reference_instructions(
        instructions_by_func,
        func_ids,
    )

    groups = list(by_func_offset.items())
    rng.shuffle(groups)
    for (func_id, offset), insts in groups:
        if len(insts) < 2:
            continue

        combos = list(itertools.combinations(range(len(insts)), 2))
        for left_idx, right_idx in sample_unique_pairs(combos, 4, rng):
            left = insts[left_idx]
            right = insts[right_idx]
            negatives = [
                cand
                for cand, cand_offset in by_func.get(func_id, [])
                if cand_offset != offset and cand.unit_id not in {left.unit_id, right.unit_id}
            ]
            if not negatives:
                continue
            matched_opcode = [cand for cand in negatives if cand.opcode == right.opcode]
            negative = rng.choice(matched_opcode or negatives)

            units[left.unit_id] = ProbeUnit(left.unit_id, "instruction", [left.text])
            units[right.unit_id] = ProbeUnit(right.unit_id, "instruction", [right.text])
            units[negative.unit_id] = ProbeUnit(negative.unit_id, "instruction", [negative.text])

            pairs.append(
                ProbePair(
                    left_id=left.unit_id,
                    right_id=right.unit_id,
                    label=1,
                    group_id=f"same_stack_offset:{func_id}:{offset}",
                    metadata={
                        "task": "same_stack_offset",
                        "func_id": func_id,
                        "stack_offset": offset,
                        "left_opcode": left.opcode,
                        "right_opcode": right.opcode,
                    },
                )
            )
            pairs.append(
                ProbePair(
                    left_id=left.unit_id,
                    right_id=negative.unit_id,
                    label=0,
                    group_id=f"diff_stack_offset:{func_id}:{left.unit_id}",
                    metadata={
                        "task": "same_stack_offset",
                        "func_id": func_id,
                        "anchor_stack_offset": offset,
                        "negative_stack_offset": extract_single_var_offset(negative.text),
                        "left_opcode": left.opcode,
                        "right_opcode": negative.opcode,
                    },
                )
            )

            if len(pairs) >= max_pairs:
                return pairs[:max_pairs], units

    return pairs[:max_pairs], units


def build_dref_retrieval_queries(
    instructions_by_func: Dict[str, List[InstructionRecord]],
    func_ids: Sequence[str],
    binary_by_func: Dict[str, str],
    max_queries: int,
    pool_size: int,
    seed: int,
) -> Tuple[List[RetrievalQuery], Dict[str, ProbeUnit]]:
    rng = random.Random(seed)
    queries: List[RetrievalQuery] = []
    units: Dict[str, ProbeUnit] = {}

    by_binary, by_binary_addr = collect_data_reference_instructions(
        instructions_by_func,
        func_ids,
        binary_by_func,
    )

    groups = list(by_binary_addr.items())
    rng.shuffle(groups)
    for (binary_id, daddr), insts in groups:
        if len(insts) < 2:
            continue

        negatives = [
            cand
            for cand, cand_daddr in by_binary.get(binary_id, [])
            if cand_daddr != daddr
        ]
        if len(negatives) < max(0, pool_size - 1):
            continue

        query_inst, positive_inst = rng.sample(insts, 2)
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
                group_id=f"dref_retrieval:{binary_id}:{daddr}:{query_inst.unit_id}",
                metadata={
                    "task": "dref_retrieval",
                    "binary_id": binary_id,
                    "data_addr": daddr,
                    "pool_size": pool_size,
                },
            )
        )
        if len(queries) >= max_queries:
            return queries, units

    return queries, units


class FrozenBackboneEncoder:
    def __init__(
        self,
        checkpoint_path: str,
        vocab: WordVocab,
        hidden: int,
        n_layers: int,
        attn_heads: int,
        seq_len: int,
        max_instructions: int,
        device: torch.device,
    ):
        self.checkpoint_path = checkpoint_path
        self.vocab = vocab
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads
        self.seq_len = seq_len
        self.max_instructions = max_instructions
        self.device = device

        self.pad_idx = vocab.stoi.get("<pad>", 0)
        self.unk_idx = vocab.stoi.get("<unk>", 1)
        self.eos_idx = vocab.stoi.get("<eos>", 2)
        self.sos_idx = vocab.stoi.get("<sos>", 3)

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict):
            sd = ckpt.get("bert_state_dict", ckpt.get("model_state_dict", ckpt))
        else:
            sd = ckpt.state_dict()
        if any(k.startswith("module.") for k in sd):
            sd = {k[len("module."):]: v for k, v in sd.items()}
        if any(k.startswith("bert.") for k in sd):
            sd = {k[len("bert."):]: v for k, v in sd.items() if k.startswith("bert.")}

        has_address = any("address_pos" in k or "address_position" in k for k in sd)
        has_var = any("var_offset" in k or "var_position" in k for k in sd)
        self.use_addr = has_address and has_var

        self.model = AddressAwareBERT(
            vocab_size=len(vocab),
            hidden=hidden,
            n_layers=n_layers,
            attn_heads=attn_heads,
            max_len=seq_len,
            use_address_embedding=self.use_addr,
            use_var_embedding=self.use_addr,
        )
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        if missing:
            print(f"[Probe] Warning: missing keys when loading {checkpoint_path}: {missing[:5]}")
        if unexpected:
            print(f"[Probe] Warning: unexpected keys when loading {checkpoint_path}: {unexpected[:5]}")

        daddr_token_ids = [
            token_id
            for token, token_id in vocab.stoi.items()
            if token == "daddr" or token.startswith("daddr_")
        ]
        if daddr_token_ids and self.use_addr:
            self.model.embedding.set_special_token_ids(sorted(daddr_token_ids))

        self.model.eval()
        self.model.to(device)

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
                raw = normalize_var_offset(vm.group(1))
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

    def _build_sequence(self, instructions: Sequence[str]):
        token_ids = [self.sos_idx]
        segment_labels = [1]
        binary_pos = [-1.0]
        function_pos = [-1.0]
        bb_pos = [-1.0]
        var_offsets = [-999999]
        section_kind_ids = [0]
        addr_type_ids = [0]
        token_texts = ["<sos>"]

        opcode_index: Optional[int] = None
        address_index: Optional[int] = None

        for inst_idx, inst in enumerate(instructions[: self.max_instructions]):
            seg_id = min(inst_idx + 1, 15)
            toks, bp, fp, bbp, var, sk, at = self._parse_instruction(inst)
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
                var_offsets.append(var[local_idx])
                section_kind_ids.append(sk[local_idx])
                addr_type_ids.append(at[local_idx])
                token_texts.append(tok)
            token_ids.append(self.eos_idx)
            segment_labels.append(seg_id)
            binary_pos.append(-1.0)
            function_pos.append(-1.0)
            bb_pos.append(-1.0)
            var_offsets.append(-999999)
            section_kind_ids.append(0)
            addr_type_ids.append(0)
            token_texts.append("<eos>")

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
        if address_index is not None and address_index >= self.seq_len:
            address_index = None

        return {
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
            "segment_labels": torch.tensor(segment_labels, dtype=torch.long),
            "binary_pos": torch.tensor(binary_pos, dtype=torch.float),
            "function_pos": torch.tensor(function_pos, dtype=torch.float),
            "bb_pos": torch.tensor(bb_pos, dtype=torch.float),
            "var_offsets": torch.tensor(var_offsets, dtype=torch.long),
            "section_kind_ids": torch.tensor(section_kind_ids, dtype=torch.long),
            "addr_type_ids": torch.tensor(addr_type_ids, dtype=torch.long),
            "opcode_index": opcode_index,
            "address_index": address_index,
        }

    @torch.no_grad()
    def encode_units(self, units: Dict[str, ProbeUnit], batch_size: int) -> Dict[str, EncodedUnit]:
        unit_ids = sorted(units.keys())
        built = {uid: self._build_sequence(units[uid].instructions) for uid in unit_ids}
        outputs: Dict[str, EncodedUnit] = {}

        for start in range(0, len(unit_ids), batch_size):
            batch_ids = unit_ids[start : start + batch_size]
            ids = torch.stack([built[uid]["input_ids"] for uid in batch_ids]).to(self.device)
            seg = torch.stack([built[uid]["segment_labels"] for uid in batch_ids]).to(self.device)
            bp = torch.stack([built[uid]["binary_pos"] for uid in batch_ids]).to(self.device)
            fp = torch.stack([built[uid]["function_pos"] for uid in batch_ids]).to(self.device)
            bb = torch.stack([built[uid]["bb_pos"] for uid in batch_ids]).to(self.device)
            vo = torch.stack([built[uid]["var_offsets"] for uid in batch_ids]).to(self.device)
            sk = torch.stack([built[uid]["section_kind_ids"] for uid in batch_ids]).to(self.device)
            at = torch.stack([built[uid]["addr_type_ids"] for uid in batch_ids]).to(self.device)

            hidden = self.model(
                ids, seg, bp, fp, bb, vo,
                section_kind_ids=sk,
                addr_type_ids=at,
            )

            for row, uid in enumerate(batch_ids):
                spec = built[uid]
                seq_ids = spec["input_ids"]
                valid_mask = (seq_ids != self.pad_idx) & (seq_ids != self.sos_idx) & (seq_ids != self.eos_idx)
                if not bool(valid_mask.any()):
                    valid_mask = seq_ids != self.pad_idx
                pooled = hidden[row][valid_mask.to(self.device)].mean(dim=0).detach().cpu()
                opcode = hidden[row, spec["opcode_index"]].detach().cpu()
                address_token = None
                if spec["address_index"] is not None:
                    address_token = hidden[row, spec["address_index"]].detach().cpu()
                outputs[uid] = EncodedUnit(pooled=pooled, opcode=opcode, address_token=address_token)

        return outputs


def cosine_score(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())


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


def balanced_accuracy(pred: Sequence[int], labels: Sequence[int]) -> float:
    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(pred, dtype=np.int64)
    pos = y == 1
    neg = y == 0
    tpr = float((p[pos] == 1).mean()) if pos.any() else 0.0
    tnr = float((p[neg] == 0).mean()) if neg.any() else 0.0
    return 0.5 * (tpr + tnr)


def f1_score(pred: Sequence[int], labels: Sequence[int]) -> float:
    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(pred, dtype=np.int64)
    tp = int(((p == 1) & (y == 1)).sum())
    fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum())
    denom = 2 * tp + fp + fn
    return 0.0 if denom == 0 else (2.0 * tp) / denom


def compute_metrics(scores: Sequence[float], labels: Sequence[int], threshold: float) -> Dict[str, float]:
    pred = [1 if s >= threshold else 0 for s in scores]
    acc = float(np.mean(np.asarray(pred) == np.asarray(labels)))
    return {
        "auc": binary_auc(scores, labels),
        "accuracy": acc,
        "balanced_accuracy": balanced_accuracy(pred, labels),
        "f1": f1_score(pred, labels),
        "threshold": float(threshold),
    }


def choose_threshold(scores: Sequence[float], labels: Sequence[int]) -> float:
    unique = sorted(set(float(x) for x in scores))
    if not unique:
        return 0.0
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


def select_pair_embeddings(task: str, left: EncodedUnit, right: EncodedUnit) -> Tuple[torch.Tensor, torch.Tensor]:
    if task == "jump_target":
        return (
            left.address_token if left.address_token is not None else left.pooled,
            right.opcode,
        )
    if task == "call_target":
        return (
            left.address_token if left.address_token is not None else left.pooled,
            right.pooled,
        )
    return left.pooled, right.pooled


def build_zero_shot_scores(task: str, pairs: Sequence[ProbePair], embeddings: Dict[str, EncodedUnit]) -> List[float]:
    scores: List[float] = []
    for pair in pairs:
        left = embeddings[pair.left_id]
        right = embeddings[pair.right_id]
        anchor, candidate = select_pair_embeddings(task, left, right)
        scores.append(cosine_score(anchor, candidate))
    return scores


def pair_features(task: str, pairs: Sequence[ProbePair], embeddings: Dict[str, EncodedUnit]) -> torch.Tensor:
    feats = []
    for pair in pairs:
        left = embeddings[pair.left_id]
        right = embeddings[pair.right_id]
        a, b = select_pair_embeddings(task, left, right)
        feats.append(torch.cat([a, b, torch.abs(a - b), a * b], dim=0))
    return torch.stack(feats, dim=0)


def compute_retrieval_metrics(
    queries: Sequence[RetrievalQuery],
    embeddings: Dict[str, EncodedUnit],
) -> Dict[str, float]:
    if not queries:
        return {}

    top1_hits = 0
    recall5_hits = 0
    mrr_total = 0.0

    for query in queries:
        query_vec = embeddings[query.query_id].pooled
        scored = []
        for candidate_id in query.candidate_ids:
            candidate_vec = embeddings[candidate_id].pooled
            scored.append((candidate_id, cosine_score(query_vec, candidate_vec)))
        scored.sort(key=lambda item: item[1], reverse=True)

        ranked_ids = [candidate_id for candidate_id, _ in scored]
        rank = ranked_ids.index(query.positive_id) + 1
        top1_hits += int(rank == 1)
        recall5_hits += int(rank <= 5)
        mrr_total += 1.0 / rank

    num_queries = len(queries)
    return {
        "top1_accuracy": top1_hits / num_queries,
        "recall_at_5": recall5_hits / num_queries,
        "mrr": mrr_total / num_queries,
        "num_queries": float(num_queries),
    }


class LinearProbe(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


def stratified_sample(pairs: Sequence[ProbePair], per_class: int, seed: int) -> List[ProbePair]:
    rng = random.Random(seed)
    pos = [pair for pair in pairs if pair.label == 1]
    neg = [pair for pair in pairs if pair.label == 0]
    if len(pos) < per_class or len(neg) < per_class:
        return []
    return rng.sample(pos, per_class) + rng.sample(neg, per_class)


def train_few_shot_probe(
    task: str,
    train_pairs: Sequence[ProbePair],
    test_pairs: Sequence[ProbePair],
    embeddings: Dict[str, EncodedUnit],
    shots_per_class: int,
    seed: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
) -> Optional[Dict[str, float]]:
    sampled = stratified_sample(train_pairs, shots_per_class, seed)
    if not sampled:
        return None

    x_train = pair_features(task, sampled, embeddings).to(device)
    y_train = torch.tensor([pair.label for pair in sampled], dtype=torch.float32, device=device)
    x_test = pair_features(task, test_pairs, embeddings).to(device)
    y_test = [pair.label for pair in test_pairs]

    model = LinearProbe(x_train.size(1)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    for _ in range(epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        logits = model(x_train)
        loss = criterion(logits, y_train)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        scores = torch.sigmoid(model(x_test)).detach().cpu().numpy().tolist()
    metrics = compute_metrics(scores, y_test, threshold=0.5)
    metrics["num_train_pairs"] = len(sampled)
    return metrics


def summarize_metric_dicts(metric_dicts: Sequence[Dict[str, float]]) -> Dict[str, float]:
    if not metric_dicts:
        return {}
    keys = sorted(metric_dicts[0].keys())
    out = {}
    for key in keys:
        vals = [m[key] for m in metric_dicts if key in m and isinstance(m[key], (int, float))]
        if not vals:
            continue
        out[f"{key}_mean"] = float(np.mean(vals))
        out[f"{key}_std"] = float(np.std(vals))
    return out


def ensure_balanced_pairs(pairs: Sequence[ProbePair]) -> List[ProbePair]:
    pos = [pair for pair in pairs if pair.label == 1]
    neg = [pair for pair in pairs if pair.label == 0]
    keep = min(len(pos), len(neg))
    return pos[:keep] + neg[:keep]


def run_task_for_model(
    model_name: str,
    encoder: FrozenBackboneEncoder,
    task: str,
    train_pairs: Sequence[ProbePair],
    test_pairs: Sequence[ProbePair],
    units: Dict[str, ProbeUnit],
    few_shots: Sequence[int],
    few_shot_seeds: Sequence[int],
    batch_size: int,
    probe_epochs: int,
    probe_lr: float,
    probe_weight_decay: float,
    device: torch.device,
) -> Dict[str, object]:
    embeddings = encoder.encode_units(units, batch_size=batch_size)
    train_scores = build_zero_shot_scores(task, train_pairs, embeddings)
    test_scores = build_zero_shot_scores(task, test_pairs, embeddings)
    train_labels = [pair.label for pair in train_pairs]
    test_labels = [pair.label for pair in test_pairs]
    threshold = choose_threshold(train_scores, train_labels)

    result: Dict[str, object] = {
        "model_name": model_name,
        "use_addr": encoder.use_addr,
        "zero_shot": compute_metrics(test_scores, test_labels, threshold),
        "few_shot": {},
    }

    for shots in few_shots:
        runs = []
        for seed in few_shot_seeds:
            metrics = train_few_shot_probe(
                task=task,
                train_pairs=train_pairs,
                test_pairs=test_pairs,
                embeddings=embeddings,
                shots_per_class=shots,
                seed=seed,
                epochs=probe_epochs,
                lr=probe_lr,
                weight_decay=probe_weight_decay,
                device=device,
            )
            if metrics is not None:
                runs.append(metrics)
        result["few_shot"][str(shots)] = {
            "runs": runs,
            "summary": summarize_metric_dicts(runs),
        }

    return result


def run_retrieval_task_for_model(
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
        "retrieval": compute_retrieval_metrics(queries, embeddings),
    }


def main():
    parser = argparse.ArgumentParser(description="PalmTree frozen-encoder probing")
    parser.add_argument("--function_blocks", required=True, help="Path to function_blocks or sampled pool_function_blocks JSON")
    parser.add_argument("--function_metadata", default=None, help="Optional JSON mapping func_id -> binary_id")
    parser.add_argument("--function_records_dir", default=None, help="Optional directory of per-binary *_functions.json files used to infer binary IDs")
    parser.add_argument("--funcsim_pairs", default=None, help="Optional path to funcsim_pairs.json to avoid same-source false negatives")
    parser.add_argument("--vocab", required=True, help="Path to vocab.pkl")
    parser.add_argument("--baseline_checkpoint", required=True, help="Non-address-aware pretrain backbone checkpoint")
    parser.add_argument("--aa_checkpoint", required=True, help="Address-aware pretrain backbone checkpoint")
    parser.add_argument("--output", required=True, help="Where to write the JSON report")
    parser.add_argument("--tasks", nargs="+", default=["same_bb", "same_function", "jump_target", "call_target", "same_dref", "same_stack_offset", "dref_retrieval"])
    parser.add_argument("--hidden", type=int, default=768)
    parser.add_argument("--n_layers", type=int, default=12)
    parser.add_argument("--attn_heads", type=int, default=12)
    parser.add_argument("--seq_len", type=int, default=20, help="Must match the pretrain checkpoint")
    parser.add_argument("--max_block_instructions", type=int, default=8, help="BB probe input truncation")
    parser.add_argument("--max_functions", type=int, default=None, help="Optional cap on functions loaded from the JSON")
    parser.add_argument("--max_train_pairs", type=int, default=2000)
    parser.add_argument("--max_test_pairs", type=int, default=1000)
    parser.add_argument("--max_train_queries", type=int, default=500)
    parser.add_argument("--max_test_queries", type=int, default=200)
    parser.add_argument("--retrieval_pool_size", type=int, default=32)
    parser.add_argument("--test_fraction", type=float, default=0.3)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--few_shots", nargs="+", type=int, default=[8, 32, 128])
    parser.add_argument("--few_shot_seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--probe_epochs", type=int, default=200)
    parser.add_argument("--probe_lr", type=float, default=1e-3)
    parser.add_argument("--probe_weight_decay", type=float, default=1e-4)
    parser.add_argument("--jump_target_opcode", default="mov", help="Keep only jump targets whose opcode matches this value; use '' for no filter")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    device = torch.device(args.device)

    print("[Probe] Loading data ...")
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

    for task in args.tasks:
        print(f"[Probe] Building task: {task}")
        if task == "same_bb":
            train_pairs, train_units = build_same_bb_pairs(
                blocks_by_func, train_func_ids, args.max_train_pairs, args.split_seed + 11
            )
            test_pairs, test_units = build_same_bb_pairs(
                blocks_by_func, test_func_ids, args.max_test_pairs, args.split_seed + 12
            )
        elif task == "same_function":
            train_pairs, train_units = build_same_function_pairs(
                blocks_by_func, train_func_ids, func_groups, args.max_train_pairs, args.split_seed + 21
            )
            test_pairs, test_units = build_same_function_pairs(
                blocks_by_func, test_func_ids, func_groups, args.max_test_pairs, args.split_seed + 22
            )
        elif task == "jump_target":
            train_pairs, train_units = build_jump_target_pairs(
                instructions_by_func, train_func_ids, args.max_train_pairs, args.split_seed + 31, args.jump_target_opcode
            )
            test_pairs, test_units = build_jump_target_pairs(
                instructions_by_func, test_func_ids, args.max_test_pairs, args.split_seed + 32, args.jump_target_opcode
            )
        elif task == "call_target":
            train_pairs, train_units = build_call_target_pairs(
                instructions_by_func, blocks_by_func, train_func_ids, binary_by_func, args.max_train_pairs, args.split_seed + 41
            )
            test_pairs, test_units = build_call_target_pairs(
                instructions_by_func, blocks_by_func, test_func_ids, binary_by_func, args.max_test_pairs, args.split_seed + 42
            )
        elif task == "same_dref":
            train_pairs, train_units = build_same_dref_pairs(
                instructions_by_func, train_func_ids, binary_by_func, args.max_train_pairs, args.split_seed + 51
            )
            test_pairs, test_units = build_same_dref_pairs(
                instructions_by_func, test_func_ids, binary_by_func, args.max_test_pairs, args.split_seed + 52
            )
        elif task == "same_stack_offset":
            train_pairs, train_units = build_same_stack_offset_pairs(
                instructions_by_func, train_func_ids, args.max_train_pairs, args.split_seed + 53
            )
            test_pairs, test_units = build_same_stack_offset_pairs(
                instructions_by_func, test_func_ids, args.max_test_pairs, args.split_seed + 54
            )
        elif task == "dref_retrieval":
            train_queries, train_units = build_dref_retrieval_queries(
                instructions_by_func,
                train_func_ids,
                binary_by_func,
                args.max_train_queries,
                args.retrieval_pool_size,
                args.split_seed + 61,
            )
            test_queries, test_units = build_dref_retrieval_queries(
                instructions_by_func,
                test_func_ids,
                binary_by_func,
                args.max_test_queries,
                args.retrieval_pool_size,
                args.split_seed + 62,
            )

            units = {}
            units.update(train_units)
            units.update(test_units)
            if not test_queries:
                report["tasks"][task] = {
                    "error": "Insufficient same-binary data-reference queries after task construction.",
                    "num_train_queries": len(train_queries),
                    "num_test_queries": len(test_queries),
                    "requires_binary_metadata": True,
                }
                continue

            task_report = {
                "num_units": len(units),
                "num_train_queries": len(train_queries),
                "num_test_queries": len(test_queries),
                "pool_size": args.retrieval_pool_size,
                "requires_binary_metadata": True,
                "models": {},
            }
            for model_name, checkpoint_path in model_specs:
                print(f"[Probe] Encoding/evaluating {task} with {model_name}: {checkpoint_path}")
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
                task_report["models"][model_name] = run_retrieval_task_for_model(
                    model_name=model_name,
                    encoder=encoder,
                    queries=test_queries,
                    units=units,
                    batch_size=args.batch_size,
                )

            report["tasks"][task] = task_report
            continue
        else:
            raise ValueError(f"Unsupported task: {task}")

        train_pairs = ensure_balanced_pairs(train_pairs)
        test_pairs = ensure_balanced_pairs(test_pairs)
        units = {}
        units.update(train_units)
        units.update(test_units)

        if not train_pairs or not test_pairs:
            report["tasks"][task] = {
                "error": "Insufficient pairs after task construction.",
                "num_train_pairs": len(train_pairs),
                "num_test_pairs": len(test_pairs),
            }
            continue

        task_report = {
            "num_units": len(units),
            "num_train_pairs": len(train_pairs),
            "num_test_pairs": len(test_pairs),
            "requires_binary_metadata": task in {"call_target", "same_dref"},
            "models": {},
        }

        for model_name, checkpoint_path in model_specs:
            print(f"[Probe] Encoding/evaluating {task} with {model_name}: {checkpoint_path}")
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
            task_report["models"][model_name] = run_task_for_model(
                model_name=model_name,
                encoder=encoder,
                task=task,
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

        report["tasks"][task] = task_report

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[Probe] Wrote report to {args.output}")


if __name__ == "__main__":
    main()
