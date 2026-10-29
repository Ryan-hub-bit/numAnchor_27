import argparse
import json
import os
import random
import re
import zipfile
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from dataset import write_corpus_from_function_blocks
from paths import ensure_parent_under_data_root, ensure_under_data_root


FILE_RE = re.compile(r"^(?P<base>.+)-(?P<opt>O0|O1|O2|O3|Os)-(?P<hash>[0-9a-f]+)\.json$")
INST_RE = re.compile(r"^\s*([0-9a-fA-F]+):\s*(.*)$")
HEX_RE = re.compile(r"\b(?:0x)?([0-9a-fA-F]+)\b")


def split_opcode_operands(disasm: str) -> Tuple[str, str]:
    if "\t" in disasm:
        parts = [p.strip() for p in disasm.split("\t") if p.strip()]
        disasm = parts[-1]
    disasm = disasm.strip()
    if not disasm:
        return "nop", ""
    if " " not in disasm:
        return disasm, ""
    opcode, operands = disasm.split(None, 1)
    return opcode, operands.strip()


def parse_instruction_lines(assembly: str) -> List[dict]:
    instructions = []
    for line in assembly.splitlines():
        m = INST_RE.match(line)
        if not m:
            continue
        addr = int(m.group(1), 16)
        body = m.group(2).strip()
        if not body:
            continue
        opcode, operands = split_opcode_operands(body)
        comment = None
        if "#" in operands:
            operands, comment = operands.split("#", 1)
            operands = operands.strip()
            comment = comment.strip()
        instructions.append(
            {
                "addr": addr,
                "opcode": opcode.strip(),
                "operands": operands.strip(),
                "comment": comment,
            }
        )
    return instructions


def is_control_transfer(opcode: str) -> bool:
    return opcode.startswith("j") or opcode.startswith("ret") or opcode.startswith("call")


def find_hex_target(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    # Prefer the first token before an objdump symbol annotation.
    if "<" in text:
        head = text.split("<", 1)[0].strip()
        if head:
            token = head.split()[0].strip(",")
            try:
                return int(token, 16)
            except ValueError:
                pass
    match = HEX_RE.search(text)
    if match:
        try:
            return int(match.group(1), 16)
        except ValueError:
            return None
    return None


def to_signed_32_hex(value: int) -> str:
    if value < 0:
        value = (1 << 32) + value
    return hex(value & 0xFFFFFFFF)


def detect_stack_var(operand: str) -> Optional[str]:
    if "%rbp" not in operand and "%rsp" not in operand and "rbp" not in operand and "rsp" not in operand:
        return None
    disp_match = re.search(r"([+-]?0x[0-9a-fA-F]+|[+-]?\d+)\((?:%?(?:rbp|rsp))\)", operand)
    if not disp_match:
        bare = re.search(r"\((?:%?(?:rbp|rsp))\)", operand)
        if bare:
            return "var(0x0)"
        return None
    raw = disp_match.group(1)
    if raw.lower().startswith("-0x"):
        value = -int(raw[3:], 16)
    elif raw.lower().startswith("0x"):
        value = int(raw, 16)
    else:
        value = int(raw, 10)
    return f"var({to_signed_32_hex(value)})"


def normalize_plain_operand(operand: str) -> List[str]:
    operand = operand.strip()
    if not operand:
        return []
    stack_var = detect_stack_var(operand)
    if stack_var is not None:
        return [stack_var]
    operand = operand.replace("%", "")
    operand = operand.replace("*", "")
    chunks = re.split(r"[\s,\[\]\(\)\+\-]+", operand)
    out = []
    for chunk in chunks:
        if not chunk:
            continue
        if HEX_RE.fullmatch(chunk):
            out.append("imm")
        else:
            out.append(chunk)
    return out or ["imm"]


def build_basic_blocks(instructions: List[dict]) -> Tuple[Dict[int, int], Dict[int, Tuple[int, int]]]:
    if not instructions:
        return {}, {}
    addr_to_index = {inst["addr"]: i for i, inst in enumerate(instructions)}
    leaders = {instructions[0]["addr"]}
    for i, inst in enumerate(instructions):
        target = None
        if inst["opcode"].startswith("j"):
            target = find_hex_target(inst["operands"])
            if target is not None and target in addr_to_index:
                leaders.add(target)
        if is_control_transfer(inst["opcode"]) and i + 1 < len(instructions):
            leaders.add(instructions[i + 1]["addr"])

    sorted_leaders = sorted(leaders)
    leader_to_range = {}
    addr_to_bb_start = {}
    for idx, leader in enumerate(sorted_leaders):
        start_idx = addr_to_index[leader]
        end_idx = addr_to_index[sorted_leaders[idx + 1]] if idx + 1 < len(sorted_leaders) else len(instructions)
        end_addr = instructions[end_idx - 1]["addr"]
        leader_to_range[leader] = (leader, end_addr)
        for inst in instructions[start_idx:end_idx]:
            addr_to_bb_start[inst["addr"]] = leader
    return addr_to_bb_start, leader_to_range


def fmt_pos(value: float) -> str:
    return f"{value:.8f}"


def annotate_function(
    func_name: str,
    func_record: dict,
    binary_min_addr: int,
    binary_max_addr: int,
    file_function_entries: Dict[int, str],
) -> Optional[List[str]]:
    assembly = func_record.get("assembly", "")
    func_addr = int(func_record.get("address", "0"), 16)
    instructions = parse_instruction_lines(assembly)
    if not instructions:
        return None

    addr_to_inst = {inst["addr"]: inst for inst in instructions}
    addr_to_bb_start, bb_ranges = build_basic_blocks(instructions)
    func_start = instructions[0]["addr"]
    func_end = instructions[-1]["addr"]
    func_span = max(1, func_end - func_start)
    binary_span = max(1, binary_max_addr - binary_min_addr)

    # Precompute target positions for instruction-local jumps.
    target_pos_map = {}
    for inst in instructions:
        addr = inst["addr"]
        bb_start = addr_to_bb_start.get(addr, func_start)
        bb_start_addr, bb_end_addr = bb_ranges.get(bb_start, (bb_start, bb_start))
        bb_span = max(1, bb_end_addr - bb_start_addr)
        target_pos_map[addr] = {
            "binary_pos": (func_addr - binary_min_addr) / float(binary_span),
            "function_pos": (bb_start - func_start) / float(func_span),
            "bb_pos": (addr - bb_start_addr) / float(bb_span),
        }

    annotated = []
    for inst in instructions:
        addr = inst["addr"]
        opcode = inst["opcode"]
        operands = inst["operands"]
        bb_start = addr_to_bb_start.get(addr, func_start)
        bb_start_addr, bb_end_addr = bb_ranges.get(bb_start, (bb_start, bb_start))
        bb_span = max(1, bb_end_addr - bb_start_addr)

        binary_pos = (func_addr - binary_min_addr) / float(binary_span)
        function_pos = (bb_start - func_start) / float(func_span)
        bb_pos = (addr - bb_start_addr) / float(bb_span)

        parts = [f"{opcode}({hex(addr)}:{fmt_pos(binary_pos)}:{fmt_pos(function_pos)}:{fmt_pos(bb_pos)})"]

        jump_or_call = opcode.startswith("j") or opcode.startswith("call")
        target_addr = find_hex_target(operands)
        used_special_target = False
        if jump_or_call and target_addr is not None:
            if target_addr in target_pos_map:
                tp = target_pos_map[target_addr]
                parts.append(
                    f"address({hex(target_addr)}:{fmt_pos(tp['binary_pos'])}:{fmt_pos(tp['function_pos'])}:{fmt_pos(tp['bb_pos'])})"
                )
                used_special_target = True
            elif target_addr in file_function_entries:
                target_func_addr = target_addr
                tp_bin = (target_func_addr - binary_min_addr) / float(binary_span)
                parts.append(f"address({hex(target_addr)}:{fmt_pos(tp_bin)}:0.00000000:0.00000000)")
                used_special_target = True

        if not used_special_target and operands:
            operand_tokens = []
            for raw_operand in [piece.strip() for piece in operands.split(",") if piece.strip()]:
                if "#" in raw_operand:
                    raw_operand = raw_operand.split("#", 1)[0].strip()
                comment_target = find_hex_target(inst.get("comment"))
                stack_var = detect_stack_var(raw_operand)
                if stack_var is not None:
                    operand_tokens.append(stack_var)
                    continue
                if comment_target is not None and comment_target not in file_function_entries:
                    operand_tokens.append(f"daddr({hex(comment_target)}:2.00000000:0.00000000:0.00000000)")
                    continue
                operand_tokens.extend(normalize_plain_operand(raw_operand))
            parts.extend(tok for tok in operand_tokens if tok)

        annotated.append(" ".join(parts))

    return annotated


def load_external_blocks_from_zip(zip_path: str, max_files: Optional[int] = None, max_functions: Optional[int] = None):
    function_blocks = {}
    equivalence_groups = defaultdict(list)

    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.endswith(".json")]
        names.sort()
        if max_files is not None:
            names = names[:max_files]

        for name in names:
            file_name = os.path.basename(name)
            match = FILE_RE.match(file_name)
            if not match:
                continue
            file_base = match.group("base")
            opt = match.group("opt")

            with zf.open(name) as f:
                raw = json.load(f)

            file_function_entries = {}
            for func_name, func_record in raw.items():
                try:
                    file_function_entries[int(func_record.get("address", "0"), 16)] = func_name
                except Exception:
                    continue

            if not file_function_entries:
                continue

            binary_min_addr = min(file_function_entries)
            binary_max_addr = max(file_function_entries)

            for func_name, func_record in raw.items():
                annotated = annotate_function(func_name, func_record, binary_min_addr, binary_max_addr, file_function_entries)
                if not annotated:
                    continue
                func_id = f"{file_base}::{opt}::{func_name}"
                function_blocks[func_id] = annotated
                group_key = f"{file_base}::{func_name}"
                equivalence_groups[group_key].append(func_id)
                if max_functions is not None and len(function_blocks) >= max_functions:
                    break
            if max_functions is not None and len(function_blocks) >= max_functions:
                break

    return function_blocks, equivalence_groups


def split_by_group(function_blocks: Dict[str, List[str]], equivalence_groups: Dict[str, List[str]], train_ratio: float, seed: int):
    groups = list(equivalence_groups.items())
    rng = random.Random(seed)
    rng.shuffle(groups)

    train_n = max(1, int(len(groups) * train_ratio))
    train_groups = dict(groups[:train_n])
    val_groups = dict(groups[train_n:])
    if not val_groups:
        val_groups = dict(groups[: min(32, len(groups))])

    train_blocks = {}
    val_blocks = {}
    for _group, ids in train_groups.items():
        for func_id in ids:
            if func_id in function_blocks:
                train_blocks[func_id] = function_blocks[func_id]
    for _group, ids in val_groups.items():
        for func_id in ids:
            if func_id in function_blocks:
                val_blocks[func_id] = function_blocks[func_id]
    return train_blocks, val_blocks, val_groups


def build_pairs_from_groups(groups: Dict[str, List[str]]):
    pairs = {}
    for _group, ids in groups.items():
        if len(ids) < 2:
            continue
        for func_id in ids:
            gt = [other for other in ids if other != func_id]
            if gt:
                pairs[func_id] = {"ground_truth": gt}
    return pairs


def save_json(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f)


def main():
    parser = argparse.ArgumentParser(description="Build address-aware corpora and real funcsim pairs from downloaded jTrans-compatible data")
    parser.add_argument("--zip", required=True, help="Downloaded external dataset zip")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_files", type=int, default=64)
    parser.add_argument("--max_functions", type=int, default=4096)
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.zip = ensure_under_data_root(args.zip, "--zip", must_exist=True)
    args.output_dir = ensure_parent_under_data_root(args.output_dir, "--output_dir")

    function_blocks, equivalence_groups = load_external_blocks_from_zip(
        args.zip,
        max_files=args.max_files,
        max_functions=args.max_functions,
    )
    if not function_blocks:
        raise RuntimeError("No address-aware functions were built from the downloaded dataset")

    train_blocks, val_blocks, val_groups = split_by_group(
        function_blocks,
        equivalence_groups,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )
    funcsim_pairs = build_pairs_from_groups(val_groups)
    funcsim_blocks = {fid: val_blocks[fid] for fid in funcsim_pairs if fid in val_blocks}

    train_blocks_path = os.path.join(args.output_dir, "train_function_blocks.json")
    val_blocks_path = os.path.join(args.output_dir, "val_function_blocks.json")
    funcsim_blocks_path = os.path.join(args.output_dir, "funcsim_function_blocks.json")
    funcsim_pairs_path = os.path.join(args.output_dir, "funcsim_pairs.json")
    train_corpus = os.path.join(args.output_dir, "pretrain_train.txt")
    val_corpus = os.path.join(args.output_dir, "pretrain_val.txt")

    save_json(train_blocks, train_blocks_path)
    save_json(val_blocks, val_blocks_path)
    save_json(funcsim_blocks, funcsim_blocks_path)
    save_json(funcsim_pairs, funcsim_pairs_path)

    write_corpus_from_function_blocks(train_blocks_path, train_corpus)
    write_corpus_from_function_blocks(val_blocks_path, val_corpus)

    print(f"Prepared {len(train_blocks)} train functions and {len(val_blocks)} val functions")
    print(f"Prepared {len(funcsim_pairs)} funcsim queries")
    print(f"Train corpus : {train_corpus}")
    print(f"Val corpus   : {val_corpus}")
    print(f"Funcsim blocks: {funcsim_blocks_path}")
    print(f"Funcsim pairs : {funcsim_pairs_path}")


if __name__ == "__main__":
    main()
