#!/usr/bin/env python3
"""
Build a function-similarity dataset variant with shuffled function-in-binary
coordinates.

For each compiled binary (*_functions.json):
  1. Collect every function's binary-level coordinate (the first float in
     opcode(address...) annotations).
  2. Shuffle those coordinates within the compiled binary only.
  3. Rewrite every opcode token to use the function's shuffled coordinate.
  4. Rewrite internal address(...) operands so their first float matches the
     shuffled coordinate of the target function.
  5. Rebuild global function_blocks.json + funcsim_pairs.json.
"""

from __future__ import annotations

import argparse
import bisect
import glob
import json
import os
import random
import re
from pathlib import Path


OPCODE_RE = re.compile(
    r"^(?P<opcode>[\w.]+)\("
    r"(?P<addr>0x[0-9a-fA-F]+):"
    r"(?P<binary>-?[0-9.]+):"
    r"(?P<func>-?[0-9.]+):"
    r"(?P<bb>-?[0-9.]+)\)"
)
ADDRESS_RE = re.compile(
    r"address\("
    r"(0x[0-9a-fA-F]+):"
    r"(-?[0-9.]+):"
    r"(-?[0-9.]+):"
    r"(-?[0-9.]+)\)"
)


def strip_opt_suffix(name: str) -> str:
    for marker in ("_O", "-O"):
        if marker in name:
            base_name, suffix = name.rsplit(marker, 1)
            if ("O" + suffix) in {"O0", "O1", "O2", "O3"}:
                return base_name
    return name


def parse_instruction_addr(inst_text: str) -> int | None:
    match = OPCODE_RE.match(inst_text)
    if not match:
        return None
    return int(match.group("addr"), 16)


def build_binary_records(source_file: str) -> list[dict]:
    with open(source_file, "r") as f:
        functions = json.load(f)

    records = []
    for order_idx, func in enumerate(functions):
        instructions = [inst for inst in func.get("funcins", "").split("\t") if inst.strip()]
        inst_addrs = [addr for addr in (parse_instruction_addr(inst) for inst in instructions) if addr is not None]

        first_inst = instructions[0] if instructions else ""
        first_match = OPCODE_RE.match(first_inst)
        if not first_match:
            raise ValueError(f"Cannot parse first instruction in {source_file}: {first_inst!r}")

        original_binary_pos = float(first_match.group("binary"))
        start_addr = inst_addrs[0] if inst_addrs else int(first_match.group("addr"), 16)
        end_addr = inst_addrs[-1] if inst_addrs else start_addr

        records.append(
            {
                "order_idx": order_idx,
                "source_file": source_file,
                "binary": func["binary"],
                "function": func["function"],
                "opt_level": func["opt_level"],
                "original_local_func_id": func.get("func_id"),
                "instructions": instructions,
                "inst_addrs": inst_addrs,
                "start_addr": start_addr,
                "end_addr": end_addr,
                "original_binary_pos": original_binary_pos,
            }
        )

    return records


def assign_shuffled_positions(records: list[dict], seed: int) -> None:
    positions = [rec["original_binary_pos"] for rec in records]
    shuffled = list(positions)
    local_rng = random.Random(f"{seed}:{Path(records[0]['source_file']).name}")
    local_rng.shuffle(shuffled)
    for rec, shuffled_pos in zip(records, shuffled):
        rec["shuffled_binary_pos"] = shuffled_pos


def build_target_lookup(records: list[dict]) -> tuple[dict[int, dict], list[int], list[dict]]:
    exact = {}
    ordered = sorted(records, key=lambda rec: rec["start_addr"])
    starts = [rec["start_addr"] for rec in ordered]

    for rec in ordered:
        for addr in rec["inst_addrs"]:
            exact[addr] = rec

    return exact, starts, ordered


def resolve_target_record(
    target_addr: int,
    exact_lookup: dict[int, dict],
    sorted_starts: list[int],
    ordered_records: list[dict],
) -> dict | None:
    direct = exact_lookup.get(target_addr)
    if direct is not None:
        return direct

    idx = bisect.bisect_right(sorted_starts, target_addr) - 1
    if idx < 0:
        return None

    candidate = ordered_records[idx]
    cover_end = candidate["end_addr"] + 15
    if candidate["start_addr"] <= target_addr <= cover_end:
        return candidate
    return None


def rewrite_instruction(
    inst_text: str,
    owner_record: dict,
    exact_lookup: dict[int, dict],
    sorted_starts: list[int],
    ordered_records: list[dict],
) -> str:
    owner_bin = owner_record["shuffled_binary_pos"]

    def rewrite_address(match: re.Match[str]) -> str:
        target_addr = int(match.group(1), 16)
        target_record = resolve_target_record(target_addr, exact_lookup, sorted_starts, ordered_records)
        if target_record is None:
            new_bin = match.group(2)
        else:
            new_bin = f"{target_record['shuffled_binary_pos']:.8f}"
        return f"address({match.group(1)}:{new_bin}:{match.group(3)}:{match.group(4)})"

    opcode_match = OPCODE_RE.match(inst_text)
    rewritten = inst_text
    if opcode_match:
        prefix = (
            f"{opcode_match.group('opcode')}("
            f"{opcode_match.group('addr')}:"
            f"{owner_bin:.8f}:"
            f"{opcode_match.group('func')}:"
            f"{opcode_match.group('bb')})"
        )
        rewritten = prefix + inst_text[opcode_match.end():]

    return ADDRESS_RE.sub(rewrite_address, rewritten)


def rewrite_binary_records(records: list[dict]) -> list[dict]:
    exact_lookup, sorted_starts, ordered_records = build_target_lookup(records)

    rewritten_records = []
    for rec in records:
        rewritten_instructions = [
            rewrite_instruction(inst, rec, exact_lookup, sorted_starts, ordered_records)
            for inst in rec["instructions"]
        ]
        rewritten_records.append({**rec, "rewritten_instructions": rewritten_instructions})
    return rewritten_records


def build_global_outputs(all_records: list[dict]) -> tuple[dict, dict, dict]:
    function_blocks = {}
    func_index = {}
    global_meta = {}

    global_id = 0
    for rec in all_records:
        gid = str(global_id)
        function_blocks[gid] = rec["rewritten_instructions"]
        global_meta[gid] = {
            "source_file": rec["source_file"],
            "binary": rec["binary"],
            "function": rec["function"],
            "opt_level": rec["opt_level"],
            "original_local_func_id": rec["original_local_func_id"],
            "start_addr": hex(rec["start_addr"]),
            "original_binary_pos": rec["original_binary_pos"],
            "shuffled_binary_pos": rec["shuffled_binary_pos"],
            "num_instructions": len(rec["rewritten_instructions"]),
        }
        base_binary = strip_opt_suffix(rec["binary"])
        func_index[(base_binary, rec["function"], rec["opt_level"])] = gid
        global_id += 1

    funcsim_pairs = {}
    for gid, meta in global_meta.items():
        base_binary = strip_opt_suffix(meta["binary"])
        gt_ids = []
        for other_opt in ("O0", "O1", "O2", "O3"):
            if other_opt == meta["opt_level"]:
                continue
            other_gid = func_index.get((base_binary, meta["function"], other_opt))
            if other_gid is not None:
                gt_ids.append(other_gid)
        if gt_ids:
            funcsim_pairs[gid] = {"ground_truth": gt_ids}

    return function_blocks, funcsim_pairs, global_meta


def infer_records_from_global_function_blocks(function_blocks_path: str) -> list[dict]:
    with open(function_blocks_path, "r") as f:
        function_blocks = json.load(f)

    records = []
    prev_binary_pos = None
    segment_id = -1

    for global_id, instructions in function_blocks.items():
        if not instructions:
            continue
        first_match = OPCODE_RE.match(instructions[0])
        if not first_match:
            raise ValueError(f"Cannot parse first instruction for func_id={global_id}")

        binary_pos = float(first_match.group("binary"))
        if prev_binary_pos is None or binary_pos < prev_binary_pos:
            segment_id += 1
        prev_binary_pos = binary_pos

        inst_addrs = [addr for addr in (parse_instruction_addr(inst) for inst in instructions) if addr is not None]
        start_addr = inst_addrs[0] if inst_addrs else int(first_match.group("addr"), 16)
        end_addr = inst_addrs[-1] if inst_addrs else start_addr

        records.append(
            {
                "global_id": str(global_id),
                "segment_id": segment_id,
                "instructions": instructions,
                "inst_addrs": inst_addrs,
                "start_addr": start_addr,
                "end_addr": end_addr,
                "original_binary_pos": binary_pos,
            }
        )

    return records


def assign_shuffled_positions_in_segment(records: list[dict], seed: int) -> None:
    positions = [rec["original_binary_pos"] for rec in records]
    shuffled = list(positions)
    local_rng = random.Random(f"{seed}:segment:{records[0]['segment_id']}")
    local_rng.shuffle(shuffled)
    for rec, shuffled_pos in zip(records, shuffled):
        rec["shuffled_binary_pos"] = shuffled_pos


def rewrite_global_segments(records: list[dict]) -> list[dict]:
    all_rewritten = []
    by_segment: dict[int, list[dict]] = {}
    for rec in records:
        by_segment.setdefault(rec["segment_id"], []).append(rec)

    for segment_records in by_segment.values():
        exact_lookup, sorted_starts, ordered_records = build_target_lookup(segment_records)
        for rec in segment_records:
            rewritten_instructions = [
                rewrite_instruction(inst, rec, exact_lookup, sorted_starts, ordered_records)
                for inst in rec["instructions"]
            ]
            all_rewritten.append({**rec, "rewritten_instructions": rewritten_instructions})

    return all_rewritten


def save_global_mode_outputs(
    rewritten_records: list[dict],
    original_funcsim_pairs_path: str,
    output_dir: str,
    seed: int,
    function_blocks_path: str,
) -> None:
    function_blocks = {
        rec["global_id"]: rec["rewritten_instructions"]
        for rec in rewritten_records
    }
    global_meta = {
        rec["global_id"]: {
            "segment_id": rec["segment_id"],
            "start_addr": hex(rec["start_addr"]),
            "original_binary_pos": rec["original_binary_pos"],
            "shuffled_binary_pos": rec["shuffled_binary_pos"],
            "num_instructions": len(rec["rewritten_instructions"]),
        }
        for rec in rewritten_records
    }
    with open(original_funcsim_pairs_path, "r") as f:
        funcsim_pairs = json.load(f)

    segment_count = len({rec["segment_id"] for rec in rewritten_records})
    shuffle_manifest = {
        "seed": seed,
        "input_mode": "global_function_blocks",
        "function_blocks": function_blocks_path,
        "funcsim_pairs": original_funcsim_pairs_path,
        "rewritten_field": "binary_pos",
        "segment_rule": "start a new segment whenever the next function's binary_pos resets",
        "segment_count": segment_count,
        "function_count": len(rewritten_records),
    }

    with open(os.path.join(output_dir, "function_blocks.json"), "w") as f:
        json.dump(function_blocks, f)
    with open(os.path.join(output_dir, "funcsim_pairs.json"), "w") as f:
        json.dump(funcsim_pairs, f)
    with open(os.path.join(output_dir, "global_function_meta.json"), "w") as f:
        json.dump(global_meta, f, indent=2)
    with open(os.path.join(output_dir, "shuffle_manifest.json"), "w") as f:
        json.dump(shuffle_manifest, f, indent=2)

    print("=" * 72)
    print("Shuffled funcsim dataset built")
    print("=" * 72)
    print(f"Output dir      : {output_dir}")
    print(f"Input mode      : global_function_blocks")
    print(f"Segments        : {segment_count}")
    print(f"Functions       : {len(function_blocks)}")
    print(f"Query functions : {len(funcsim_pairs)}")
    print("=" * 72)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build shuffled-coordinate funcsim data")
    parser.add_argument(
        "--function_blocks",
        default=None,
        help="Existing global function_blocks.json to rewrite in-place style",
    )
    parser.add_argument(
        "--funcsim_pairs",
        default=None,
        help="Existing global funcsim_pairs.json to carry over in global mode",
    )
    parser.add_argument(
        "--input_dir",
        default=None,
        help="Directory containing *_functions.json files",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory for rebuilt function_blocks / funcsim_pairs",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.function_blocks and args.funcsim_pairs:
        records = infer_records_from_global_function_blocks(args.function_blocks)
        by_segment: dict[int, list[dict]] = {}
        for rec in records:
            by_segment.setdefault(rec["segment_id"], []).append(rec)
        for segment_records in by_segment.values():
            assign_shuffled_positions_in_segment(segment_records, args.seed)
        rewritten_records = rewrite_global_segments(records)
        save_global_mode_outputs(
            rewritten_records=rewritten_records,
            original_funcsim_pairs_path=args.funcsim_pairs,
            output_dir=args.output_dir,
            seed=args.seed,
            function_blocks_path=args.function_blocks,
        )
        return

    if not args.input_dir:
        raise ValueError("Either --function_blocks/--funcsim_pairs or --input_dir must be provided.")

    source_files = sorted(glob.glob(os.path.join(args.input_dir, "*_functions.json")))
    if not source_files:
        raise FileNotFoundError(f"No *_functions.json files found in {args.input_dir}")

    all_records = []
    shuffle_manifest = {
        "seed": args.seed,
        "input_dir": args.input_dir,
        "rewritten_field": "binary_pos",
        "grouping_rule": "per compiled binary file",
        "source_file_count": len(source_files),
        "files": [],
    }

    for source_file in source_files:
        records = build_binary_records(source_file)
        if not records:
            continue
        assign_shuffled_positions(records, args.seed)
        rewritten_records = rewrite_binary_records(records)
        all_records.extend(rewritten_records)

        shuffle_manifest["files"].append(
            {
                "source_file": source_file,
                "binary": rewritten_records[0]["binary"],
                "opt_level": rewritten_records[0]["opt_level"],
                "function_count": len(rewritten_records),
            }
        )

    function_blocks, funcsim_pairs, global_meta = build_global_outputs(all_records)

    with open(os.path.join(args.output_dir, "function_blocks.json"), "w") as f:
        json.dump(function_blocks, f)
    with open(os.path.join(args.output_dir, "funcsim_pairs.json"), "w") as f:
        json.dump(funcsim_pairs, f)
    with open(os.path.join(args.output_dir, "global_function_meta.json"), "w") as f:
        json.dump(global_meta, f, indent=2)
    with open(os.path.join(args.output_dir, "shuffle_manifest.json"), "w") as f:
        json.dump(shuffle_manifest, f, indent=2)

    print("=" * 72)
    print("Shuffled funcsim dataset built")
    print("=" * 72)
    print(f"Output dir      : {args.output_dir}")
    print(f"Source files    : {len(source_files)}")
    print(f"Functions       : {len(function_blocks)}")
    print(f"Query functions : {len(funcsim_pairs)}")
    print("=" * 72)


if __name__ == "__main__":
    main()
