#!/usr/bin/env python3
"""
Build Trex-style multi-field corpora from the local Stateformer x86-64 dataset.

This adapter intentionally uses the information available in the inline
Stateformer corpus:
  - opcode / operand tokens
  - instruction / code / data addresses embedded in inline tokens
  - hierarchical address-aware operand markers such as address(...) and daddr(...)

The original Trex pretraining recipe relies on micro-trace values. The
Stateformer corpus in this repository does not ship those traces, so this
adapter repurposes the four byte fields as a lightweight numeric side-channel:
  - address-aware mode: opcode/address/var tokens emit 32-bit little-endian
    numeric bytes derived from the inline address or stack offset
  - baseline mode: all byte fields are blank ("##"), matching a non-address-aware
    variant
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import re
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple


FIELD_ORDER = [
    "static",
    "inst_pos_emb",
    "op_pos_emb",
    "arch_emb",
    "byte1",
    "byte2",
    "byte3",
    "byte4",
]
AA_FIELD_ORDER = [
    "aa_role_ids",
    "binary_pos",
    "function_pos",
    "bb_pos",
    "var_offset",
]

REAL_BYTE_TOKENS = [f"{value:02x}" for value in range(256)]
BYTE_BLANK = "##"
ARCH_TOKEN = "x64"
NON_ADDR_SENTINEL = -1.0
NON_VAR_SENTINEL = -999999

OPCODE_RE = re.compile(
    r"^([A-Za-z_.$@?][^()]*)\((0x[0-9a-fA-F]+):([0-9]*\.?[0-9]+):([0-9]*\.?[0-9]+):([0-9]*\.?[0-9]+)\)$"
)
ADDRESS_RE = re.compile(
    r"^(address|daddr(?:_[a-z]+)?)\((0x[0-9a-fA-F]+):([0-9]*\.?[0-9]+):([0-9]*\.?[0-9]+):([0-9]*\.?[0-9]+)\)$"
)
VAR_RE = re.compile(r"^var\((0x[0-9a-fA-F]+)\)$")


def iter_lines(path: Path) -> Iterator[str]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield line


def fast_random_line_sample(path: Path, sample_size: int, seed: int) -> List[str]:
    if sample_size <= 0:
        return []

    rng = random.Random(seed)
    file_size = path.stat().st_size
    sample: List[str] = []
    seen = set()
    max_attempts = max(sample_size * 20, 1000)

    with path.open("rb") as handle:
        attempts = 0
        while len(sample) < sample_size and attempts < max_attempts:
            attempts += 1
            offset = rng.randrange(0, max(file_size, 1))
            handle.seek(offset)
            if offset > 0:
                handle.readline()
            line = handle.readline()
            if not line:
                handle.seek(0)
                line = handle.readline()
            if not line:
                break
            decoded = line.decode("utf-8", errors="ignore").strip()
            if not decoded or decoded in seen:
                continue
            seen.add(decoded)
            sample.append(decoded)

    if len(sample) < sample_size:
        for line in iter_lines(path):
            if line in seen:
                continue
            sample.append(line)
            seen.add(line)
            if len(sample) >= sample_size:
                break

    return sample


def parse_hexish(value: str) -> int:
    raw = int(value, 16)
    return raw & 0xFFFFFFFF


def parse_signed_hex(value: str) -> int:
    digits = value.lower().replace("0x", "")
    raw = int(digits, 16)
    bits = max(1, len(digits) * 4)
    sign_bit = 1 << (bits - 1)
    if raw & sign_bit:
        raw -= 1 << bits
    return raw


def little_endian_bytes_u32(value: int) -> Tuple[str, str, str, str]:
    return tuple(f"{(value >> (8 * shift)) & 0xFF:02x}" for shift in range(4))  # type: ignore[return-value]


def normalize_static_token(raw_token: str, mode: str) -> str:
    address_match = ADDRESS_RE.match(raw_token)
    if address_match:
        return "daddr" if address_match.group(1).startswith("daddr") else "address"

    var_match = VAR_RE.match(raw_token)
    if var_match:
        return "var"

    opcode_match = OPCODE_RE.match(raw_token)
    if opcode_match:
        return opcode_match.group(1).lower()

    return raw_token.lower()


def numeric_side_channel(raw_token: str, static_token: str, mode: str) -> Tuple[str, str, str, str]:
    if mode != "addressaware":
        return (BYTE_BLANK, BYTE_BLANK, BYTE_BLANK, BYTE_BLANK)

    opcode_match = OPCODE_RE.match(raw_token)
    if opcode_match:
        return little_endian_bytes_u32(parse_hexish(opcode_match.group(2)))

    address_match = ADDRESS_RE.match(raw_token)
    if address_match:
        return little_endian_bytes_u32(parse_hexish(address_match.group(2)))

    var_match = VAR_RE.match(raw_token)
    if var_match:
        return little_endian_bytes_u32(parse_hexish(var_match.group(1)))

    return (BYTE_BLANK, BYTE_BLANK, BYTE_BLANK, BYTE_BLANK)


def format_float_token(value: float) -> str:
    return f"{value:.8f}"


def aa_side_channel(raw_token: str) -> Dict[str, str]:
    address_match = ADDRESS_RE.match(raw_token)
    if address_match:
        role_id = "3" if address_match.group(1).startswith("daddr") else "2"
        return {
            "aa_role_ids": role_id,
            "binary_pos": format_float_token(float(address_match.group(3)) if role_id == "3" else NON_ADDR_SENTINEL),
            "function_pos": format_float_token(float(address_match.group(4))),
            "bb_pos": format_float_token(float(address_match.group(5))),
            "var_offset": str(NON_VAR_SENTINEL),
        }

    var_match = VAR_RE.match(raw_token)
    if var_match:
        return {
            "aa_role_ids": "4",
            "binary_pos": format_float_token(NON_ADDR_SENTINEL),
            "function_pos": format_float_token(NON_ADDR_SENTINEL),
            "bb_pos": format_float_token(NON_ADDR_SENTINEL),
            "var_offset": str(parse_signed_hex(var_match.group(1))),
        }

    opcode_match = OPCODE_RE.match(raw_token)
    if opcode_match:
        return {
            "aa_role_ids": "1",
            "binary_pos": format_float_token(NON_ADDR_SENTINEL),
            "function_pos": format_float_token(float(opcode_match.group(4))),
            "bb_pos": format_float_token(float(opcode_match.group(5))),
            "var_offset": str(NON_VAR_SENTINEL),
        }

    return {
        "aa_role_ids": "0",
        "binary_pos": format_float_token(NON_ADDR_SENTINEL),
        "function_pos": format_float_token(NON_ADDR_SENTINEL),
        "bb_pos": format_float_token(NON_ADDR_SENTINEL),
        "var_offset": str(NON_VAR_SENTINEL),
    }


def convert_inline_sequence(inline_text: str, mode: str) -> Dict[str, List[str]]:
    fields = {name: [] for name in FIELD_ORDER}
    instructions = [chunk for chunk in inline_text.split("\t") if chunk.strip()]

    for inst_index, instruction in enumerate(instructions):
        raw_tokens = instruction.strip().split()
        for op_index, raw_token in enumerate(raw_tokens):
            static_token = normalize_static_token(raw_token, mode=mode)
            byte_values = numeric_side_channel(raw_token, static_token=static_token, mode=mode)

            fields["static"].append(static_token)
            fields["inst_pos_emb"].append(str(inst_index))
            fields["op_pos_emb"].append(str(op_index))
            fields["arch_emb"].append(ARCH_TOKEN)
            for byte_field, byte_value in zip(("byte1", "byte2", "byte3", "byte4"), byte_values):
                fields[byte_field].append(byte_value)

    return fields


def convert_inline_sequence_with_aa(inline_text: str, mode: str) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    fields = {name: [] for name in FIELD_ORDER}
    aa_fields = {name: [] for name in AA_FIELD_ORDER}
    instructions = [chunk for chunk in inline_text.split("\t") if chunk.strip()]

    for inst_index, instruction in enumerate(instructions):
        raw_tokens = instruction.strip().split()
        for op_index, raw_token in enumerate(raw_tokens):
            static_token = normalize_static_token(raw_token, mode=mode)
            byte_values = numeric_side_channel(raw_token, static_token=static_token, mode=mode)
            aa_values = aa_side_channel(raw_token)

            fields["static"].append(static_token)
            fields["inst_pos_emb"].append(str(inst_index))
            fields["op_pos_emb"].append(str(op_index))
            fields["arch_emb"].append(ARCH_TOKEN)
            for byte_field, byte_value in zip(("byte1", "byte2", "byte3", "byte4"), byte_values):
                fields[byte_field].append(byte_value)
            for aa_field in AA_FIELD_ORDER:
                aa_fields[aa_field].append(aa_values[aa_field])

    return fields, aa_fields


def write_parallel_fields(lines: Sequence[str], output_prefix: Path, mode: str) -> Dict[str, collections.Counter]:
    counters: Dict[str, collections.Counter] = {
        field: collections.Counter() for field in FIELD_ORDER
    }
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    writers = {
        field: (output_prefix.parent / f"{output_prefix.name}.{field}").open("w", encoding="utf-8")
        for field in FIELD_ORDER
    }
    try:
        for inline_text in lines:
            converted = convert_inline_sequence(inline_text, mode=mode)
            for field in FIELD_ORDER:
                tokens = converted[field]
                writers[field].write(" ".join(tokens))
                writers[field].write("\n")
                counters[field].update(tokens)
    finally:
        for handle in writers.values():
            handle.close()
    return counters


def write_aa_sidecar(lines: Sequence[str], output_prefix: Path, mode: str) -> None:
    writers = {
        field: (output_prefix.parent / f"{output_prefix.name}.{field}").open("w", encoding="utf-8")
        for field in AA_FIELD_ORDER
    }
    try:
        for inline_text in lines:
            _, aa_fields = convert_inline_sequence_with_aa(inline_text, mode=mode)
            for field in AA_FIELD_ORDER:
                writers[field].write(" ".join(aa_fields[field]))
                writers[field].write("\n")
    finally:
        for handle in writers.values():
            handle.close()


def build_dictionary_tokens(field: str, counter: collections.Counter, top_k: int | None = None) -> List[Tuple[str, int]]:
    if field.startswith("byte"):
        ordered = [(BYTE_BLANK, counter.get(BYTE_BLANK, 1))]
        ordered.extend((token, counter.get(token, 1)) for token in REAL_BYTE_TOKENS)
        return ordered

    items = counter.most_common(top_k)
    return [(token, count) for token, count in items]


def write_dicts(dict_root: Path, counters: Dict[str, collections.Counter], max_static_vocab: int, max_position_vocab: int) -> None:
    dict_root.mkdir(parents=True, exist_ok=True)
    for field, counter in counters.items():
        field_dir = dict_root / field
        field_dir.mkdir(parents=True, exist_ok=True)
        if field == "static":
            items = build_dictionary_tokens(field, counter, top_k=max_static_vocab)
        elif field in {"inst_pos_emb", "op_pos_emb"}:
            items = build_dictionary_tokens(field, counter, top_k=max_position_vocab)
        else:
            items = build_dictionary_tokens(field, counter)
        with (field_dir / "dict.txt").open("w", encoding="utf-8") as handle:
            for token, count in items:
                handle.write(f"{token} {count}\n")


def merge_counters(parts: Iterable[Dict[str, collections.Counter]]) -> Dict[str, collections.Counter]:
    merged = {field: collections.Counter() for field in FIELD_ORDER}
    for part in parts:
        for field in FIELD_ORDER:
            merged[field].update(part[field])
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Trex corpora from Stateformer x64 inline CFG data")
    parser.add_argument("--train-corpus", type=Path, default=Path("/path/to/numAnchor_artifact/stateformer_x64_dataset/pretrain_ctx2/splits/train_cfg.txt"))
    parser.add_argument("--valid-corpus", type=Path, default=Path("/path/to/numAnchor_artifact/stateformer_x64_dataset/pretrain_ctx2/splits/val_cfg.txt"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["addressaware", "baseline", "addressaware_aa"], required=True)
    parser.add_argument("--train-samples", type=int, default=200000)
    parser.add_argument("--valid-samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-static-vocab", type=int, default=50000)
    parser.add_argument("--max-position-vocab", type=int, default=2048)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_lines = fast_random_line_sample(args.train_corpus, args.train_samples, seed=args.seed)
    valid_lines = fast_random_line_sample(args.valid_corpus, args.valid_samples, seed=args.seed + 1)

    train_counters = write_parallel_fields(train_lines, args.output_dir / "train", mode=args.mode)
    valid_counters = write_parallel_fields(valid_lines, args.output_dir / "valid", mode=args.mode)
    if args.mode == "addressaware_aa":
        write_aa_sidecar(train_lines, args.output_dir / "train", mode=args.mode)
        write_aa_sidecar(valid_lines, args.output_dir / "valid", mode=args.mode)
    merged = merge_counters([train_counters, valid_counters])
    write_dicts(args.output_dir / "dicts", merged, max_static_vocab=args.max_static_vocab, max_position_vocab=args.max_position_vocab)

    metadata = {
        "mode": args.mode,
        "seed": args.seed,
        "train_samples": len(train_lines),
        "valid_samples": len(valid_lines),
        "train_corpus": str(args.train_corpus),
        "valid_corpus": str(args.valid_corpus),
    }
    with (args.output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)

    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
