#!/usr/bin/env python3
"""
Build Trex-native pretraining text fields from Stateformer x64 raw binaries.

This script follows Trex's original byte/trace representation:
  1. Extract function bytes from ELF binaries with objdump
  2. Run Trex micro-tracing on sampled functions
  3. Emit Trex pretraining fields:
       static / inst_pos_emb / op_pos_emb / arch_emb / byte1..byte4

The output is intentionally compatible with Trex's existing fairseq task.
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import io
import json
import random
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

from capstone import CS_ARCH_X86, CS_MODE_64, Cs, CsError

from command.stateformer_x64.prepare_stateformer_trex import (
    ARCH_TOKEN,
    BYTE_BLANK,
    FIELD_ORDER,
    REAL_BYTE_TOKENS,
    merge_counters,
    write_dicts,
)
from micro_trace.tracing import EX


FUNC_RE = re.compile(r"\<(.*)\>:")
BYTES_RE = re.compile(r":\t(.*?)\t")
BYTES_NOP_RE = re.compile(r":\t(.*?)\s+$")
TOKEN_HEX_RE = re.compile(r"0[xX][0-9a-fA-F]+")


@dataclass
class FunctionCandidate:
    binary_path: Path
    function_name: str
    byte_tokens: List[str]


def objdump_text_section(binary_path: Path) -> str:
    result = subprocess.run(
        ["objdump", "-d", str(binary_path), "-j", ".text"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def parse_objdump_output(text: str) -> Dict[str, List[str]]:
    current = ""
    functions: Dict[str, List[str]] = {}
    for line in text.splitlines():
        func_match = FUNC_RE.findall(line)
        if len(func_match) == 1:
            current = func_match[0]
            functions[current] = []
            continue
        if not current:
            continue
        byte_match = BYTES_RE.findall(line)
        if len(byte_match) == 1:
            functions[current].extend(byte_match[0].strip().split())
            continue
        nop_match = BYTES_NOP_RE.findall(line)
        if len(nop_match) == 1:
            functions[current].extend(nop_match[0].strip().split())
    return functions


def tokenize(text: str) -> List[str]:
    text = text.replace(",", " , ")
    text = text.replace("[", " [ ")
    text = text.replace("]", " ] ")
    text = text.replace(":", " : ")
    text = text.replace("*", " * ")
    text = text.replace("(", " ( ")
    text = text.replace(")", " ) ")
    text = text.replace("{", " { ")
    text = text.replace("}", " } ")
    text = text.replace("#", "")
    text = text.replace("$", "")
    text = text.replace("!", " ! ")
    text = re.sub(r"-(0[xX][0-9a-fA-F]+)", r"- \1", text)
    text = re.sub(r"-([0-9a-fA-F]+)", r"- \1", text)
    return text.split()


def hex2trace_word(value: str) -> str:
    number = value.replace("0x", "")[-8:]
    return number.rjust(8, "0")


def num_inst(code_bytes: bytes, md: Cs) -> int:
    return len(list(md.disasm_lite(code_bytes, 0x400000)))


def bytes_to_trace_fields(value_list: Sequence[str]) -> Tuple[List[str], List[str], List[str], List[str]]:
    bytes_per_token = [[value[i : i + 2] for i in range(0, len(value), 2)] for value in value_list]
    return (
        [chunk[0] for chunk in bytes_per_token],
        [chunk[1] for chunk in bytes_per_token],
        [chunk[2] for chunk in bytes_per_token],
        [chunk[3] for chunk in bytes_per_token],
    )


def get_trace_fields(static_instructions: Sequence[str], registers_list: Sequence[Dict[str, int]]) -> Tuple[List[str], List[str], List[str], List[str]]:
    static_tokens: List[str] = []
    trace_values: List[str] = []
    inst_indices: List[str] = []
    op_indices: List[str] = []

    for inst_index, (instruction, registers) in enumerate(zip(static_instructions, registers_list)):
        for token_index, token in enumerate(tokenize(instruction)):
            upper = token.upper()
            if upper in registers:
                static_tokens.append(token.lower())
                trace_values.append(hex2trace_word(hex(registers[upper])))
            elif TOKEN_HEX_RE.search(token):
                static_tokens.append("hexvar")
                trace_values.append(hex2trace_word(token.lower()))
            elif token.lower().isdigit():
                static_tokens.append("num")
                trace_values.append(hex2trace_word(hex(int(token))))
            else:
                static_tokens.append(token.lower())
                trace_values.append(BYTE_BLANK * 4)
            inst_indices.append(str(inst_index))
            op_indices.append(str(token_index))

    return static_tokens, trace_values, inst_indices, op_indices


def build_example(static_tokens: Sequence[str], trace_values: Sequence[str], inst_indices: Sequence[str], op_indices: Sequence[str]) -> Dict[str, List[str]]:
    byte1, byte2, byte3, byte4 = bytes_to_trace_fields(trace_values)
    length = len(static_tokens)
    return {
        "static": list(static_tokens),
        "inst_pos_emb": list(inst_indices),
        "op_pos_emb": list(op_indices),
        "arch_emb": [ARCH_TOKEN] * length,
        "byte1": byte1,
        "byte2": byte2,
        "byte3": byte3,
        "byte4": byte4,
    }


def dummy_trace_values(length: int) -> List[str]:
    return [BYTE_BLANK * 4] * length


def trace_function(candidate: FunctionCandidate, md: Cs, num_traces: int, timeout: int) -> List[Dict[str, List[str]]]:
    code_bytes = bytes.fromhex("".join(candidate.byte_tokens))
    traced_examples: List[Dict[str, List[str]]] = []
    last_static_tokens: List[str] = []
    last_inst_indices: List[str] = []
    last_op_indices: List[str] = []

    for _ in range(num_traces):
        try:
            ex = EX(code_bytes, mode="x86_64", md=md)
            with contextlib.redirect_stdout(io.StringIO()):
                ex.run(timeout=timeout)
        except Exception:
            ex = None

        if ex is None or not getattr(ex.mu, "static", None) or not getattr(ex.mu, "trace", None):
            continue

        static_tokens, trace_values, inst_indices, op_indices = get_trace_fields(ex.mu.static, ex.mu.trace)
        if not static_tokens:
            continue

        last_static_tokens = static_tokens
        last_inst_indices = inst_indices
        last_op_indices = op_indices
        traced_examples.append(build_example(static_tokens, trace_values, inst_indices, op_indices))

    if last_static_tokens:
        traced_examples.append(
            build_example(
                last_static_tokens,
                dummy_trace_values(len(last_static_tokens)),
                last_inst_indices,
                last_op_indices,
            )
        )

    return traced_examples


def iter_binaries(input_root: Path, opts: Sequence[str]) -> Iterator[Path]:
    for opt in opts:
        for opt_dir in sorted(input_root.glob(f"*-{opt}")):
            if not opt_dir.is_dir():
                continue
            for binary in sorted(opt_dir.iterdir()):
                if binary.is_file() and binary.suffix != ".i64":
                    yield binary


def collect_candidates(input_root: Path, opts: Sequence[str], seed: int, max_binaries: int, max_functions: int, min_inst: int, max_inst: int) -> List[FunctionCandidate]:
    md = Cs(CS_ARCH_X86, CS_MODE_64)
    rng = random.Random(seed)
    binaries = list(iter_binaries(input_root, opts))
    rng.shuffle(binaries)
    if max_binaries > 0:
        binaries = binaries[:max_binaries]

    candidates: List[FunctionCandidate] = []
    for binary in binaries:
        try:
            parsed = parse_objdump_output(objdump_text_section(binary))
        except subprocess.CalledProcessError:
            continue
        for function_name, byte_tokens in parsed.items():
            if function_name.startswith("_") or not byte_tokens:
                continue
            try:
                inst_count = num_inst(bytes.fromhex("".join(byte_tokens)), md)
            except CsError:
                continue
            if inst_count < min_inst or inst_count > max_inst:
                continue
            candidates.append(
                FunctionCandidate(
                    binary_path=binary,
                    function_name=function_name,
                    byte_tokens=byte_tokens,
                )
            )

    rng.shuffle(candidates)
    if max_functions > 0:
        candidates = candidates[:max_functions]
    return candidates


def write_examples(examples: Sequence[Dict[str, List[str]]], output_prefix: Path) -> Dict[str, collections.Counter]:
    counters = {field: collections.Counter() for field in FIELD_ORDER}
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    writers = {
        field: (output_prefix.parent / f"{output_prefix.name}.{field}").open("w", encoding="utf-8")
        for field in FIELD_ORDER
    }
    try:
        for example in examples:
            for field in FIELD_ORDER:
                writers[field].write(" ".join(example[field]))
                writers[field].write("\n")
                counters[field].update(example[field])
    finally:
        for handle in writers.values():
            handle.close()
    return counters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Trex-native Stateformer x64 pretraining data from raw binaries")
    parser.add_argument("--input-root", type=Path, default=Path("/path/to/numAnchor_artifact/stateformer_x64_dataset/raw_input/x86-64"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--opts", nargs="*", default=["O0", "O1", "O2", "O3"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-binaries", type=int, default=64)
    parser.add_argument("--max-functions", type=int, default=4000)
    parser.add_argument("--min-inst", type=int, default=5)
    parser.add_argument("--max-inst", type=int, default=128)
    parser.add_argument("--num-traces", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=200000)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--max-static-vocab", type=int, default=50000)
    parser.add_argument("--max-position-vocab", type=int, default=2048)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    md = Cs(CS_ARCH_X86, CS_MODE_64)
    candidates = collect_candidates(
        input_root=args.input_root,
        opts=args.opts,
        seed=args.seed,
        max_binaries=args.max_binaries,
        max_functions=args.max_functions,
        min_inst=args.min_inst,
        max_inst=args.max_inst,
    )

    examples: List[Dict[str, List[str]]] = []
    traced_functions = 0
    for candidate in candidates:
        function_examples = trace_function(candidate, md=md, num_traces=args.num_traces, timeout=args.timeout)
        if not function_examples:
            continue
        traced_functions += 1
        examples.extend(function_examples)

    if not examples:
        raise RuntimeError("No pretraining examples were generated.")

    rng = random.Random(args.seed)
    rng.shuffle(examples)
    valid_size = max(1, int(len(examples) * args.valid_ratio))
    valid_examples = examples[:valid_size]
    train_examples = examples[valid_size:]
    if not train_examples:
        raise RuntimeError("Validation split consumed all examples; increase the dataset size.")

    train_counters = write_examples(train_examples, args.output_dir / "train")
    valid_counters = write_examples(valid_examples, args.output_dir / "valid")
    merged = merge_counters([train_counters, valid_counters])
    write_dicts(
        args.output_dir / "dicts",
        merged,
        max_static_vocab=args.max_static_vocab,
        max_position_vocab=args.max_position_vocab,
    )

    metadata = {
        "input_root": str(args.input_root),
        "opts": list(args.opts),
        "seed": args.seed,
        "candidate_functions": len(candidates),
        "traced_functions": traced_functions,
        "train_examples": len(train_examples),
        "valid_examples": len(valid_examples),
        "num_traces": args.num_traces,
        "timeout": args.timeout,
        "max_binaries": args.max_binaries,
        "max_functions": args.max_functions,
        "min_inst": args.min_inst,
        "max_inst": args.max_inst,
    }
    with (args.output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
