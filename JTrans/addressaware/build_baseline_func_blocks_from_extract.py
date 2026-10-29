#!/usr/bin/env python3
"""
Build baseline jTrans function blocks from official *_extract.pkl files.

The output is keyed by the same canonical variant ID format used by the
address-aware pipeline:

    <binary_name>::<opt_level>::<function_name>

Each value contains baseline-tokenized instructions suitable for the
JSON-based AAE finetune/eval pipeline.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import re
from pathlib import Path


JTRANS_READIDADATA = Path("/path/to/numAnchor/JTrans/readidadata.py")

FILENAME_RE = re.compile(
    r"^(?P<binary>.+)-(?P<opt>O0|O1|O2|O3|Os|Og|Ofast)-(?P<hash>[0-9a-f]{32})_extract\.pkl$"
)

SKIP_FUNCTIONS = {".plt", "extern", ".init", ".fini"}


def load_readidadata_module():
    spec = importlib.util.spec_from_file_location(
        "jtrans_readidadata", JTRANS_READIDADATA
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load readidadata module from {JTRANS_READIDADATA}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def gen_funcstr(func_data, readidadata_module, max_len=512, convert_jump=True):
    """
    Reproduce the original jTrans `gen_funcstr` behavior on official extract.pkl
    entries: (func_addr, asm_list, rawbytes_list, cfg, bai_feature).
    """
    if isinstance(func_data, dict):
        asm_list = func_data.get("asm", [])
        cfg = func_data.get("cfg")
    else:
        _, asm_list, _, cfg, _ = func_data

    if not asm_list or cfg is None:
        return ""

    code_lst = []
    map_id = {}
    bb_ls = sorted(cfg.nodes)

    for bb in bb_ls:
        asm = cfg.nodes[bb]["asm"]
        map_id[bb] = len(code_lst)
        for code in asm:
            operator, operand1, operand2, operand3, _ = readidadata_module.parse_asm(code)
            code_lst.append(operator)
            if operand1 is not None:
                code_lst.append(operand1)
            if operand2 is not None:
                code_lst.append(operand2)
            if operand3 is not None:
                code_lst.append(operand3)

    for idx, token in enumerate(code_lst):
        if not token.startswith("hex_"):
            continue
        jumpaddr = int(token[4:], 16)
        if jumpaddr in map_id:
            jumpid = map_id[jumpaddr]
            code_lst[idx] = f"JUMP_ADDR_{jumpid}" if jumpid < max_len else "JUMP_ADDR_EXCEEDED"
        else:
            code_lst[idx] = "UNK_JUMP_ADDR"
        if not convert_jump:
            code_lst[idx] = "CONST"

    return " ".join(code_lst)


def parse_extract_name(path: Path):
    match = FILENAME_RE.match(path.name)
    if not match:
        return None
    return match.group("binary"), match.group("opt"), match.group("hash")


def main():
    parser = argparse.ArgumentParser(description="Build baseline func_blocks JSON")
    parser.add_argument("--extract_root", required=True, help="Root directory with *_extract.pkl files")
    parser.add_argument("--output", required=True, help="Output func_blocks_baseline.json path")
    parser.add_argument("--min_instructions", type=int, default=5, help="Minimum instructions per function")
    args = parser.parse_args()

    extract_root = Path(args.extract_root)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    readidadata_module = load_readidadata_module()

    func_blocks = {}
    total_files = 0
    total_funcs = 0

    for pkl_path in sorted(extract_root.glob("**/*_extract.pkl")):
        parsed = parse_extract_name(pkl_path)
        if not parsed:
            continue
        binary_name, opt_level, _ = parsed
        total_files += 1

        with open(pkl_path, "rb") as fh:
            data = pickle.load(fh)

        for func_name, func_data in data.items():
            if func_name in SKIP_FUNCTIONS:
                continue

            if isinstance(func_data, dict):
                asm_list = func_data.get("asm", [])
            else:
                asm_list = func_data[1]
            if len(asm_list) < args.min_instructions:
                continue

            token_str = gen_funcstr(func_data, readidadata_module)
            if not token_str:
                continue

            variant_id = f"{binary_name}::{opt_level}::{func_name}"
            func_blocks[variant_id] = {
                "binary_name": binary_name,
                "function_name": func_name,
                "opt": opt_level,
                "tokens": token_str,
                "num_instructions": len(asm_list),
            }
            total_funcs += 1

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(func_blocks, fh)

    print(f"Processed {total_files} extract files")
    print(f"Wrote {total_funcs} baseline function blocks to {output_path}")


if __name__ == "__main__":
    main()
