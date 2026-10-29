#!/usr/bin/env python3
"""
End-to-end smoke test for the address-aware PalmTree instruction pipeline.

It builds a tiny synthetic CFG-only context-2 corpus, trains:
  1. a tiny baseline checkpoint without address/var embeddings
  2. a tiny address-aware checkpoint

and then runs the full intrinsic probing suite on the resulting checkpoints.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
PROBE_DIR = ROOT / "probing"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PROBE_DIR) not in sys.path:
    sys.path.insert(0, str(PROBE_DIR))

from probe_eval import (  # noqa: E402
    build_call_target_pairs,
    build_dref_retrieval_queries,
    build_jump_target_pairs,
    build_same_bb_pairs,
    build_same_dref_pairs,
    build_same_function_pairs,
    reconstruct_blocks,
    split_function_ids,
)


def fmt8(value: float) -> str:
    return f"{value:.8f}"


def code_inst(opcode: str, addr: int, binary_pos: float, function_pos: float, bb_pos: float, *operands: str) -> str:
    header = f"{opcode}(0x{addr:x}:{fmt8(binary_pos)}:{fmt8(function_pos)}:{fmt8(bb_pos)})"
    if not operands:
        return header
    return header + " " + " ".join(operands)


def addr_token(addr: int, binary_pos: float, function_pos: float, bb_pos: float) -> str:
    return f"address(0x{addr:x}:{fmt8(binary_pos)}:{fmt8(function_pos)}:{fmt8(bb_pos)})"


def daddr_token(token_name: str, addr: int, section_pos: float, offset_in_section: float) -> str:
    return f"{token_name}(0x{addr:x}:{fmt8(section_pos)}:{fmt8(offset_in_section)})"


def build_synthetic_function_records() -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    binary_specs = [
        ("binA", 0x1000, "daddr_rodata", 0x5000, "daddr_data", 0x5100),
        ("binB", 0x3000, "daddr_bss", 0x7000, "daddr_other", 0x7100),
    ]

    for binary_name, base_addr, even_token, even_addr, odd_token, odd_addr in binary_specs:
        binary_positions = [0.06 + idx * 0.14 for idx in range(6)]
        for idx in range(6):
            func_id = f"{binary_name}_f{idx}"
            entry = base_addr + idx * 0x100
            jump_target = entry + 0x10
            call_site = entry + 0x14
            callee_idx = (idx + 1) % 6
            callee_entry = base_addr + callee_idx * 0x100
            callee_binary_pos = binary_positions[callee_idx]

            if idx % 2 == 0:
                primary_token = even_token
                primary_addr = even_addr
                primary_section_pos = 0.20
                primary_offset = 0.10
                secondary_token = odd_token
                secondary_addr = odd_addr
                secondary_section_pos = 0.65
                secondary_offset = 0.25
            else:
                primary_token = odd_token
                primary_addr = odd_addr
                primary_section_pos = 0.65
                primary_offset = 0.35
                secondary_token = even_token
                secondary_addr = even_addr
                secondary_section_pos = 0.20
                secondary_offset = 0.55

            bp = binary_positions[idx]
            block0_fp = 0.10
            block1_fp = 0.60

            instructions = [
                code_inst(
                    "mov",
                    entry,
                    bp,
                    block0_fp,
                    0.00,
                    "rax",
                    daddr_token(primary_token, primary_addr, primary_section_pos, primary_offset),
                ),
                code_inst("add", entry + 0x4, bp, block0_fp, 0.50, "rbx", "rcx"),
                code_inst(
                    "jmp",
                    entry + 0x8,
                    bp,
                    block0_fp,
                    1.00,
                    addr_token(jump_target, bp, block1_fp, 0.00),
                ),
                code_inst(
                    "mov",
                    jump_target,
                    bp,
                    block1_fp,
                    0.00,
                    "rcx",
                    daddr_token(secondary_token, secondary_addr, secondary_section_pos, secondary_offset),
                ),
                code_inst(
                    "call",
                    call_site,
                    bp,
                    block1_fp,
                    0.50,
                    addr_token(callee_entry, callee_binary_pos, 0.10, 0.00),
                ),
            ]
            records.append(
                {
                    "func_id": func_id,
                    "binary": binary_name,
                    "instructions": instructions,
                }
            )
    return records


def build_cfg_ctx2_lines(records: Sequence[Dict[str, object]]) -> List[str]:
    lines: List[str] = []
    for record in records:
        instructions = list(record["instructions"])
        for idx in range(len(instructions) - 1):
            lines.append("\t".join(instructions[idx : idx + 2]))
    return lines


def write_lines(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(line.rstrip() + "\n")


def find_viable_split_seed(
    function_blocks: Dict[str, List[str]],
    binary_by_func: Dict[str, str],
    test_fraction: float,
    max_pairs: int,
    max_queries: int,
    pool_size: int,
) -> int:
    instructions_by_func, blocks_by_func = reconstruct_blocks(function_blocks)
    func_ids = sorted(function_blocks.keys())

    for split_seed in range(256):
        train_func_ids, test_func_ids = split_function_ids(func_ids, test_fraction, split_seed)
        if not train_func_ids or not test_func_ids:
            continue

        func_groups: Dict[str, int] = {}
        checks: List[Tuple[bool, bool]] = []

        train_same_bb, _ = build_same_bb_pairs(blocks_by_func, train_func_ids, max_pairs, split_seed + 11)
        test_same_bb, _ = build_same_bb_pairs(blocks_by_func, test_func_ids, max_pairs, split_seed + 12)
        checks.append((bool(train_same_bb), bool(test_same_bb)))

        train_same_fn, _ = build_same_function_pairs(blocks_by_func, train_func_ids, func_groups, max_pairs, split_seed + 21)
        test_same_fn, _ = build_same_function_pairs(blocks_by_func, test_func_ids, func_groups, max_pairs, split_seed + 22)
        checks.append((bool(train_same_fn), bool(test_same_fn)))

        train_jump, _ = build_jump_target_pairs(instructions_by_func, train_func_ids, max_pairs, split_seed + 31, "mov")
        test_jump, _ = build_jump_target_pairs(instructions_by_func, test_func_ids, max_pairs, split_seed + 32, "mov")
        checks.append((bool(train_jump), bool(test_jump)))

        train_call, _ = build_call_target_pairs(
            instructions_by_func, blocks_by_func, train_func_ids, binary_by_func, max_pairs, split_seed + 41
        )
        test_call, _ = build_call_target_pairs(
            instructions_by_func, blocks_by_func, test_func_ids, binary_by_func, max_pairs, split_seed + 42
        )
        checks.append((bool(train_call), bool(test_call)))

        train_dref, _ = build_same_dref_pairs(instructions_by_func, train_func_ids, binary_by_func, max_pairs, split_seed + 51)
        test_dref, _ = build_same_dref_pairs(instructions_by_func, test_func_ids, binary_by_func, max_pairs, split_seed + 52)
        checks.append((bool(train_dref), bool(test_dref)))

        train_ret, _ = build_dref_retrieval_queries(
            instructions_by_func, train_func_ids, binary_by_func, max_queries, pool_size, split_seed + 61
        )
        test_ret, _ = build_dref_retrieval_queries(
            instructions_by_func, test_func_ids, binary_by_func, max_queries, pool_size, split_seed + 62
        )
        checks.append((bool(train_ret), bool(test_ret)))

        if all(train_ok and test_ok for train_ok, test_ok in checks):
            return split_seed

    raise RuntimeError("Could not find a split seed that supports every probing task in both train and test splits.")


def run(cmd: Sequence[str], cwd: Path) -> None:
    print("[Smoke] Running:", " ".join(cmd))
    subprocess.run(list(cmd), cwd=str(cwd), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Address-aware PalmTree end-to-end smoke")
    parser.add_argument(
        "--workdir",
        default=str(Path("output/smoke/addressaware_e2e").resolve()),
        help="Directory for synthetic data, checkpoints, and reports",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--clean", action="store_true", help="Remove the workdir before running")
    args = parser.parse_args()

    workdir = Path(args.workdir).resolve()
    if args.clean and workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    data_dir = workdir / "data"
    baseline_out = workdir / "baseline_pretrain"
    aa_out = workdir / "addressaware_pretrain"
    baseline_logs = workdir / "logs_baseline"
    aa_logs = workdir / "logs_addressaware"
    probe_output = workdir / "probe_report.json"
    vocab_path = workdir / "vocab.pkl"

    records = build_synthetic_function_records()
    function_blocks = {str(record["func_id"]): list(record["instructions"]) for record in records}
    binary_by_func = {str(record["func_id"]): str(record["binary"]) for record in records}
    cfg_lines = build_cfg_ctx2_lines(records)
    split_point = max(4, int(len(cfg_lines) * 0.75))
    cfg_train_lines = cfg_lines[:split_point]
    cfg_val_lines = cfg_lines[split_point:]

    write_lines(data_dir / "cfg_ctx2_train.txt", cfg_train_lines)
    write_lines(data_dir / "cfg_ctx2_val.txt", cfg_val_lines)
    with (data_dir / "function_blocks.json").open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2)

    split_seed = find_viable_split_seed(
        function_blocks=function_blocks,
        binary_by_func=binary_by_func,
        test_fraction=0.5,
        max_pairs=16,
        max_queries=4,
        pool_size=4,
    )
    print(f"[Smoke] Selected probing split_seed={split_seed}")

    common_train_args = [
        sys.executable,
        str(ROOT / "train.py"),
        "--cfg_train",
        str(data_dir / "cfg_ctx2_train.txt"),
        "--cfg_val",
        str(data_dir / "cfg_ctx2_val.txt"),
        "--vocab",
        str(vocab_path),
        "--hidden",
        "32",
        "--layers",
        "2",
        "--attn_heads",
        "4",
        "--seq_len",
        "32",
        "--epochs",
        "1",
        "--batch_size",
        "8",
        "--num_workers",
        "0",
        "--lr",
        "5e-4",
        "--enable_imc",
        "--instruction_mask_prob",
        "0.20",
        "--token_mask_prob",
        "0.15",
        "--early_stopping_patience",
        "0",
        "--seed",
        "7",
    ]

    run(
        common_train_args
        + [
            "--output_dir",
            str(baseline_out),
            "--log_dir",
            str(baseline_logs),
            "--disable_address_embedding",
            "--disable_var_embedding",
        ],
        cwd=ROOT,
    )
    run(
        common_train_args
        + [
            "--output_dir",
            str(aa_out),
            "--log_dir",
            str(aa_logs),
        ],
        cwd=ROOT,
    )

    baseline_ckpt = baseline_out / "best_bert.pt"
    aa_ckpt = aa_out / "best_bert.pt"
    if not baseline_ckpt.exists():
        baseline_ckpt = baseline_out / "latest_bert.pt"
    if not aa_ckpt.exists():
        aa_ckpt = aa_out / "latest_bert.pt"

    run(
        [
            sys.executable,
            str(PROBE_DIR / "probe_eval.py"),
            "--function_blocks",
            str(data_dir / "function_blocks.json"),
            "--vocab",
            str(vocab_path),
            "--baseline_checkpoint",
            str(baseline_ckpt),
            "--aa_checkpoint",
            str(aa_ckpt),
            "--output",
            str(probe_output),
            "--tasks",
            "same_bb",
            "same_function",
            "jump_target",
            "call_target",
            "same_dref",
            "dref_retrieval",
            "--hidden",
            "32",
            "--n_layers",
            "2",
            "--attn_heads",
            "4",
            "--seq_len",
            "32",
            "--max_train_pairs",
            "16",
            "--max_test_pairs",
            "8",
            "--max_train_queries",
            "4",
            "--max_test_queries",
            "2",
            "--retrieval_pool_size",
            "4",
            "--test_fraction",
            "0.5",
            "--split_seed",
            str(split_seed),
            "--few_shots",
            "2",
            "--few_shot_seeds",
            "0",
            "--probe_epochs",
            "5",
            "--probe_lr",
            "1e-3",
            "--probe_weight_decay",
            "1e-4",
            "--batch_size",
            "8",
            "--device",
            args.device,
        ],
        cwd=ROOT,
    )

    with probe_output.open("r", encoding="utf-8") as handle:
        report = json.load(handle)

    failing_tasks = {
        task: payload.get("error")
        for task, payload in report.get("tasks", {}).items()
        if isinstance(payload, dict) and payload.get("error")
    }
    if failing_tasks:
        raise RuntimeError(f"Probe smoke reported task construction failures: {failing_tasks}")

    summary = {
        "workdir": str(workdir),
        "baseline_checkpoint": str(baseline_ckpt),
        "addressaware_checkpoint": str(aa_ckpt),
        "probe_output": str(probe_output),
        "split_seed": split_seed,
        "tasks": sorted(report.get("tasks", {}).keys()),
    }
    with (workdir / "smoke_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("[Smoke] Success")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
