#!/usr/bin/env python3
"""
Batch-export ELF binaries into address-aware SymLM JSON using IDA Pro.

This is meant for the official SymLM binary release as well as local ELF trees.
It only processes ELF files, skips files whose JSON already exists, and writes
one IDA log per source binary.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

from tqdm import tqdm


OPT_LEVELS = ("O0", "O1", "O2", "O3")
ARCH_NAMES = {"x86", "x64", "arm", "mips"}


def sanitize_component(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", name.strip())
    return cleaned.strip("_") or "unknown"


def infer_opt_level(binary_path: Path) -> str:
    for candidate in [binary_path.parent] + list(binary_path.parents):
        name = candidate.name
        if name in OPT_LEVELS:
            return name
        for opt in OPT_LEVELS:
            if name.endswith("_" + opt):
                return opt
    return "unknown"


def build_source_tag(binary_path: Path, opt_level: str) -> str:
    arch = None
    for parent in binary_path.parents:
        if parent.name in ARCH_NAMES:
            arch = parent.name
            break

    parent_name = binary_path.parent.name
    grandparent_name = binary_path.parent.parent.name if binary_path.parent.parent != binary_path.parent else ""

    parts = []
    if arch:
        parts.append(arch)
    if opt_level != "unknown":
        parts.append(opt_level)
    if grandparent_name and grandparent_name not in ARCH_NAMES and grandparent_name not in OPT_LEVELS:
        parts.append(grandparent_name)
    if parent_name and parent_name not in ARCH_NAMES and parent_name not in OPT_LEVELS:
        parts.append(parent_name)
    parts.append(binary_path.name)

    sanitized = [sanitize_component(part) for part in parts if part]
    deduped = []
    for part in sanitized:
        if not deduped or deduped[-1] != part:
            deduped.append(part)
    return "_".join(deduped)


def is_elf(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            return f.read(4) == b"\x7fELF"
    except OSError:
        return False


def collect_elf_files(input_root: Path) -> list[Path]:
    files = []
    for path in input_root.rglob("*"):
        if path.is_file() and is_elf(path):
            files.append(path)
    files.sort(key=lambda p: (p.stat().st_size, str(p)))
    return files


def main():
    parser = argparse.ArgumentParser(description="Export official SymLM binaries with IDA in address-aware format")
    parser.add_argument("--input-root", required=True, help="Root directory of binaries to scan")
    parser.add_argument("--output-dir", required=True, help="Directory for exported JSON files")
    parser.add_argument("--logs-dir", required=True, help="Directory for IDA logs")
    parser.add_argument("--ida-path", default="/path/to/ida-pro-9.0/idat", help="Path to idat")
    parser.add_argument("--ida-script", required=True, help="Path to process_address_aware_ida.py")
    parser.add_argument("--symbols-file", required=True, help="Top-symbols file for smart merge")
    parser.add_argument("--disable-smart-merge", action="store_true", help="Disable smart-merge symbol normalization")
    parser.add_argument("--limit", type=int, default=None, help="Process at most this many ELF files")
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    logs_dir = Path(args.logs_dir)
    ida_script = Path(args.ida_script).resolve()

    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    elf_files = collect_elf_files(input_root)
    if args.limit is not None:
        elf_files = elf_files[: args.limit]

    print(f"[INFO] Found {len(elf_files)} ELF files under {input_root}")
    print(f"[INFO] Output dir: {output_dir}")
    print(f"[INFO] Logs dir: {logs_dir}")

    success = 0
    skipped = 0
    failed = 0
    failures = []
    smart_merge_flag = "false" if args.disable_smart_merge else "true"

    for binary_path in tqdm(elf_files, desc="IDA export"):
        opt_level = infer_opt_level(binary_path)
        source_tag = build_source_tag(binary_path, opt_level)
        out_json = output_dir / f"{source_tag}_addressaware_symlm.json"
        log_file = logs_dir / f"{source_tag}.log"

        if out_json.exists():
            skipped += 1
            continue

        cmd = [
            args.ida_path,
            "-A",
            f"-S{ida_script} {output_dir} {args.symbols_file} {smart_merge_flag}",
            str(binary_path),
        ]
        with log_file.open("w", encoding="utf-8") as logf:
            proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, text=True)

        if proc.returncode == 0 and out_json.exists():
            success += 1
        else:
            failed += 1
            failures.append({
                "binary": str(binary_path),
                "expected_output": str(out_json),
                "log_file": str(log_file),
                "returncode": proc.returncode,
            })

    summary = {
        "input_root": str(input_root),
        "num_elf_files": len(elf_files),
        "success": success,
        "skipped": skipped,
        "failed": failed,
        "failures": failures,
    }
    summary_path = output_dir / "export_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"[INFO] Success: {success}")
    print(f"[INFO] Skipped: {skipped}")
    print(f"[INFO] Failed : {failed}")
    print(f"[INFO] Summary written to {summary_path}")


if __name__ == "__main__":
    main()
