#!/usr/bin/env python3
"""
Build a section-aware Arch CFG pretrain subset from existing ctx2 inline files.

This script rewrites legacy tokens of the form:
    daddr(0xADDR:old_pos1:old_pos2:old_pos3)

into section-specific data-address tokens:
    daddr_rodata(0xADDR:section_pos:offset_in_section)
    daddr_bss(...)
    daddr_data(...)
    daddr_other(...)

using the raw ELF section layout of the corresponding binary.

The output is intended for address-aware PalmTree pretraining with CFG-only,
context=2 data.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from section_utils import classify_data_section, format_daddr_operand


DADDR_RE = re.compile(r"daddr\((0x[0-9a-fA-F]+):([0-9.]+):([0-9.]+):([0-9.]+)\)")
SECTION_RE = re.compile(
    r"^\s*\[\s*\d+\]\s+(\S+)\s+\S+\s+([0-9A-Fa-f]+)\s+[0-9A-Fa-f]+\s+([0-9A-Fa-f]+)\s+\S+\s+([A-Z]+)\b"
)


@dataclass(frozen=True)
class SectionRecord:
    name: str
    start: int
    end: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build section-aware Arch CFG ctx2 subset")
    parser.add_argument("--cfg_root", required=True, help="Root containing *_cfg_2_inline.txt files")
    parser.add_argument("--raw_root", required=True, help="Root containing raw ELF binaries")
    parser.add_argument("--train_out", required=True, help="Output train corpus path")
    parser.add_argument("--val_out", required=True, help="Output val corpus path")
    parser.add_argument("--manifest_out", required=True, help="Output manifest JSON path")
    parser.add_argument("--train_line_budget", type=int, default=3000000, help="Maximum rewritten train lines")
    parser.add_argument("--val_line_budget", type=int, default=200000, help="Maximum rewritten val lines")
    parser.add_argument("--max_files", type=int, default=None, help="Optional cap on processed cfg files")
    parser.add_argument(
        "--max_lines_per_file",
        type=int,
        default=20000,
        help="Optional cap on how many rewritten lines to take from each cfg file",
    )
    return parser.parse_args()


def parse_sections_from_readelf(binary_path: Path) -> Tuple[List[SectionRecord], int, int]:
    proc = subprocess.run(
        ["readelf", "-W", "-S", str(binary_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    alloc_sections: List[SectionRecord] = []
    for line in proc.stdout.splitlines():
        match = SECTION_RE.match(line)
        if not match:
            continue
        name = match.group(1)
        start = int(match.group(2), 16)
        size = int(match.group(3), 16)
        flags = match.group(4)
        if "A" not in flags or size <= 0:
            continue
        alloc_sections.append(SectionRecord(name=name, start=start, end=start + size))

    if not alloc_sections:
        return [], 0, 0

    min_addr = min(section.start for section in alloc_sections)
    max_addr = max(section.end for section in alloc_sections)
    return alloc_sections, min_addr, max_addr


def find_section(sections: List[SectionRecord], address: int) -> Optional[SectionRecord]:
    for section in sections:
        if section.start <= address < section.end:
            return section
    return None


def rewrite_line(line: str, sections: List[SectionRecord], min_addr: int, max_addr: int) -> str:
    def repl(match: re.Match[str]) -> str:
        address_hex = match.group(1)
        address = int(address_hex, 16)
        section = find_section(sections, address)
        if section is None:
            return format_daddr_operand(address_hex, 0.0, 0.0, 4)

        is_data_like, section_kind = classify_data_section(section.name)
        if max_addr > min_addr:
            section_pos = max(0.0, min(1.0, (section.start - min_addr) / float(max_addr - min_addr)))
        else:
            section_pos = 0.0

        if section.end > section.start:
            offset_in_section = max(0.0, min(1.0, (address - section.start) / float(section.end - section.start)))
        else:
            offset_in_section = 0.0

        if not is_data_like:
            section_kind = 4
        return format_daddr_operand(address_hex, section_pos, offset_in_section, section_kind)

    return DADDR_RE.sub(repl, line.rstrip("\n"))


def iter_cfg_files(cfg_root: Path) -> Iterable[Path]:
    yield from sorted(cfg_root.rglob("*_cfg_2_inline.txt"))


def binary_path_for_cfg(cfg_root: Path, raw_root: Path, cfg_path: Path) -> Optional[Path]:
    rel_dir = cfg_path.parent.relative_to(cfg_root)
    binary_name = cfg_path.name[: -len("_cfg_2_inline.txt")]
    candidate = raw_root / rel_dir / binary_name
    return candidate if candidate.exists() else None


def main() -> None:
    args = parse_args()

    cfg_root = Path(args.cfg_root).resolve()
    raw_root = Path(args.raw_root).resolve()
    train_out = Path(args.train_out).resolve()
    val_out = Path(args.val_out).resolve()
    manifest_out = Path(args.manifest_out).resolve()

    train_out.parent.mkdir(parents=True, exist_ok=True)
    val_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.parent.mkdir(parents=True, exist_ok=True)

    section_cache: Dict[str, Tuple[List[SectionRecord], int, int]] = {}
    manifest: Dict[str, object] = {
        "cfg_root": str(cfg_root),
        "raw_root": str(raw_root),
        "train_out": str(train_out),
        "val_out": str(val_out),
        "train_line_budget": args.train_line_budget,
        "val_line_budget": args.val_line_budget,
        "processed_files": [],
        "skipped_files": [],
    }

    train_written = 0
    val_written = 0
    processed_files = 0

    files_by_dir: Dict[Path, List[Path]] = {}
    for cfg_path in iter_cfg_files(cfg_root):
        files_by_dir.setdefault(cfg_path.parent, []).append(cfg_path)
    directory_order = sorted(files_by_dir.keys())

    with train_out.open("w", encoding="utf-8") as train_handle, val_out.open("w", encoding="utf-8") as val_handle:
        while directory_order:
            if args.max_files is not None and processed_files >= args.max_files:
                break
            if train_written >= args.train_line_budget and val_written >= args.val_line_budget:
                break

            next_round: List[Path] = []
            for directory in directory_order:
                queue = files_by_dir[directory]
                if not queue:
                    continue
                cfg_path = queue.pop(0)
                if queue:
                    next_round.append(directory)

                if args.max_files is not None and processed_files >= args.max_files:
                    break
                if train_written >= args.train_line_budget and val_written >= args.val_line_budget:
                    break

                binary_path = binary_path_for_cfg(cfg_root, raw_root, cfg_path)
                if binary_path is None:
                    manifest["skipped_files"].append({"cfg_path": str(cfg_path), "reason": "missing_raw_binary"})
                    continue

                cache_key = str(binary_path)
                if cache_key not in section_cache:
                    try:
                        section_cache[cache_key] = parse_sections_from_readelf(binary_path)
                    except subprocess.CalledProcessError as exc:
                        manifest["skipped_files"].append(
                            {"cfg_path": str(cfg_path), "reason": f"readelf_failed:{exc.returncode}"}
                        )
                        continue

                sections, min_addr, max_addr = section_cache[cache_key]
                if not sections:
                    manifest["skipped_files"].append({"cfg_path": str(cfg_path), "reason": "no_alloc_sections"})
                    continue

                file_train = 0
                file_val = 0
                file_total = 0
                with cfg_path.open("r", encoding="utf-8") as source:
                    for raw_line in source:
                        rewritten = rewrite_line(raw_line, sections, min_addr, max_addr)
                        if train_written < args.train_line_budget:
                            train_handle.write(rewritten + "\n")
                            train_written += 1
                            file_train += 1
                        elif val_written < args.val_line_budget:
                            val_handle.write(rewritten + "\n")
                            val_written += 1
                            file_val += 1
                        else:
                            break

                        file_total += 1
                        if args.max_lines_per_file is not None and file_total >= args.max_lines_per_file:
                            break

                manifest["processed_files"].append(
                    {
                        "cfg_path": str(cfg_path),
                        "binary_path": str(binary_path),
                        "train_lines": file_train,
                        "val_lines": file_val,
                    }
                )
                processed_files += 1

            directory_order = next_round

    manifest["train_lines_written"] = train_written
    manifest["val_lines_written"] = val_written
    manifest["num_processed_files"] = processed_files
    manifest["num_skipped_files"] = len(manifest["skipped_files"])

    with manifest_out.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(json.dumps(
        {
            "train_lines_written": train_written,
            "val_lines_written": val_written,
            "num_processed_files": processed_files,
            "num_skipped_files": len(manifest["skipped_files"]),
            "train_out": str(train_out),
            "val_out": str(val_out),
            "manifest_out": str(manifest_out),
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
