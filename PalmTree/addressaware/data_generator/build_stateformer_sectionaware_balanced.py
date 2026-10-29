#!/usr/bin/env python3
"""
Build a balanced section-aware CFG corpus from existing ctx2 inline files.

Unlike the budget-based subset builder, this script guarantees that every
available binary can contribute up to a fixed number of train/validation lines.
This is useful for large all-binary pretraining runs where we want broad binary
coverage while keeping the epoch time bounded.
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
    parser = argparse.ArgumentParser(description="Build balanced section-aware CFG ctx2 corpus")
    parser.add_argument("--cfg_root", required=True, help="Root containing *_cfg_2_inline.txt files")
    parser.add_argument("--raw_root", required=True, help="Root containing raw ELF binaries")
    parser.add_argument("--train_out", required=True, help="Output train corpus path")
    parser.add_argument("--val_out", required=True, help="Output val corpus path")
    parser.add_argument("--manifest_out", required=True, help="Output manifest JSON path")
    parser.add_argument(
        "--train_lines_per_file",
        type=int,
        default=900,
        help="Maximum rewritten training lines to take from each cfg file",
    )
    parser.add_argument(
        "--val_lines_per_file",
        type=int,
        default=100,
        help="Maximum rewritten validation lines to take from each cfg file after training lines",
    )
    parser.add_argument("--max_files", type=int, default=None, help="Optional cap on processed cfg files")
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
        "train_lines_per_file": args.train_lines_per_file,
        "val_lines_per_file": args.val_lines_per_file,
        "processed_files": [],
        "skipped_files": [],
    }

    processed_files = 0
    train_written = 0
    val_written = 0

    with train_out.open("w", encoding="utf-8") as train_handle, val_out.open("w", encoding="utf-8") as val_handle:
        for cfg_path in iter_cfg_files(cfg_root):
            if args.max_files is not None and processed_files >= args.max_files:
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
            with cfg_path.open("r", encoding="utf-8") as source:
                for raw_line in source:
                    rewritten = rewrite_line(raw_line, sections, min_addr, max_addr)
                    if file_train < args.train_lines_per_file:
                        train_handle.write(rewritten + "\n")
                        file_train += 1
                        train_written += 1
                    elif file_val < args.val_lines_per_file:
                        val_handle.write(rewritten + "\n")
                        file_val += 1
                        val_written += 1
                    else:
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

    manifest["train_lines_written"] = train_written
    manifest["val_lines_written"] = val_written
    manifest["num_processed_files"] = processed_files
    manifest["num_skipped_files"] = len(manifest["skipped_files"])

    with manifest_out.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(
        json.dumps(
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
        )
    )


if __name__ == "__main__":
    main()
