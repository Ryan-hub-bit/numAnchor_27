#!/usr/bin/env python3
"""
Convert address-aware SymLM JSON exports into PalmTree-style fixed-width
pretraining windows.

For each exported JSON shard:
- read each function's TAB-joined inline instructions
- generate sliding windows of exactly K instructions
- write one line per window, matching PalmTree's *_cfg_{K}_inline.txt format

This keeps the final MLM corpus aligned with the user's existing PalmTree
pretraining input shape for both 2-inline and 8-inline settings.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def iter_windows(funcins: str, window_size: int):
    instructions = [inst for inst in funcins.split("\t") if inst.strip()]
    if len(instructions) < window_size:
        return
    for start in range(0, len(instructions) - window_size + 1):
        yield "\t".join(instructions[start:start + window_size])


def main():
    parser = argparse.ArgumentParser(description="Build PalmTree-style cfg windows from address-aware SymLM exports")
    parser.add_argument("--input-dir", required=True, help="Directory containing *_addressaware_symlm.json files")
    parser.add_argument("--output-dir", required=True, help="Directory where *_cfg_{K}_inline.txt files will be written")
    parser.add_argument("--window-size", type=int, required=True, choices=[2, 8], help="Instruction window size")
    parser.add_argument("--min-instructions", type=int, default=None, help="Minimum instructions per function; defaults to window size")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    min_instructions = args.min_instructions or args.window_size

    total_functions = 0
    total_windows = 0
    written_files = 0

    for json_path in sorted(input_dir.glob("*.json")):
        try:
            records = json.loads(json_path.read_text())
        except Exception as exc:
            print(f"[WARNING] Failed to read {json_path}: {exc}")
            continue

        out_name = json_path.name.replace("_addressaware_symlm.json", f"_cfg_{args.window_size}_inline.txt")
        out_path = output_dir / out_name

        num_windows = 0
        with out_path.open("w", encoding="utf-8") as out_f:
            for record in records:
                funcins = record.get("funcins", "").strip()
                num_instr = record.get("num_instructions", 0)
                if not funcins or num_instr < min_instructions:
                    continue
                total_functions += 1
                for window in iter_windows(funcins, args.window_size):
                    out_f.write(window + "\n")
                    num_windows += 1

        total_windows += num_windows
        written_files += 1

    print(f"[INFO] Input dir        : {input_dir}")
    print(f"[INFO] Output dir       : {output_dir}")
    print(f"[INFO] Window size      : {args.window_size}")
    print(f"[INFO] Functions used   : {total_functions}")
    print(f"[INFO] Windows produced : {total_windows}")
    print(f"[INFO] Files written    : {written_files}")


if __name__ == "__main__":
    main()
