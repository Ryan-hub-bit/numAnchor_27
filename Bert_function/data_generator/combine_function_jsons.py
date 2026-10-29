#!/usr/bin/env python3
"""
Build function-level pretraining corpora from ``*_functions.json`` files.

Expected input format:
- each JSON file is a list of function records
- each record contains a ``funcins`` field with one full function encoded as
  tab-separated instructions

This script streams records file-by-file so it works well on larger corpora.
"""

import json
import os
import random
import re
from collections import Counter
from pathlib import Path

from tqdm import tqdm


def collect_all_files(base_dir, recursive=False):
    """Collect all *_functions.json files from a directory."""
    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"[WARNING] Directory not found: {base_dir}")
        return []

    files = (
        list(base_path.rglob("*_functions.json"))
        if recursive
        else list(base_path.glob("*_functions.json"))
    )
    mode = "recursively" if recursive else "in top level"
    print(f"[INFO] Found {len(files)} files {mode} under {base_dir}")
    return files


def iter_funcins_from_json(path):
    """Yield non-empty funcins strings from one JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)

    if not isinstance(records, list):
        raise ValueError(f"{path} does not contain a JSON list")

    for record in records:
        funcins = record.get("funcins", "").strip()
        if funcins:
            yield funcins


def get_top_symbols(file_list, top_n=400, output_file=None):
    """Count dot-prefixed symbols across all function-inline records."""
    pattern = re.compile(r"\.([a-zA-Z_][a-zA-Z0-9_]*)")
    counts = Counter()

    print(f"[INFO] Analyzing symbol frequencies across {len(file_list)} files...")
    for filepath in tqdm(file_list, desc="Analyzing symbols"):
        for funcins in iter_funcins_from_json(filepath):
            counts.update(pattern.findall(funcins))

    top_symbols = set(symbol for symbol, _ in counts.most_common(top_n))

    print(f"\n[INFO] Found {len(counts)} unique symbols")
    print(f"[INFO] Keeping top {top_n} symbols, replacing {max(0, len(counts) - top_n)} with .plt")

    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            for symbol, count in counts.most_common(top_n):
                f.write(f".{symbol}\t{count}\n")
        print(f"[INFO] Saved top {top_n} symbols to: {output_file}")

    return top_symbols, pattern


def load_top_symbols(symbols_file):
    """Load a precomputed top-symbol list."""
    with open(symbols_file, "r", encoding="utf-8") as f:
        top_symbols = set(line.split("\t")[0].lstrip(".") for line in f if line.strip())
    pattern = re.compile(r"\.([a-zA-Z_][a-zA-Z0-9_]*)")
    print(f"[INFO] Loaded {len(top_symbols)} symbols from {symbols_file}")
    return top_symbols, pattern


def smart_merge_line(line, top_symbols, pattern):
    """Replace low-frequency symbols with .plt in one line."""

    def replace_symbol(match):
        symbol_name = match.group(1)
        if symbol_name in top_symbols:
            return match.group(0)
        return ".plt"

    return pattern.sub(replace_symbol, line)


def count_total_functions(file_list):
    """Count functions by iterating through funcins entries."""
    total = 0
    for filepath in tqdm(file_list, desc="Counting functions"):
        for _ in iter_funcins_from_json(filepath):
            total += 1
    return total


def main():
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Combine *_functions.json files into function-level train/val corpora")
    parser.add_argument("--input_dir", required=True, help="Directory containing *_functions.json files")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--train_output_name", default="train_functions.txt", help="Train output file name")
    parser.add_argument("--val_output_name", default="val_functions.txt", help="Validation output file name")
    parser.add_argument("--val_ratio", type=float, default=0.05, help="Validation split ratio")
    parser.add_argument("--top_n", type=int, default=400, help="Number of top symbols to keep")
    parser.add_argument("--sample_ratio", type=float, default=1.0, help="Sampling ratio")
    parser.add_argument("--enable_smart_merge", action="store_true", help="Collapse low-frequency symbols into .plt")
    parser.add_argument("--recursive", action="store_true", help="Recursively search for *_functions.json files")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed")
    args = parser.parse_args()

    random.seed(args.seed)

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    train_output = os.path.join(output_dir, args.train_output_name)
    val_output = os.path.join(output_dir, args.val_output_name)
    symbols_file = os.path.join(output_dir, "top_symbols_functions.txt")

    print("=" * 70)
    print("FUNCTION-LEVEL JSON CORPUS COMBINATION")
    print("=" * 70)
    print(f"Input directory: {args.input_dir}")
    print(f"Train output: {train_output}")
    print(f"Val output: {val_output}")
    print(f"Recursive search: {'ENABLED' if args.recursive else 'DISABLED'}")
    print(f"Sample ratio: {args.sample_ratio:.1%}")
    print(f"Validation ratio: {args.val_ratio:.1%}")
    print(f"Smart merge: {'ENABLED' if args.enable_smart_merge else 'DISABLED'}")
    print("=" * 70)

    all_files = collect_all_files(args.input_dir, recursive=args.recursive)
    if not all_files:
        print("[ERROR] No *_functions.json files found in input directory!")
        sys.exit(1)

    top_symbols = None
    pattern = None
    if args.enable_smart_merge:
        if os.path.exists(symbols_file):
            top_symbols, pattern = load_top_symbols(symbols_file)
        else:
            top_symbols, pattern = get_top_symbols(all_files, args.top_n, symbols_file)

    print("\n[STEP] Counting total functions...")
    total_functions = count_total_functions(all_files)
    print(f"[INFO] Total functions: {total_functions:,}")

    shuffled_files = list(all_files)
    random.shuffle(shuffled_files)

    print("\n[STEP] Writing train/val outputs...")
    train_count = 0
    val_count = 0
    seen_count = 0
    sampled_count = 0

    with open(train_output, "w", encoding="utf-8") as train_f, open(val_output, "w", encoding="utf-8") as val_f:
        for filepath in tqdm(shuffled_files, desc="Processing files"):
            records = list(iter_funcins_from_json(filepath))
            random.shuffle(records)

            for line in records:
                seen_count += 1
                if args.sample_ratio < 1.0 and random.random() >= args.sample_ratio:
                    continue

                if top_symbols is not None and pattern is not None:
                    line = smart_merge_line(line, top_symbols, pattern)

                sampled_count += 1
                if args.val_ratio > 0.0 and random.random() < args.val_ratio:
                    val_f.write(line + "\n")
                    val_count += 1
                else:
                    train_f.write(line + "\n")
                    train_count += 1

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Input files: {len(all_files)}")
    print(f"Input functions: {total_functions:,}")
    print(f"Functions visited: {seen_count:,}")
    print(f"Functions sampled: {sampled_count:,}")
    print(f"Train functions: {train_count:,}")
    print(f"Val functions: {val_count:,}")
    print(f"Seed: {args.seed}")
    if args.enable_smart_merge:
        print(f"Top symbols file: {symbols_file}")
    print("=" * 70)
    print("[SUCCESS] Function-level train/val corpora are ready")


if __name__ == "__main__":
    main()
