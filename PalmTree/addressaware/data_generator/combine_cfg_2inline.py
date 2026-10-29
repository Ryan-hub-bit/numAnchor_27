#!/usr/bin/env python3
"""
Combine CFG 2-inline files for PRETRAINING (no train/val/test split).
Combines ALL CFG lines from all *_2_inline.txt files into a single file.
Includes smart_merge: keeps only top 400 most common symbols, replaces others with .plt
"""

import os
import random
import re
from pathlib import Path
from tqdm import tqdm
from collections import Counter


def collect_all_files(base_dir, recursive=False):
    """Collect all *_2_inline.txt files from a directory."""
    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"[WARNING] Directory not found: {base_dir}")
        return []

    files = list(base_path.rglob("*_2_inline.txt")) if recursive else list(base_path.glob("*_2_inline.txt"))
    mode = "recursively" if recursive else "in top level"
    print(f"[INFO] Found {len(files)} files {mode} under {base_dir}")
    return files


def get_top_symbols(file_list, top_n=400, output_file=None):
    """
    Analyze all files and find the top N most common symbols (dot-prefixed names).
    Pattern matches: .symbol_name (e.g., .plt, .got, .text, etc.)
    Saves to file for consistency across runs.
    
    Args:
        file_list: List of file paths to analyze
        top_n: Number of top symbols to keep (default: 400)
        output_file: Path to save top symbols (optional)
    
    Returns:
        Set of top N symbol names, compiled regex pattern
    """
    # Regex: Matches . followed by a letter or underscore, then any word characters
    pattern = re.compile(r'\.([a-zA-Z_][a-zA-Z0-9_]*)')
    counts = Counter()
    
    print(f"[INFO] Analyzing symbol frequencies across {len(file_list)} files...")
    for filepath in tqdm(file_list, desc="Analyzing symbols"):
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                counts.update(pattern.findall(line))
    
    # Get top N symbols
    top_symbols = set(symbol for symbol, _ in counts.most_common(top_n))
    
    print(f"\n[INFO] Found {len(counts)} unique symbols")
    print(f"[INFO] Keeping top {top_n} symbols, replacing {len(counts) - top_n} with .plt")
    print(f"\n[INFO] Top 20 symbols:")
    for symbol, count in counts.most_common(20):
        status = "KEEP" if symbol in top_symbols else "REPLACE"
        print(f"  .{symbol}: {count:,} occurrences [{status}]")
    
    # Save to file
    if output_file:
        with open(output_file, 'w') as f:
            for symbol, count in counts.most_common(top_n):
                f.write(f".{symbol}\t{count}\n")
        print(f"\n[INFO] Saved top {top_n} symbols to: {output_file}")
    
    return top_symbols, pattern


def load_top_symbols(symbols_file):
    """Load top symbols from file."""
    try:
        with open(symbols_file, 'r') as f:
            # Extract symbol names (without the leading .)
            top_symbols = set(line.split('\t')[0].lstrip('.') for line in f if line.strip())
        pattern = re.compile(r'\.([a-zA-Z_][a-zA-Z0-9_]*)')
        print(f"[INFO] Loaded {len(top_symbols)} symbols from {symbols_file}")
        return top_symbols, pattern
    except Exception as e:
        print(f"[ERROR] Failed to load symbols: {e}")
        return None, None


def smart_merge_line(line, top_symbols, pattern):
    """Replace low-frequency symbols with .plt in a single line."""
    def replace_symbol(match):
        symbol_name = match.group(1)
        if symbol_name in top_symbols:
            return match.group(0)  # Keep as-is
        else:
            return '.plt'  # Replace with .plt
    
    return pattern.sub(replace_symbol, line)


def count_lines_in_file(filepath):
    """Count lines in a file."""
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        return sum(1 for _ in f)


def sample_lines_from_file(filepath, sample_ratio, top_symbols=None, pattern=None):
    """
    Read lines from a file, apply optional smart_merge, and sample.
    
    Args:
        filepath: Path to the file
        sample_ratio: Ratio of lines to sample (0.0 to 1.0)
        top_symbols: Set of symbols to keep (for smart_merge)
        pattern: Compiled regex pattern (for smart_merge)
    
    Returns:
        List of sampled lines
    """
    lines = []
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            # Apply smart_merge if enabled
            if top_symbols is not None and pattern is not None:
                line = smart_merge_line(line, top_symbols, pattern)
            
            # Sample with probability = sample_ratio
            if sample_ratio >= 1.0 or random.random() < sample_ratio:
                lines.append(line.rstrip('\n'))
    
    return lines


def write_lines(lines, output_path):
    """Write lines to output file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(line + '\n')
    print(f"[INFO] Wrote {len(lines):,} lines to {output_path}")


def main():
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Combine CFG 2-inline files for pretraining")
    parser.add_argument("--input_dir", required=True, help="Directory containing *_2_inline.txt files")
    parser.add_argument("--output_dir", required=True, help="Output directory for combined file")
    parser.add_argument("--top_n", type=int, default=400, help="Number of top symbols to keep (default: 400)")
    parser.add_argument("--sample_ratio", type=float, default=1.0, help="Sampling ratio (0.0-1.0, default: 1.0 for all data)")
    parser.add_argument("--enable_smart_merge", action="store_true", help="Enable smart merge (replace low-freq symbols with .plt)")
    parser.add_argument("--output_name", default="cfg_2inline_pretrain.txt", help="Output file name (default: cfg_2inline_pretrain.txt)")
    parser.add_argument("--recursive", action="store_true", help="Recursively search for *_2_inline.txt files")
    
    args = parser.parse_args()
    
    # Paths
    input_dir = args.input_dir
    output_dir = args.output_dir
    output_file = os.path.join(output_dir, args.output_name)
    symbols_file = os.path.join(output_dir, "top_symbols_cfg_2inline.txt")
    
    # Configuration
    TOP_N_SYMBOLS = args.top_n
    SAMPLE_RATIO = args.sample_ratio
    ENABLE_SMART_MERGE = args.enable_smart_merge
    
    # Print configuration
    print("="*70)
    print("CFG 2-INLINE PRETRAINING DATA COMBINATION")
    print("="*70)
    print(f"Input directory: {input_dir}")
    print(f"Output file: {output_file}")
    print(f"Recursive search: {'ENABLED' if args.recursive else 'DISABLED'}")
    print(f"Sample ratio: {SAMPLE_RATIO:.1%} (for pretraining, typically use 1.0 = all data)")
    print(f"Smart merge: {'ENABLED' if ENABLE_SMART_MERGE else 'DISABLED'}")
    if ENABLE_SMART_MERGE:
        print(f"  - Keep top {TOP_N_SYMBOLS} symbols, replace others with .plt")
    print("="*70)
    
    # Step 1: Collect all files from input directory
    print("\n[STEP 1] Collecting files from input directory...")
    all_files = collect_all_files(input_dir, recursive=args.recursive)
    
    if len(all_files) == 0:
        print("[ERROR] No *_2_inline.txt files found in input directory!")
        sys.exit(1)
    
    print(f"\n[INFO] Files collected: {len(all_files)} files")
    
    # Step 2: Get top symbols for smart_merge (analyze ALL files together)
    top_symbols = None
    pattern = None
    if ENABLE_SMART_MERGE:
        print("\n[STEP 2] Analyzing symbols for smart_merge...")
        
        # Check if symbol file exists
        if os.path.exists(symbols_file):
            print(f"[INFO] Found existing symbol file: {symbols_file}")
            top_symbols, pattern = load_top_symbols(symbols_file)
            if top_symbols is None:
                print(f"[WARNING] Failed to load symbols, re-analyzing...")
                top_symbols, pattern = get_top_symbols(all_files, TOP_N_SYMBOLS, symbols_file)
        else:
            print(f"[INFO] No existing symbol file, analyzing and saving to: {symbols_file}")
            top_symbols, pattern = get_top_symbols(all_files, TOP_N_SYMBOLS, symbols_file)
    
    # Step 3: Count total lines
    step_num = 3 if ENABLE_SMART_MERGE else 2
    print(f"\n[STEP {step_num}] Counting total CFG lines...")
    
    total_lines = sum(count_lines_in_file(f) for f in tqdm(all_files, desc="Counting"))
    
    print(f"\n[INFO] Total CFG lines: {total_lines:,}")
    print(f"[INFO] Expected output (with {SAMPLE_RATIO:.1%} sampling): ~{int(total_lines * SAMPLE_RATIO):,} lines")
    
    # Step 4: Sample from ALL files and combine
    step_num = 4 if ENABLE_SMART_MERGE else 3
    print(f"\n[STEP {step_num}] Sampling and combining all files...")
    if ENABLE_SMART_MERGE:
        print(f"  Applying smart_merge: keeping top {TOP_N_SYMBOLS} symbols, replacing others with .plt")
    
    all_lines = []
    
    for filepath in tqdm(all_files, desc="Processing"):
        all_lines.extend(sample_lines_from_file(filepath, SAMPLE_RATIO, top_symbols=top_symbols, pattern=pattern))
    
    print(f"\n[INFO] Collected {len(all_lines):,} CFG lines")
    
    # Step 5: Shuffle to mix different binaries/optimizations
    print(f"\n[STEP 5] Shuffling data...")
    random.shuffle(all_lines)
    print(f"[INFO] Data shuffled")
    
    # Step 6: Write to output file
    print(f"\n[STEP 6] Writing output file...")
    write_lines(all_lines, output_file)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Input statistics:")
    print(f"  - Total files: {len(all_files)}")
    print(f"  - Total CFG lines (2-inline): {total_lines:,}")
    print(f"\nSampling:")
    print(f"  - Sampling ratio: {SAMPLE_RATIO:.1%}")
    print(f"  - Lines sampled: {len(all_lines):,} ({len(all_lines)/total_lines:.1%})")
    if ENABLE_SMART_MERGE:
        print(f"\nSmart merge:")
        print(f"  - Kept top {TOP_N_SYMBOLS} symbols")
        print(f"  - Replaced others with .plt")
        print(f"  - Symbol list saved to: {symbols_file}")
    print(f"\nOutput file:")
    print(f"  {output_file}")
    print(f"  CFG lines (2-inline): {len(all_lines):,}")
    print("="*70)
    print("[SUCCESS] CFG 2-inline pretraining data ready!")


if __name__ == "__main__":
    main()
