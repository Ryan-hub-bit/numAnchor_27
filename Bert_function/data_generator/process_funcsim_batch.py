#!/usr/bin/env python3
"""
Batch processor for funcsim dataset.

Orchestrates IDA batch processing across all binaries, then assembles:
  function_blocks.json  : {func_id -> [instr_str, ...]}
  funcsim_pairs.json    : {func_id -> {"ground_truth": [other_func_id, ...]}}

These are the exact formats expected by PalmTree/funcsim/dataloader.py.

Directory structure:
  /path/to/numAnchor_artifact/addressaware/funcsim_dataset/
    binary_O0/  binary_O1/  binary_O2/  binary_O3/

Each IDA run produces: OUTPUT_DIR/{binary_name}_functions.json
  [{func_id, binary, function, opt_level, funcins}, ...]
  where funcins = TAB-joined instruction strings (func_gen.py format)
"""

import argparse
import os
import json
import subprocess
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm


# Configuration
FUNCSIM_BASE = "/path/to/numAnchor_artifact/addressaware/funcsim_dataset"
OUTPUT_DIR = "/path/to/numAnchor_artifact/addressaware/funcsim_block_8"
IDA_PATH = "/path/to/ida-pro-9.0/idat"
IDA_SCRIPT = Path(__file__).parent / "process_funcsim_ida.py"
SYMBOLS_FILE = "/path/to/numAnchor_artifact/addressaware/cfg_train/top_symbols_cfg_8inline.txt"
ENABLE_SMART_MERGE = True


def split_variant_dir_name(dir_name):
    """
    Parse dataset directory names into (base_name, opt_level).

    Supports both naming styles:
      - bind_O0
      - coreutils-7.6-O0

    Returns:
      (base_name, opt_level) or (None, None) if the name is not an optimization
      variant directory.
    """
    for marker in ("_O", "-O"):
        if marker in dir_name:
            base_name, suffix = dir_name.rsplit(marker, 1)
            opt_level = "O" + suffix
            if opt_level in ["O0", "O1", "O2", "O3"]:
                return base_name, opt_level
    return None, None


def strip_opt_suffix(name):
    """
    Remove a trailing optimization suffix from a dataset/binary name.

    Examples:
      bind_O0           -> bind
      coreutils-7.6-O2  -> coreutils-7.6
    """
    for marker in ("_O", "-O"):
        if marker in name:
            base_name, suffix = name.rsplit(marker, 1)
            if ("O" + suffix) in ["O0", "O1", "O2", "O3"]:
                return base_name
    return name


def process_single_binary(binary_path, opt_level, parent_dir, output_dir, symbols_file, enable_smart_merge):
    """
    Run IDA on a single binary.
    Returns path to the generated JSON file.
    
    Args:
        binary_path: Full path to binary
        opt_level: Optimization level (O0, O1, O2, O3)
        parent_dir: Directory containing the binary (used for output naming)
        output_dir: Where to save output
        symbols_file: Path to symbols file
        enable_smart_merge: Whether to use smart merge
    """
    binary_name = Path(binary_path).name
    
    # Build IDA command - pass opt_level as additional argument
    smart_merge_str = "true" if enable_smart_merge else "false"
    cmd = [
        IDA_PATH,
        "-A",
        f"-S{IDA_SCRIPT} {output_dir} {symbols_file} {smart_merge_str}",
        binary_path
    ]
    
    # Return expected output path (include opt_level to avoid overwriting across opt levels)
    output_json = Path(output_dir) / f"{binary_name}_{opt_level}_functions.json"
    if output_json.exists():
        print(f"  [SKIP] Already exists: {output_json.name}")
        return output_json

    print(f"  Running IDA on {binary_name} ({opt_level})...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"  [ERROR] IDA failed on {binary_name}")
        print(result.stderr)
        return None
    
    if output_json.exists():
        return output_json
    else:
        print(f"  [WARNING] Expected output not found: {output_json}")
        return None


def collect_all_binaries(base_dir):
    """
    Collect all binaries from directories named {binary_name}_O[0-3].
    Returns: {(base_name, binary_file): {O0: (dir_path, opt_level), O1: ..., ...}}
    
    Directory structure: /funcsim_dataset/{binary_name}_O0/{executables}
    """
    binary_groups = defaultdict(dict)
    
    base_path = Path(base_dir)
    
    # Scan all directories
    for dir_path in base_path.iterdir():
        if not dir_path.is_dir():
            continue
        
        dir_name = dir_path.name
        
        # Extract opt level from directory name.
        base_name, opt_level = split_variant_dir_name(dir_name)
        if opt_level is not None:
            # Find all executables in this directory (skip .i64 files)
            for binary_path in dir_path.iterdir():
                if binary_path.is_file() and not binary_path.name.endswith('.i64'):
                    # Group by (base_name, binary_file)
                    key = (base_name, binary_path.name)
                    binary_groups[key][opt_level] = (str(binary_path), opt_level, str(dir_path))
    
    return binary_groups


def process_all_binaries(binary_groups, output_dir, symbols_file, enable_smart_merge):
    """
    Process all binaries through IDA.
    Returns list of output JSON file paths.
    """
    output_files = []
    
    print(f"[INFO] Processing {len(binary_groups)} binary groups...")
    
    for (base_name, binary_file), opt_dict in tqdm(binary_groups.items(), desc="Processing binaries"):
        for opt_level in ['O0', 'O1', 'O2', 'O3']:
            if opt_level not in opt_dict:
                print(f"[WARNING] Missing {opt_level} for {base_name}/{binary_file}")
                continue
            
            binary_path, opt, parent_dir = opt_dict[opt_level]
            output_file = process_single_binary(
                binary_path, opt_level, parent_dir, output_dir, symbols_file, enable_smart_merge
            )
            
            if output_file:
                output_files.append(output_file)
    
    print(f"[INFO] Successfully processed {len(output_files)} binaries")
    return output_files


def load_all_functions(output_files):
    """
    Load all functions from individual JSON files.
    Returns: list of all functions with reassigned func_ids
    """
    all_functions = []
    func_id = 0
    
    for json_file in tqdm(output_files, desc="Loading functions"):
        try:
            with open(json_file, 'r') as f:
                functions = json.load(f)
            
            # Reassign func_ids globally
            for func in functions:
                func['func_id'] = func_id
                all_functions.append(func)
                func_id += 1
        except Exception as e:
            print(f"[ERROR] Failed to load {json_file}: {e}")
    
    print(f"[INFO] Loaded {len(all_functions)} total functions")
    return all_functions


def build_function_index(all_functions):
    """
    Build index: {(base_binary_name, function_name, opt_level): func_id}
    """
    func_index = {}
    
    for func in all_functions:
        binary = func['binary']
        function = func['function']
        opt_level = func['opt_level']
        func_id = func['func_id']
        
        # Extract base binary name (remove optimization suffix if present)
        base_binary = strip_opt_suffix(binary)
        
        key = (base_binary, function, opt_level)
        func_index[key] = func_id
    
    return func_index


def generate_symmetric_pairs(all_functions, func_index):
    """
    Generate symmetric function pairs.
    
    For each function at optimization level X, pair it with the same function
    at all other optimization levels (if they exist).
    
    Returns: list of {func_id, gt_pairs: [other_func_ids]}
    """
    function_pairs = []
    
    print("[INFO] Generating symmetric pairs...")
    
    for func in tqdm(all_functions, desc="Generating pairs"):
        binary = func['binary']
        function = func['function']
        opt_level = func['opt_level']
        func_id = func['func_id']
        
        # Extract base binary name
        base_binary = strip_opt_suffix(binary)
        
        # Find all other optimization levels
        gt_pairs = []
        for other_opt in ['O0', 'O1', 'O2', 'O3']:
            if other_opt == opt_level:
                continue
            
            key = (base_binary, function, other_opt)
            if key in func_index:
                gt_pairs.append(func_index[key])

        if gt_pairs:
            function_pairs.append({
                'func_id': func_id,
                'gt_pairs': gt_pairs
            })

    print(f"[INFO] Generated {len(function_pairs)} function pairs")
    return function_pairs


def build_dataloader_outputs(all_functions, function_pairs):
    """
    Convert raw lists into the formats expected by funcsim/dataloader.py:

      function_blocks.json  : {str(func_id): [instr_str, instr_str, ...]}
        - funcins (TAB-joined) is split back into a list of instruction strings

      funcsim_pairs.json    : {str(func_id): {"ground_truth": [func_id, ...]}}
        - func_id keys are strings; ground_truth values are int lists
    """
    # function_blocks: split TAB-joined funcins into instruction list
    function_blocks = {}
    for func in all_functions:
        fid     = str(func['func_id'])
        funcins = func.get('funcins', '')
        instrs  = [s for s in funcins.split('\t') if s.strip()]
        function_blocks[fid] = instrs

    # funcsim_pairs: {func_id -> {"ground_truth": [...]}}
    # ground_truth IDs must be strings to match function_blocks keys in dataloader
    funcsim_pairs = {}
    for entry in function_pairs:
        fid = str(entry['func_id'])
        funcsim_pairs[fid] = {"ground_truth": [str(x) for x in entry['gt_pairs']]}

    return function_blocks, funcsim_pairs


def parse_args():
    parser = argparse.ArgumentParser(description="Build PalmTree funcsim data from a directory of optimization-level subfolders")
    parser.add_argument("--base_dir", default=FUNCSIM_BASE, help="Root directory containing *_O0/*_O1/*_O2/*_O3 subdirectories")
    parser.add_argument("--output_dir", default=OUTPUT_DIR, help="Output directory for function_blocks.json and funcsim_pairs.json")
    parser.add_argument("--ida_path", default=IDA_PATH, help="Path to IDA batch binary")
    parser.add_argument("--symbols_file", default=SYMBOLS_FILE, help="Top-symbols file used for smart merge")
    parser.add_argument("--enable_smart_merge", dest="enable_smart_merge", action="store_true", help="Enable smart merge for external symbols")
    parser.add_argument("--disable_smart_merge", dest="enable_smart_merge", action="store_false", help="Disable smart merge for external symbols")
    parser.set_defaults(enable_smart_merge=ENABLE_SMART_MERGE)
    return parser.parse_args()


def main():
    global FUNCSIM_BASE, OUTPUT_DIR, IDA_PATH, SYMBOLS_FILE, ENABLE_SMART_MERGE
    args = parse_args()
    FUNCSIM_BASE = args.base_dir
    OUTPUT_DIR = args.output_dir
    IDA_PATH = args.ida_path
    SYMBOLS_FILE = args.symbols_file
    ENABLE_SMART_MERGE = args.enable_smart_merge

    print("=" * 80)
    print("Funcsim Dataset Batch Processor")
    print("=" * 80)
    print(f"Base dir           : {FUNCSIM_BASE}")
    print(f"Output dir         : {OUTPUT_DIR}")
    print(f"IDA path           : {IDA_PATH}")
    print(f"Symbols file       : {SYMBOLS_FILE}")
    print(f"Smart merge        : {ENABLE_SMART_MERGE}")

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    fb_file = Path(OUTPUT_DIR) / "function_blocks.json"
    fp_file = Path(OUTPUT_DIR) / "funcsim_pairs.json"
    if fb_file.exists() and fp_file.exists():
        print(f"[SKIP] Final outputs already exist, nothing to do:")
        print(f"  {fb_file}")
        print(f"  {fp_file}")
        return

    # Step 1: collect binaries
    print("\n[STEP 1] Collecting binaries...")
    binary_groups = collect_all_binaries(FUNCSIM_BASE)
    print(f"Found {len(binary_groups)} binary groups")

    # Step 2: run IDA on each binary
    print("\n[STEP 2] Processing binaries through IDA...")
    output_files = process_all_binaries(
        binary_groups, OUTPUT_DIR, SYMBOLS_FILE, ENABLE_SMART_MERGE
    )
    if not output_files:
        print("[ERROR] No binaries processed successfully")
        return

    # Step 3: load per-binary JSON files
    print("\n[STEP 3] Loading functions...")
    all_functions = load_all_functions(output_files)

    # Step 4: build lookup index
    print("\n[STEP 4] Building function index...")
    func_index = build_function_index(all_functions)

    # Step 5: generate cross-opt symmetric pairs
    print("\n[STEP 5] Generating symmetric pairs...")
    function_pairs = generate_symmetric_pairs(all_functions, func_index)

    # Step 6: convert to dataloader format and write
    print("\n[STEP 6] Building dataloader outputs...")
    function_blocks, funcsim_pairs = build_dataloader_outputs(all_functions, function_pairs)

    print("\n[STEP 7] Writing outputs...")
    fb_file = Path(OUTPUT_DIR) / "function_blocks.json"
    fp_file = Path(OUTPUT_DIR) / "funcsim_pairs.json"

    if fb_file.exists():
        print(f"[SKIP] Already exists: {fb_file}")
    else:
        with open(fb_file, 'w') as f:
            json.dump(function_blocks, f)
        print(f"Wrote: {fb_file}  ({len(function_blocks)} functions)")

    if fp_file.exists():
        print(f"[SKIP] Already exists: {fp_file}")
    else:
        with open(fp_file, 'w') as f:
            json.dump(funcsim_pairs, f)
        print(f"Wrote: {fp_file}  ({len(funcsim_pairs)} query functions)")

    # Stats
    print("\n" + "=" * 80)
    print("STATISTICS")
    print("=" * 80)
    print(f"Total functions  : {len(all_functions)}")
    print(f"With pairs       : {len(function_pairs)}")
    opt_counts = defaultdict(int)
    for func in all_functions:
        opt_counts[func['opt_level']] += 1
    for opt in sorted(opt_counts):
        print(f"  {opt}: {opt_counts[opt]:,} functions")
    print("=" * 80)
    print("Done!")
    print(f"\nOutputs ready for funcsim/dataloader.py:")
    print(f"  --function_blocks {fb_file}")
    print(f"  --funcsim_pairs   {fp_file}")


if __name__ == "__main__":
    main()
