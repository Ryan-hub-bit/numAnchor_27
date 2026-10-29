"""
Generate evaluation pool data files for fair model comparison.

Creates two files:
1. pool_ids_10k.json - Pool function IDs and query IDs
2. pool_function_blocks_10k.json - Function blocks for pool (only 10k functions)

Compared with the original version, this script enforces two extra rules:
1. Functions in the same funcsim equivalence group as a query/ground-truth
   are removed from the distractor pool.
2. Functions with the same exact-content hash as a query/ground-truth are
   removed from the distractor pool, and distractors are deduplicated by hash.

This avoids the case where retrieval returns an instruction-identical function
that is not listed in the explicit ground_truth set.
"""

import argparse
import hashlib
import json
import os
import random
from collections import defaultdict
from datetime import datetime


def _build_equivalence_groups(function_blocks, funcsim_pairs):
    """Union-find over funcsim_pairs to recover source-function groups."""
    parent = {}

    def find(x):
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for query_id, pair_data in funcsim_pairs.items():
        for gt_id in pair_data.get("ground_truth", []):
            union(str(query_id), str(gt_id))

    root_to_gid = {}
    func_to_group = {}
    group_to_members = defaultdict(list)
    counter = 0

    for func_id in function_blocks.keys():
        root = find(func_id)
        if root not in root_to_gid:
            root_to_gid[root] = counter
            counter += 1
        gid = root_to_gid[root]
        func_to_group[func_id] = gid
        group_to_members[gid].append(func_id)

    return func_to_group, dict(group_to_members)


def _compute_content_hash(instructions):
    """Stable hash for the exact instruction list of one function."""
    payload = json.dumps(instructions, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _build_content_hashes(function_blocks):
    func_to_hash = {}
    hash_to_members = defaultdict(list)
    for func_id, instructions in function_blocks.items():
        content_hash = _compute_content_hash(instructions)
        func_to_hash[func_id] = content_hash
        hash_to_members[content_hash].append(func_id)
    return func_to_hash, dict(hash_to_members)


def _func_id_sort_key(func_id):
    s = str(func_id)
    return (0, int(s)) if s.isdigit() else (1, s)


def _select_queries_and_ground_truth(
    all_query_ids,
    funcsim_pairs,
    function_blocks,
    func_to_group,
    func_to_hash,
    num_queries,
    gt_per_query,
):
    """
    Select queries such that:
    1. Each query keeps exactly `gt_per_query` GT functions.
    2. The selected GTs are hash-unique within the query.
    3. Query hash + GT hashes are globally unique across the whole pool seed set.
    4. Query groups are unique across the whole pool.
    """
    sampled_query_ids = []
    selected_gt_by_query = {}
    used_groups = set()
    used_hashes = set()

    for query_id in all_query_ids:
        query_id = str(query_id)
        if query_id not in function_blocks:
            continue

        query_group = func_to_group.get(query_id)
        query_hash = func_to_hash.get(query_id)
        if query_group in used_groups or query_hash in used_hashes:
            continue

        gt_candidates = []
        seen_local_hashes = {query_hash}
        for gt_id in sorted(funcsim_pairs.get(query_id, {}).get("ground_truth", []), key=_func_id_sort_key):
            gt_id = str(gt_id)
            if gt_id not in function_blocks:
                continue
            gt_hash = func_to_hash.get(gt_id)
            if gt_hash in seen_local_hashes:
                continue
            gt_candidates.append(gt_id)
            seen_local_hashes.add(gt_hash)

        if len(gt_candidates) < gt_per_query:
            continue

        selected_gts = []
        selected_gt_hashes = set()
        for gt_id in gt_candidates:
            gt_hash = func_to_hash.get(gt_id)
            if gt_hash in used_hashes:
                continue
            selected_gts.append(gt_id)
            selected_gt_hashes.add(gt_hash)
            if len(selected_gts) >= gt_per_query:
                break

        if len(selected_gts) < gt_per_query:
            continue

        sampled_query_ids.append(query_id)
        selected_gt_by_query[query_id] = selected_gts
        used_groups.add(query_group)
        used_hashes.add(query_hash)
        used_hashes.update(selected_gt_hashes)

        if len(sampled_query_ids) >= num_queries:
            break

    return sampled_query_ids, selected_gt_by_query


def generate_pool_data(
    function_blocks_file,
    funcsim_pairs_file,
    pool_size,
    output_dir,
    seed=42,
    num_queries=None,
    gt_per_query=None,
    output_suffix="",
):
    """
    Generate pool data files.

    Args:
        function_blocks_file: Path to function_blocks.json
        funcsim_pairs_file: Path to funcsim_pairs.json
        pool_size: Total number of functions in the pool
        output_dir: Directory to save output files
        seed: Random seed for reproducibility
    """
    print(f"Loading function blocks from {function_blocks_file}...")
    with open(function_blocks_file, "r") as f:
        function_blocks = json.load(f)
    print(f"Loaded {len(function_blocks)} function blocks")

    print(f"Loading funcsim pairs from {funcsim_pairs_file}...")
    with open(funcsim_pairs_file, "r") as f:
        funcsim_pairs = json.load(f)
    print(f"Loaded {len(funcsim_pairs)} query functions")

    random.seed(seed)

    all_func_ids = list(function_blocks.keys())
    all_query_ids = [str(qid) for qid in funcsim_pairs.keys()]
    func_to_group, _ = _build_equivalence_groups(function_blocks, funcsim_pairs)
    func_to_hash, _ = _build_content_hashes(function_blocks)

    # Sample queries first (aim for ~10% of pool size for queries unless
    # overridden), but avoid repeated equivalence groups and repeated hashes.
    target_num_queries = num_queries or min(max(100, pool_size // 10), len(all_query_ids))
    random.shuffle(all_query_ids)
    if gt_per_query is None:
        sampled_query_ids = []
        selected_gt_by_query = {}
        seen_query_groups = set()
        seen_query_hashes = set()
        for query_id in all_query_ids:
            group_id = func_to_group.get(query_id)
            content_hash = func_to_hash.get(query_id)
            if group_id in seen_query_groups or content_hash in seen_query_hashes:
                continue
            sampled_query_ids.append(query_id)
            selected_gt_by_query[query_id] = [
                str(gt_id)
                for gt_id in funcsim_pairs.get(query_id, {}).get("ground_truth", [])
                if str(gt_id) in function_blocks
            ]
            seen_query_groups.add(group_id)
            seen_query_hashes.add(content_hash)
            if len(sampled_query_ids) >= target_num_queries:
                break
    else:
        sampled_query_ids, selected_gt_by_query = _select_queries_and_ground_truth(
            all_query_ids=all_query_ids,
            funcsim_pairs=funcsim_pairs,
            function_blocks=function_blocks,
            func_to_group=func_to_group,
            func_to_hash=func_to_hash,
            num_queries=target_num_queries,
            gt_per_query=gt_per_query,
        )

    print(f"\nSampling {len(sampled_query_ids)} queries from {len(all_query_ids)} total...")

    # Queries + explicit positives must always stay in the pool.
    required_func_ids = set(sampled_query_ids)
    for query_id in sampled_query_ids:
        for gt_id in selected_gt_by_query.get(query_id, []):
            gt_id_str = str(gt_id)
            if gt_id_str in function_blocks:
                required_func_ids.add(gt_id_str)

    required_groups = {func_to_group[fid] for fid in required_func_ids if fid in func_to_group}
    required_hashes = {func_to_hash[fid] for fid in required_func_ids if fid in func_to_hash}

    print(f"Required functions (queries + ground truth): {len(required_func_ids)}")

    same_group_candidates = 0
    same_hash_candidates = 0
    remaining = []
    for fid in all_func_ids:
        if fid in required_func_ids:
            continue
        if func_to_group.get(fid) in required_groups:
            same_group_candidates += 1
            continue
        if func_to_hash.get(fid) in required_hashes:
            same_hash_candidates += 1
            continue
        remaining.append(fid)

    random.shuffle(remaining)
    num_distractors = max(0, pool_size - len(required_func_ids))

    if len(required_func_ids) > pool_size:
        print(
            f"WARNING: Required functions ({len(required_func_ids)}) exceed "
            f"pool_size ({pool_size})"
        )
        print("Using all required functions. Consider increasing pool_size.")

    selected_distractors = []
    seen_distractor_groups = set()
    seen_distractor_hashes = set()
    duplicate_group_distractors = 0
    duplicate_hash_distractors = 0
    for fid in remaining:
        group_id = func_to_group.get(fid)
        content_hash = func_to_hash.get(fid)
        if group_id in seen_distractor_groups:
            duplicate_group_distractors += 1
            continue
        if content_hash in seen_distractor_hashes:
            duplicate_hash_distractors += 1
            continue
        selected_distractors.append(fid)
        seen_distractor_groups.add(group_id)
        seen_distractor_hashes.add(content_hash)
        if len(selected_distractors) >= num_distractors:
            break

    if len(selected_distractors) < num_distractors:
        print(
            f"WARNING: Only {len(selected_distractors)} valid distractors available "
            f"after group/hash filtering (requested {num_distractors})"
        )

    selected_func_ids = list(required_func_ids) + selected_distractors

    print(f"Adding {len(selected_distractors)} filtered distractors")
    print(f"Total pool size: {len(selected_func_ids)} functions")

    print("\nExtracting function blocks for pool...")
    pool_function_blocks = {
        fid: function_blocks[fid]
        for fid in selected_func_ids
        if fid in function_blocks
    }

    pool_ids_data = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "seed": seed,
            "pool_size": len(selected_func_ids),
            "num_queries": len(sampled_query_ids),
            "num_required": len(required_func_ids),
            "num_required_groups": len(required_groups),
            "num_required_hashes": len(required_hashes),
            "num_distractors": len(selected_distractors),
            "gt_per_query": gt_per_query,
            "same_group_candidates_removed": same_group_candidates,
            "same_hash_candidates_removed": same_hash_candidates,
            "duplicate_group_distractors_removed": duplicate_group_distractors,
            "duplicate_hash_distractors_removed": duplicate_hash_distractors,
        },
        "query_ids": sampled_query_ids,
        "selected_ground_truth_ids": selected_gt_by_query,
        "pool_function_ids": selected_func_ids,
    }

    os.makedirs(output_dir, exist_ok=True)

    if pool_size >= 1000 and pool_size % 1000 == 0:
        size_tag = f"{pool_size // 1000}k"
    else:
        size_tag = str(pool_size)

    suffix = f"_{output_suffix}" if output_suffix else ""
    pool_ids_file = os.path.join(output_dir, f"pool_ids_{size_tag}{suffix}.json")
    pool_blocks_file = os.path.join(output_dir, f"pool_function_blocks_{size_tag}{suffix}.json")

    print(f"\nSaving pool IDs to {pool_ids_file}...")
    with open(pool_ids_file, "w") as f:
        json.dump(pool_ids_data, f, indent=2)

    print(f"Saving pool function blocks to {pool_blocks_file}...")
    with open(pool_blocks_file, "w") as f:
        json.dump(pool_function_blocks, f)

    print("\n" + "=" * 60)
    print("Pool Data Generated Successfully!")
    print("=" * 60)
    print(f"Pool size: {len(selected_func_ids)} functions")
    print(f"Queries: {len(sampled_query_ids)}")
    print(f"Required (queries + ground truth): {len(required_func_ids)}")
    print(f"Required groups: {len(required_groups)}")
    print(f"Required hashes: {len(required_hashes)}")
    print(f"Distractors: {len(selected_distractors)}")
    print(f"Removed same-group candidates: {same_group_candidates}")
    print(f"Removed same-hash candidates: {same_hash_candidates}")
    print(f"Removed duplicate-group distractors: {duplicate_group_distractors}")
    print(f"Removed duplicate-hash distractors: {duplicate_hash_distractors}")
    print("\nOutput files:")
    print(f"  IDs:     {pool_ids_file}")
    print(f"  Blocks:  {pool_blocks_file}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Generate pool data files for evaluation")

    parser.add_argument("--function_blocks", type=str, required=True,
                       help="Path to function_blocks.json")
    parser.add_argument("--funcsim_pairs", type=str, required=True,
                       help="Path to funcsim_pairs.json")
    parser.add_argument("--pool_size", type=int, default=10000,
                       help="Target pool size (default: 10000)")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for pool files")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (default: 42)")
    parser.add_argument("--num_queries", type=int, default=None,
                       help="Fixed number of queries to sample")
    parser.add_argument("--gt_per_query", type=int, default=None,
                       help="Require exactly this many unique GT representatives per query")
    parser.add_argument("--output_suffix", type=str, default="",
                       help="Optional suffix appended to output filenames")

    args = parser.parse_args()

    generate_pool_data(
        function_blocks_file=args.function_blocks,
        funcsim_pairs_file=args.funcsim_pairs,
        pool_size=args.pool_size,
        output_dir=args.output_dir,
        seed=args.seed,
        num_queries=args.num_queries,
        gt_per_query=args.gt_per_query,
        output_suffix=args.output_suffix,
    )


if __name__ == "__main__":
    main()
