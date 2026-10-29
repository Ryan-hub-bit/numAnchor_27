#!/usr/bin/env python3
"""
Reconstruct the sampled fine-tune subset, build retrieval pools, and optionally
run evaluation for multiple pool sizes.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import subprocess
import sys
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import random_split


CUR_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.join(CUR_DIR, "..")
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from vocab import WordVocab

DL_SPEC = importlib.util.spec_from_file_location(
    "funcsim_dataloader_local", os.path.join(CUR_DIR, "dataloader.py")
)
DL_MOD = importlib.util.module_from_spec(DL_SPEC)
DL_SPEC.loader.exec_module(DL_MOD)
FunctionSimilarityDataset = DL_MOD.FunctionSimilarityDataset

POOL_SPEC = importlib.util.spec_from_file_location(
    "generate_pool_local", os.path.join(CUR_DIR, "generate_pool_data.py")
)
POOL_MOD = importlib.util.module_from_spec(POOL_SPEC)
POOL_SPEC.loader.exec_module(POOL_MOD)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sort_func_id(func_id: str) -> tuple[int, object]:
    s = str(func_id)
    return (0, int(s)) if s.isdigit() else (1, s)


def reconstruct_pair_splits(args: argparse.Namespace):
    vocab = WordVocab.load_vocab(args.vocab)

    set_seed(args.seed)
    full_ds = FunctionSimilarityDataset(
        function_blocks_file=args.function_blocks,
        funcsim_pairs_file=args.funcsim_pairs,
        vocab=vocab,
        seq_len=args.seq_len,
        max_instructions=args.max_instructions,
        negative_samples=args.negative_samples,
    )

    full_pair_count = len(full_ds)
    if args.dataset_fraction < 1.0:
        keep = int(full_pair_count * args.dataset_fraction)
        fraction_ds, _ = random_split(
            full_ds,
            [keep, full_pair_count - keep],
            generator=torch.Generator().manual_seed(args.seed),
        )
        fraction_indices = list(fraction_ds.indices)
    else:
        fraction_ds = full_ds
        fraction_indices = list(range(full_pair_count))

    use_test = args.train_split < 1.0
    if use_test:
        test_n = int(len(fraction_ds) * (1 - args.train_split))
        tv_n = len(fraction_ds) - test_n
        tv_ds, test_ds = random_split(
            fraction_ds,
            [tv_n, test_n],
            generator=torch.Generator().manual_seed(args.seed),
        )
    else:
        tv_ds, test_ds = fraction_ds, None

    use_val = args.val_split > 0.0
    if use_val:
        val_n = int(len(tv_ds) * args.val_split)
        train_n = len(tv_ds) - val_n
        train_ds, val_ds = random_split(
            tv_ds,
            [train_n, val_n],
            generator=torch.Generator().manual_seed(args.seed),
        )
    else:
        train_ds, val_ds = tv_ds, None

    all_pairs = full_ds.training_pairs
    fraction_pairs = [all_pairs[idx] for idx in fraction_indices]

    test_indices_local = list(test_ds.indices) if test_ds is not None else []
    test_pair_indices = [fraction_indices[idx] for idx in test_indices_local]
    test_pairs = [all_pairs[idx] for idx in test_pair_indices]

    return {
        "dataset": full_ds,
        "all_pairs": all_pairs,
        "fraction_indices": fraction_indices,
        "fraction_pairs": fraction_pairs,
        "train_count": len(train_ds),
        "val_count": len(val_ds) if val_ds is not None else 0,
        "test_count": len(test_ds) if test_ds is not None else 0,
        "test_pair_indices": test_pair_indices,
        "test_pairs": test_pairs,
    }


def build_subset_files(args: argparse.Namespace, split_data: dict) -> tuple[str, str, dict]:
    dataset = split_data["dataset"]
    fraction_pairs = split_data["fraction_pairs"]
    test_pairs = split_data["test_pairs"]

    subset_function_ids = sorted(
        {str(fid) for pair in fraction_pairs for fid in pair[:2]},
        key=sort_func_id,
    )
    subset_function_id_set = set(subset_function_ids)

    subset_function_blocks = {
        fid: dataset.function_blocks[fid]
        for fid in subset_function_ids
        if fid in dataset.function_blocks
    }

    candidate_test_queries = []
    seen = set()
    for anchor_id, _, _ in test_pairs:
        anchor_id = str(anchor_id)
        if anchor_id in seen:
            continue
        seen.add(anchor_id)
        candidate_test_queries.append(anchor_id)

    subset_funcsim_pairs = {}
    for query_id in candidate_test_queries:
        gt_ids = [
            str(gt_id)
            for gt_id in dataset.funcsim_pairs.get(query_id, {}).get("ground_truth", [])
            if str(gt_id) in subset_function_id_set
        ]
        if gt_ids:
            subset_funcsim_pairs[query_id] = {"ground_truth": gt_ids}

    os.makedirs(args.output_dir, exist_ok=True)
    tag = f"df{int(round(args.dataset_fraction * 1000)):03d}"
    subset_blocks_path = os.path.join(args.output_dir, f"function_blocks_{tag}_subset.json")
    subset_pairs_path = os.path.join(args.output_dir, f"funcsim_pairs_{tag}_testqueries.json")

    with open(subset_blocks_path, "w") as f:
        json.dump(subset_function_blocks, f)
    with open(subset_pairs_path, "w") as f:
        json.dump(subset_funcsim_pairs, f)

    manifest = {
        "seed": args.seed,
        "dataset_fraction": args.dataset_fraction,
        "train_split": args.train_split,
        "val_split": args.val_split,
        "negative_samples": args.negative_samples,
        "drop_binary_pos": args.drop_binary_pos,
        "drop_code_binary_pos": args.drop_code_binary_pos,
        "fraction_pair_count": len(fraction_pairs),
        "test_pair_count": len(test_pairs),
        "train_pair_count": split_data["train_count"],
        "val_pair_count": split_data["val_count"],
        "subset_function_count": len(subset_function_blocks),
        "test_query_count": len(subset_funcsim_pairs),
        "subset_function_blocks": subset_blocks_path,
        "subset_funcsim_pairs": subset_pairs_path,
    }

    manifest_path = os.path.join(args.output_dir, "subset_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return subset_blocks_path, subset_pairs_path, manifest


def select_master_queries(function_blocks: dict, funcsim_pairs: dict, query_fraction: float, pool_sizes: list[int], gt_per_query: int, seed: int):
    all_query_ids = [str(qid) for qid in funcsim_pairs.keys()]
    func_to_group, _ = POOL_MOD._build_equivalence_groups(function_blocks, funcsim_pairs)
    func_to_hash, _ = POOL_MOD._build_content_hashes(function_blocks)

    random.seed(seed)
    random.shuffle(all_query_ids)

    target_from_fraction = max(1, int(len(all_query_ids) * query_fraction))
    target_from_pool = max(1, max(pool_sizes) // 10)
    master_target = min(target_from_fraction, target_from_pool)

    query_ids, selected_gt = POOL_MOD._select_queries_and_ground_truth(
        all_query_ids=all_query_ids,
        funcsim_pairs=funcsim_pairs,
        function_blocks=function_blocks,
        func_to_group=func_to_group,
        func_to_hash=func_to_hash,
        num_queries=master_target,
        gt_per_query=gt_per_query,
    )
    return query_ids, selected_gt


def build_fixed_pool(function_blocks: dict, funcsim_pairs: dict, query_ids: list[str], selected_gt_by_query: dict[str, list[str]], pool_size: int, seed: int) -> tuple[dict, dict]:
    func_to_group, _ = POOL_MOD._build_equivalence_groups(function_blocks, funcsim_pairs)
    func_to_hash, _ = POOL_MOD._build_content_hashes(function_blocks)

    all_func_ids = list(function_blocks.keys())
    required_func_ids = set(query_ids)
    for query_id in query_ids:
        required_func_ids.update(selected_gt_by_query.get(query_id, []))

    required_groups = {func_to_group[fid] for fid in required_func_ids if fid in func_to_group}
    required_hashes = {func_to_hash[fid] for fid in required_func_ids if fid in func_to_hash}

    remaining = []
    same_group_candidates = 0
    same_hash_candidates = 0
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

    random.seed(seed)
    random.shuffle(remaining)

    num_distractors = max(0, pool_size - len(required_func_ids))
    selected_distractors = []
    seen_groups = set()
    seen_hashes = set()
    duplicate_group_distractors = 0
    duplicate_hash_distractors = 0
    for fid in remaining:
        if len(selected_distractors) >= num_distractors:
            break
        group_id = func_to_group.get(fid)
        content_hash = func_to_hash.get(fid)
        if group_id in seen_groups:
            duplicate_group_distractors += 1
            continue
        if content_hash in seen_hashes:
            duplicate_hash_distractors += 1
            continue
        selected_distractors.append(fid)
        seen_groups.add(group_id)
        seen_hashes.add(content_hash)

    selected_func_ids = sorted(required_func_ids, key=sort_func_id) + selected_distractors
    pool_blocks = {fid: function_blocks[fid] for fid in selected_func_ids if fid in function_blocks}
    pool_ids = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "seed": seed,
            "pool_size": len(selected_func_ids),
            "num_queries": len(query_ids),
            "num_required": len(required_func_ids),
            "num_required_groups": len(required_groups),
            "num_required_hashes": len(required_hashes),
            "num_distractors": len(selected_distractors),
            "gt_per_query": len(next(iter(selected_gt_by_query.values()))) if selected_gt_by_query else 0,
            "same_group_candidates_removed": same_group_candidates,
            "same_hash_candidates_removed": same_hash_candidates,
            "duplicate_group_distractors_removed": duplicate_group_distractors,
            "duplicate_hash_distractors_removed": duplicate_hash_distractors,
        },
        "query_ids": query_ids,
        "selected_ground_truth_ids": {qid: selected_gt_by_query[qid] for qid in query_ids},
        "pool_function_ids": selected_func_ids,
    }
    return pool_ids, pool_blocks


def maybe_run_evaluation(args: argparse.Namespace, pool_files: list[tuple[int, str]]) -> list[dict]:
    if not args.checkpoint:
        return []

    eval_outputs = []
    eval_log_dir = os.path.join(args.output_dir, "eval_logs")
    os.makedirs(eval_log_dir, exist_ok=True)

    for pool_size, pool_ids_path in pool_files:
        eval_output = os.path.join(args.output_dir, f"eval_subsetpool_{pool_size}.json")
        task_name = f"{args.task_name_prefix}_pool{pool_size}"
        cmd = [
            sys.executable,
            os.path.join(CUR_DIR, "evaluate.py"),
            "--function_blocks", args.subset_function_blocks,
            "--funcsim_pairs", args.subset_funcsim_pairs,
            "--vocab", args.vocab,
            "--checkpoint", args.checkpoint,
            "--eval_pool", pool_ids_path,
            "--hidden", str(args.hidden),
            "--n_layers", str(args.n_layers),
            "--attn_heads", str(args.attn_heads),
            "--seq_len", str(args.seq_len),
            "--max_instructions", str(args.max_instructions),
            "--embedding_dim", str(args.embedding_dim),
            "--window_subbatch", str(args.window_subbatch),
            "--batch_size", str(args.eval_batch_size),
            "--output", eval_output,
            "--log_dir", eval_log_dir,
            "--task_name", task_name,
            "--device", args.device,
            "--num_workers", str(args.num_workers),
        ]
        if args.drop_binary_pos:
            cmd.append("--drop_binary_pos")
        if args.drop_code_binary_pos:
            cmd.append("--drop_code_binary_pos")
        subprocess.run(cmd, check=True)
        eval_outputs.append({"pool_size": pool_size, "output": eval_output})

    return eval_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build subset retrieval pools and run eval")
    parser.add_argument("--function_blocks", required=True)
    parser.add_argument("--funcsim_pairs", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dataset_fraction", type=float, default=0.05)
    parser.add_argument("--train_split", type=float, default=0.8)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--negative_samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--max_instructions", type=int, default=80)
    parser.add_argument("--query_fraction", type=float, default=0.1)
    parser.add_argument("--gt_per_query", type=int, default=3)
    parser.add_argument("--pool_sizes", type=int, nargs="+", default=[100, 1000, 10000])
    parser.add_argument("--drop_binary_pos", action="store_true",
                        help="Ignore function-in-binary coordinates during evaluation.")
    parser.add_argument("--drop_code_binary_pos", action="store_true",
                        help="Ignore the binary-level coordinate only for code-side address tokens.")

    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--hidden", type=int, default=768)
    parser.add_argument("--n_layers", type=int, default=12)
    parser.add_argument("--attn_heads", type=int, default=12)
    parser.add_argument("--embedding_dim", type=int, default=256)
    parser.add_argument("--window_subbatch", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--task_name_prefix", default="shufflecoord_eval")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    split_data = reconstruct_pair_splits(args)
    subset_blocks_path, subset_pairs_path, manifest = build_subset_files(args, split_data)

    with open(subset_blocks_path, "r") as f:
        subset_function_blocks = json.load(f)
    with open(subset_pairs_path, "r") as f:
        subset_funcsim_pairs = json.load(f)

    master_queries, master_selected_gt = select_master_queries(
        function_blocks=subset_function_blocks,
        funcsim_pairs=subset_funcsim_pairs,
        query_fraction=args.query_fraction,
        pool_sizes=args.pool_sizes,
        gt_per_query=args.gt_per_query,
        seed=args.seed,
    )
    if not master_queries:
        raise RuntimeError(
            "No queries satisfy the requested gt_per_query constraint. "
            "Try a larger dataset_fraction or a smaller gt_per_query."
        )

    args.subset_function_blocks = subset_blocks_path
    args.subset_funcsim_pairs = subset_pairs_path

    pool_files = []
    summary = {
        **manifest,
        "master_query_count": len(master_queries),
        "pools": [],
    }

    for pool_size in sorted(args.pool_sizes):
        query_budget = max(1, pool_size // 10)
        query_ids = master_queries[: min(query_budget, len(master_queries))]
        selected_gt = {qid: master_selected_gt[qid] for qid in query_ids}
        pool_ids, pool_blocks = build_fixed_pool(
            function_blocks=subset_function_blocks,
            funcsim_pairs=subset_funcsim_pairs,
            query_ids=query_ids,
            selected_gt_by_query=selected_gt,
            pool_size=pool_size,
            seed=args.seed,
        )

        size_tag = f"{pool_size // 1000}k" if pool_size >= 1000 and pool_size % 1000 == 0 else str(pool_size)
        pool_ids_path = os.path.join(args.output_dir, f"pool_ids_{size_tag}.json")
        pool_blocks_path = os.path.join(args.output_dir, f"pool_function_blocks_{size_tag}.json")
        with open(pool_ids_path, "w") as f:
            json.dump(pool_ids, f, indent=2)
        with open(pool_blocks_path, "w") as f:
            json.dump(pool_blocks, f)

        pool_files.append((pool_size, pool_ids_path))
        summary["pools"].append(
            {
                "pool_size": pool_size,
                "query_count": len(query_ids),
                "pool_ids_path": pool_ids_path,
                "pool_function_blocks_path": pool_blocks_path,
            }
        )

    eval_outputs = maybe_run_evaluation(args, pool_files)
    summary["evaluation_outputs"] = eval_outputs

    summary_path = os.path.join(args.output_dir, "subset_eval_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("=" * 72)
    print("Subset retrieval evaluation assets built")
    print("=" * 72)
    print(f"Output dir         : {args.output_dir}")
    print(f"Subset functions   : {summary['subset_function_count']}")
    print(f"Test queries       : {summary['test_query_count']}")
    print(f"Master query count : {summary['master_query_count']}")
    for pool in summary["pools"]:
        print(f"Pool {pool['pool_size']:>5}: {pool['query_count']:>4} queries -> {pool['pool_ids_path']}")
    if eval_outputs:
        print("Evaluation outputs:")
        for item in eval_outputs:
            print(f"  pool {item['pool_size']:>5}: {item['output']}")
    print("=" * 72)


if __name__ == "__main__":
    main()
