#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from command.stateformer_x64.prepare_stateformer_trex import (
    AA_FIELD_ORDER,
    FIELD_ORDER,
    convert_inline_sequence,
    convert_inline_sequence_with_aa,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Trex similarity finetune pairs from the local stateformer_x64 funcsim dataset"
    )
    parser.add_argument(
        "--function-blocks",
        type=Path,
        default=Path("/path/to/numAnchor_artifact/stateformer_x64_dataset/funcsim_block_ctx2/function_blocks.json"),
    )
    parser.add_argument(
        "--funcsim-pairs",
        type=Path,
        default=Path("/path/to/numAnchor_artifact/stateformer_x64_dataset/funcsim_block_ctx2/funcsim_pairs.json"),
    )
    parser.add_argument(
        "--exclude-query-pool",
        type=Path,
        default=Path("/path/to/numAnchor_artifact/stateformer_x64_dataset/funcsim_block_ctx2/pool_ids_10k.json"),
        help="query pool whose equivalence groups should be held out from train/valid finetune data",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["baseline", "addressaware", "addressaware_aa"], required=True)
    parser.add_argument(
        "--train-positive-samples",
        type=int,
        default=50000,
        help="number of positive train pairs to sample before adding negatives",
    )
    parser.add_argument(
        "--valid-positive-samples",
        type=int,
        default=5000,
        help="number of positive valid pairs to sample before adding negatives",
    )
    parser.add_argument(
        "--negatives-per-positive",
        type=int,
        default=1,
        help="how many negatives to sample for each positive pair",
    )
    parser.add_argument(
        "--valid-fraction",
        type=float,
        default=0.02,
        help="fraction of non-held-out equivalence groups reserved for validation",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="truncate each function representation to this many tokens before writing",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_equivalence_groups(
    function_blocks: Dict[str, List[str]],
    funcsim_pairs: Dict[str, Dict[str, List[str]]],
) -> Tuple[Dict[str, int], Dict[int, List[str]]]:
    parent: Dict[str, str] = {}

    def find(node: str) -> str:
        root = parent.get(node, node)
        while parent.get(root, root) != root:
            root = parent[root]
        while parent.get(node, node) != node:
            next_node = parent[node]
            parent[node] = root
            node = next_node
        return root

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for query_id, pair_data in funcsim_pairs.items():
        if query_id not in function_blocks:
            continue
        for gt_id in pair_data.get("ground_truth", []):
            gt_id = str(gt_id)
            if gt_id in function_blocks:
                union(query_id, gt_id)

    root_to_group: Dict[str, int] = {}
    func_to_group: Dict[str, int] = {}
    group_to_funcs: Dict[int, List[str]] = defaultdict(list)
    for func_id in function_blocks:
        root = find(func_id)
        group_id = root_to_group.setdefault(root, len(root_to_group))
        func_to_group[func_id] = group_id
        group_to_funcs[group_id].append(func_id)
    return func_to_group, dict(group_to_funcs)


def build_hash_groups(function_blocks: Dict[str, List[str]]) -> Dict[str, str]:
    func_to_hash: Dict[str, str] = {}
    for func_id, instructions in function_blocks.items():
        payload = json.dumps(instructions, separators=(",", ":"), ensure_ascii=False)
        func_to_hash[func_id] = hashlib.sha1(payload.encode("utf-8")).hexdigest()
    return func_to_hash


def build_positive_index(
    function_blocks: Dict[str, List[str]],
    funcsim_pairs: Dict[str, Dict[str, List[str]]],
) -> Dict[int, List[Tuple[str, str]]]:
    positives: Dict[int, List[Tuple[str, str]]] = defaultdict(list)
    for query_id, pair_data in funcsim_pairs.items():
        if query_id not in function_blocks:
            continue
        for gt_id in pair_data.get("ground_truth", []):
            gt_id = str(gt_id)
            if gt_id in function_blocks:
                positives[query_id].append((query_id, gt_id))
    return positives


def sample_pairs(
    positive_pairs: Sequence[Tuple[str, str]],
    sample_count: int,
    negatives_per_positive: int,
    split_func_ids: Sequence[str],
    func_to_group: Dict[str, int],
    func_to_hash: Dict[str, str],
    rng: random.Random,
) -> List[Tuple[str, str, float]]:
    if not positive_pairs:
        return []

    if sample_count <= 0:
        selected_positives = list(positive_pairs)
    elif sample_count >= len(positive_pairs):
        selected_positives = list(positive_pairs)
        while len(selected_positives) < sample_count:
            selected_positives.append(rng.choice(positive_pairs))
    else:
        selected_positives = rng.sample(list(positive_pairs), sample_count)

    samples: List[Tuple[str, str, float]] = []
    for query_id, pos_id in selected_positives:
        samples.append((query_id, pos_id, 1.0))
        query_group = func_to_group[query_id]
        query_hash = func_to_hash[query_id]
        for _ in range(negatives_per_positive):
            neg_id = None
            for _attempt in range(64):
                candidate = rng.choice(split_func_ids)
                if candidate == query_id:
                    continue
                if func_to_group[candidate] == query_group:
                    continue
                if func_to_hash[candidate] == query_hash:
                    continue
                neg_id = candidate
                break
            if neg_id is not None:
                samples.append((query_id, neg_id, -1.0))

    rng.shuffle(samples)
    return samples


def truncate_fields(fields: Dict[str, List[str]], max_tokens: int) -> Dict[str, List[str]]:
    if max_tokens <= 0:
        return fields
    return {field: values[:max_tokens] for field, values in fields.items()}


def convert_function(
    instructions: Sequence[str],
    mode: str,
    max_tokens: int,
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]] | None]:
    inline_text = "\t".join(instructions)
    if mode == "addressaware_aa":
        fields, aa_fields = convert_inline_sequence_with_aa(inline_text, mode=mode)
        return truncate_fields(fields, max_tokens), truncate_fields(aa_fields, max_tokens)
    fields = convert_inline_sequence(inline_text, mode=mode)
    return truncate_fields(fields, max_tokens), None


def write_split(
    split: str,
    samples: Sequence[Tuple[str, str, float]],
    cache: Dict[str, Tuple[Dict[str, List[str]], Dict[str, List[str]] | None]],
    output_dir: Path,
    mode: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    token_writers = {}
    for input_name in ("input0", "input1"):
        for field in FIELD_ORDER:
            path = output_dir / f"{split}.{field}.{input_name}"
            token_writers[(input_name, field)] = path.open("w", encoding="utf-8")

    aa_writers = {}
    if mode == "addressaware_aa":
        for input_name in ("input0", "input1"):
            aa_root = output_dir / "aa" / input_name
            aa_root.mkdir(parents=True, exist_ok=True)
            for field in AA_FIELD_ORDER:
                path = aa_root / f"{split}.{field}"
                aa_writers[(input_name, field)] = path.open("w", encoding="utf-8")

    label_path = output_dir / f"{split}.label"
    label_writer = label_path.open("w", encoding="utf-8")

    try:
        for input0_id, input1_id, label in samples:
            for input_name, func_id in (("input0", input0_id), ("input1", input1_id)):
                fields, aa_fields = cache[func_id]
                for field in FIELD_ORDER:
                    token_writers[(input_name, field)].write(" ".join(fields[field]))
                    token_writers[(input_name, field)].write("\n")
                if mode == "addressaware_aa" and aa_fields is not None:
                    for field in AA_FIELD_ORDER:
                        aa_writers[(input_name, field)].write(" ".join(aa_fields[field]))
                        aa_writers[(input_name, field)].write("\n")
            label_writer.write(f"{label:.1f}\n")
    finally:
        for handle in token_writers.values():
            handle.close()
        for handle in aa_writers.values():
            handle.close()
        label_writer.close()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    function_blocks = load_json(args.function_blocks)
    funcsim_pairs = load_json(args.funcsim_pairs)
    pool_spec = load_json(args.exclude_query_pool) if args.exclude_query_pool else None

    func_to_group, group_to_funcs = build_equivalence_groups(function_blocks, funcsim_pairs)
    func_to_hash = build_hash_groups(function_blocks)
    query_to_positive_pairs = build_positive_index(function_blocks, funcsim_pairs)

    eligible_query_ids = sorted(query_to_positive_pairs.keys())
    eligible_groups = sorted({func_to_group[query_id] for query_id in eligible_query_ids})

    excluded_groups = set()
    if pool_spec is not None:
        excluded_groups = {
            func_to_group[str(query_id)]
            for query_id in pool_spec.get("query_ids", [])
            if str(query_id) in func_to_group
        }

    train_valid_groups = [group_id for group_id in eligible_groups if group_id not in excluded_groups]
    rng.shuffle(train_valid_groups)

    valid_group_count = int(round(len(train_valid_groups) * args.valid_fraction))
    if valid_group_count <= 0 and train_valid_groups:
        valid_group_count = 1
    valid_groups = set(train_valid_groups[:valid_group_count])
    train_groups = set(train_valid_groups[valid_group_count:])

    train_positive_pairs = [
        pair
        for query_id, pairs in query_to_positive_pairs.items()
        if func_to_group[query_id] in train_groups
        for pair in pairs
    ]
    valid_positive_pairs = [
        pair
        for query_id, pairs in query_to_positive_pairs.items()
        if func_to_group[query_id] in valid_groups
        for pair in pairs
    ]

    train_func_ids = [func_id for group_id in train_groups for func_id in group_to_funcs[group_id]]
    valid_func_ids = [func_id for group_id in valid_groups for func_id in group_to_funcs[group_id]]

    train_samples = sample_pairs(
        positive_pairs=train_positive_pairs,
        sample_count=args.train_positive_samples,
        negatives_per_positive=args.negatives_per_positive,
        split_func_ids=train_func_ids,
        func_to_group=func_to_group,
        func_to_hash=func_to_hash,
        rng=rng,
    )
    valid_samples = sample_pairs(
        positive_pairs=valid_positive_pairs,
        sample_count=args.valid_positive_samples,
        negatives_per_positive=args.negatives_per_positive,
        split_func_ids=valid_func_ids,
        func_to_group=func_to_group,
        func_to_hash=func_to_hash,
        rng=rng,
    )

    required_func_ids = {
        func_id
        for samples in (train_samples, valid_samples)
        for input0_id, input1_id, _label in samples
        for func_id in (input0_id, input1_id)
    }
    cache = {
        func_id: convert_function(function_blocks[func_id], mode=args.mode, max_tokens=args.max_tokens)
        for func_id in sorted(required_func_ids)
    }

    write_split("train", train_samples, cache, args.output_dir, mode=args.mode)
    write_split("valid", valid_samples, cache, args.output_dir, mode=args.mode)

    manifest = {
        "mode": args.mode,
        "seed": args.seed,
        "max_tokens": args.max_tokens,
        "train_positive_samples": args.train_positive_samples,
        "valid_positive_samples": args.valid_positive_samples,
        "negatives_per_positive": args.negatives_per_positive,
        "train_pairs": len(train_samples),
        "valid_pairs": len(valid_samples),
        "train_groups": len(train_groups),
        "valid_groups": len(valid_groups),
        "excluded_groups": len(excluded_groups),
        "unique_functions_materialized": len(required_func_ids),
        "function_blocks": str(args.function_blocks),
        "funcsim_pairs": str(args.funcsim_pairs),
        "exclude_query_pool": str(args.exclude_query_pool) if args.exclude_query_pool else None,
    }
    with (args.output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
