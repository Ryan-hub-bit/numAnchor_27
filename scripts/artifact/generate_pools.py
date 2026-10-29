#!/usr/bin/env python3
"""Generate deterministic retrieval pools from ground-truth pairs only."""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_ground_truth(path: Path) -> Dict[str, Dict[str, List[str]]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict) and "pairs" in payload:
        return convert_pair_list(payload["pairs"])
    if isinstance(payload, dict):
        return payload
    raise ValueError(f"unsupported ground-truth format: {path}")


def convert_pair_list(pairs: Sequence[dict]) -> Dict[str, Dict[str, List[str]]]:
    """Convert JTrans-style optimisation group lists into query -> ground_truth."""
    result: Dict[str, Dict[str, List[str]]] = {}
    opt_keys = ("O0", "O1", "O2", "O3", "Os")
    for pair in pairs:
        ids = [str(pair[key]) for key in opt_keys if pair.get(key) is not None]
        for query_id in ids:
            result[query_id] = {"ground_truth": [item for item in ids if item != query_id]}
    return result


def all_function_ids(pairs: Dict[str, Dict[str, List[str]]]) -> List[str]:
    seen = {}
    for query_id, item in pairs.items():
        seen[str(query_id)] = None
        for gt_id in item.get("ground_truth", []):
            seen[str(gt_id)] = None
    return list(seen)


def eligible_queries(pairs: Dict[str, Dict[str, List[str]]], min_ground_truth: int) -> List[str]:
    return [
        str(query_id)
        for query_id, item in pairs.items()
        if len([gt for gt in item.get("ground_truth", []) if gt is not None]) >= min_ground_truth
    ]


def build_pool(args: argparse.Namespace) -> dict:
    pairs = load_ground_truth(args.ground_truth)
    rng = random.Random(args.seed)

    queries = eligible_queries(pairs, args.min_ground_truth)
    rng.shuffle(queries)
    queries = queries[: args.query_count]
    if not queries:
        raise ValueError("no eligible queries found")

    selected_gt = {}
    required = set(queries)
    for query_id in queries:
        gt_ids = [str(item) for item in pairs[query_id].get("ground_truth", []) if item is not None]
        rng.shuffle(gt_ids)
        chosen = gt_ids[: args.ground_truth_per_query]
        selected_gt[query_id] = chosen
        required.update(chosen)

    universe = all_function_ids(pairs)
    distractors = [item for item in universe if item not in required]
    rng.shuffle(distractors)
    target_distractors = max(0, args.pool_size - len(required))
    pool_ids = list(required) + distractors[:target_distractors]
    rng.shuffle(pool_ids)

    if args.strict_size and len(pool_ids) != args.pool_size:
        raise ValueError(f"requested pool size {args.pool_size}, generated {len(pool_ids)}")

    return {
        "metadata": {
            "generated_at_utc": utc_now(),
            "source_ground_truth": str(args.ground_truth),
            "seed": args.seed,
            "pool_size_requested": args.pool_size,
            "pool_size": len(pool_ids),
            "query_count_requested": args.query_count,
            "query_count": len(queries),
            "ground_truth_per_query": args.ground_truth_per_query,
            "min_ground_truth": args.min_ground_truth,
        },
        "query_ids": queries,
        "selected_ground_truth_ids": selected_gt,
        "pool_function_ids": pool_ids,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pool-size", type=int, default=10000)
    parser.add_argument("--query-count", type=int, default=1000)
    parser.add_argument("--ground-truth-per-query", type=int, default=1)
    parser.add_argument("--min-ground-truth", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strict-size", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pool = build_pool(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(pool, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"wrote {args.output}")
    print(f"queries={len(pool['query_ids'])} pool={len(pool['pool_function_ids'])}")


if __name__ == "__main__":
    main()
