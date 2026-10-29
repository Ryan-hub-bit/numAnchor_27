#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from evaluate_pool import embed_functions, load_checkpoint_model


METADATA_FILE_RE = re.compile(r"^(?P<binary>.+)_(?P<opt>O[0-3])_functions\.json$")


def build_opt_levels_by_global_id(
    function_blocks: Dict[str, List[str]],
    function_ids: Sequence[str],
    metadata_dir: Path,
) -> Dict[str, Optional[str]]:
    signature_to_ids: Dict[str, List[str]] = defaultdict(list)
    for function_id in function_ids:
        blocks = function_blocks.get(function_id)
        if not blocks:
            continue
        signature_to_ids["\t".join(blocks)].append(function_id)

    signature_to_opts: Dict[str, set[str]] = defaultdict(set)
    for path in sorted(metadata_dir.glob("*_functions.json")):
        match = METADATA_FILE_RE.match(path.name)
        if not match:
            continue
        opt_level = match.group("opt")
        with path.open("r", encoding="utf-8") as handle:
            rows = json.load(handle)
        for row in rows:
            signature = row.get("funcins")
            if signature in signature_to_ids:
                signature_to_opts[signature].add(row.get("opt_level", opt_level))

    opt_by_id: Dict[str, Optional[str]] = {}
    for signature, ids in signature_to_ids.items():
        opts = signature_to_opts.get(signature, set())
        resolved_opt = next(iter(opts)) if len(opts) == 1 else None
        for function_id in ids:
            opt_by_id[function_id] = resolved_opt
    return opt_by_id


def compute_metrics_for_queries(
    embeddings: Dict[str, torch.Tensor],
    query_ids: Sequence[str],
    pool_ids: Sequence[str],
    ground_truth_by_query: Dict[str, Sequence[str]],
) -> Dict[str, float]:
    pool = [func_id for func_id in pool_ids if func_id in embeddings]
    pool_matrix = torch.stack([embeddings[func_id] for func_id in pool], dim=0)
    pool_index = {func_id: idx for idx, func_id in enumerate(pool)}

    recalls = {1: [], 5: [], 10: [], 20: []}
    reciprocal_ranks: List[float] = []

    for query_id in query_ids:
        if query_id not in embeddings:
            continue
        ground_truth = {func_id for func_id in ground_truth_by_query.get(query_id, []) if func_id in pool_index}
        if not ground_truth:
            continue

        query = embeddings[query_id].unsqueeze(0)
        sims = (query @ pool_matrix.T).squeeze(0)
        if query_id in pool_index:
            sims[pool_index[query_id]] = -2.0

        ranking = torch.argsort(sims, descending=True).tolist()
        ranked_ids = [pool[index] for index in ranking]

        for k in recalls:
            recalls[k].append(float(bool(set(ranked_ids[:k]) & ground_truth)))

        rr = 0.0
        for rank, candidate_id in enumerate(ranked_ids, start=1):
            if candidate_id in ground_truth:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

    result = {
        f"recall@{k}": float(np.mean(values)) if values else 0.0
        for k, values in recalls.items()
    }
    result["mrr"] = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0
    result["num_queries"] = len(reciprocal_ranks)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Trex checkpoints on the 10k pool with optimisation-level splits")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--function-blocks", type=Path, default=Path("/path/to/numAnchor_artifact/stateformer_x64_dataset/funcsim_block_ctx2/function_blocks.json"))
    parser.add_argument("--funcsim-pairs", type=Path, default=Path("/path/to/numAnchor_artifact/stateformer_x64_dataset/funcsim_block_ctx2/funcsim_pairs.json"))
    parser.add_argument("--pool-file", type=Path, default=Path("/path/to/numAnchor_artifact/stateformer_x64_dataset/funcsim_block_ctx2/pool_ids_10k.json"))
    parser.add_argument("--metadata-dir", type=Path, default=Path("/path/to/numAnchor_artifact/stateformer_x64_dataset/funcsim_block_ctx2"))
    parser.add_argument("--mode", choices=["addressaware", "baseline", "native", "addressaware_aa"], required=True)
    parser.add_argument("--max-positions", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    model, task = load_checkpoint_model(args.checkpoint, args.data_dir)
    model.to(device)

    with args.function_blocks.open("r", encoding="utf-8") as handle:
        function_blocks = json.load(handle)
    with args.funcsim_pairs.open("r", encoding="utf-8") as handle:
        funcsim_pairs = json.load(handle)
    with args.pool_file.open("r", encoding="utf-8") as handle:
        pool_spec = json.load(handle)

    query_ids = [str(func_id) for func_id in pool_spec["query_ids"]]
    pool_ids = [str(func_id) for func_id in pool_spec["pool_function_ids"]]
    all_ids = list(dict.fromkeys(pool_ids + query_ids))

    embeddings = embed_functions(
        model=model,
        dictionaries=task.source_dictionary,
        function_blocks=function_blocks,
        func_ids=all_ids,
        mode=args.mode,
        max_positions=args.max_positions,
        batch_size=args.batch_size,
        device=device,
    )

    overall_ground_truth = {
        str(query_id): [str(func_id) for func_id in data["ground_truth"]]
        for query_id, data in funcsim_pairs.items()
    }

    relevant_ids = list(
        dict.fromkeys(
            all_ids
            + [
                str(func_id)
                for targets in overall_ground_truth.values()
                for func_id in targets
            ]
        )
    )
    opt_by_id = build_opt_levels_by_global_id(
        function_blocks=function_blocks,
        function_ids=relevant_ids,
        metadata_dir=args.metadata_dir,
    )

    opt_levels = sorted({opt for opt in opt_by_id.values() if opt and opt.startswith("O")})

    by_query_opt = {}
    for opt_level in opt_levels:
        opt_queries = [query_id for query_id in query_ids if opt_by_id.get(query_id) == opt_level]
        by_query_opt[opt_level] = compute_metrics_for_queries(
            embeddings=embeddings,
            query_ids=opt_queries,
            pool_ids=pool_ids,
            ground_truth_by_query=overall_ground_truth,
        )

    by_opt_pair = {}
    for src_opt in opt_levels:
        by_opt_pair[src_opt] = {}
        for dst_opt in opt_levels:
            if src_opt == dst_opt:
                continue
            pair_ground_truth = {}
            pair_queries = []
            for query_id in query_ids:
                if opt_by_id.get(query_id) != src_opt:
                    continue
                target_ids = [
                    func_id
                    for func_id in overall_ground_truth.get(query_id, [])
                    if opt_by_id.get(func_id) == dst_opt
                ]
                if not target_ids:
                    continue
                pair_queries.append(query_id)
                pair_ground_truth[query_id] = target_ids

            by_opt_pair[src_opt][dst_opt] = compute_metrics_for_queries(
                embeddings=embeddings,
                query_ids=pair_queries,
                pool_ids=pool_ids,
                ground_truth_by_query=pair_ground_truth,
            )

    unresolved_ids = [function_id for function_id in relevant_ids if opt_by_id.get(function_id) is None]

    payload = {
        "checkpoint": str(args.checkpoint),
        "data_dir": str(args.data_dir),
        "mode": args.mode,
        "pool_file": str(args.pool_file),
        "split_coverage": {
            "resolved_query_ids": sum(1 for query_id in query_ids if opt_by_id.get(query_id) is not None),
            "total_query_ids": len(query_ids),
            "resolved_relevant_ids": sum(1 for function_id in relevant_ids if opt_by_id.get(function_id) is not None),
            "total_relevant_ids": len(relevant_ids),
            "ambiguous_or_missing_count": len(unresolved_ids),
            "ambiguous_or_missing_sample_ids": unresolved_ids[:50],
        },
        "overall": compute_metrics_for_queries(
            embeddings=embeddings,
            query_ids=query_ids,
            pool_ids=pool_ids,
            ground_truth_by_query=overall_ground_truth,
        ),
        "by_query_opt": by_query_opt,
        "by_opt_pair": by_opt_pair,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
