#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

TREX_ROOT = Path(__file__).resolve().parents[2]
if str(TREX_ROOT) not in sys.path:
    sys.path.insert(0, str(TREX_ROOT))

from fairseq import checkpoint_utils

from command.stateformer_x64.funcsim_retrieval_common import (
    collate_aa_batch,
    collate_src_batch,
    load_json,
    materialize_encodings,
)
from command.stateformer_x64.train_funcsim_retrieval import TrexRetrievalModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Trex retrieval-style funcsim checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pool-file", type=Path, required=True)
    parser.add_argument("--function-blocks", type=Path, default=None)
    parser.add_argument("--funcsim-pairs", type=Path, default=None)
    parser.add_argument("--pretrained-checkpoint", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def compute_metrics(embeddings, funcsim_pairs, query_ids, pool_ids):
    pool = [func_id for func_id in pool_ids if func_id in embeddings]
    pool_matrix = torch.stack([embeddings[func_id] for func_id in pool], dim=0)
    recalls = {1: [], 5: [], 10: [], 20: []}
    reciprocal_ranks = []

    for query_id in query_ids:
        if query_id not in embeddings or query_id not in funcsim_pairs:
            continue
        query = embeddings[query_id].unsqueeze(0)
        sims = (query @ pool_matrix.T).squeeze(0)
        if query_id in pool:
            sims[pool.index(query_id)] = -2.0
        ranking = torch.argsort(sims, descending=True).tolist()
        ranked_ids = [pool[index] for index in ranking]
        ground_truth = set(str(item) for item in funcsim_pairs[query_id]["ground_truth"])
        if not ground_truth:
            continue
        for k in recalls:
            recalls[k].append(float(bool(set(ranked_ids[:k]) & ground_truth)))
        rr = 0.0
        for rank, candidate_id in enumerate(ranked_ids, start=1):
            if candidate_id in ground_truth:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

    result = {f"recall@{k}": float(np.mean(values)) if values else 0.0 for k, values in recalls.items()}
    result["mrr"] = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0
    result["num_queries"] = len(reciprocal_ranks)
    return result


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    function_blocks_path = Path(args.function_blocks or config["function_blocks"])
    funcsim_pairs_path = Path(args.funcsim_pairs or config["funcsim_pairs"])
    pretrained_checkpoint = Path(args.pretrained_checkpoint or config["pretrained_checkpoint"])
    data_dir = Path(args.data_dir or config["data_dir"])

    models, _cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        [str(pretrained_checkpoint)],
        arg_overrides={"data": str(data_dir)},
    )
    backbone = models[0]
    model = TrexRetrievalModel(
        backbone=backbone,
        static_pad_idx=task.source_dictionary["static"].pad(),
        embedding_dim=int(config["embedding_dim"]),
        dropout=float(config["dropout"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = model.to(device)
    model.eval()

    function_blocks = load_json(function_blocks_path)
    funcsim_pairs = load_json(funcsim_pairs_path)
    pool_spec = load_json(args.pool_file)
    query_ids = [str(query_id) for query_id in pool_spec["query_ids"]]
    pool_ids = [str(func_id) for func_id in pool_spec["pool_function_ids"]]
    all_ids = list(dict.fromkeys(query_ids + pool_ids))
    encoding_mode = "addressaware_aa" if config["mode"] == "numanchor_aa" else config["mode"]

    encoded = materialize_encodings(
        function_blocks=function_blocks,
        func_ids=all_ids,
        dictionaries=task.source_dictionary,
        mode=encoding_mode,
        max_positions=int(config["max_positions"]),
    )

    embeddings = {}
    for start in range(0, len(all_ids), args.batch_size):
        batch_ids = all_ids[start : start + args.batch_size]
        batch_src = collate_src_batch([encoded[func_id][0] for func_id in batch_ids], dictionaries=task.source_dictionary)
        batch_aa = collate_aa_batch([encoded[func_id][1] for func_id in batch_ids])
        batch_src = {field: tensor.to(device) for field, tensor in batch_src.items()}
        if batch_aa is not None:
            batch_aa = {field: tensor.to(device) for field, tensor in batch_aa.items()}
        batch_emb = model(batch_src, batch_aa).cpu()
        for func_id, embedding in zip(batch_ids, batch_emb):
            embeddings[func_id] = embedding

    metrics = compute_metrics(embeddings, funcsim_pairs, query_ids, pool_ids)
    payload = {
        "checkpoint": str(args.checkpoint),
        "pool_file": str(args.pool_file),
        "pretrained_checkpoint": str(pretrained_checkpoint),
        "data_dir": str(data_dir),
        "config": config,
        "metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
