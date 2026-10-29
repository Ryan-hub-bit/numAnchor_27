import argparse
import json
import os
import random
import sys

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from paper_funcsim import AddressAwarePaperFunctionModel, AddressAwareTripletDataset, compute_embeddings
from paths import ensure_parent_under_data_root, ensure_under_data_root
from vocab import WordVocab


DEFAULT_COMBOS = [("O0", "O3"), ("O0", "Os"), ("O1", "Os"), ("O1", "O3"), ("O2", "Os"), ("O2", "O3")]


def parse_func_id(func_id: str):
    binary, opt, func_name = func_id.split("::", 2)
    return binary, opt, func_name


def build_opt_pairs(funcsim_pairs):
    grouped = {}
    for query_id, pair_data in funcsim_pairs.items():
        binary, opt, func_name = parse_func_id(query_id)
        key = f"{binary}::{func_name}"
        grouped.setdefault(key, {})[opt] = query_id
        for gt in pair_data.get("ground_truth", []):
            gt = str(gt)
            binary2, opt2, func_name2 = parse_func_id(gt)
            if binary2 == binary and func_name2 == func_name:
                grouped.setdefault(key, {})[opt2] = gt
    return grouped


def eval_combo(pair_index, embs, src_opt, dst_opt, poolsize, seed):
    pairs = []
    for opt_map in pair_index.values():
        src_id = opt_map.get(src_opt)
        dst_id = opt_map.get(dst_opt)
        if src_id and dst_id and src_id in embs and dst_id in embs:
            pairs.append((src_id, dst_id))

    if len(pairs) < 2:
        return {"num_pairs": len(pairs), "poolsize": poolsize, "mrr": 0.0, "recall@1": 0.0}

    rng = random.Random(seed)
    rng.shuffle(pairs)
    actual_pool = min(poolsize, len(pairs))
    usable = len(pairs) - (len(pairs) % actual_pool)
    pairs = pairs[:usable]

    mrrs = []
    recalls = []
    for start in range(0, len(pairs), actual_pool):
        batch = pairs[start : start + actual_pool]
        src = torch.stack([embs[src_id] for src_id, _ in batch])
        dst = torch.stack([embs[dst_id] for _, dst_id in batch])
        sim = torch.mm(src, dst.T).cpu().numpy()
        for i in range(actual_pool):
            order = np.argsort(-sim[i])
            rank = int(np.where(order == i)[0][0]) + 1
            recalls.append(1 if rank == 1 else 0)
            mrrs.append(1.0 / rank)

    return {
        "num_pairs": len(pairs),
        "poolsize": actual_pool,
        "mrr": float(np.mean(mrrs)) if mrrs else 0.0,
        "recall@1": float(np.mean(recalls)) if recalls else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate paper-aligned address-aware jTrans on optimisation pairs")
    parser.add_argument("--function_blocks", required=True)
    parser.add_argument("--funcsim_pairs", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--poolsizes", default="32,10000")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.function_blocks = ensure_under_data_root(args.function_blocks, "--function_blocks", must_exist=True)
    args.funcsim_pairs = ensure_under_data_root(args.funcsim_pairs, "--funcsim_pairs", must_exist=True)
    args.vocab = ensure_under_data_root(args.vocab, "--vocab", must_exist=True)
    args.checkpoint = ensure_under_data_root(args.checkpoint, "--checkpoint", must_exist=True)
    args.output = ensure_parent_under_data_root(args.output, "--output")

    vocab = WordVocab.load_vocab(args.vocab)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = ckpt.get("config", {})
    use_address_embedding = config.get("use_address_embedding", True)
    use_var_embedding = config.get("use_var_embedding", True)
    use_address_var = use_address_embedding or use_var_embedding
    seq_len = config.get("seq_len", 512)
    max_instructions = config.get("max_instructions", 100000)
    address_scale = config.get("address_scale", 10000.0)
    learnable_address_scale = config.get("learnable_address_scale", False)

    dataset = AddressAwareTripletDataset(
        function_blocks_file=args.function_blocks,
        funcsim_pairs_file=args.funcsim_pairs,
        vocab=vocab,
        seq_len=seq_len,
        max_instructions=max_instructions,
        use_address_var=use_address_var,
        seed=args.seed,
    )

    model = AddressAwarePaperFunctionModel(
        vocab_size=len(vocab),
        hidden=config.get("hidden", 256),
        n_layers=config.get("n_layers", 4),
        attn_heads=config.get("attn_heads", 4),
        seq_len=seq_len,
        dropout=config.get("dropout", 0.1),
        use_address_embedding=use_address_embedding,
        use_var_embedding=use_var_embedding,
        address_scale=address_scale,
        learnable_address_scale=learnable_address_scale,
    )
    model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)

    daddr_id = vocab.stoi.get("daddr")
    if daddr_id is not None and use_address_embedding:
        model.bert.embedding.set_special_token_ids(daddr_id)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device)

    func_ids = sorted(dataset.base.function_blocks)
    embs = compute_embeddings(model, dataset, func_ids, device, args.batch_size)
    pair_index = build_opt_pairs(dataset.base.funcsim_pairs)

    poolsizes = [int(p) for p in args.poolsizes.split(",") if p.strip()]
    results = {
        "checkpoint": args.checkpoint,
        "use_addr": use_address_embedding,
        "use_var": use_var_embedding,
        "poolsizes": poolsizes,
        "combos": {},
    }
    for poolsize in poolsizes:
        pool_key = f"pool_{poolsize}"
        results["combos"][pool_key] = {}
        for src_opt, dst_opt in DEFAULT_COMBOS:
            combo_key = f"{src_opt}-{dst_opt}"
            results["combos"][pool_key][combo_key] = eval_combo(pair_index, embs, src_opt, dst_opt, poolsize, args.seed)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
