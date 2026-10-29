import argparse
import json
import os
import random
import sys
import importlib.util
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from paths import ensure_parent_under_data_root, ensure_under_data_root
from vocab import WordVocab
import vocab as addressaware_vocab


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PALMTREE_ROOT = os.path.join(ROOT, "PalmTree", "addressaware")
if PALMTREE_ROOT not in sys.path:
    sys.path.append(PALMTREE_ROOT)

CUR = os.path.join(ROOT, "PalmTree", "addressaware", "funcsim")

# Make PalmTree funcsim helpers reuse the addressaware vocab loader so the
# lightweight JTrans vocab format resolves to a WordVocab-compatible object.
sys.modules["vocab"] = addressaware_vocab


def _load_local(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(CUR, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_dl = _load_local("funcsim_dataloader_opt_eval", "dataloader.py")
_mdl = _load_local("funcsim_model_opt_eval", "model.py")

FunctionSimilarityDataset = _dl.FunctionSimilarityDataset
FunctionSimilarityModel = _mdl.FunctionSimilarityModel

DEFAULT_COMBOS = [("O0", "O3"), ("O0", "Os"), ("O1", "Os"), ("O1", "O3"), ("O2", "Os"), ("O2", "O3")]


def setup_log(log_dir: str):
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(log_dir, f"eval_opt_pairs_{ts}.log")


def parse_func_id(func_id: str):
    binary, opt, func_name = func_id.split("::", 2)
    return binary, opt, func_name


@torch.no_grad()
def compute_embeddings(model, dataset, func_ids, device, batch_size):
    model.eval()
    embs = {}
    for start in tqdm(range(0, len(func_ids), batch_size), desc="Encoding", leave=False):
        batch_ids = func_ids[start : start + batch_size]
        encoded = [dataset.encode_function(fid) for fid in batch_ids]
        ids_t = torch.stack([e["input_ids"] for e in encoded]).to(device)
        seg_t = torch.stack([e["segment"] for e in encoded]).to(device)
        bp_t = torch.stack([e["binary_pos"] for e in encoded]).to(device)
        fp_t = torch.stack([e["function_pos"] for e in encoded]).to(device)
        bb_t = torch.stack([e["bb_pos"] for e in encoded]).to(device)
        vo_t = torch.stack([e["var_offsets"] for e in encoded]).to(device)
        ni = [e["num_instructions"] for e in encoded]
        bn = [e["instruction_boundaries"] for e in encoded]
        out = model(ids_t, seg_t, bp_t, fp_t, bb_t, vo_t, num_instructions=ni, instruction_boundaries=bn)
        for fid, emb in zip(batch_ids, out.cpu()):
            embs[fid] = F.normalize(emb, p=2, dim=0)
    return embs


def build_opt_pairs(funcsim_pairs):
    grouped = {}
    for query_id, pair_data in funcsim_pairs.items():
        binary, opt, func_name = parse_func_id(query_id)
        key = f"{binary}::{func_name}"
        grouped.setdefault(key, {})[opt] = query_id
        for gt in pair_data.get("ground_truth", []):
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

    mrrs = []
    recalls = []
    usable = len(pairs) - (len(pairs) % actual_pool)
    pairs = pairs[:usable]

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
    parser = argparse.ArgumentParser(description="Evaluate address-aware jTrans on paper-style optimisation pairs")
    parser.add_argument("--function_blocks", required=True)
    parser.add_argument("--funcsim_pairs", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--log_dir", required=True)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--attn_heads", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--max_instructions", type=int, default=64)
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--window_size", type=int, default=32)
    parser.add_argument("--window_subbatch", type=int, default=256)
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
    args.log_dir = ensure_parent_under_data_root(args.log_dir, "--log_dir")

    vocab = WordVocab.load_vocab(args.vocab)
    dataset = FunctionSimilarityDataset(
        function_blocks_file=args.function_blocks,
        funcsim_pairs_file=args.funcsim_pairs,
        vocab=vocab,
        seq_len=args.seq_len,
        max_instructions=args.max_instructions,
        negative_samples=1,
    )

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    has_address = any("address_pos" in k or "address_position" in k for k in sd)
    has_var = any("var_offset" in k or "var_position" in k for k in sd)
    use_addr = has_address and has_var
    dataset.use_address_var = use_addr

    model = FunctionSimilarityModel(
        vocab_size=len(vocab),
        hidden=args.hidden,
        n_layers=args.n_layers,
        attn_heads=args.attn_heads,
        max_len=args.window_size,
        window_size=args.window_size,
        instrs_per_window=2,
        sos_idx=vocab.stoi.get("<sos>", 3),
        eos_idx=vocab.stoi.get("<eos>", 2),
        pad_idx=vocab.stoi.get("<pad>", 0),
        use_address_embedding=use_addr,
        use_var_embedding=use_addr,
        embedding_dim=args.embedding_dim,
        window_subbatch=args.window_subbatch,
    )
    model.load_state_dict(sd, strict=False)
    model.to(torch.device(args.device if torch.cuda.is_available() else "cpu"))

    func_ids = sorted(dataset.function_blocks)
    embs = compute_embeddings(model, dataset, func_ids, next(model.parameters()).device, args.batch_size)
    pair_index = build_opt_pairs(dataset.funcsim_pairs)

    results = {
        "checkpoint": args.checkpoint,
        "use_addr": use_addr,
        "poolsizes": [int(p) for p in args.poolsizes.split(",") if p.strip()],
        "combos": {},
    }
    for poolsize in results["poolsizes"]:
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
