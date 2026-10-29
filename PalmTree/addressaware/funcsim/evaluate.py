"""
Evaluation script for Function Similarity

Computes retrieval metrics (Recall@1/5/10/20 and MRR) on a pool of functions.

Usage:
    python evaluate.py \\
        --function_blocks /path/to/funcsim/function_blocks.json \\
        --funcsim_pairs   /path/to/funcsim/funcsim_pairs.json \\
        --vocab           /path/to/vocab.pkl \\
        --checkpoint      ../../output/funcsim/best_model.pt \\
        --output          ../../output/funcsim/results.json

Optional pool file (pre-sampled for reproducible evaluation):
    --eval_pool /path/to/funcsim/pool_1k.json

The pool JSON must have the structure:
    {
      "query_ids":       ["id1", "id2", ...],
      "pool_function_ids": ["id1", "id2", ..., "idN"],
      "metadata": { "seed": 42, "generated_at": "..." }
    }
"""

import argparse
import json
import logging
import os
import sys
import importlib.util
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# ── path setup ────────────────────────────────────────────────────────────────
_parent = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
if _parent not in sys.path:
    sys.path.insert(0, _parent)
from vocab import WordVocab

_cur = os.path.dirname(os.path.abspath(__file__))

def _load_local(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_cur, fname))
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_dl  = _load_local("funcsim_dataloader", "dataloader.py")
_mdl = _load_local("funcsim_model",      "model.py")

FunctionSimilarityDataset = _dl.FunctionSimilarityDataset
FunctionSimilarityModel   = _mdl.FunctionSimilarityModel
# ─────────────────────────────────────────────────────────────────────────────


def setup_logging(log_dir: str):
    os.makedirs(log_dir, exist_ok=True)
    ts       = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'eval_{ts}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s  %(levelname)s  %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return logging.getLogger('funcsim_eval')


# ── embedding computation ─────────────────────────────────────────────────────

@torch.no_grad()
def compute_embeddings(model, dataset, func_ids, device, batch_size=64,
                       cache_path=None):
    """
    Encode every function in func_ids using the sliding-window BERT strategy.

    Results are cached to cache_path (a .pt file) to avoid re-running.
    Returns:
        dict { func_id -> embedding_tensor [embedding_dim] }
    """
    # try cache
    embs = {}
    if cache_path and os.path.exists(cache_path):
        cached = torch.load(cache_path, map_location='cpu')
        embs = cached.get('embeddings', {})
        missing = [fid for fid in func_ids if fid not in embs]
        if not missing:
            print(f"[Eval] Loaded {len(embs)} embeddings from cache: {cache_path}")
            return embs
        print(f"[Eval] Cache incomplete ({len(missing)} missing). Reusing {len(embs)} cached embeddings and computing the rest …")
        func_ids = missing

    model.eval()

    # encode in mini-batches
    for start in tqdm(range(0, len(func_ids), batch_size), desc="Encoding"):
        batch_ids = func_ids[start : start + batch_size]
        encoded   = [dataset.encode_function(fid) for fid in batch_ids]

        ids_t  = torch.stack([e['input_ids']    for e in encoded]).to(device)
        seg_t  = torch.stack([e['segment']      for e in encoded]).to(device)
        bp_t   = torch.stack([e['binary_pos']   for e in encoded]).to(device)
        fp_t   = torch.stack([e['function_pos'] for e in encoded]).to(device)
        bb_t   = torch.stack([e['bb_pos']       for e in encoded]).to(device)
        vo_t   = torch.stack([e['var_offsets']  for e in encoded]).to(device)
        sk_t   = torch.stack([e['section_kind_ids'] for e in encoded]).to(device)
        at_t   = torch.stack([e['addr_type_ids'] for e in encoded]).to(device)

        # boundary metadata stays as Python lists (variable-length per sample)
        ni     = [e['num_instructions']       for e in encoded]
        bn     = [e['instruction_boundaries'] for e in encoded]

        out = model(ids_t, seg_t, bp_t, fp_t, bb_t, vo_t, sk_t, at_t,
                    num_instructions=ni, instruction_boundaries=bn)   # [B, D]
        for fid, emb in zip(batch_ids, out.cpu()):
            embs[fid] = emb

    # save cache
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save({'embeddings': embs}, cache_path)
        print(f"[Eval] Embeddings cached to {cache_path}")

    return embs


# ── retrieval metrics ─────────────────────────────────────────────────────────

def compute_retrieval_metrics(embs, query_ids, funcsim_pairs, pool_ids,
                               top_k=(1, 5, 10, 20), logger=None):
    """
    For each query, rank all pool functions by cosine similarity and compute
    Recall@K and MRR.

    Args:
        embs         : { func_id -> tensor [D] }
        query_ids    : list of query function IDs
        funcsim_pairs: { func_id -> { 'ground_truth': [...] } }
        pool_ids     : list of all candidate function IDs (includes queries)
        top_k        : tuple of K values
    Returns:
        dict with 'recall@K', 'mrr', 'num_queries' keys
    """
    # build matrix [N_pool, D]
    valid_pool = [fid for fid in pool_ids if fid in embs]
    pool_mat   = torch.stack([embs[fid] for fid in valid_pool])  # [P, D]
    pool_mat   = F.normalize(pool_mat, p=2, dim=1)               # unit sphere

    recall_hits = {k: [] for k in top_k}
    rr_list     = []

    log = (lambda msg: logger.info(msg)) if logger else print

    for query_id in tqdm(query_ids, desc="Ranking"):
        if query_id not in embs or query_id not in funcsim_pairs:
            continue
        gt_raw = funcsim_pairs[query_id].get('ground_truth', [])
        gt_set = set(str(g) for g in gt_raw)
        if not gt_set:
            continue

        q_emb = F.normalize(embs[query_id].unsqueeze(0), p=2, dim=1).to(pool_mat.device)
        sims  = (q_emb @ pool_mat.T).squeeze(0)                  # [P]

        # mask self
        try:
            self_idx = valid_pool.index(query_id)
            sims[self_idx] = -2.0
        except ValueError:
            pass

        ranked_indices = sims.argsort(descending=True).tolist()
        ranked_ids     = [valid_pool[i] for i in ranked_indices]

        # Recall@K
        for k in top_k:
            hit = int(bool(set(ranked_ids[:k]) & gt_set))
            recall_hits[k].append(hit)

        # MRR
        rr = 0.0
        for rank, cid in enumerate(ranked_ids, start=1):
            if cid in gt_set:
                rr = 1.0 / rank
                break
        rr_list.append(rr)

    results = {f'recall@{k}': float(np.mean(recall_hits[k]))
               for k in top_k if recall_hits[k]}
    results['mrr']         = float(np.mean(rr_list)) if rr_list else 0.0
    results['num_queries'] = len(rr_list)
    return results


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Funcsim evaluation")

    # Data
    parser.add_argument("--function_blocks", required=True,
                        help="Path to function_blocks.json")
    parser.add_argument("--funcsim_pairs",   required=True,
                        help="Path to funcsim_pairs.json")
    parser.add_argument("--vocab",           required=True,
                        help="Path to vocab.pkl")
    parser.add_argument("--eval_pool",       default=None,
                        help="Path to pre-sampled pool JSON "
                             "(query_ids + pool_function_ids). "
                             "If omitted, all functions are used.")

    # Model
    parser.add_argument("--checkpoint",    required=True)
    parser.add_argument("--hidden",        type=int, default=768)
    parser.add_argument("--n_layers",      type=int, default=12)
    parser.add_argument("--attn_heads",    type=int, default=12)
    parser.add_argument("--seq_len",       type=int, default=420)
    parser.add_argument("--max_instructions", type=int, default=50,
                        help="Max number of instructions per function")
    parser.add_argument("--embedding_dim", type=int, default=256)
    parser.add_argument("--window_size", type=int, default=32,
                        help="Instruction-pair BERT window length; should match pretrain seq_len")
    parser.add_argument("--window_subbatch", type=int, default=256,
                        help="Max number of BERT windows processed per internal sub-batch")

    # Evaluation
    parser.add_argument("--batch_size",  type=int,   default=64)
    parser.add_argument("--pool_size",   type=int,   default=None,
                        help="Random subset of pool functions (None = all)")

    # Output
    parser.add_argument("--output",  default="../../output/funcsim/eval_results.json")
    parser.add_argument("--log_dir", default="../../log/funcsim")
    parser.add_argument("--task_name", default=None,
                        help="Tag used in the embedding cache filename")

    # Device
    parser.add_argument("--device",      default="cuda")
    parser.add_argument("--num_workers", type=int, default=4)

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger = setup_logging(args.log_dir)

    logger.info("=" * 70)
    logger.info("Function Similarity Evaluation")
    logger.info("=" * 70)
    logger.info(f"device     : {device}")
    logger.info(f"checkpoint : {args.checkpoint}")
    logger.info("=" * 70)

    # ── vocabulary ────────────────────────────────────────────────────────────
    vocab = WordVocab.load_vocab(args.vocab)
    logger.info(f"Vocab size: {len(vocab)}")

    # ── load & inspect checkpoint ─────────────────────────────────────────────
    logger.info("Loading checkpoint …")
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    sd   = ckpt.get('model_state_dict', ckpt)

    has_address = any('address_pos' in k or 'address_position' in k for k in sd)
    has_var     = any('var_offset'  in k or 'var_position'     in k for k in sd)
    use_addr    = has_address and has_var

    # read embedding_dim from checkpoint metadata if available
    emb_dim = ckpt.get('embedding_dim', args.embedding_dim)

    logger.info(f"Checkpoint  epoch={ckpt.get('epoch','?')}  "
                f"val_acc={ckpt.get('val_acc', float('nan')):.4f}")
    logger.info(f"use_addr={use_addr}  embedding_dim={emb_dim}")

    # ── build a dataset object (no pairs needed — we call encode_function) ────
    logger.info("Loading function blocks …")
    dataset = FunctionSimilarityDataset(
        function_blocks_file=args.function_blocks,
        funcsim_pairs_file=None,    # not needed for encoding
        vocab=vocab,
        seq_len=args.seq_len,
        max_instructions=args.max_instructions,
        negative_samples=0,
    )
    dataset.use_address_var = use_addr

    logger.info("Loading funcsim pairs …")
    with open(args.funcsim_pairs, 'r') as f:
        funcsim_pairs = json.load(f)

    # ── determine pool & query sets ───────────────────────────────────────────
    if args.eval_pool:
        logger.info(f"Loading eval pool from {args.eval_pool}")
        with open(args.eval_pool, 'r') as f:
            pool_data = json.load(f)
        query_ids    = pool_data['query_ids']
        pool_ids     = pool_data['pool_function_ids']
        selected_gt  = pool_data.get('selected_ground_truth_ids')
        # filter to functions we actually have blocks for
        pool_ids     = [fid for fid in pool_ids     if fid in dataset.function_blocks]
        query_ids    = [fid for fid in query_ids    if fid in dataset.function_blocks
                                                    and fid in funcsim_pairs]
        if selected_gt:
            # When an eval pool pins a specific GT subset, prefer that exact
            # subset over the full dataset-level GT list.
            for qid in query_ids:
                if qid in selected_gt:
                    funcsim_pairs[qid]['ground_truth'] = selected_gt[qid]
        logger.info(f"Pool: {len(pool_ids)} functions | Queries: {len(query_ids)}")
    else:
        import random
        all_ids = list(dataset.function_blocks.keys())
        if args.pool_size and args.pool_size < len(all_ids):
            random.seed(42)
            # make sure every query + its ground-truth is in the pool
            query_ids = [fid for fid in funcsim_pairs if fid in dataset.function_blocks]
            required  = set(query_ids)
            for qid in query_ids:
                required.update(str(g) for g in funcsim_pairs[qid].get('ground_truth', []))
            distractors = [f for f in all_ids if f not in required]
            random.shuffle(distractors)
            n_dist   = max(0, args.pool_size - len(required))
            pool_ids = list(required) + distractors[:n_dist]
        else:
            query_ids = [fid for fid in funcsim_pairs if fid in dataset.function_blocks]
            pool_ids  = all_ids
        logger.info(f"Pool: {len(pool_ids)} functions | Queries: {len(query_ids)}")

    # ── build model ───────────────────────────────────────────────────────────
    logger.info("Building model …")
    model = FunctionSimilarityModel(
        vocab_size=len(vocab),
        hidden=args.hidden,
        n_layers=args.n_layers,
        attn_heads=args.attn_heads,
        max_len=args.window_size,
        window_size=args.window_size,
        instrs_per_window=2,                 # 2 instructions per BERT window
        sos_idx=vocab.stoi.get('<sos>', 3),
        eos_idx=vocab.stoi.get('<eos>', 2),
        pad_idx=vocab.stoi.get('<pad>', 0),
        use_address_embedding=use_addr,
        use_var_embedding=use_addr,
        embedding_dim=emb_dim,
        freeze_bert=False,
        gradient_checkpointing=False,
        window_subbatch=args.window_subbatch,
    )

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        logger.warning(f"Missing keys: {missing[:5]}")
    if unexpected:
        logger.warning(f"Unexpected keys: {unexpected[:5]}")

    daddr_ids = [
        token_id
        for token, token_id in vocab.stoi.items()
        if token == 'daddr' or token.startswith('daddr_')
    ]
    if daddr_ids and use_addr:
        model.bert.embedding.set_special_token_ids(sorted(daddr_ids))

    model = model.to(device)
    model.eval()

    # ── compute / load embeddings ─────────────────────────────────────────────
    cache_path = None
    if args.task_name:
        cache_dir  = os.path.join(os.path.dirname(args.checkpoint), 'embeddings')
        cache_path = os.path.join(cache_dir,
                                  f'{args.task_name}_pool{len(pool_ids)}.pt')

    logger.info("Computing embeddings …")
    embs = compute_embeddings(model, dataset, pool_ids, device,
                              batch_size=args.batch_size,
                              cache_path=cache_path)
    logger.info(f"Embeddings computed for {len(embs)} functions")

    # ── retrieval metrics ─────────────────────────────────────────────────────
    logger.info("Computing Recall@K and MRR …")
    results = compute_retrieval_metrics(
        embs=embs,
        query_ids=query_ids,
        funcsim_pairs=funcsim_pairs,
        pool_ids=pool_ids,
        top_k=(1, 5, 10, 20),
        logger=logger,
    )

    logger.info("\n" + "=" * 60)
    logger.info("Results:")
    logger.info("=" * 60)
    logger.info(f"Queries evaluated : {results['num_queries']}")
    for k in (1, 5, 10, 20):
        key = f'recall@{k}'
        if key in results:
            logger.info(f"  Recall@{k:<3}  = {results[key]:.4f}")
    logger.info(f"  MRR        = {results['mrr']:.4f}")
    logger.info("=" * 60)

    # ── save results ──────────────────────────────────────────────────────────
    out_data = {
        'checkpoint': args.checkpoint,
        'pool_size':  len(pool_ids),
        'results':    results,
        'checkpoint_info': {
            'epoch':    ckpt.get('epoch'),
            'val_acc':  ckpt.get('val_acc'),
            'use_addr': use_addr,
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(out_data, f, indent=2)
    logger.info(f"Results saved to {args.output}")


if __name__ == '__main__':
    main()
