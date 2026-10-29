#!/usr/bin/env python3
"""
Re-evaluate baseline or address-aware retrieval pools and summarize metrics by
project (binary name).

Outputs:
- project_opt_pair.json / .csv: one row per project and optimization pair
- project_overall.json / .csv: all opt-pair queries merged per project
- global.json: global metrics for validation against existing summaries
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np


AAE_ROOT = Path("/path/to/numAnchor/JTrans")


def eprint(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def load_module(path: Path, module_name: str):
    parent = str(path.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_more(fp, buffer: str, chunk_size: int) -> Tuple[str, bool]:
    chunk = fp.read(chunk_size)
    if chunk == "":
        return buffer, True
    return buffer + chunk, False


def iter_top_level_object(path: str, chunk_size: int = 1024 * 1024) -> Iterator[Tuple[str, object]]:
    decoder = json.JSONDecoder()
    with open(path, "r", encoding="utf-8") as fp:
        buffer = ""
        pos = 0
        eof = False

        def refill() -> None:
            nonlocal buffer, eof
            if eof:
                return
            buffer, eof = _read_more(fp, buffer, chunk_size)

        def skip_ws() -> None:
            nonlocal pos
            while True:
                while pos < len(buffer) and buffer[pos].isspace():
                    pos += 1
                if pos < len(buffer) or eof:
                    return
                refill()

        def decode_at(start: int):
            while True:
                try:
                    return decoder.raw_decode(buffer, start)
                except json.JSONDecodeError:
                    if eof:
                        raise
                    refill()

        refill()
        skip_ws()
        if pos >= len(buffer) or buffer[pos] != "{":
            raise ValueError(f"{path} is not a JSON object")
        pos += 1

        while True:
            skip_ws()
            while pos < len(buffer) and buffer[pos] == ",":
                pos += 1
                skip_ws()

            if pos >= len(buffer):
                if eof:
                    raise ValueError(f"Unexpected EOF while parsing {path}")
                refill()
                continue

            if buffer[pos] == "}":
                break

            key, end = decode_at(pos)
            pos = end
            skip_ws()
            while pos >= len(buffer):
                if eof:
                    raise ValueError(f"Unexpected EOF after key in {path}")
                refill()
                skip_ws()
            if buffer[pos] != ":":
                raise ValueError(f"Expected ':' after key in {path}")
            pos += 1
            skip_ws()
            value, end = decode_at(pos)
            pos = end
            yield key, value

            if pos > chunk_size * 4:
                buffer = buffer[pos:]
                pos = 0


def parse_variant_id(variant_id: str) -> Tuple[str, str, str]:
    parts = variant_id.split("::", 2)
    if len(parts) != 3:
        return "", "", variant_id
    return parts[0], parts[1], parts[2]


def normalize_metric_row(row: Dict) -> Dict:
    q = row["query_count"]
    return {
        **row,
        "MRR": row["rr_sum"] / q if q else 0.0,
        "Recall@1": row["hit_1"] / q if q else 0.0,
        "Recall@5": row["hit_5"] / q if q else 0.0,
        "Recall@10": row["hit_10"] / q if q else 0.0,
    }


def load_selected_func_blocks(func_blocks_path: str, needed_ids: set[str]) -> Dict[str, object]:
    eprint(f"[func_blocks] scanning {func_blocks_path} for {len(needed_ids)} needed functions")
    selected = {}
    for idx, (func_id, payload) in enumerate(iter_top_level_object(func_blocks_path), start=1):
        if func_id in needed_ids:
            selected[func_id] = payload
            if len(selected) % 5000 == 0:
                eprint(f"[func_blocks] loaded {len(selected)} selected functions")
            if len(selected) == len(needed_ids):
                break
        if idx % 100000 == 0:
            eprint(f"[func_blocks] scanned {idx} entries")
    missing = sorted(needed_ids - set(selected))
    if missing:
        raise RuntimeError(f"Missing {len(missing)} required functions; first few: {missing[:5]}")
    eprint(f"[func_blocks] loaded all {len(selected)} required functions")
    return selected


def normalize_embeddings(emb: np.ndarray) -> np.ndarray:
    return emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)


def compute_ranks(query_emb: np.ndarray, pool_emb: np.ndarray, gt_indices: Sequence[int]) -> List[int]:
    sim = np.matmul(query_emb, pool_emb.T)
    ranks = np.argsort(-sim, axis=1)
    result = []
    for i, gt_idx in enumerate(gt_indices):
        rank_pos = int(np.where(ranks[i] == gt_idx)[0][0]) + 1
        result.append(rank_pos)
    return result


def ensure_embeddings(
    missing_ids: Sequence[str],
    cache: Dict[str, np.ndarray],
    generate_fn,
) -> None:
    if not missing_ids:
        return
    emb = generate_fn(list(missing_ids))
    for func_id, vec in zip(missing_ids, emb):
        cache[func_id] = vec


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def write_csv(path: Path, rows: List[Dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def metric_fieldnames(include_opt_pair: bool) -> List[str]:
    fields = ["project"]
    if include_opt_pair:
        fields.append("opt_pair")
    fields.extend(["query_count", "MRR", "Recall@1", "Recall@5", "Recall@10"])
    return fields


def summarize_rows(rows: List[Dict]) -> Dict:
    total_q = sum(r["query_count"] for r in rows)
    total_rr = sum(r["rr_sum"] for r in rows)
    total_h1 = sum(r["hit_1"] for r in rows)
    total_h5 = sum(r["hit_5"] for r in rows)
    total_h10 = sum(r["hit_10"] for r in rows)
    return {
        "query_count": total_q,
        "MRR": total_rr / total_q if total_q else 0.0,
        "Recall@1": total_h1 / total_q if total_q else 0.0,
        "Recall@5": total_h5 / total_q if total_q else 0.0,
        "Recall@10": total_h10 / total_q if total_q else 0.0,
    }


def filter_pool_files(pool_files: Iterable[Path], pool_size: Optional[int], opt_pair: Optional[str]) -> List[Path]:
    files = sorted(pool_files)
    if pool_size is not None:
        files = [p for p in files if f"{pool_size}" in p.name]
    if opt_pair is not None:
        files = [p for p in files if opt_pair in p.name]
    return files


def run_baseline(args: argparse.Namespace) -> Dict:
    module = load_module(AAE_ROOT / "evaluate_baseline_pools.py", "eval_baseline_pools_mod")

    tokenizer = module.load_vocab(args.vocab)
    model, config = module.load_model(args.model, tokenizer, args.device)
    model_vocab_size = config.vocab_size

    pool_dir = Path(args.pool_dir)
    pool_files = filter_pool_files(pool_dir.glob("pool_*.json"), args.pool_size, args.opt_pair)
    if not pool_files:
        raise RuntimeError("No baseline pool files matched the provided filters")

    pool_specs = []
    needed_ids: set[str] = set()
    for pool_file in pool_files:
        pool_size = int(pool_file.stem.split("_")[1])
        opt_pair = "_".join(pool_file.stem.split("_")[2:])
        query_size = int(pool_size * 0.2)
        query_file = pool_dir / f"query_{query_size}_from_pool_{pool_size}_{opt_pair}.json"
        if not query_file.exists():
            raise RuntimeError(f"Missing query file for {pool_file}: {query_file}")

        with open(pool_file, "r", encoding="utf-8") as f:
            pool_data = json.load(f)
        with open(query_file, "r", encoding="utf-8") as f:
            query_data = json.load(f)

        pool_ids = pool_data["pool"]
        query_ids = query_data["queries"]
        gt_indices = query_data["ground_truth"]
        needed_ids.update(pool_ids)
        needed_ids.update(query_ids)
        pool_specs.append(
            {
                "pool_file": pool_file.name,
                "opt_pair": opt_pair,
                "pool_ids": pool_ids,
                "query_ids": query_ids,
                "gt_indices": gt_indices,
            }
        )

    func_blocks = load_selected_func_blocks(args.func_blocks, needed_ids)
    emb_cache: Dict[str, np.ndarray] = {}

    def generate(func_ids: List[str]) -> np.ndarray:
        return module.generate_embeddings(
            model,
            func_ids,
            func_blocks,
            tokenizer,
            args.device,
            args.max_len,
            args.batch_size,
            model_vocab_size,
            fair_token_budget=args.fair_token_budget,
        )

    detail_groups: Dict[Tuple[str, str], Dict] = {}

    for spec in pool_specs:
        pool_ids = spec["pool_ids"]
        query_ids = spec["query_ids"]
        gt_indices = spec["gt_indices"]
        missing = [fid for fid in set(pool_ids + query_ids) if fid not in emb_cache]
        if missing:
            eprint(f"[baseline] generating {len(missing)} uncached embeddings for {spec['pool_file']}")
            ensure_embeddings(missing, emb_cache, generate)

        pool_emb = normalize_embeddings(np.vstack([emb_cache[fid] for fid in pool_ids]))
        query_emb = normalize_embeddings(np.vstack([emb_cache[fid] for fid in query_ids]))
        ranks = compute_ranks(query_emb, pool_emb, gt_indices)

        for query_id, rank in zip(query_ids, ranks):
            project, _opt, _func = parse_variant_id(query_id)
            key = (project, spec["opt_pair"])
            bucket = detail_groups.setdefault(
                key,
                {
                    "project": project,
                    "opt_pair": spec["opt_pair"],
                    "query_count": 0,
                    "rr_sum": 0.0,
                    "hit_1": 0,
                    "hit_5": 0,
                    "hit_10": 0,
                },
            )
            bucket["query_count"] += 1
            bucket["rr_sum"] += 1.0 / rank
            bucket["hit_1"] += int(rank == 1)
            bucket["hit_5"] += int(rank <= 5)
            bucket["hit_10"] += int(rank <= 10)

    return finalize_results(detail_groups, args.run_label)


def load_addressaware_vocab_stoi(vocab_dir: str) -> Dict[str, int]:
    vocab_stoi = {}
    with open(Path(vocab_dir) / "vocab.txt", "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            vocab_stoi[line.strip()] = idx
    return vocab_stoi


def run_addressaware(args: argparse.Namespace) -> Dict:
    module = load_module(AAE_ROOT / "evaluate_addressaware_pools.py", "eval_addressaware_pools_mod")

    tokenizer = module.load_vocab(args.vocab_path)
    vocab_stoi = load_addressaware_vocab_stoi(args.vocab_path)
    model, _config = module.load_model(args.checkpoint, vocab_stoi, args.device)

    pool_dir = Path(args.pool_dir)
    pool_files = filter_pool_files(pool_dir.glob("pool_*.json"), args.pool_size, args.opt_pair)
    if not pool_files:
        raise RuntimeError("No address-aware pool files matched the provided filters")

    pool_specs = []
    needed_ids: set[str] = set()
    for pool_file in pool_files:
        with open(pool_file, "r", encoding="utf-8") as f:
            pool_data = json.load(f)

        if "queries" in pool_data:
            pool_ids = [item["high_id"] for item in pool_data["pool"]]
            query_ids = [item["query_id"] for item in pool_data["queries"]]
            gt_ids = [item["gt_id"] for item in pool_data["queries"]]
            gt_map = {func_id: idx for idx, func_id in enumerate(pool_ids)}
            gt_indices = [gt_map[gt_id] for gt_id in gt_ids]
            opt_pair = pool_data.get("opt_pair") or pool_file.stem.replace("pool_", "")
        else:
            pool_ids = pool_data["pool"]
            pool_name = pool_file.stem
            query_file = pool_dir / f"query_{pool_name.replace('pool_', '')}.json"
            if not query_file.exists():
                raise RuntimeError(f"Missing query file for {pool_file}: {query_file}")
            with open(query_file, "r", encoding="utf-8") as f:
                query_data = json.load(f)
            query_ids = [item["query_id"] for item in query_data["queries"]]
            gt_ids = [item["gt_id"] for item in query_data["queries"]]
            gt_map = {func_id: idx for idx, func_id in enumerate(pool_ids)}
            gt_indices = [gt_map[gt_id] for gt_id in gt_ids]
            opt_pair = pool_file.stem.replace("pool_", "")

        needed_ids.update(pool_ids)
        needed_ids.update(query_ids)
        pool_specs.append(
            {
                "pool_file": pool_file.name,
                "opt_pair": opt_pair,
                "pool_ids": pool_ids,
                "query_ids": query_ids,
                "gt_indices": gt_indices,
            }
        )

    func_blocks = load_selected_func_blocks(args.func_blocks, needed_ids)
    emb_cache: Dict[str, np.ndarray] = {}

    def generate(func_ids: List[str]) -> np.ndarray:
        return module.generate_embeddings(
            model,
            func_ids,
            func_blocks,
            tokenizer,
            vocab_stoi,
            args.device,
            args.max_len,
            args.batch_size,
            args.mask_binary_pos,
            args.multi_layer_pool,
        )

    detail_groups: Dict[Tuple[str, str], Dict] = {}

    for spec in pool_specs:
        pool_ids = spec["pool_ids"]
        query_ids = spec["query_ids"]
        gt_indices = spec["gt_indices"]
        missing = [fid for fid in set(pool_ids + query_ids) if fid not in emb_cache]
        if missing:
            eprint(f"[addressaware] generating {len(missing)} uncached embeddings for {spec['pool_file']}")
            ensure_embeddings(missing, emb_cache, generate)

        pool_emb = normalize_embeddings(np.vstack([emb_cache[fid] for fid in pool_ids]))
        query_emb = normalize_embeddings(np.vstack([emb_cache[fid] for fid in query_ids]))
        ranks = compute_ranks(query_emb, pool_emb, gt_indices)

        for query_id, rank in zip(query_ids, ranks):
            project, _opt, _func = parse_variant_id(query_id)
            key = (project, spec["opt_pair"])
            bucket = detail_groups.setdefault(
                key,
                {
                    "project": project,
                    "opt_pair": spec["opt_pair"],
                    "query_count": 0,
                    "rr_sum": 0.0,
                    "hit_1": 0,
                    "hit_5": 0,
                    "hit_10": 0,
                },
            )
            bucket["query_count"] += 1
            bucket["rr_sum"] += 1.0 / rank
            bucket["hit_1"] += int(rank == 1)
            bucket["hit_5"] += int(rank <= 5)
            bucket["hit_10"] += int(rank <= 10)

    return finalize_results(detail_groups, args.run_label)


def finalize_results(detail_groups: Dict[Tuple[str, str], Dict], run_label: str) -> Dict:
    detail_rows_raw = sorted(detail_groups.values(), key=lambda x: (x["project"], x["opt_pair"]))
    detail_rows = [normalize_metric_row(row) for row in detail_rows_raw]

    overall_groups: Dict[str, Dict] = {}
    for row in detail_rows_raw:
        bucket = overall_groups.setdefault(
            row["project"],
            {
                "project": row["project"],
                "query_count": 0,
                "rr_sum": 0.0,
                "hit_1": 0,
                "hit_5": 0,
                "hit_10": 0,
            },
        )
        bucket["query_count"] += row["query_count"]
        bucket["rr_sum"] += row["rr_sum"]
        bucket["hit_1"] += row["hit_1"]
        bucket["hit_5"] += row["hit_5"]
        bucket["hit_10"] += row["hit_10"]

    overall_rows_raw = sorted(overall_groups.values(), key=lambda x: x["project"])
    overall_rows = [normalize_metric_row(row) for row in overall_rows_raw]

    global_by_opt_pair = []
    by_opt_pair: Dict[str, List[Dict]] = defaultdict(list)
    for row in detail_rows_raw:
        by_opt_pair[row["opt_pair"]].append(row)
    for opt_pair, rows in sorted(by_opt_pair.items()):
        metrics = summarize_rows(rows)
        global_by_opt_pair.append({"opt_pair": opt_pair, **metrics})

    global_overall = summarize_rows(detail_rows_raw)
    return {
        "run_label": run_label,
        "project_opt_pair": detail_rows,
        "project_overall": overall_rows,
        "global_by_opt_pair": global_by_opt_pair,
        "global_overall": global_overall,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export project-level retrieval results")
    parser.add_argument("--mode", choices=["baseline", "addressaware"], required=True)
    parser.add_argument("--run_label", required=True, help="Short label used in output metadata")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pool_dir", required=True)
    parser.add_argument("--func_blocks", required=True)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pool_size", type=int, default=10000)
    parser.add_argument("--opt_pair", default=None)

    parser.add_argument("--model", help="Baseline finetuned checkpoint")
    parser.add_argument("--vocab", help="Baseline tokenizer path")
    parser.add_argument("--fair_token_budget", action="store_true")

    parser.add_argument("--checkpoint", help="Address-aware finetuned checkpoint")
    parser.add_argument("--vocab_path", help="Address-aware tokenizer path")
    parser.add_argument("--mask_binary_pos", action="store_true")
    parser.add_argument("--multi_layer_pool", type=int, default=0)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "baseline":
        if not args.model or not args.vocab:
            raise RuntimeError("--model and --vocab are required for baseline mode")
        results = run_baseline(args)
    else:
        if not args.checkpoint or not args.vocab_path:
            raise RuntimeError("--checkpoint and --vocab_path are required for addressaware mode")
        results = run_addressaware(args)

    write_json(output_dir / "project_opt_pair.json", results["project_opt_pair"])
    write_csv(
        output_dir / "project_opt_pair.csv",
        results["project_opt_pair"],
        metric_fieldnames(include_opt_pair=True),
    )
    write_json(output_dir / "project_overall.json", results["project_overall"])
    write_csv(
        output_dir / "project_overall.csv",
        results["project_overall"],
        metric_fieldnames(include_opt_pair=False),
    )
    write_json(
        output_dir / "global.json",
        {
            "run_label": results["run_label"],
            "global_by_opt_pair": results["global_by_opt_pair"],
            "global_overall": results["global_overall"],
        },
    )
    eprint(f"[done] wrote project-level summaries to {output_dir}")


if __name__ == "__main__":
    main()
