#!/usr/bin/env python3
"""
Synthetic daddr-identity sanity check.

For each sampled instruction containing exactly one daddr token, we create:
  - positive: an exact copy of the instruction
  - negative: the same instruction with only the daddr operand replaced
              by a different daddr of the same daddr token type

This isolates whether the encoder can distinguish daddr identity when the
instruction template is otherwise unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from probe_eval import (  # noqa: E402
    DADDR_RE,
    FrozenBackboneEncoder,
    ProbePair,
    ProbeUnit,
    WordVocab,
    choose_threshold,
    compute_metrics,
)


def extract_single_daddr_match(inst_text: str) -> Optional[Tuple[str, str]]:
    matches = list(DADDR_RE.finditer(inst_text))
    if len(matches) != 1:
        return None
    match = matches[0]
    return match.group(1), match.group(0)


def load_unique_daddr_instructions(path: str) -> List[str]:
    seen = set()
    instructions: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            for inst_text in parts:
                if not inst_text or inst_text in seen:
                    continue
                if extract_single_daddr_match(inst_text) is None:
                    continue
                seen.add(inst_text)
                instructions.append(inst_text)
    return instructions


def build_synthetic_pairs(
    instructions: Sequence[str],
    max_pairs: int,
    seed: int,
) -> Tuple[List[ProbePair], Dict[str, ProbeUnit]]:
    rng = random.Random(seed)
    by_token_type: Dict[str, List[Tuple[str, str]]] = {}

    for inst_text in instructions:
        parsed = extract_single_daddr_match(inst_text)
        if parsed is None:
            continue
        token_type, full_daddr = parsed
        by_token_type.setdefault(token_type, []).append((inst_text, full_daddr))

    pairs: List[ProbePair] = []
    units: Dict[str, ProbeUnit] = {}
    counters = {"anchor": 0, "pos": 0, "neg": 0}

    token_types = list(by_token_type.keys())
    rng.shuffle(token_types)
    for token_type in token_types:
        entries = by_token_type[token_type]
        if len(entries) < 2:
            continue

        shuffled_entries = list(entries)
        rng.shuffle(shuffled_entries)
        for inst_text, full_daddr in shuffled_entries:
            replacement_candidates = [
                other_daddr for _, other_daddr in entries if other_daddr != full_daddr
            ]
            if not replacement_candidates:
                continue
            replacement = rng.choice(replacement_candidates)
            negative_text = inst_text.replace(full_daddr, replacement, 1)

            anchor_id = f"anchor:{counters['anchor']}"
            pos_id = f"pos:{counters['pos']}"
            neg_id = f"neg:{counters['neg']}"
            counters["anchor"] += 1
            counters["pos"] += 1
            counters["neg"] += 1

            units[anchor_id] = ProbeUnit(anchor_id, "instruction", [inst_text])
            units[pos_id] = ProbeUnit(pos_id, "instruction", [inst_text])
            units[neg_id] = ProbeUnit(neg_id, "instruction", [negative_text])

            pairs.append(
                ProbePair(
                    left_id=anchor_id,
                    right_id=pos_id,
                    label=1,
                    group_id=f"synthetic_daddr:{token_type}:{anchor_id}",
                    metadata={
                        "task": "synthetic_daddr_identity",
                        "daddr_token": token_type,
                        "anchor_daddr": full_daddr,
                        "candidate_daddr": full_daddr,
                    },
                )
            )
            pairs.append(
                ProbePair(
                    left_id=anchor_id,
                    right_id=neg_id,
                    label=0,
                    group_id=f"synthetic_daddr_neg:{token_type}:{anchor_id}",
                    metadata={
                        "task": "synthetic_daddr_identity",
                        "daddr_token": token_type,
                        "anchor_daddr": full_daddr,
                        "candidate_daddr": replacement,
                    },
                )
            )

            if len(pairs) >= max_pairs:
                return pairs[:max_pairs], units

    return pairs[:max_pairs], units


def cosine_score(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())


def build_scores(pairs: Sequence[ProbePair], embeddings: Dict[str, object]) -> List[float]:
    scores: List[float] = []
    for pair in pairs:
        left = embeddings[pair.left_id].pooled
        right = embeddings[pair.right_id].pooled
        scores.append(cosine_score(left, right))
    return scores


def summarize_scores(scores: Sequence[float], labels: Sequence[int]) -> Dict[str, float]:
    pos = [score for score, label in zip(scores, labels) if label == 1]
    neg = [score for score, label in zip(scores, labels) if label == 0]
    return {
        "positive_mean": float(sum(pos) / len(pos)) if pos else float("nan"),
        "negative_mean": float(sum(neg) / len(neg)) if neg else float("nan"),
        "margin": float((sum(pos) / len(pos)) - (sum(neg) / len(neg))) if pos and neg else float("nan"),
    }


def run_for_model(
    checkpoint_path: str,
    vocab: WordVocab,
    hidden: int,
    n_layers: int,
    attn_heads: int,
    seq_len: int,
    batch_size: int,
    device: torch.device,
    train_pairs: Sequence[ProbePair],
    test_pairs: Sequence[ProbePair],
    units: Dict[str, ProbeUnit],
) -> Dict[str, object]:
    encoder = FrozenBackboneEncoder(
        checkpoint_path=checkpoint_path,
        vocab=vocab,
        hidden=hidden,
        n_layers=n_layers,
        attn_heads=attn_heads,
        seq_len=seq_len,
        max_instructions=8,
        device=device,
    )
    embeddings = encoder.encode_units(units, batch_size=batch_size)
    train_scores = build_scores(train_pairs, embeddings)
    test_scores = build_scores(test_pairs, embeddings)
    train_labels = [pair.label for pair in train_pairs]
    test_labels = [pair.label for pair in test_pairs]
    threshold = choose_threshold(train_scores, train_labels)
    metrics = compute_metrics(test_scores, test_labels, threshold)
    metrics["score_summary"] = summarize_scores(test_scores, test_labels)
    return {
        "use_addr": encoder.use_addr,
        "zero_shot": metrics,
    }


def main():
    parser = argparse.ArgumentParser(description="Synthetic daddr identity sanity check")
    parser.add_argument("--train_cfg", required=True)
    parser.add_argument("--val_cfg", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--baseline_checkpoint", required=True)
    parser.add_argument("--aa_checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--hidden", type=int, default=768)
    parser.add_argument("--n_layers", type=int, default=12)
    parser.add_argument("--attn_heads", type=int, default=12)
    parser.add_argument("--seq_len", type=int, default=32)
    parser.add_argument("--max_train_pairs", type=int, default=2000)
    parser.add_argument("--max_test_pairs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    train_insts = load_unique_daddr_instructions(args.train_cfg)
    val_insts = load_unique_daddr_instructions(args.val_cfg)
    train_pairs, train_units = build_synthetic_pairs(train_insts, args.max_train_pairs, args.seed + 1)
    test_pairs, test_units = build_synthetic_pairs(val_insts, args.max_test_pairs, args.seed + 2)
    units = {}
    units.update(train_units)
    units.update(test_units)

    vocab = WordVocab.load_vocab(args.vocab)
    device = torch.device(args.device)

    report = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "config": vars(args),
        "num_train_pairs": len(train_pairs),
        "num_test_pairs": len(test_pairs),
        "num_units": len(units),
        "models": {
            "baseline": run_for_model(
                checkpoint_path=args.baseline_checkpoint,
                vocab=vocab,
                hidden=args.hidden,
                n_layers=args.n_layers,
                attn_heads=args.attn_heads,
                seq_len=args.seq_len,
                batch_size=args.batch_size,
                device=device,
                train_pairs=train_pairs,
                test_pairs=test_pairs,
                units=units,
            ),
            "address_aware": run_for_model(
                checkpoint_path=args.aa_checkpoint,
                vocab=vocab,
                hidden=args.hidden,
                n_layers=args.n_layers,
                attn_heads=args.attn_heads,
                seq_len=args.seq_len,
                batch_size=args.batch_size,
                device=device,
                train_pairs=train_pairs,
                test_pairs=test_pairs,
                units=units,
            ),
        },
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[SyntheticDaddr] Wrote report to {args.output}")


if __name__ == "__main__":
    main()
