#!/usr/bin/env python3
"""
Hard data-xref identity probing from section-aware CFG corpora.

This evaluation is intentionally stricter than same_dref:
  - only uses single-instruction mov-based data accesses
  - positives and negatives must share the same instruction template
  - positives additionally require different next-instruction context

Tasks:
  - same_load_xref_hard
  - same_store_xref_hard
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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


HEADER_PREFIX_RE = re.compile(r"^(\w+)\([^)]*\)")
ADDRESS_RE = re.compile(r"address\([^)]*\)")
VAR_RE = re.compile(r"var\([^)]*\)")


@dataclass
class XrefOccurrence:
    unit_id: str
    inst_text: str
    next_text: str
    mode: str
    opcode: str
    daddr: str
    template: str
    next_template: str


def canonicalize_instruction(inst_text: str) -> str:
    inst = HEADER_PREFIX_RE.sub(lambda m: m.group(1), inst_text, count=1)
    inst = ADDRESS_RE.sub("address", inst)

    def repl_daddr(match: re.Match[str]) -> str:
        return match.group(1)

    inst = DADDR_RE.sub(repl_daddr, inst)
    inst = VAR_RE.sub("var", inst)
    return inst.strip()


def parse_mov_daddr_occurrence(inst_text: str, next_text: str, prefix: str) -> Optional[XrefOccurrence]:
    header_match = HEADER_PREFIX_RE.match(inst_text)
    if not header_match:
        return None
    opcode = header_match.group(1)
    if opcode != "mov":
        return None

    remainder = inst_text[header_match.end():].strip()
    parts = remainder.split()
    if len(parts) != 2:
        return None

    daddr_match = DADDR_RE.search(inst_text)
    if not daddr_match:
        return None
    daddr = daddr_match.group(2)

    left, right = parts
    if left.startswith("daddr_") and not right.startswith("daddr_"):
        mode = "store"
    elif right.startswith("daddr_") and not left.startswith("daddr_"):
        mode = "load"
    else:
        return None

    template = canonicalize_instruction(inst_text)
    next_template = canonicalize_instruction(next_text)
    unit_id = f"{prefix}:{mode}:{daddr}:{abs(hash((inst_text, next_text)))}"
    return XrefOccurrence(
        unit_id=unit_id,
        inst_text=inst_text,
        next_text=next_text,
        mode=mode,
        opcode=opcode,
        daddr=daddr,
        template=template,
        next_template=next_template,
    )


def load_occurrences_from_cfg(path: str, prefix: str) -> Dict[str, List[XrefOccurrence]]:
    by_mode: Dict[str, List[XrefOccurrence]] = {"load": [], "store": []}
    seen: set[Tuple[str, str]] = set()

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 2:
                continue
            inst_text, next_text = parts
            key = (inst_text, next_text)
            if key in seen:
                continue
            seen.add(key)

            occ = parse_mov_daddr_occurrence(inst_text, next_text, prefix)
            if occ is None:
                continue
            by_mode[occ.mode].append(occ)
    return by_mode


def sample_pair_candidates(
    occurrences: Sequence[XrefOccurrence],
    max_pairs: int,
    seed: int,
    task_name: str,
) -> Tuple[List[ProbePair], Dict[str, ProbeUnit]]:
    rng = random.Random(seed)
    by_template: Dict[str, List[XrefOccurrence]] = {}
    by_template_addr: Dict[Tuple[str, str], List[XrefOccurrence]] = {}

    for occ in occurrences:
        by_template.setdefault(occ.template, []).append(occ)
        by_template_addr.setdefault((occ.template, occ.daddr), []).append(occ)

    group_keys = list(by_template_addr.keys())
    rng.shuffle(group_keys)

    pairs: List[ProbePair] = []
    units: Dict[str, ProbeUnit] = {}

    for template, daddr in group_keys:
        group = by_template_addr[(template, daddr)]
        if len(group) < 2:
            continue
        candidate_pool = by_template.get(template, [])
        different_addr = [occ for occ in candidate_pool if occ.daddr != daddr]
        if not different_addr:
            continue

        unique_group: List[XrefOccurrence] = list({(occ.inst_text, occ.next_text): occ for occ in group}.values())
        rng.shuffle(unique_group)
        for i in range(len(unique_group)):
            for j in range(i + 1, len(unique_group)):
                left = unique_group[i]
                right = unique_group[j]
                if left.next_template == right.next_template:
                    continue

                negatives = [
                    occ
                    for occ in different_addr
                    if occ.next_template != left.next_template
                    and occ.unit_id not in {left.unit_id, right.unit_id}
                ]
                if not negatives:
                    negatives = [
                        occ for occ in different_addr
                        if occ.unit_id not in {left.unit_id, right.unit_id}
                    ]
                if not negatives:
                    continue
                negative = rng.choice(negatives)

                units[left.unit_id] = ProbeUnit(left.unit_id, "instruction", [left.inst_text])
                units[right.unit_id] = ProbeUnit(right.unit_id, "instruction", [right.inst_text])
                units[negative.unit_id] = ProbeUnit(negative.unit_id, "instruction", [negative.inst_text])

                pairs.append(
                    ProbePair(
                        left_id=left.unit_id,
                        right_id=right.unit_id,
                        label=1,
                        group_id=f"{task_name}:{template}:{daddr}",
                        metadata={
                            "task": task_name,
                            "template": template,
                            "data_addr": daddr,
                            "left_next": left.next_template,
                            "right_next": right.next_template,
                        },
                    )
                )
                pairs.append(
                    ProbePair(
                        left_id=left.unit_id,
                        right_id=negative.unit_id,
                        label=0,
                        group_id=f"{task_name}_neg:{template}:{left.unit_id}",
                        metadata={
                            "task": task_name,
                            "template": template,
                            "anchor_data_addr": daddr,
                            "negative_data_addr": negative.daddr,
                            "left_next": left.next_template,
                            "right_next": negative.next_template,
                        },
                    )
                )

                if len(pairs) >= max_pairs:
                    return pairs[:max_pairs], units

    return pairs[:max_pairs], units


def cosine_score(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())


def build_scores(pairs: Sequence[ProbePair], embeddings: Dict[str, object]) -> List[float]:
    return [cosine_score(embeddings[p.left_id].pooled, embeddings[p.right_id].pooled) for p in pairs]


def run_zero_shot(
    encoder: FrozenBackboneEncoder,
    train_pairs: Sequence[ProbePair],
    test_pairs: Sequence[ProbePair],
    units: Dict[str, ProbeUnit],
    batch_size: int,
) -> Dict[str, float]:
    embeddings = encoder.encode_units(units, batch_size=batch_size)
    train_scores = build_scores(train_pairs, embeddings)
    test_scores = build_scores(test_pairs, embeddings)
    train_labels = [pair.label for pair in train_pairs]
    test_labels = [pair.label for pair in test_pairs]
    threshold = choose_threshold(train_scores, train_labels)
    return compute_metrics(test_scores, test_labels, threshold)


def main():
    parser = argparse.ArgumentParser(description="Hard data-xref identity probing on CFG corpora")
    parser.add_argument("--train_cfg", required=True)
    parser.add_argument("--val_cfg", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--baseline_checkpoint", required=True)
    parser.add_argument("--aa_checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tasks", nargs="+", default=["same_load_xref_hard", "same_store_xref_hard"])
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
    device = torch.device(args.device)

    train_occ = load_occurrences_from_cfg(args.train_cfg, "train")
    val_occ = load_occurrences_from_cfg(args.val_cfg, "val")
    vocab = WordVocab.load_vocab(args.vocab)

    report: Dict[str, object] = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "config": vars(args),
        "tasks": {},
    }

    model_specs = [
        ("baseline", args.baseline_checkpoint),
        ("address_aware", args.aa_checkpoint),
    ]

    task_to_mode = {
        "same_load_xref_hard": "load",
        "same_store_xref_hard": "store",
    }

    for task in args.tasks:
        if task not in task_to_mode:
            raise ValueError(f"Unsupported task: {task}")
        mode = task_to_mode[task]
        train_pairs, train_units = sample_pair_candidates(
            train_occ[mode], args.max_train_pairs, args.seed + (11 if mode == "load" else 21), task
        )
        test_pairs, test_units = sample_pair_candidates(
            val_occ[mode], args.max_test_pairs, args.seed + (12 if mode == "load" else 22), task
        )

        if not train_pairs or not test_pairs:
            report["tasks"][task] = {
                "error": "Insufficient hard xref pairs after task construction.",
                "num_train_pairs": len(train_pairs),
                "num_test_pairs": len(test_pairs),
            }
            continue

        units: Dict[str, ProbeUnit] = {}
        units.update(train_units)
        units.update(test_units)

        task_report = {
            "num_train_pairs": len(train_pairs),
            "num_test_pairs": len(test_pairs),
            "num_units": len(units),
            "models": {},
        }

        for model_name, checkpoint_path in model_specs:
            encoder = FrozenBackboneEncoder(
                checkpoint_path=checkpoint_path,
                vocab=vocab,
                hidden=args.hidden,
                n_layers=args.n_layers,
                attn_heads=args.attn_heads,
                seq_len=args.seq_len,
                max_instructions=8,
                device=device,
            )
            task_report["models"][model_name] = {
                "model_name": model_name,
                "use_addr": encoder.use_addr,
                "zero_shot": run_zero_shot(
                    encoder=encoder,
                    train_pairs=train_pairs,
                    test_pairs=test_pairs,
                    units=units,
                    batch_size=args.batch_size,
                ),
            }

        report["tasks"][task] = task_report

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[HardXref] Wrote report to {args.output}")


if __name__ == "__main__":
    main()
