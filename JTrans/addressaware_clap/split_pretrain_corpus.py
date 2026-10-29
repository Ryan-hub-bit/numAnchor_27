import argparse
import json
import os
import random
import re

from paths import ensure_parent_under_data_root, ensure_under_data_root


INSTR_ADDR_RE = re.compile(r"\b\w+\((0x[0-9a-fA-F]+):")
TARGET_ADDR_RE = re.compile(r"\baddress\((0x[0-9a-fA-F]+):")


def has_valid_jtp_candidate(line: str, seq_len: int) -> bool:
    instructions = [inst.strip() for inst in line.split("\t") if inst.strip()]
    token_budget = 1  # <sos>
    opcode_addrs = set()
    target_addrs = []

    for inst in instructions:
        match = INSTR_ADDR_RE.match(inst)
        if match is not None:
            opcode_addrs.add(match.group(1).lower())
        token_budget += len(inst.split())
        if token_budget >= seq_len:
            break
        target_addrs.extend(addr.lower() for addr in TARGET_ADDR_RE.findall(inst))

    return any(addr in opcode_addrs for addr in target_addrs)


def main():
    parser = argparse.ArgumentParser(description="Split address-aware pretrain corpus with JTP-aware validation coverage")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--train_output", required=True)
    parser.add_argument("--val_output", required=True)
    parser.add_argument("--stats_output", default=None)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--val_size", type=int, default=5000)
    parser.add_argument("--min_val_jtp", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.corpus = ensure_under_data_root(args.corpus, "--corpus", must_exist=True)
    args.train_output = ensure_parent_under_data_root(args.train_output, "--train_output")
    args.val_output = ensure_parent_under_data_root(args.val_output, "--val_output")
    if args.stats_output:
        args.stats_output = ensure_parent_under_data_root(args.stats_output, "--stats_output")

    with open(args.corpus, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f if line.strip()]

    rng = random.Random(args.seed)
    indexed = list(enumerate(lines))
    rng.shuffle(indexed)

    positive = []
    negative = []
    for idx, line in indexed:
        if has_valid_jtp_candidate(line, args.seq_len):
            positive.append((idx, line))
        else:
            negative.append((idx, line))

    val_size = min(args.val_size, len(indexed))
    target_positive = min(len(positive), int(round(val_size * args.min_val_jtp)))
    val_rows = positive[:target_positive]

    remaining = positive[target_positive:] + negative
    rng.shuffle(remaining)
    val_rows.extend(remaining[: max(0, val_size - len(val_rows))])
    selected_ids = {idx for idx, _line in val_rows}
    train_rows = [(idx, line) for idx, line in indexed if idx not in selected_ids]

    os.makedirs(os.path.dirname(args.train_output), exist_ok=True)
    with open(args.train_output, "w", encoding="utf-8") as f:
        for _idx, line in train_rows:
            f.write(line + "\n")
    with open(args.val_output, "w", encoding="utf-8") as f:
        for _idx, line in val_rows:
            f.write(line + "\n")

    val_jtp = sum(1 for _idx, line in val_rows if has_valid_jtp_candidate(line, args.seq_len))
    train_jtp = sum(1 for _idx, line in train_rows if has_valid_jtp_candidate(line, args.seq_len))
    stats = {
        "corpus": args.corpus,
        "seq_len": args.seq_len,
        "seed": args.seed,
        "total_functions": len(indexed),
        "train_functions": len(train_rows),
        "val_functions": len(val_rows),
        "train_jtp_functions": train_jtp,
        "val_jtp_functions": val_jtp,
        "train_jtp_ratio": train_jtp / len(train_rows) if train_rows else 0.0,
        "val_jtp_ratio": val_jtp / len(val_rows) if val_rows else 0.0,
    }

    if args.stats_output:
        os.makedirs(os.path.dirname(args.stats_output), exist_ok=True)
        with open(args.stats_output, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)

    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
