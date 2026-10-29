#!/usr/bin/env python3

import argparse
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Split combined PalmTree pretrain corpus into train/val")
    parser.add_argument("--input", required=True)
    parser.add_argument("--train_output", required=True)
    parser.add_argument("--val_output", required=True)
    parser.add_argument("--val_ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Missing input corpus: {input_path}")

    lines = [line.rstrip("\n") for line in input_path.open("r", encoding="utf-8") if line.strip()]
    rng = random.Random(args.seed)
    rng.shuffle(lines)

    val_size = max(1, int(len(lines) * args.val_ratio))
    val_lines = lines[:val_size]
    train_lines = lines[val_size:]

    train_path = Path(args.train_output)
    val_path = Path(args.val_output)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    val_path.parent.mkdir(parents=True, exist_ok=True)

    train_path.write_text("\n".join(train_lines) + "\n", encoding="utf-8")
    val_path.write_text("\n".join(val_lines) + "\n", encoding="utf-8")

    print(f"Input lines : {len(lines)}")
    print(f"Train lines : {len(train_lines)}")
    print(f"Val lines   : {len(val_lines)}")
    print(f"Train file  : {train_path}")
    print(f"Val file    : {val_path}")


if __name__ == "__main__":
    main()
