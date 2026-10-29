#!/usr/bin/env python3
"""
Export an address-aware WordVocab pickle into a HuggingFace-style tokenizer dir.
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
from pathlib import Path


def load_word_vocab_module(vocab_py: Path):
    spec = importlib.util.spec_from_file_location("aae_address_vocab", vocab_py)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load vocab module from {vocab_py}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description="Write HF tokenizer dir from WordVocab pickle")
    parser.add_argument("--vocab_pkl", required=True, help="Path to address-aware vocab pickle")
    parser.add_argument("--output_dir", required=True, help="Output tokenizer directory")
    parser.add_argument(
        "--template_dir",
        default=str(Path(__file__).resolve().parent),
        help="Template directory containing tokenizer_config.json",
    )
    args = parser.parse_args()

    template_dir = Path(args.template_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vocab_module = load_word_vocab_module(template_dir / "vocab.py")
    vocab = vocab_module.WordVocab.load_vocab(args.vocab_pkl)

    with open(output_dir / "vocab.txt", "w", encoding="utf-8") as fh:
        for token in vocab.itos:
            fh.write(f"{token}\n")

    for filename in ["tokenizer_config.json", "special_tokens_map.json"]:
        src = template_dir / filename
        if src.exists():
            shutil.copy2(src, output_dir / filename)

    print(f"Wrote tokenizer dir with {len(vocab)} tokens to {output_dir}")


if __name__ == "__main__":
    main()
