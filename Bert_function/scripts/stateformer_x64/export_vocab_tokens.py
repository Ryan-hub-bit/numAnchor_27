#!/usr/bin/env python3
"""
Export a PalmTree pickle vocabulary to text files.

Outputs:
1. vocab_tokens.txt      -> "index<TAB>token"
2. vocab_tokens_only.txt -> plain token per line
"""

import argparse
import os
import sys
import types


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab", required=True, help="Path to vocab pickle")
    parser.add_argument(
        "--output",
        required=True,
        help="Path to indexed token output (index<TAB>token)",
    )
    parser.add_argument(
        "--tokens-only-output",
        required=True,
        help="Path to plain token-per-line output",
    )
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    # vocab.py imports tqdm at module import time, but exporting tokens only
    # needs the class definitions for pickle loading. Provide a tiny fallback
    # so this script works even outside the training conda env.
    if "tqdm" not in sys.modules:
        try:
            import tqdm  # noqa: F401  # pylint: disable=unused-import
        except ModuleNotFoundError:
            sys.modules["tqdm"] = types.SimpleNamespace(tqdm=lambda iterable: iterable)

    from vocab import WordVocab  # pylint: disable=import-error

    vocab = WordVocab.load_vocab(args.vocab)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    os.makedirs(os.path.dirname(args.tokens_only_output), exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as fout:
        for idx, token in enumerate(vocab.itos):
            fout.write(f"{idx}\t{token}\n")

    with open(args.tokens_only_output, "w", encoding="utf-8") as fout:
        for token in vocab.itos:
            fout.write(f"{token}\n")

    print(f"Exported {len(vocab.itos)} tokens from {args.vocab}")
    print(f"Indexed tokens: {args.output}")
    print(f"Plain tokens:   {args.tokens_only_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
