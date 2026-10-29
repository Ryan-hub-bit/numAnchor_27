import argparse
import os
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from paths import ensure_parent_under_data_root, ensure_under_data_root
from vocab import WordVocab


def main():
    parser = argparse.ArgumentParser(description="Export all tokens from a vocab pickle to a text file")
    parser.add_argument("--vocab", required=True, help="Path to vocab .pkl file")
    parser.add_argument("--output", required=True, help="Path to output text file")
    args = parser.parse_args()

    args.vocab = ensure_under_data_root(args.vocab, "--vocab", must_exist=True)
    args.output = ensure_parent_under_data_root(args.output, "--output")

    vocab = WordVocab.load_vocab(args.vocab)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        for idx, tok in enumerate(vocab.itos):
            f.write(f"{idx}\t{tok}\n")

    print(f"vocab_path={args.vocab}")
    print(f"token_count={len(vocab)}")
    print(f"output_path={args.output}")


if __name__ == "__main__":
    main()
