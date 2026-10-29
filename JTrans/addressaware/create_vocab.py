import argparse
import os
import re
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from paths import ensure_parent_under_data_root, ensure_under_data_root
from vocab import WordVocab


ADDRESS_SUFFIX_PATTERN = re.compile(r"\(0x[0-9a-fA-F]+:[0-9.]+:[0-9.]+:[0-9.]+\)")
VAR_SUFFIX_PATTERN = re.compile(r"\(0x[0-9a-fA-F]+\)")
HEX_TOKEN_PATTERN = re.compile(r"^0x[0-9a-fA-F]+$")


def preprocess_line(line: str):
    """
    Strip address payloads so vocab matches dataset.py tokenization.

    Examples:
        mov(0x401000:0.5:0.3:0.2) eax ebx -> mov eax ebx
        address(0x123:0.5:0.3:0.2) -> address
        var(0x10028) -> var
        lea rax 0x12345 -> lea rax
    """
    cleaned = ADDRESS_SUFFIX_PATTERN.sub("", line)
    cleaned = VAR_SUFFIX_PATTERN.sub("", cleaned)

    tokens = []
    for tok in cleaned.replace("\t", " ").split():
        if tok and not HEX_TOKEN_PATTERN.match(tok):
            tokens.append(tok)
    return tokens


class PreprocessedFile:
    def __init__(self, file_handle):
        self.file_handle = file_handle

    def __iter__(self):
        for line in self.file_handle:
            yield preprocess_line(line)


def main():
    parser = argparse.ArgumentParser(description="Create vocab for address-aware JTrans")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_size", type=int, default=20000)
    parser.add_argument("--min_freq", type=int, default=1)
    args = parser.parse_args()

    args.corpus = ensure_under_data_root(args.corpus, "--corpus", must_exist=True)
    args.output = ensure_parent_under_data_root(args.output, "--output")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.corpus, "r", encoding="utf-8") as f:
        vocab = WordVocab([PreprocessedFile(f)], max_size=args.max_size, min_freq=args.min_freq)
    vocab.save_vocab(args.output)
    print(f"Saved vocab with {len(vocab)} tokens to {args.output}")


if __name__ == "__main__":
    main()
