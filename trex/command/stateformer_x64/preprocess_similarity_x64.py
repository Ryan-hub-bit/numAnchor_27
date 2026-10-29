#!/usr/bin/env python3

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from command.stateformer_x64.prepare_stateformer_trex import FIELD_ORDER


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Binarize Trex similarity_x64 pair data")
    parser.add_argument("--text-dir", type=Path, required=True)
    parser.add_argument("--src-dict-root", type=Path, required=True)
    parser.add_argument("--dest-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def preprocess_one(args: argparse.Namespace, split_field: str, input_name: str) -> None:
    destdir = args.dest_dir / input_name / split_field
    destdir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "fairseq_cli.preprocess",
            "--only-source",
            "--srcdict",
            str(args.src_dict_root / split_field / "dict.txt"),
            "--trainpref",
            str(args.text_dir / f"train.{split_field}.{input_name}"),
            "--validpref",
            str(args.text_dir / f"valid.{split_field}.{input_name}"),
            "--destdir",
            str(destdir),
            "--workers",
            str(args.workers),
        ],
        check=True,
    )


def main() -> None:
    args = parse_args()
    args.dest_dir.mkdir(parents=True, exist_ok=True)

    for input_name in ("input0", "input1"):
        for field in FIELD_ORDER:
            preprocess_one(args, field, input_name)

    label_dir = args.dest_dir / "label"
    label_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.text_dir / "train.label", label_dir / "train.label")
    shutil.copy2(args.text_dir / "valid.label", label_dir / "valid.label")


if __name__ == "__main__":
    main()
