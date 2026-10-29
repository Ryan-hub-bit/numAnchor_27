#!/usr/bin/env python3
"""
Extract aligned baseline and address-aware function data into JSONL.

This script is designed for the very large paper-sized func_blocks JSON files.
It incrementally parses the top-level JSON object, writes temporary key/value
TSV shards, externally sorts them, and then performs a streaming merge-join.

Each output line contains one canonical variant id:

    <binary_name>::<opt_level>::<function_name>

with both baseline and address-aware payloads attached when available.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional, Tuple


def eprint(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def open_text(path: str, mode: str):
    if path.endswith(".gz"):
        return gzip.open(path, mode, encoding="utf-8")
    return open(path, mode, encoding="utf-8")


def compact_json(obj: Dict) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def _read_more(fp, buffer: str, chunk_size: int) -> Tuple[str, bool]:
    chunk = fp.read(chunk_size)
    if chunk == "":
        return buffer, True
    return buffer + chunk, False


def iter_top_level_object(path: str, chunk_size: int = 1024 * 1024) -> Iterator[Tuple[str, object]]:
    """
    Incrementally iterate key/value pairs from a top-level JSON object without
    loading the entire file into memory.
    """
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
            nonlocal buffer, pos
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


def normalize_baseline_entry(variant_id: str, raw_value) -> Dict:
    if not isinstance(raw_value, dict):
        return {
            "variant_id": variant_id,
            "binary_name": "",
            "function_name": variant_id,
            "opt": "",
            "tokens": str(raw_value),
            "num_instructions": 0,
        }

    return {
        "variant_id": variant_id,
        "binary_name": raw_value.get("binary_name", ""),
        "function_name": raw_value.get("function_name", ""),
        "opt": raw_value.get("opt", ""),
        "tokens": raw_value.get("tokens", ""),
        "num_instructions": raw_value.get("num_instructions", 0),
    }


def normalize_addressaware_entry(variant_id: str, raw_value) -> Dict:
    if isinstance(raw_value, dict):
        instructions = raw_value.get("instructions")
        if instructions is None and isinstance(raw_value.get("tokens"), list):
            instructions = "\t".join(raw_value["tokens"])
        elif instructions is None:
            instructions = raw_value.get("tokens", "")
        num_instructions = raw_value.get("num_instructions")
        if num_instructions is None and isinstance(instructions, str) and instructions:
            num_instructions = instructions.count("\t") + 1
        elif num_instructions is None:
            num_instructions = 0
        return {
            "variant_id": variant_id,
            "binary_name": raw_value.get("binary_name", ""),
            "function_name": raw_value.get("function_name", ""),
            "opt": raw_value.get("opt", ""),
            "instructions": instructions if isinstance(instructions, str) else str(instructions),
            "num_instructions": num_instructions,
        }

    if isinstance(raw_value, list):
        instructions = "\t".join(raw_value)
        num_instructions = len(raw_value)
    else:
        instructions = str(raw_value)
        num_instructions = instructions.count("\t") + 1 if instructions else 0

    return {
        "variant_id": variant_id,
        "binary_name": "",
        "function_name": variant_id,
        "opt": "",
        "instructions": instructions,
        "num_instructions": num_instructions,
    }


def write_tsv(
    input_path: str,
    output_path: str,
    normalizer,
    label: str,
    limit: Optional[int] = None,
) -> int:
    count = 0
    with open(output_path, "w", encoding="utf-8") as fout:
        for variant_id, raw_value in iter_top_level_object(input_path):
            normalized = normalizer(variant_id, raw_value)
            fout.write(variant_id)
            fout.write("\t")
            fout.write(compact_json(normalized))
            fout.write("\n")
            count += 1
            if count % 100000 == 0:
                eprint(f"[{label}] wrote {count} records")
            if limit is not None and count >= limit:
                break
    eprint(f"[{label}] finished with {count} records")
    return count


def sort_tsv(input_path: str, output_path: str, temp_dir: str) -> None:
    env = os.environ.copy()
    env["LC_ALL"] = "C"
    subprocess.run(
        [
            "sort",
            "-T",
            temp_dir,
            "-t",
            "\t",
            "-k1,1",
            input_path,
            "-o",
            output_path,
        ],
        check=True,
        env=env,
    )


def iter_sorted_tsv(path: str) -> Iterator[Tuple[str, Dict]]:
    with open(path, "r", encoding="utf-8") as fp:
        for line in fp:
            line = line.rstrip("\n")
            if not line:
                continue
            key, payload = line.split("\t", 1)
            yield key, json.loads(payload)


def _function_key(entry: Dict) -> str:
    binary_name = entry.get("binary_name", "")
    function_name = entry.get("function_name", "")
    if binary_name or function_name:
        return f"{binary_name}::{function_name}"
    return entry.get("variant_id", "")


def merge_join(
    baseline_sorted: str,
    addressaware_sorted: str,
    output_path: str,
    baseline_only_path: Optional[str] = None,
    addressaware_only_path: Optional[str] = None,
) -> Dict[str, int]:
    stats = {"common": 0, "baseline_only": 0, "addressaware_only": 0}
    b_iter = iter_sorted_tsv(baseline_sorted)
    a_iter = iter_sorted_tsv(addressaware_sorted)

    b_item = next(b_iter, None)
    a_item = next(a_iter, None)

    with open_text(output_path, "wt") as out_fp:
        b_only_fp = open_text(baseline_only_path, "wt") if baseline_only_path else None
        a_only_fp = open_text(addressaware_only_path, "wt") if addressaware_only_path else None
        try:
            while b_item is not None and a_item is not None:
                b_key, b_val = b_item
                a_key, a_val = a_item
                if b_key == a_key:
                    record = {
                        "variant_id": b_key,
                        "function_key": _function_key(b_val) or _function_key(a_val),
                        "binary_name": b_val.get("binary_name") or a_val.get("binary_name", ""),
                        "function_name": b_val.get("function_name") or a_val.get("function_name", ""),
                        "opt": b_val.get("opt") or a_val.get("opt", ""),
                        "baseline": {
                            "tokens": b_val.get("tokens", ""),
                            "num_instructions": b_val.get("num_instructions", 0),
                        },
                        "addressaware": {
                            "instructions": a_val.get("instructions", ""),
                            "num_instructions": a_val.get("num_instructions", 0),
                        },
                    }
                    out_fp.write(compact_json(record))
                    out_fp.write("\n")
                    stats["common"] += 1
                    if stats["common"] % 100000 == 0:
                        eprint(f"[merge] wrote {stats['common']} aligned records")
                    b_item = next(b_iter, None)
                    a_item = next(a_iter, None)
                elif b_key < a_key:
                    if b_only_fp is not None:
                        b_only_fp.write(compact_json(b_val))
                        b_only_fp.write("\n")
                    stats["baseline_only"] += 1
                    b_item = next(b_iter, None)
                else:
                    if a_only_fp is not None:
                        a_only_fp.write(compact_json(a_val))
                        a_only_fp.write("\n")
                    stats["addressaware_only"] += 1
                    a_item = next(a_iter, None)

            while b_item is not None:
                _b_key, b_val = b_item
                if b_only_fp is not None:
                    b_only_fp.write(compact_json(b_val))
                    b_only_fp.write("\n")
                stats["baseline_only"] += 1
                b_item = next(b_iter, None)

            while a_item is not None:
                _a_key, a_val = a_item
                if a_only_fp is not None:
                    a_only_fp.write(compact_json(a_val))
                    a_only_fp.write("\n")
                stats["addressaware_only"] += 1
                a_item = next(a_iter, None)
        finally:
            if b_only_fp is not None:
                b_only_fp.close()
            if a_only_fp is not None:
                a_only_fp.close()

    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract aligned baseline and address-aware function data")
    parser.add_argument("--baseline", required=True, help="Path to func_blocks_baseline.json")
    parser.add_argument("--addressaware", required=True, help="Path to func_blocks_addr.json or funcsim_function_blocks.json")
    parser.add_argument("--output", required=True, help="Output JSONL or JSONL.GZ path for common keys")
    parser.add_argument("--baseline_only", help="Optional JSONL or JSONL.GZ path for baseline-only keys")
    parser.add_argument("--addressaware_only", help="Optional JSONL or JSONL.GZ path for address-aware-only keys")
    parser.add_argument("--work_dir", help="Optional work directory for temporary TSV and sort files")
    parser.add_argument("--keep_temp", action="store_true", help="Keep intermediate TSV files")
    parser.add_argument("--limit", type=int, default=None, help="Only parse the first N records from each input")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.work_dir:
        work_dir = Path(args.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        cleanup = False
    else:
        work_dir = Path(tempfile.mkdtemp(prefix="extract_baseline_addressaware_"))
        cleanup = True

    eprint(f"[work] using {work_dir}")
    baseline_tsv = work_dir / "baseline.tsv"
    baseline_sorted = work_dir / "baseline.sorted.tsv"
    addressaware_tsv = work_dir / "addressaware.tsv"
    addressaware_sorted = work_dir / "addressaware.sorted.tsv"

    try:
        write_tsv(
            input_path=args.baseline,
            output_path=str(baseline_tsv),
            normalizer=normalize_baseline_entry,
            label="baseline",
            limit=args.limit,
        )
        sort_tsv(str(baseline_tsv), str(baseline_sorted), str(work_dir))
        eprint("[baseline] sort complete")

        write_tsv(
            input_path=args.addressaware,
            output_path=str(addressaware_tsv),
            normalizer=normalize_addressaware_entry,
            label="addressaware",
            limit=args.limit,
        )
        sort_tsv(str(addressaware_tsv), str(addressaware_sorted), str(work_dir))
        eprint("[addressaware] sort complete")

        stats = merge_join(
            baseline_sorted=str(baseline_sorted),
            addressaware_sorted=str(addressaware_sorted),
            output_path=args.output,
            baseline_only_path=args.baseline_only,
            addressaware_only_path=args.addressaware_only,
        )
        eprint(f"[done] {json.dumps(stats, ensure_ascii=False)}")
    finally:
        if cleanup and not args.keep_temp:
            shutil.rmtree(work_dir, ignore_errors=True)
        elif args.keep_temp:
            eprint(f"[work] kept temp files under {work_dir}")


if __name__ == "__main__":
    main()
