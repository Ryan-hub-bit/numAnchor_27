#!/usr/bin/env python3
"""
Group canonical variant IDs into AAE-compatible ground_truth JSON.

Expected key format:
    <binary_name>::<opt_level>::<function_name>
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


VARIANT_RE = re.compile(
    r"^(?P<binary>.+)::(?P<opt>O0|O1|O2|O3|Os|Og|Ofast)::(?P<func>.+)$"
)


def main():
    parser = argparse.ArgumentParser(description="Build grouped ground truth from canonical variant IDs")
    parser.add_argument("--func_blocks", required=True, help="Input func_blocks JSON path")
    parser.add_argument("--output", required=True, help="Output ground_truth.json path")
    args = parser.parse_args()

    with open(args.func_blocks, "r", encoding="utf-8") as fh:
        func_blocks = json.load(fh)

    grouped = defaultdict(dict)
    for variant_id in func_blocks:
        match = VARIANT_RE.match(variant_id)
        if not match:
            continue
        binary_name = match.group("binary")
        function_name = match.group("func")
        opt_level = match.group("opt")
        grouped[(binary_name, function_name)][opt_level] = variant_id

    pairs = []
    opt_coverage = defaultdict(int)
    for (binary_name, function_name), opt_map in sorted(grouped.items()):
        if len(opt_map) < 2:
            continue
        pair = {
            "binary_name": binary_name,
            "function_name": function_name,
        }
        pair.update(opt_map)
        pairs.append(pair)
        for opt_level in opt_map:
            opt_coverage[opt_level] += 1

    output = {
        "pairs": pairs,
        "total_pairs": len(pairs),
        "statistics": {
            "optimization_coverage": dict(sorted(opt_coverage.items())),
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2)

    print(f"Grouped {len(pairs)} function families into {output_path}")


if __name__ == "__main__":
    main()
