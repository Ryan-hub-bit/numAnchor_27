#!/usr/bin/env python3
"""
Wrap the current address-aware funcsim blocks into the JSON format expected by
AAE's finetune/eval pipeline.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


VARIANT_RE = re.compile(
    r"^(?P<binary>.+)::(?P<opt>O0|O1|O2|O3|Os|Og|Ofast)::(?P<func>.+)$"
)


def normalize_entry(variant_id: str, raw_value):
    match = VARIANT_RE.match(variant_id)
    binary_name = match.group("binary") if match else ""
    opt_level = match.group("opt") if match else ""
    function_name = match.group("func") if match else variant_id

    if isinstance(raw_value, dict):
        entry = dict(raw_value)
        if "instructions" in entry and isinstance(entry["instructions"], list):
            entry["instructions"] = "\t".join(entry["instructions"])
        elif "instructions" not in entry and "tokens" in entry and isinstance(entry["tokens"], list):
            entry["instructions"] = "\t".join(entry["tokens"])
        if "num_instructions" not in entry:
            if "instructions" in entry and isinstance(entry["instructions"], str) and entry["instructions"]:
                entry["num_instructions"] = entry["instructions"].count("\t") + 1
        entry.setdefault("binary_name", binary_name)
        entry.setdefault("function_name", function_name)
        entry.setdefault("opt", opt_level)
        return entry

    if isinstance(raw_value, list):
        instructions = "\t".join(raw_value)
        num_instructions = len(raw_value)
    else:
        instructions = str(raw_value)
        num_instructions = instructions.count("\t") + 1 if instructions else 0

    return {
        "binary_name": binary_name,
        "function_name": function_name,
        "opt": opt_level,
        "instructions": instructions,
        "num_instructions": num_instructions,
    }


def main():
    parser = argparse.ArgumentParser(description="Wrap AA func_blocks into AAE format")
    parser.add_argument("--input", required=True, help="Input funcsim_function_blocks.json path")
    parser.add_argument("--output", required=True, help="Output AAE-style func_blocks.json path")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as fh:
        func_blocks = json.load(fh)

    wrapped = {
        variant_id: normalize_entry(variant_id, raw_value)
        for variant_id, raw_value in func_blocks.items()
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(wrapped, fh)

    print(f"Wrote {len(wrapped)} wrapped address-aware blocks to {output_path}")


if __name__ == "__main__":
    main()
