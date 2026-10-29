#!/usr/bin/env python3
"""
Normalize AAE pool-evaluation JSON into a compact paper-style summary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


OPT_MAP = {
    "O0_vs_O3": "O0-O3",
    "O0_vs_Os": "O0-Os",
    "O1_vs_O3": "O1-O3",
    "O1_vs_Os": "O1-Os",
    "O2_vs_O3": "O2-O3",
    "O2_vs_Os": "O2-Os",
}


def normalize_opt_pair(raw: str) -> str:
    for key, value in OPT_MAP.items():
        if key in raw:
            return value
    return raw.replace("_vs_", "-").replace("_filtered", "")


def main():
    parser = argparse.ArgumentParser(description="Summarize AAE pool results")
    parser.add_argument("--input", required=True, help="Input evaluation_results.json path")
    parser.add_argument("--output", required=True, help="Output summary JSON path")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as fh:
        raw = json.load(fh)

    if isinstance(raw, dict) and "combos" in raw:
        summary = {"results": {}, "average": {}}
        for pool_key, opt_results in raw["combos"].items():
            summary["results"][pool_key] = {}
            for opt_pair, metrics in opt_results.items():
                summary["results"][pool_key][normalize_opt_pair(opt_pair)] = {
                    "MRR": metrics["mrr"],
                    "Recall@1": metrics["recall@1"],
                    "Recall@5": metrics.get("recall@5"),
                    "Recall@10": metrics.get("recall@10"),
                }
    else:
        rows = raw
        summary = {"results": {}, "average": {}}
        for row in rows:
            pool_size = row.get("pool_size")
            opt_pair = normalize_opt_pair(row.get("opt_pair", row.get("pool_file", "unknown")))
            pool_key = f"pool_{pool_size}"
            summary["results"].setdefault(pool_key, {})
            summary["results"][pool_key][opt_pair] = {
                "MRR": row["MRR"],
                "Recall@1": row["Recall@1"],
                "Recall@5": row["Recall@5"],
                "Recall@10": row["Recall@10"],
            }

    for pool_key, opt_results in summary["results"].items():
        if not opt_results:
            continue
        keys = ["MRR", "Recall@1", "Recall@5", "Recall@10"]
        averages = {}
        for key in keys:
            values = [item[key] for item in opt_results.values() if item.get(key) is not None]
            averages[key] = (sum(values) / len(values)) if values else None
        summary["average"][pool_key] = averages

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(f"Wrote summary to {output_path}")


if __name__ == "__main__":
    main()
