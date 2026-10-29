#!/usr/bin/env python3
"""Create paper-style intrinsic evaluation assets for PalmTree.

This script reads the latest hierarchy and control-flow probe JSON reports and
produces:

1. A grouped-bar figure for hierarchy accuracy.
2. A markdown table for control-flow AUC.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


HIERARCHY_LABELS = {
    "instruction_in_binary": "Inst. in Binary",
    "instruction_in_function": "Inst. in Function",
    "instruction_in_block": "Inst. in Block",
}

CONTROLFLOW_LABELS = {
    "jump_target": "Jump-target recovery",
    "call_target": "Call-target recovery",
}


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_hierarchy_accuracy(report: Dict) -> List[Tuple[str, float, float]]:
    rows: List[Tuple[str, float, float]] = []
    for key in (
        "instruction_in_binary",
        "instruction_in_function",
        "instruction_in_block",
    ):
        task = report["tasks"][key]
        baseline = task["models"]["baseline"]["probe_metrics"]["accuracy"]
        aa = task["models"]["address_aware"]["probe_metrics"]["accuracy"]
        rows.append((HIERARCHY_LABELS[key], float(baseline), float(aa)))
    return rows


def extract_controlflow_auc(report: Dict) -> List[Tuple[str, float, float, int]]:
    rows: List[Tuple[str, float, float, int]] = []
    for key in ("jump_target", "call_target"):
        task = report["tasks"][key]
        baseline = task["models"]["baseline"]["zero_shot"]["auc"]
        aa = task["models"]["address_aware"]["zero_shot"]["auc"]
        rows.append((CONTROLFLOW_LABELS[key], float(baseline), float(aa), int(task["num_test_pairs"])))
    return rows


def plot_hierarchy_accuracy(rows: List[Tuple[str, float, float]], output_base: str) -> None:
    labels = [row[0] for row in rows]
    baseline_vals = np.array([row[1] for row in rows], dtype=float)
    aa_vals = np.array([row[2] for row in rows], dtype=float)

    x = np.arange(len(labels))
    width = 0.34

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.titlesize": 16,
            "axes.labelsize": 12,
            "legend.fontsize": 11,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
        }
    )

    fig, ax = plt.subplots(figsize=(8.6, 4.6), constrained_layout=True)
    baseline_color = "#9AA4B2"
    aa_color = "#D55E00"

    bars1 = ax.bar(x - width / 2, baseline_vals, width, label="Baseline", color=baseline_color, edgecolor="black", linewidth=0.8)
    bars2 = ax.bar(x + width / 2, aa_vals, width, label="Address-aware", color=aa_color, edgecolor="black", linewidth=0.8)

    ax.axhline(0.25, color="black", linestyle="--", linewidth=1.0, alpha=0.55, label="Chance")
    ax.set_ylim(0.2, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_title("Hierarchical Structure Verification")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=3, loc="upper left")

    for bars in (bars1, bars2):
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.015,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    fig.savefig(output_base + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(output_base + ".pdf", bbox_inches="tight")
    plt.close(fig)


def write_controlflow_table(rows: List[Tuple[str, float, float, int]], path: str) -> None:
    lines = [
        "# Control-Flow Recovery Table",
        "",
        "| Task | Metric | Baseline | Address-aware | Delta | Test Size |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for label, baseline, aa, test_size in rows:
        lines.append(
            f"| {label} | Zero-shot AUC | {baseline:.3f} | {aa:.3f} | {aa - baseline:+.3f} | {test_size} |"
        )
    lines.extend(
        [
            "",
            "Suggested table note:",
            "",
            "`Control-flow recovery is reported as zero-shot AUC on frozen instruction embeddings.`",
        ]
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot paper-style PalmTree intrinsic results")
    parser.add_argument("--hierarchy_json", required=True)
    parser.add_argument("--controlflow_json", required=True)
    parser.add_argument("--figure_output_base", required=True)
    parser.add_argument("--auc_table_output", required=True)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.figure_output_base)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.auc_table_output)), exist_ok=True)

    hierarchy_report = load_json(args.hierarchy_json)
    controlflow_report = load_json(args.controlflow_json)

    hierarchy_rows = extract_hierarchy_accuracy(hierarchy_report)
    controlflow_rows = extract_controlflow_auc(controlflow_report)

    plot_hierarchy_accuracy(hierarchy_rows, args.figure_output_base)
    write_controlflow_table(controlflow_rows, args.auc_table_output)


if __name__ == "__main__":
    main()
