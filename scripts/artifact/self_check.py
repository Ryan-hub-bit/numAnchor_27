#!/usr/bin/env python3
"""Run structural self-checks for the staged numAnchor artifact."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


DEFAULT_ROOT = Path(os.environ.get("NUMANCHOR_ARTIFACT_ROOT", "numAnchor_artifact/staging"))
DEFAULT_WORK = Path(os.environ.get("NUMANCHOR_WORK_DIR", "numAnchor_artifact/work/results_tmp"))
SCRIPT_DIR = Path(__file__).resolve().parent


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_cmd(cmd: List[str]) -> Dict[str, object]:
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return {"cmd": cmd, "returncode": proc.returncode, "output": proc.stdout}


def load_manifest(root: Path) -> dict:
    with (root / "artifact_manifest.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def model_matrix(manifest: dict) -> Dict[str, List[str]]:
    matrix: Dict[str, List[str]] = {}
    for entry in manifest.get("entries", []):
        if entry.get("kind") != "model" or entry.get("status") == "missing_optional":
            continue
        matrix.setdefault(entry.get("component", "unknown"), []).append(entry["target"])
    return matrix


def generate_sample_pool(root: Path, work: Path, name: str, ground_truth: str, pool_size: int, query_count: int) -> Dict[str, object]:
    output = work / "pools" / f"{name}_pool.json"
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "generate_pools.py"),
        "--ground-truth",
        str(root / ground_truth),
        "--output",
        str(output),
        "--pool-size",
        str(pool_size),
        "--query-count",
        str(query_count),
        "--seed",
        "42",
    ]
    result = run_cmd(cmd)
    if output.exists():
        with output.open("r", encoding="utf-8") as handle:
            pool = json.load(handle)
        result["pool_file"] = str(output)
        result["query_count"] = len(pool.get("query_ids", []))
        result["pool_size"] = len(pool.get("pool_function_ids", []))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--cleanup-pools", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)

    checks: Dict[str, object] = {"generated_at_utc": utc_now(), "artifact_root": str(args.artifact_root)}

    manifest_check = run_cmd(
        [
            sys.executable,
            str(SCRIPT_DIR / "check_manifest.py"),
            "--artifact-root",
            str(args.artifact_root),
            "--scan-text",
            "--strict-name-scan",
        ]
    )
    checks["manifest_check"] = manifest_check

    manifest = load_manifest(args.artifact_root)
    checks["model_matrix"] = model_matrix(manifest)

    pool_specs = [
        ("shared_stateformer_x64", "data/shared_stateformer_x64/ground_truth.json", 128, 16),
        ("bert_function_x64", "data/bert_function_x64/ground_truth.json", 128, 16),
        ("jtrans_full_test", "data/jtrans_full/numanchor/test/ground_truth.json", 128, 16),
    ]
    pool_results = []
    for name, ground_truth, pool_size, query_count in pool_specs:
        if (args.artifact_root / ground_truth).exists():
            pool_results.append(generate_sample_pool(args.artifact_root, args.work_dir, name, ground_truth, pool_size, query_count))
    checks["pool_generation"] = pool_results

    checks["passed"] = manifest_check["returncode"] == 0 and all(item["returncode"] == 0 for item in pool_results)

    report_path = args.work_dir / "self_check_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(checks, handle, indent=2, sort_keys=True)
        handle.write("\n")

    if args.cleanup_pools:
        shutil.rmtree(args.work_dir / "pools", ignore_errors=True)

    print(json.dumps({"passed": checks["passed"], "report": str(report_path)}, indent=2))
    if not checks["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
