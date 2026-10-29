#!/usr/bin/env python3
"""Validate the staged numAnchor artifact manifest and public path hygiene."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


DEFAULT_ROOT = Path(os.environ.get("NUMANCHOR_ARTIFACT_ROOT", "numAnchor_artifact/staging"))
TEXT_SUFFIXES = {".json", ".jsonl", ".log", ".md", ".txt", ".sh", ".py", ".yaml", ".yml", ".cfg"}
LEGACY_MISSPELLING = "addres" + "aware"
PATH_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"/home/[^/\s]+",
        r"/data/[^/\s]+",
        r"/path/to/",
        r"addressaware",
        re.escape(LEGACY_MISSPELLING),
        r"epoch",
        r"\bep[0-9]\b",
    )
]
TEXT_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (r"/home/[^/\s]+", r"/data/[^/\s]+", r"/path/to/", re.escape(LEGACY_MISSPELLING))
]


def load_manifest(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def summarize_path(path: Path) -> Tuple[int, int]:
    if path.is_file():
        return 1, path.stat().st_size
    count = 0
    size = 0
    for item in path.rglob("*"):
        if item.is_file():
            count += 1
            size += item.stat().st_size
    return count, size


def iter_public_files(root: Path) -> Iterable[Path]:
    for item in root.rglob("*"):
        if item.is_file():
            yield item


def scan_paths(root: Path, manifest: Dict[str, Any]) -> List[str]:
    issues: List[str] = []
    manifest_text = json.dumps(manifest, sort_keys=True)
    for pattern in PATH_PATTERNS:
        if pattern.search(manifest_text):
            issues.append(f"manifest contains forbidden token pattern: {pattern.pattern}")
    for item in root.rglob("*"):
        rel = item.relative_to(root).as_posix()
        for pattern in PATH_PATTERNS:
            if pattern.search(rel):
                issues.append(f"path contains forbidden token pattern {pattern.pattern}: {rel}")
    return issues


def scan_text(root: Path, max_bytes: int, strict_name_scan: bool) -> List[str]:
    patterns = list(TEXT_PATTERNS)
    if strict_name_scan:
        patterns.append(re.compile(r"addressaware", re.IGNORECASE))
    issues: List[str] = []
    for item in iter_public_files(root):
        if item.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if item.stat().st_size > max_bytes:
            continue
        try:
            text = item.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        rel = item.relative_to(root).as_posix()
        for pattern in patterns:
            if pattern.search(text):
                issues.append(f"text contains forbidden token pattern {pattern.pattern}: {rel}")
    return issues


def check_entries(root: Path, manifest: Dict[str, Any]) -> List[str]:
    issues: List[str] = []
    for entry in manifest.get("entries", []):
        if entry.get("status") == "missing_optional":
            continue
        target = entry.get("target")
        if not target:
            issues.append(f"manifest entry missing target: {entry}")
            continue
        path = root / target
        if not path.exists():
            issues.append(f"missing target: {target}")
            continue
        count, size = summarize_path(path)
        if entry.get("file_count") != count:
            issues.append(f"file count mismatch for {target}: manifest={entry.get('file_count')} actual={count}")
        if entry.get("bytes") != size:
            issues.append(f"byte size mismatch for {target}: manifest={entry.get('bytes')} actual={size}")
        if path.is_symlink():
            issues.append(f"target is a symlink: {target}")
    return issues


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--scan-text", action="store_true")
    parser.add_argument("--strict-name-scan", action="store_true")
    parser.add_argument("--max-text-bytes", type=int, default=2_000_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest or args.artifact_root / "artifact_manifest.json"
    manifest = load_manifest(manifest_path)
    issues = []
    issues.extend(check_entries(args.artifact_root, manifest))
    issues.extend(scan_paths(args.artifact_root, manifest))
    if args.scan_text:
        issues.extend(scan_text(args.artifact_root, args.max_text_bytes, args.strict_name_scan))

    if issues:
        print("manifest check failed:")
        for issue in issues:
            print(f"- {issue}")
        raise SystemExit(1)

    entries = manifest.get("entries", [])
    total_bytes = sum(entry.get("bytes", 0) for entry in entries)
    total_files = sum(entry.get("file_count", 0) for entry in entries)
    print("manifest check passed")
    print(f"entries={len(entries)} files={total_files} bytes={total_bytes}")


if __name__ == "__main__":
    main()
