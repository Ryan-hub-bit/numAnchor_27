#!/usr/bin/env python3
"""Stage the private numAnchor artifact payload into a public directory tree."""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


DEFAULT_SELECTION = Path(os.environ.get("NUMANCHOR_SOURCE_SELECTION", "source_selection.local.json"))
DEFAULT_DEST = Path(os.environ.get("NUMANCHOR_ARTIFACT_ROOT", "numAnchor_artifact/staging"))


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def safe_target(root: Path, target: str) -> Path:
    target_path = Path(target)
    if target_path.is_absolute() or ".." in target_path.parts:
        raise ValueError(f"target must be a relative path under the artifact root: {target}")
    return root / target_path


def iter_source_files(source: Path) -> Iterable[Tuple[Path, Path]]:
    if source.is_file():
        yield source, Path(source.name)
        return
    if not source.is_dir():
        raise FileNotFoundError(source)
    for item in source.rglob("*"):
        if item.is_file():
            yield item, item.relative_to(source)


def remove_existing(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def transfer_file(source: Path, dest: Path, mode: str, overwrite: bool) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        if not overwrite:
            return "kept"
        remove_existing(dest)

    if mode == "symlink":
        dest.symlink_to(source)
        return "symlinked"
    if mode == "copy":
        shutil.copy2(source, dest)
        return "copied"
    if mode in {"hardlink", "auto"}:
        try:
            os.link(source, dest)
            return "hardlinked"
        except OSError as exc:
            if mode == "hardlink":
                raise
            if exc.errno not in {errno.EXDEV, errno.EPERM, errno.EACCES}:
                raise
            shutil.copy2(source, dest)
            return "copied"
    raise ValueError(f"unknown link mode: {mode}")


def summarize_path(path: Path, hash_files: bool = False) -> Dict[str, Any]:
    if path.is_file():
        payload: Dict[str, Any] = {"file_count": 1, "bytes": path.stat().st_size}
        if hash_files:
            payload["sha256"] = sha256_file(path)
        return payload

    total_size = 0
    file_count = 0
    for item in path.rglob("*"):
        if item.is_file():
            file_count += 1
            total_size += item.stat().st_size
    return {"file_count": file_count, "bytes": total_size}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stage_entry(
    entry: Dict[str, Any],
    root: Path,
    mode: str,
    overwrite: bool,
    include_sources: bool,
    hash_files: bool,
    dry_run: bool,
) -> Dict[str, Any]:
    source = Path(entry["source"]).expanduser()
    target_rel = entry["target"]
    target = safe_target(root, target_rel)
    if not source.exists():
        if entry.get("required", True):
            raise FileNotFoundError(source)
        return {
            "component": entry.get("component", "unknown"),
            "kind": entry.get("kind", "unknown"),
            "variant": entry.get("variant", "unspecified"),
            "target": target_rel,
            "status": "missing_optional",
        }

    if not dry_run:
        if source.is_dir() and target.exists() and overwrite:
            remove_existing(target)
        for source_file, relative in iter_source_files(source):
            dest_file = target if source.is_file() else target / relative
            transfer_file(source_file, dest_file, mode, overwrite)

    payload = {
        "component": entry.get("component", "unknown"),
        "kind": entry.get("kind", "unknown"),
        "variant": entry.get("variant", "unspecified"),
        "target": target_rel,
        "is_directory": source.is_dir(),
        "status": "dry_run" if dry_run else "staged",
    }
    if "description" in entry:
        payload["description"] = entry["description"]
    if not dry_run:
        payload.update(summarize_path(target, hash_files=hash_files and target.is_file()))
    else:
        payload.update(summarize_path(source, hash_files=False))
    if include_sources:
        payload["source"] = str(source)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--link-mode", choices=["auto", "hardlink", "copy", "symlink"], default="auto")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--include-local-sources", action="store_true")
    parser.add_argument("--hash-files", action="store_true", help="Hash single-file entries. This is slow for large models.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selection = load_json(args.selection)
    entries: List[Dict[str, Any]] = selection.get("entries", [])
    if not entries:
        raise SystemExit(f"no entries found in {args.selection}")

    args.dest.mkdir(parents=True, exist_ok=True)
    staged_entries = [
        stage_entry(
            entry=entry,
            root=args.dest,
            mode=args.link_mode,
            overwrite=args.overwrite,
            include_sources=args.include_local_sources,
            hash_files=args.hash_files,
            dry_run=args.dry_run,
        )
        for entry in entries
    ]

    manifest = {
        "artifact_name": "numAnchor",
        "artifact_version": selection.get("artifact_version", "artifact-evaluation"),
        "generated_at_utc": utc_now(),
        "paths_are_relative_to": ".",
        "large_payload_location": "Zenodo artifact archive",
        "entries": staged_entries,
    }
    manifest_path = args.dest / "artifact_manifest.json"
    if not args.dry_run:
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")

    total_bytes = sum(item.get("bytes", 0) for item in staged_entries)
    print(f"staged entries: {len(staged_entries)}")
    print(f"total bytes: {total_bytes}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
