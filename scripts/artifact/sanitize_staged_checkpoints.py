#!/usr/bin/env python3
"""Remove local filesystem paths from staged PyTorch checkpoints.

Run this on the staged artifact copy, not on training output directories. The
script saves through a temporary file and replaces the staged checkpoint so
hard-linked sources are not modified.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from argparse import Namespace
from pathlib import Path
from typing import Any

import torch

try:
    from omegaconf import DictConfig, ListConfig, OmegaConf
except Exception:  # pragma: no cover - optional dependency for non-fairseq ckpts
    DictConfig = ()  # type: ignore[assignment]
    ListConfig = ()  # type: ignore[assignment]
    OmegaConf = None  # type: ignore[assignment]


DEFAULT_ROOT = Path(os.environ.get("NUMANCHOR_ARTIFACT_ROOT", "numAnchor_artifact/staging"))
LEGACY_MISSPELLING = "addres" + "aware"
LOCAL_PATTERNS = (
    "/home/",
    "/data/",
    "/path/to/",
    LEGACY_MISSPELLING,
)
PATH_RE = re.compile(r"(/home/[^/\s]+|/data/[^/\s]+|/path/to)[^\s'\",)}\]]*")


def contains_local_path(value: str) -> bool:
    low = value.lower()
    return any(pattern in low for pattern in LOCAL_PATTERNS)


def relative_for_checkpoint(path: Path, key: str, value: str) -> str:
    parts = path.parts
    text = value.lower()

    if "/models/trex/" in path.as_posix():
        variant = "numanchor" if "numanchor" in path.name else "baseline"
        if key.endswith("pretrained_checkpoint"):
            return f"models/trex/{variant}_pretrain.pt"
        if key.endswith("data_dir") or key.endswith("data"):
            return f"data/trex/{variant}_pretrain_bin"
        if key.endswith("aa_data_dir"):
            return "data/trex/numanchor_pretrain_bin"
        if key.endswith("function_blocks"):
            return "data/shared_stateformer_x64/function_blocks.json"
        if key.endswith("funcsim_pairs"):
            return "data/shared_stateformer_x64/ground_truth.json"
        if key.endswith("exclude_query_pool") or "pool_ids" in text:
            return "work/pools/trex_pool_10k.json"
        if key.endswith("save_dir"):
            return "models/trex"

    if "/models/stateformer/" in path.as_posix():
        opt = next((part for part in parts if part in {"O0", "O1", "O2", "O3"}), "OPT")
        variant = "numanchor" if "numanchor" in path.name else "baseline"
        if key.endswith("restore_file"):
            return f"models/stateformer/{opt}/{variant}_pretrain.pt"
        if key.endswith("data"):
            split = "downstream_sparse" if "finetune" in path.name else "pretrain_dataset"
            return f"data/stateformer_strict_opt/{opt}/{split}"
        if key.endswith("save_dir"):
            stem = path.stem
            return f"models/stateformer/{opt}/{stem}"

    replacements = {
        "stateformer_x64_dataset/funcsim_block_ctx2/function_blocks.json": "data/shared_stateformer_x64/function_blocks.json",
        "stateformer_x64_dataset/funcsim_block_ctx2/funcsim_pairs.json": "data/shared_stateformer_x64/ground_truth.json",
        "stateformer_x64_dataset/funcsim_block_ctx2/pool_ids_10k.json": "work/pools/pool_10k.json",
    }
    for suffix, new in replacements.items():
        if suffix in value:
            return new

    return PATH_RE.sub("<redacted-local-path>", value).replace(LEGACY_MISSPELLING, "numAnchor")


def sanitize_obj(obj: Any, checkpoint_path: Path, prefix: str = "") -> tuple[Any, int]:
    changed = 0
    if isinstance(obj, str):
        if "/models/trex/" in checkpoint_path.as_posix() and prefix.endswith("mode") and obj == "addressaware_aa":
            return "numanchor_aa", 1
        if contains_local_path(obj):
            return relative_for_checkpoint(checkpoint_path, prefix, obj), 1
        return obj, 0

    if isinstance(obj, dict):
        for key in list(obj.keys()):
            new_value, delta = sanitize_obj(obj[key], checkpoint_path, f"{prefix}.{key}" if prefix else str(key))
            if delta:
                obj[key] = new_value
                changed += delta
        return obj, changed

    if isinstance(obj, list):
        for index, value in enumerate(obj):
            new_value, delta = sanitize_obj(value, checkpoint_path, f"{prefix}[{index}]")
            if delta:
                obj[index] = new_value
                changed += delta
        return obj, changed

    if isinstance(obj, tuple):
        items = []
        for index, value in enumerate(obj):
            new_value, delta = sanitize_obj(value, checkpoint_path, f"{prefix}[{index}]")
            items.append(new_value)
            changed += delta
        return tuple(items), changed

    if isinstance(obj, Namespace):
        for key, value in vars(obj).items():
            new_value, delta = sanitize_obj(value, checkpoint_path, f"{prefix}.{key}" if prefix else key)
            if delta:
                setattr(obj, key, new_value)
                changed += delta
        return obj, changed

    if OmegaConf is not None and isinstance(obj, (DictConfig, ListConfig)):
        keys = range(len(obj)) if isinstance(obj, ListConfig) else list(obj.keys())
        for key in keys:
            try:
                value = obj[key]
            except Exception:
                continue
            new_value, delta = sanitize_obj(value, checkpoint_path, f"{prefix}.{key}" if prefix else str(key))
            if delta:
                obj[key] = new_value
                changed += delta
        return obj, changed

    return obj, 0


def sanitize_checkpoint(path: Path, dry_run: bool) -> int:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    checkpoint, changed = sanitize_obj(checkpoint, path)
    if changed and not dry_run:
        tmp = path.with_suffix(path.suffix + ".sanitized.tmp")
        torch.save(checkpoint, tmp)
        os.replace(tmp, path)
    return changed


def summarize_path(path: Path) -> tuple[int, int]:
    if path.is_file():
        return 1, path.stat().st_size
    count = 0
    size = 0
    for item in path.rglob("*"):
        if item.is_file():
            count += 1
            size += item.stat().st_size
    return count, size


def refresh_manifest(root: Path) -> None:
    manifest_path = root / "artifact_manifest.json"
    if not manifest_path.exists():
        return
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    for entry in manifest.get("entries", []):
        if entry.get("status") == "missing_optional":
            continue
        target = entry.get("target")
        if not target:
            continue
        path = root / target
        if not path.exists():
            continue
        file_count, size = summarize_path(path)
        entry["file_count"] = file_count
        entry["bytes"] = size
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-refresh-manifest", action="store_true")
    parser.add_argument("--refresh-manifest-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.refresh_manifest_only:
        refresh_manifest(args.artifact_root)
        print("manifest refreshed")
        return
    model_root = args.artifact_root / "models"
    targets = sorted(model_root.rglob("*.pt"))
    total = 0
    touched = 0
    for path in targets:
        changed = sanitize_checkpoint(path, args.dry_run)
        if changed:
            touched += 1
            total += changed
            print(f"{path.relative_to(args.artifact_root)}: sanitized {changed} metadata strings")
    if total and not args.dry_run and not args.no_refresh_manifest:
        refresh_manifest(args.artifact_root)
    print(f"checkpoints_touched={touched} strings_sanitized={total}")


if __name__ == "__main__":
    main()
