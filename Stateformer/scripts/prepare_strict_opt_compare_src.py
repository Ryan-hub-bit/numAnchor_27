#!/usr/bin/env python3

import argparse
import json
from collections import Counter
from pathlib import Path


OPTS = ("O0", "O1", "O2", "O3")


def is_elf(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            return handle.read(4) == b"\x7fELF"
    except OSError:
        return False


def collect_by_opt(raw_root: Path):
    by_opt = {opt: {} for opt in OPTS}
    for path in sorted(raw_root.rglob("*")):
        if not path.is_file() or not is_elf(path):
            continue
        parent = path.parent.name
        opt = None
        for candidate in OPTS:
            if f"-{candidate}" in parent:
                opt = candidate
                break
        if opt is None:
            continue
        base_dir = parent.replace(f"-{opt}", "")
        key = (base_dir, path.name)
        by_opt[opt][key] = str(path.resolve())
    return by_opt


def main():
    parser = argparse.ArgumentParser(description="Prepare strict O0/O1/O2/O3 source sets using the exact binary intersection.")
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    raw_root = Path(args.raw_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    by_opt = collect_by_opt(raw_root)
    common = sorted(set(by_opt["O0"]) & set(by_opt["O1"]) & set(by_opt["O2"]) & set(by_opt["O3"]))

    summary = {
        "common_exact_binaries": len(common),
        "opt_counts": {opt: len(by_opt[opt]) for opt in OPTS},
        "family_counts": dict(Counter(base for base, _ in common)),
        "opts": {},
    }

    for opt in OPTS:
        src_dir = output_root / opt / "src"
        src_dir.mkdir(parents=True, exist_ok=True)
        for base_dir, program in common:
            target = Path(by_opt[opt][(base_dir, program)])
            link_name = f"{base_dir}-{opt}__{program}"
            link_path = src_dir / link_name
            if link_path.is_symlink() or link_path.exists():
                if link_path.resolve() == target:
                    continue
                link_path.unlink()
            link_path.symlink_to(target)

        summary["opts"][opt] = {
            "src_dir": str(src_dir),
            "binary_count": len(common),
        }

    manifest_path = output_root / "strict_opt_common_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
