#!/usr/bin/env python3

import argparse
import collections
import os
from pathlib import Path


def human_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{num_bytes} B"


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect downloaded x86-64 dataset contents")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    manifest_path = Path(args.manifest)
    summary_path = Path(args.summary)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    files = []
    ext_counter = collections.Counter()
    dir_counter = collections.Counter()
    total_bytes = 0

    for path in sorted(input_dir.rglob("*")):
        rel = path.relative_to(input_dir)
        if path.is_dir():
            dir_counter[str(rel.parent)] += 1
            continue
        try:
            size = path.stat().st_size
        except OSError:
            size = -1
        total_bytes += max(size, 0)
        ext_counter[path.suffix or "<no_ext>"] += 1
        files.append((str(rel), size))

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        for rel, size in files:
            f.write(f"{size}\t{rel}\n")

    top_ext_lines = "\n".join(
        f"- `{ext}`: {count}" for ext, count in ext_counter.most_common(20)
    )
    sample_lines = "\n".join(f"- `{rel}`" for rel, _ in files[:40])

    summary = f"""# x86-64 Download Summary

- Root: `{input_dir}`
- File count: `{len(files)}`
- Total size: `{human_size(total_bytes)}`

## Top Extensions
{top_ext_lines or '- none'}

## Sample Files
{sample_lines or '- none'}
"""

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(summary, encoding="utf-8")

    print(f"Wrote manifest: {manifest_path}")
    print(f"Wrote summary : {summary_path}")
    print(f"Files         : {len(files)}")
    print(f"Total size    : {human_size(total_bytes)}")


if __name__ == "__main__":
    main()
