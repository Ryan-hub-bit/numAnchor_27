"""
Streaming deduplication for large corpus files.

Uses xxhash (fast) or hashlib (fallback) to hash each line and track seen hashes.
Memory usage: ~8 bytes per unique line (just the hash, not the line itself).

Usage:
    python dedup.py \
        --input  /path/to/numAnchor_artifact/addressaware/cfg_train/cfg_8inline_pretrain.txt \
        --output /path/to/numAnchor_artifact/addressaware/cfg_train/cfg_8inline_pretrain_nodup.txt \
        [--report_every 1000000]
"""

import argparse
import sys
import time

def get_hasher():
    """Use xxhash if available (10x faster), fall back to hashlib sha1."""
    try:
        import xxhash
        return lambda line: xxhash.xxh64(line).intdigest()
    except ImportError:
        import hashlib
        return lambda line: int(hashlib.sha1(line).hexdigest(), 16)

def dedup(input_path, output_path, report_every=1_000_000):
    hasher = get_hasher()
    seen = set()

    total = 0
    kept = 0
    dupes = 0

    t0 = time.time()

    with open(input_path, 'rb') as fin, open(output_path, 'wb') as fout:
        for raw in fin:
            total += 1
            h = hasher(raw.rstrip(b'\n'))
            if h not in seen:
                seen.add(h)
                fout.write(raw)
                kept += 1
            else:
                dupes += 1

            if total % report_every == 0:
                elapsed = time.time() - t0
                rate = total / elapsed / 1_000_000
                pct_dupes = 100.0 * dupes / total
                print(
                    f"[{total:,}] kept={kept:,}  dupes={dupes:,} ({pct_dupes:.1f}%)  "
                    f"speed={rate:.2f}M lines/s  elapsed={elapsed:.0f}s",
                    flush=True
                )

    elapsed = time.time() - t0
    print(f"\n=== Done ===")
    print(f"Total lines  : {total:,}")
    print(f"Kept (unique): {kept:,}")
    print(f"Duplicates   : {dupes:,}  ({100.0*dupes/total:.2f}%)")
    print(f"Output       : {output_path}")
    print(f"Time         : {elapsed:.1f}s  ({total/elapsed/1e6:.2f}M lines/s)")

def main():
    parser = argparse.ArgumentParser(description="Stream-deduplicate a large text corpus.")
    parser.add_argument('--input',  required=True,  help="Path to input file")
    parser.add_argument('--output', required=True,  help="Path to output file")
    parser.add_argument('--report_every', type=int, default=1_000_000,
                        help="Print progress every N lines (default: 1000000)")
    args = parser.parse_args()

    print(f"Input : {args.input}")
    print(f"Output: {args.output}")
    print(f"Hashing with: ", end='')
    try:
        import xxhash
        print("xxhash (fast)")
    except ImportError:
        print("hashlib sha1 (install xxhash for 10x speedup: pip install xxhash)")

    dedup(args.input, args.output, args.report_every)

if __name__ == '__main__':
    main()
