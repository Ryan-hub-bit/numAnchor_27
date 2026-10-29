#!/usr/bin/env python3

import os
import subprocess
import time
from pathlib import Path


RAW_ROOT = Path(os.environ.get("RAW_ROOT", "/path/to/numAnchor_artifact/stateformer_x64_dataset/raw_input/x86-64"))
CFG_ROOT = Path(os.environ.get("CFG_ROOT", "/path/to/numAnchor_artifact/stateformer_x64_dataset/cfg_ctx2"))
MAIN_LOG = Path(os.environ.get("CFG_MAIN_LOG", "/path/to/numAnchor_artifact/log/Palmtree/generate_cfg_ctx2_recursive.log"))
WATCHDOG_LOG = Path(os.environ.get("CFG_WATCHDOG_LOG", "/path/to/numAnchor_artifact/log/Palmtree/generate_cfg_ctx2_watchdog.log"))
POLL_SECONDS = int(os.environ.get("CFG_WATCHDOG_POLL_SECONDS", "60"))


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def log(message: str) -> None:
    WATCHDOG_LOG.parent.mkdir(parents=True, exist_ok=True)
    with WATCHDOG_LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"[{utc_now()}] {message}\n")


def pids_for(pattern: str) -> list[str]:
    result = subprocess.run(["pgrep", "-af", pattern], check=False, capture_output=True, text=True)
    pids = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        pid = line.split(maxsplit=1)[0]
        if pid.isdigit():
            pids.append(pid)
    return pids


def active_state() -> tuple[list[str], list[str], list[str]]:
    return (
        pids_for("generate_cfg_ctx2_recursive.sh"),
        pids_for("run_cfg_batch.sh 2"),
        pids_for("idat -A -S/path/to/numAnchor/PalmTree/data_generator/cfg_gen.py"),
    )


def total_groups() -> int:
    if not RAW_ROOT.exists():
        return 0
    return sum(1 for path in RAW_ROOT.iterdir() if path.is_dir())


def completed_groups() -> int:
    if not CFG_ROOT.exists():
        return 0
    count = 0
    for path in CFG_ROOT.iterdir():
        if path.is_dir() and any(path.glob("*_cfg_2_inline.txt")):
            count += 1
    return count


def completed_cfg_files() -> int:
    if not CFG_ROOT.exists():
        return 0
    return sum(1 for _ in CFG_ROOT.rglob("*_cfg_2_inline.txt"))


def current_target() -> str:
    if not MAIN_LOG.exists():
        return "unknown"
    try:
        lines = MAIN_LOG.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return "unknown"
    for line in reversed(lines):
        if "RUN " in line and " -> " in line:
            return line.split("RUN ", 1)[1]
    return "unknown"


def complete() -> bool:
    if not MAIN_LOG.exists():
        return False
    try:
        text = MAIN_LOG.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    return "Finished recursive ctx=2 CFG generation" in text


def main() -> None:
    log("CFG watchdog started")
    last_snapshot = None
    total = total_groups()

    while True:
        gen_pids, batch_pids, ida_pids = active_state()
        snapshot = (
            tuple(gen_pids),
            tuple(batch_pids),
            tuple(ida_pids),
            completed_groups(),
            completed_cfg_files(),
            current_target(),
            total,
            complete(),
        )
        if snapshot != last_snapshot:
            log(
                "Progress "
                f"groups_done={snapshot[3]}/{snapshot[6]} "
                f"cfg_files={snapshot[4]} "
                f"current={snapshot[5]} "
                f"gen_pids={','.join(gen_pids) if gen_pids else 'none'} "
                f"batch_pids={','.join(batch_pids) if batch_pids else 'none'} "
                f"ida_pids={','.join(ida_pids) if ida_pids else 'none'}"
            )
            last_snapshot = snapshot

        if snapshot[7]:
            log(f"CFG generation complete; groups_done={snapshot[3]}/{snapshot[6]} cfg_files={snapshot[4]}")
            return

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
