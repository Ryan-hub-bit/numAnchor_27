#!/usr/bin/env python3

import os
import subprocess
import time
from pathlib import Path


FOLDER_ID = os.environ.get("X64_FOLDER_ID", "1gvGb_d9Fqkam2fm3LKgWhhWfhPxa_UsM")
ROOT_NAME = os.environ.get("ROOT_NAME", "x86-64")
TARGET_ROOT = Path(os.environ.get("TARGET_ROOT", "/path/to/numAnchor_artifact/stateformer_x64_dataset/raw_input"))
TARGET_DIR = TARGET_ROOT / ROOT_NAME
DOWNLOAD_SCRIPT = Path(
    os.environ.get(
        "DOWNLOAD_SCRIPT",
        "/path/to/numAnchor/PalmTree/scripts/stateformer_x64/download_x86_64_from_drive.sh",
    )
)
DOWNLOAD_LOG = Path(os.environ.get("DOWNLOAD_LOG", "/path/to/numAnchor_artifact/log/Palmtree/download_stateformer_x64.log"))
WATCHDOG_LOG = Path(os.environ.get("WATCHDOG_LOG", "/path/to/numAnchor_artifact/log/Palmtree/stateformer_x64_watchdog.log"))
POLL_SECONDS = int(os.environ.get("WATCHDOG_POLL_SECONDS", "60"))


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def log(message: str) -> None:
    WATCHDOG_LOG.parent.mkdir(parents=True, exist_ok=True)
    with WATCHDOG_LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"[{utc_now()}] {message}\n")


def has_completion_marker() -> bool:
    if not DOWNLOAD_LOG.exists():
        return False
    try:
        content = DOWNLOAD_LOG.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    return "Download completed successfully" in content or f"DONE {TARGET_DIR}" in content


def active_pids() -> list[str]:
    pattern = f"drive_recursive_download.py --folder_id {FOLDER_ID}"
    result = subprocess.run(
        ["pgrep", "-af", pattern],
        check=False,
        capture_output=True,
        text=True,
    )
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    pids: list[str] = []
    for line in lines:
        pid = line.split(maxsplit=1)[0]
        if pid.isdigit():
            pids.append(pid)
    return pids


def file_count() -> int:
    if not TARGET_DIR.exists():
        return 0
    return sum(1 for _ in TARGET_DIR.rglob("*") if _.is_file())


def dir_size_mb() -> int:
    if not TARGET_DIR.exists():
        return 0
    total = 0
    for path in TARGET_DIR.rglob("*"):
        if path.is_file():
            try:
                total += path.stat().st_size
            except OSError:
                pass
    return total // (1024 * 1024)


def infected_count() -> int:
    if not DOWNLOAD_LOG.exists():
        return 0
    try:
        return sum(1 for line in DOWNLOAD_LOG.open("r", encoding="utf-8", errors="ignore") if line.startswith("SKIP_INFECTED "))
    except OSError:
        return 0


def relaunch_download() -> None:
    env = os.environ.copy()
    env["X64_FOLDER_ID"] = FOLDER_ID
    env["ROOT_NAME"] = ROOT_NAME
    env["TARGET_ROOT"] = str(TARGET_ROOT)
    subprocess.Popen(
        [str(DOWNLOAD_SCRIPT)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        env=env,
    )
    log("Download process missing; relaunched downloader")


def main() -> None:
    WATCHDOG_LOG.parent.mkdir(parents=True, exist_ok=True)
    log("Watchdog started")
    last_snapshot: tuple[int, int, int] | None = None

    while True:
        if has_completion_marker():
            log(
                f"Download complete; size_mb={dir_size_mb()} files={file_count()} infected_skips={infected_count()}"
            )
            return

        pids = active_pids()
        if not pids:
            relaunch_download()
            time.sleep(5)
            pids = active_pids()

        snapshot = (dir_size_mb(), file_count(), infected_count())
        if snapshot != last_snapshot:
            size_mb, files, infected = snapshot
            log(f"Progress pids={','.join(pids) if pids else 'none'} size_mb={size_mb} files={files} infected_skips={infected}")
            last_snapshot = snapshot

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
