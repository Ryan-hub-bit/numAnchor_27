#!/usr/bin/env python3

import argparse
import re
import time
from pathlib import Path
from typing import Iterable

import requests
from bs4 import BeautifulSoup


EMBEDDED_VIEW = "https://drive.google.com/embeddedfolderview?id={folder_id}#list"
FILE_DOWNLOAD_URL = "https://drive.google.com/uc?export=download&id={file_id}"
FILE_ID_RE = re.compile(r"/file/d/([A-Za-z0-9_-]+)")
FOLDER_ID_RE = re.compile(r"/drive/folders/([A-Za-z0-9_-]+)")


class InfectedFileError(RuntimeError):
    """Google Drive blocks public download for infected files."""


def sanitize_name(name: str) -> str:
    cleaned = name.strip().replace("/", "_").replace("\x00", "")
    return cleaned or "unnamed"


def list_folder(folder_id: str, retries: int = 3) -> list[tuple[str, str, str]]:
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(EMBEDDED_VIEW.format(folder_id=folder_id), timeout=60)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            break
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt == retries:
                raise
            print(f"RETRY folder {folder_id}: attempt {attempt}/{retries} failed: {exc}", flush=True)
            time.sleep(min(10, attempt * 2))
    else:
        raise RuntimeError(f"Unable to list folder {folder_id}: {last_error}")

    items: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str]] = set()

    for anchor in soup.find_all("a", href=True):
        href = anchor["href"]
        label = sanitize_name(" ".join(anchor.stripped_strings))

        file_match = FILE_ID_RE.search(href)
        if file_match:
            key = ("file", file_match.group(1))
            if key not in seen:
                items.append(("file", file_match.group(1), label))
                seen.add(key)
            continue

        folder_match = FOLDER_ID_RE.search(href)
        if folder_match:
            key = ("folder", folder_match.group(1))
            if key not in seen:
                items.append(("folder", folder_match.group(1), label))
                seen.add(key)

    return items


def extract_confirm_token(resp: requests.Response) -> str | None:
    for key, value in resp.cookies.items():
        if key.startswith("download_warning"):
            return value

    match = re.search(r'name="confirm"\s+value="([^"]+)"', resp.text)
    if match:
        return match.group(1)

    match = re.search(r'confirm=([0-9A-Za-z_]+)', resp.text)
    if match:
        return match.group(1)

    return None


def stream_to_file(resp: requests.Response, output_path: Path) -> None:
    tmp_path = output_path.with_suffix(output_path.suffix + ".part")
    with tmp_path.open("wb") as handle:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if chunk:
                handle.write(chunk)
    tmp_path.replace(output_path)


def direct_download(file_id: str, output_path: Path, session: requests.Session) -> None:
    params = {"export": "download", "id": file_id}
    resp = session.get(FILE_DOWNLOAD_URL.format(file_id=file_id), params=params, stream=True, timeout=120)
    resp.raise_for_status()

    content_type = (resp.headers.get("content-type") or "").lower()
    content_disp = resp.headers.get("content-disposition") or ""
    is_binary = "attachment" in content_disp.lower() or "application/octet-stream" in content_type

    if not is_binary:
        body = resp.text
        lower_body = body.lower()
        if "infected file" in lower_body or "only the owner is allowed to download infected files" in lower_body:
            raise InfectedFileError(f"Google Drive blocked infected file download for file id={file_id}")
        token = extract_confirm_token(resp)
        if token is None:
            raise RuntimeError(
                f"Unable to obtain Google Drive confirm token for file id={file_id}; "
                f"content-type={content_type!r}, first 200 chars={body[:200]!r}"
            )

        resp.close()
        params["confirm"] = token
        resp = session.get(FILE_DOWNLOAD_URL.format(file_id=file_id), params=params, stream=True, timeout=120)
        resp.raise_for_status()

    try:
        stream_to_file(resp, output_path)
    finally:
        resp.close()


def download_file(file_id: str, output_path: Path, retries: int = 3) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"SKIP {output_path}: already exists ({output_path.stat().st_size} bytes)", flush=True)
        return
    last_error = None
    for attempt in range(1, retries + 1):
        print(f"DOWNLOAD {output_path} [attempt {attempt}/{retries}]", flush=True)
        try:
            with requests.Session() as session:
                direct_download(file_id=file_id, output_path=output_path, session=session)
            return
        except InfectedFileError as exc:
            print(f"SKIP_INFECTED {output_path}: {exc}", flush=True)
            return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt == retries:
                raise
            print(f"RETRY file {output_path}: attempt {attempt}/{retries} failed: {exc}", flush=True)
            time.sleep(min(10, attempt * 2))
    raise RuntimeError(f"Failed to download file id={file_id} -> {output_path}: {last_error}")


def walk_folder(folder_id: str, output_dir: Path, visited: set[str]) -> None:
    if folder_id in visited:
        print(f"SKIP folder {folder_id}: already visited")
        return

    visited.add(folder_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    items = list_folder(folder_id)
    print(f"FOLDER {folder_id} -> {output_dir} ({len(items)} items)")

    for kind, item_id, label in items:
        if kind == "folder":
            walk_folder(item_id, output_dir / label, visited)
        else:
            download_file(item_id, output_dir / label)


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Recursively download a public Google Drive folder")
    parser.add_argument("--folder_id", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--root_name", default="x86-64")
    args = parser.parse_args(argv)

    root = Path(args.output_root) / sanitize_name(args.root_name)
    walk_folder(args.folder_id, root, set())
    print(f"DONE {root}")


if __name__ == "__main__":
    main()
