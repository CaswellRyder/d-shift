"""Bounded authorized public COCO export, quarantined ZIP only; no credentials saved."""
import argparse
import getpass
import hashlib
from pathlib import Path
import re
import shutil
import time
from urllib.parse import urlparse
import zipfile

import requests

from dtr.data import write_json
from scripts.download_dtr import fetch_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", required=True, help="workspace/project")
    p.add_argument("--version", type=int, required=True)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--max-bytes", type=int, default=1_000_000_000)
    args = p.parse_args()
    if (not re.fullmatch(r"[a-z0-9-]+/[a-z0-9-]+", args.project)
            or args.version < 1 or not 1 <= args.max_bytes <= 1_000_000_000):
        p.error("Require public project slug, positive version and <=1 GB bound")
    if args.output.exists():
        raise FileExistsError(args.output)
    if shutil.disk_usage(Path.cwd()).free < args.max_bytes + 2_000_000_000:
        raise RuntimeError("Insufficient free disk margin")
    key = getpass.getpass("Roboflow API key (hidden, memory only): ")
    base = f"https://api.roboflow.com/{args.project}/{args.version}"
    meta = fetch_json(base, key).get("version", {})
    public = {k: meta.get(k) for k in ("id", "name", "images", "splits", "preprocessing", "augmentation")}
    for attempt in range(60):
        result = fetch_json(f"{base}/coco", key)
        link = result.get("export", {}).get("link")
        if result.get("ready") is not False and link:
            break
        if attempt % 6 == 0:
            print("Export preparing", flush=True)
        time.sleep(5)
    else:
        raise RuntimeError("Export preparation timeout")
    if urlparse(link).scheme != "https":
        raise ValueError("Require HTTPS export")
    args.output.mkdir(parents=True)
    dest = args.output / "dataset.zip"
    digest, size = hashlib.sha256(), 0
    with requests.get(link, stream=True, timeout=(30, 60)) as response:
        if response.status_code != 200:
            raise RuntimeError("Export download failed")
        length = int(response.headers.get("Content-Length", 0))
        if length > args.max_bytes:
            raise ValueError("Archive exceeds requested bound")
        with dest.open("xb") as stream:
            for chunk in response.iter_content(1024*1024):
                size += len(chunk)
                if size > args.max_bytes:
                    raise ValueError("Archive exceeds requested bound")
                stream.write(chunk)
                digest.update(chunk)
                if size // 50_000_000 != (size-len(chunk)) // 50_000_000:
                    print(f"Downloaded {size:,} bytes", flush=True)
    # Do not extract third-party files or execute anything from the archive.
    with zipfile.ZipFile(dest) as archive:
        members = archive.infolist()
        uncompressed_bytes = sum(r.file_size for r in members)
    receipt = dict(source=f"https://universe.roboflow.com/{args.project}/dataset/{args.version}",
                   metadata=public, archive_sha256=digest.hexdigest(), archive_bytes=size,
                   uncompressed_bytes=uncompressed_bytes, members=len(members), format="coco",
                   credential_stored=False, extracted=False, training_approved=False,
                   review_required=True, deployment_approved=False)
    write_json(args.output / "download-receipt.json", receipt)
    print(dict(bytes=size, sha256=digest.hexdigest(), members=len(members), training_approved=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Download stopped ({type(exc).__name__}); credentials and links suppressed", flush=True)
        raise SystemExit(1) from None
