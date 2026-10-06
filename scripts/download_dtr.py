"""Download authorized COCO exports without storing credentials or signed links.

Uses the same documented export endpoint as the official Roboflow Python SDK.
Existing requests dependency only; does not change the pinned training environment.
"""

import getpass
import hashlib
import json
import time
import zipfile
from pathlib import Path
from urllib.parse import urlparse

import requests

BASE = "https://api.roboflow.com/cheese-geozd/cats-and-dogs-bmity"


def fetch_json(url, key):
    response = requests.get(url, params={"api_key": key}, timeout=60)
    if response.status_code not in (200, 202):
        raise RuntimeError(f"Roboflow returned HTTP {response.status_code}; details suppressed")
    return response.json()


def main():
    key = getpass.getpass("Roboflow API key (hidden; memory only): ")
    for version in (11, 10):
        root = Path(f"data/raw/roboflow-dtr-v{version}")
        if root.exists():
            raise FileExistsError(f"Refusing to overwrite {root}")
        metadata = fetch_json(f"{BASE}/{version}", key).get("version", {})
        # Allowlist metadata: API responses may also contain credentials or signed URLs.
        public = {
            k: metadata.get(k)
            for k in ("id", "name", "images", "splits", "augmentation", "preprocessing", "created")
        }
        print(json.dumps({"version": version, "metadata": public}), flush=True)
        for attempt in range(60):
            export = fetch_json(f"{BASE}/{version}/coco", key)
            if export.get("ready") is not False and export.get("export", {}).get("link"):
                break
            if attempt % 6 == 0:
                print(f"V{version}: export preparing", flush=True)
            time.sleep(5)
        else:
            raise RuntimeError("Export not ready after bounded wait")
        link = export["export"]["link"]
        if urlparse(link).scheme != "https":
            raise ValueError("Expected HTTPS download")
        root.mkdir(parents=True)
        archive = root / "dataset.zip"
        digest = hashlib.sha256()
        size = 0
        with requests.get(link, stream=True, timeout=(30, 90)) as response:
            if response.status_code != 200:
                raise RuntimeError(f"Download returned HTTP {response.status_code}")
            with archive.open("xb") as stream:
                for chunk in response.iter_content(1024 * 1024):
                    size += len(chunk)
                    if size > 3_000_000_000:
                        raise ValueError("Archive exceeds 3 GB safety limit")
                    digest.update(chunk)
                    stream.write(chunk)
        with zipfile.ZipFile(archive) as zipped:
            if sum(x.file_size for x in zipped.infolist()) > 8_000_000_000:
                raise ValueError("Expanded archive exceeds 8 GB")
            for member in zipped.infolist():
                dest = (root / "coco" / member.filename).resolve()
                if not dest.is_relative_to((root / "coco").resolve()):
                    raise ValueError("Unsafe archive path")
                if (member.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError("Archive symlink rejected")
            zipped.extractall(root / "coco")
        receipt = {
            "source": f"https://universe.roboflow.com/cheese-geozd/cats-and-dogs-bmity/dataset/{version}",
            "version": version,
            "format": "coco",
            "archive_sha256": digest.hexdigest(),
            "archive_bytes": size,
            "metadata": public,
            "credential_stored": False,
        }
        (root / "download-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(f"V{version} downloaded and extracted: {size:,} bytes", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # Never print exception text/traceback: requests errors may contain authenticated URLs.
        print(f"Download stopped ({type(exc).__name__}); no credentials logged.", flush=True)
        raise SystemExit(1) from None
