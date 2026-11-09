"""Check frozen v1 model artifacts in the evolving source tree. No network.

This verifies model bytes, not the identity of the historical v1 source tree.
Use verify_release.py from the v1.0.0 tag for strict release reproduction.
"""

import argparse
import json
from pathlib import Path

if __package__:
    from .verify_release import check_file, smoke
else:
    from verify_release import check_file, smoke


def verify(root):
    root = Path(root)
    manifest = json.loads((root / "models/v1.0.0/manifest.json").read_text())
    if manifest["deployment_approved"] is not False:
        raise ValueError("This snapshot must remain unapproved for deployment")
    files = manifest["files"]
    if not isinstance(files, dict) or not files:
        raise ValueError("Model artifact manifest must contain files")
    for name, info in files.items():
        check_file(root, name, info)
    return {
        "files_verified": len(files),
        "source_files_verified": 0,
        "source_identity_verified": False,
        "assets_verified": 0,
        "deployment_approved": False,
        "scope": "Committed v1 artifact bytes only; no frozen source identity or flight approval",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true",
                        help="Also run generated inference probes with local ML dependencies")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    result = verify(root)
    if args.smoke:
        result["generated_probe_checks"] = smoke(root)
        result["scope"] += "; generated model probes are not recognition accuracy or Pi timing"
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
