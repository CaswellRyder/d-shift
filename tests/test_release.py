import hashlib
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "verify_release", Path(__file__).resolve().parents[1] / "scripts/verify_release.py")
release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release)


def test_release_file_integrity(tmp_path):
    (tmp_path / "weights").write_bytes(b"model")
    info = {"bytes": 5, "sha256": hashlib.sha256(b"model").hexdigest()}
    assert release.check_file(tmp_path, "weights", info) == b"model"
    (tmp_path / "weights").write_bytes(b"other")
    with pytest.raises(ValueError, match="mismatch"):
        release.check_file(tmp_path, "weights", info)


def test_release_rejects_path_escape(tmp_path):
    with pytest.raises(ValueError, match="escapes"):
        release.check_file(tmp_path, "../secret", {})
    with pytest.raises(ValueError, match="escapes"):
        release.check_file(tmp_path, str(tmp_path / "absolute"), {})
