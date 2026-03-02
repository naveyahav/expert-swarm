"""
tests/test_validate_expert.py
Unit tests for router.validate_expert().

Each test monkeypatches router.PROJECT_ROOT to a pytest tmp_path so the
directory-traversal guard is exercised without touching the real project tree.

Run with:
    pytest tests/test_validate_expert.py -v
"""

import hashlib
import sys
from pathlib import Path

# Ensure project root is importable regardless of launch directory.
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import router


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_adapter(base: Path, name: str, files: dict[str, bytes] = None) -> Path:
    """Create a fake adapter directory under *base* with the given files."""
    files = files or {"adapter_config.json": b'{"r": 8}', "weights.bin": b"\x00" * 64}
    adapter = base / name
    adapter.mkdir(parents=True, exist_ok=True)
    for filename, content in files.items():
        (adapter / filename).write_bytes(content)
    return adapter


def _real_hash(directory: Path) -> str:
    """
    Mirrors router._hash_directory exactly — kept in sync manually.
    Used to produce correct expected hashes for the happy-path tests.
    """
    sha = hashlib.sha256()
    for file_path in sorted(directory.rglob("*")):
        if file_path.is_file():
            sha.update(file_path.relative_to(directory).as_posix().encode())
            with file_path.open("rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    sha.update(chunk)
    return sha.hexdigest()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestValidateExpert:

    def test_valid_adapter_returns_true(self, tmp_path, monkeypatch):
        """Correct hash + path inside project root → True."""
        monkeypatch.setattr(router, "PROJECT_ROOT", tmp_path)
        adapter = _make_adapter(tmp_path, "coder")
        good_hash = _real_hash(adapter)

        assert router.validate_expert(str(adapter), good_hash) is True

    def test_wrong_hash_returns_false(self, tmp_path, monkeypatch):
        """Correct path but wrong hash → False (tamper detection)."""
        monkeypatch.setattr(router, "PROJECT_ROOT", tmp_path)
        adapter = _make_adapter(tmp_path, "coder")
        wrong_hash = "a" * 64  # valid hex length, wrong value

        assert router.validate_expert(str(adapter), wrong_hash) is False

    def test_hash_is_case_insensitive(self, tmp_path, monkeypatch):
        """Manifest stores uppercase hash → should still match."""
        monkeypatch.setattr(router, "PROJECT_ROOT", tmp_path)
        adapter = _make_adapter(tmp_path, "coder")
        good_hash = _real_hash(adapter).upper()

        assert router.validate_expert(str(adapter), good_hash) is True

    def test_placeholder_hash_returns_false(self, tmp_path, monkeypatch):
        """Unconfigured entries with REPLACE_WITH_* hash → always rejected."""
        monkeypatch.setattr(router, "PROJECT_ROOT", tmp_path)
        adapter = _make_adapter(tmp_path, "coder")

        assert router.validate_expert(str(adapter), "REPLACE_WITH_ACTUAL_SHA256") is False

    def test_empty_hash_returns_false(self, tmp_path, monkeypatch):
        """Empty string hash → rejected (treat as unconfigured)."""
        monkeypatch.setattr(router, "PROJECT_ROOT", tmp_path)
        adapter = _make_adapter(tmp_path, "coder")

        assert router.validate_expert(str(adapter), "") is False

    def test_nonexistent_path_returns_false(self, tmp_path, monkeypatch):
        """Path that does not exist → False."""
        monkeypatch.setattr(router, "PROJECT_ROOT", tmp_path)
        missing = str(tmp_path / "does_not_exist")

        assert router.validate_expert(missing, "a" * 64) is False

    def test_file_instead_of_directory_returns_false(self, tmp_path, monkeypatch):
        """Passing a file path instead of a directory → False."""
        monkeypatch.setattr(router, "PROJECT_ROOT", tmp_path)
        a_file = tmp_path / "model.bin"
        a_file.write_bytes(b"data")

        assert router.validate_expert(str(a_file), "a" * 64) is False

    def test_directory_traversal_returns_false(self, tmp_path, monkeypatch):
        """Adapter path outside PROJECT_ROOT → blocked, returns False."""
        project = tmp_path / "project"
        project.mkdir()
        monkeypatch.setattr(router, "PROJECT_ROOT", project)

        # Adapter lives in tmp_path, which is the *parent* of our fake root.
        outside_adapter = _make_adapter(tmp_path, "evil_adapter")
        good_hash = _real_hash(outside_adapter)

        assert router.validate_expert(str(outside_adapter), good_hash) is False

    def test_tampered_file_returns_false(self, tmp_path, monkeypatch):
        """Hash captured before a file is modified → rejected after tampering."""
        monkeypatch.setattr(router, "PROJECT_ROOT", tmp_path)
        adapter = _make_adapter(tmp_path, "coder")
        hash_before = _real_hash(adapter)

        # Simulate tampering by modifying a weight file after hashing.
        (adapter / "weights.bin").write_bytes(b"\xff" * 64)

        assert router.validate_expert(str(adapter), hash_before) is False

    def test_added_file_invalidates_hash(self, tmp_path, monkeypatch):
        """Adding a new file to the adapter dir invalidates the stored hash."""
        monkeypatch.setattr(router, "PROJECT_ROOT", tmp_path)
        adapter = _make_adapter(tmp_path, "coder")
        hash_before = _real_hash(adapter)

        (adapter / "injected.py").write_bytes(b"import os; os.system('rm -rf /')")

        assert router.validate_expert(str(adapter), hash_before) is False
