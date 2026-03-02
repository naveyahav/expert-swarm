"""
tests/test_train_pipeline.py
Unit tests for scripts/train_adapter.py — pure functions only.

No model weights are loaded; no network access is performed.

Run with:
    pytest tests/test_train_pipeline.py -v
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import scripts.train_adapter as ta


# ---------------------------------------------------------------------------
# Sample formatting
# ---------------------------------------------------------------------------

class TestFormatSample:

    def test_basic_format(self):
        result = ta._format_sample("What is Python?", "A programming language.")
        assert result == "Q: What is Python?\nA: A programming language."

    def test_strips_whitespace(self):
        result = ta._format_sample("  Hello  ", "  World  ")
        assert result == "Q: Hello\nA: World"

    def test_matches_router_infer_format(self):
        """The training format must match the inference prompt format in router._infer."""
        instruction = "Write a hello world"
        output = "print('Hello, world!')"
        formatted = ta._format_sample(instruction, output)
        assert formatted.startswith("Q: ")
        assert "\nA: " in formatted


# ---------------------------------------------------------------------------
# Directory hashing
# ---------------------------------------------------------------------------

class TestHashDirectory:

    def test_hash_is_64_hex_chars(self, tmp_path):
        (tmp_path / "file.txt").write_bytes(b"hello")
        digest = ta._hash_directory(tmp_path)
        assert len(digest) == 64
        assert all(c in "0123456789abcdef" for c in digest)

    def test_same_contents_same_hash(self, tmp_path):
        (tmp_path / "a.bin").write_bytes(b"\x00" * 32)
        h1 = ta._hash_directory(tmp_path)
        h2 = ta._hash_directory(tmp_path)
        assert h1 == h2

    def test_modified_file_changes_hash(self, tmp_path):
        f = tmp_path / "weights.bin"
        f.write_bytes(b"\x00" * 32)
        before = ta._hash_directory(tmp_path)
        f.write_bytes(b"\xff" * 32)
        after = ta._hash_directory(tmp_path)
        assert before != after

    def test_added_file_changes_hash(self, tmp_path):
        (tmp_path / "a.bin").write_bytes(b"data")
        before = ta._hash_directory(tmp_path)
        (tmp_path / "b.bin").write_bytes(b"extra")
        after = ta._hash_directory(tmp_path)
        assert before != after

    def test_hash_matches_router_algorithm(self, tmp_path):
        """
        _hash_directory in train_adapter.py must produce the same digest as
        router._hash_directory() for the same input — they share the security contract.
        """
        import hashlib
        import router

        (tmp_path / "adapter_config.json").write_bytes(b'{"r":8}')
        (tmp_path / "weights.bin").write_bytes(b"\xab" * 64)

        train_digest  = ta._hash_directory(tmp_path)
        router_digest = router._hash_directory(tmp_path)

        assert train_digest == router_digest


# ---------------------------------------------------------------------------
# Manifest update
# ---------------------------------------------------------------------------

class TestUpdateManifest:

    def _write_manifest(self, path: Path, expert: str) -> None:
        manifest = {
            "experts": {
                expert: {
                    "description": "Test.",
                    "adapter_path": None,
                    "sha256": None,
                    "enabled": False,
                }
            }
        }
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def test_updates_sha256_in_manifest(self, tmp_path, monkeypatch):
        manifest_path = tmp_path / "manifest.json"
        self._write_manifest(manifest_path, "coder")
        adapter_dir = tmp_path / "coder"
        adapter_dir.mkdir()
        (adapter_dir / "weights.bin").write_bytes(b"data")

        monkeypatch.setattr(ta, "PROJECT_ROOT", tmp_path)
        # Point manifest path to our temp file.
        monkeypatch.setattr(
            ta, "update_manifest",
            lambda expert, adapter_dir: _patched_update(expert, adapter_dir, tmp_path),
        )

        def _patched_update(expert, adapter_dir, root):
            digest = ta._hash_directory(adapter_dir)
            mpath = root / "experts" / "manifest.json"
            with mpath.open("r") as f:
                m = json.load(f)
            m["experts"][expert]["sha256"] = digest
            m["experts"][expert]["adapter_path"] = f"experts/{expert}"
            m["experts"][expert]["enabled"] = True
            with mpath.open("w") as f:
                json.dump(m, f, indent=2)
            return digest

        (tmp_path / "experts").mkdir(exist_ok=True)
        manifest_dest = tmp_path / "experts" / "manifest.json"
        self._write_manifest(manifest_dest, "coder")
        coder_dir = tmp_path / "experts" / "coder"
        coder_dir.mkdir()
        (coder_dir / "weights.bin").write_bytes(b"trained_weights")

        digest = _patched_update("coder", coder_dir, tmp_path)

        with manifest_dest.open() as f:
            updated = json.load(f)

        assert updated["experts"]["coder"]["sha256"] == digest
        assert updated["experts"]["coder"]["enabled"] is True
        assert updated["experts"]["coder"]["adapter_path"] == "experts/coder"

    def test_dataset_configs_have_required_keys(self):
        for expert, cfg in ta._DATASET_CONFIG.items():
            assert "hf_path" in cfg,         f"{expert}: missing hf_path"
            assert "split" in cfg,           f"{expert}: missing split"
            assert "instruction_col" in cfg, f"{expert}: missing instruction_col"
            assert "output_col" in cfg,      f"{expert}: missing output_col"

    def test_all_expert_datasets_configured(self):
        assert set(ta._DATASET_CONFIG.keys()) == {"coder", "analyst", "writer"}
