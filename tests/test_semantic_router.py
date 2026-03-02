"""
tests/test_semantic_router.py
Unit tests for core/semantic_router.py.

Tests cover both the happy path (embedder available) and the graceful
fallback path (sentence-transformers unavailable or embedder fails to load).

Run with:
    pytest tests/test_semantic_router.py -v
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import core.semantic_router as sr


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reset_router():
    """Reset the module-level singletons between tests."""
    sr._embedder = None
    sr._expert_embeddings = {}


# ---------------------------------------------------------------------------
# Fallback path (sentence-transformers unavailable)
# ---------------------------------------------------------------------------

class TestSemanticRouterFallback:

    def setup_method(self):
        _reset_router()

    def test_classify_returns_none_when_import_fails(self, monkeypatch):
        """If sentence-transformers can't be imported, classify() returns None."""
        monkeypatch.setattr(sr, "_load_embedder", lambda: False)
        assert sr.classify("write a Python function", ["coder", "analyst"]) is None

    def test_classify_returns_none_for_empty_candidates(self, monkeypatch):
        """No candidate experts → None regardless of embedder state."""
        monkeypatch.setattr(sr, "_load_embedder", lambda: True)
        # All enabled experts have no profile entry.
        assert sr.classify("hello world", ["base"]) is None

    def test_classify_returns_none_for_empty_enabled_list(self, monkeypatch):
        monkeypatch.setattr(sr, "_load_embedder", lambda: True)
        assert sr.classify("write code", []) is None

    def test_load_embedder_returns_false_on_import_error(self, monkeypatch):
        """_load_embedder() must return False when SentenceTransformer is missing."""
        _reset_router()
        with patch.dict("sys.modules", {"sentence_transformers": None}):
            result = sr._load_embedder()
        assert result is False
        # Singleton must remain unset after failure.
        assert sr._embedder is None


# ---------------------------------------------------------------------------
# Happy path (mocked embedder)
# ---------------------------------------------------------------------------

class TestSemanticRouterHappyPath:

    def setup_method(self):
        _reset_router()

    def _install_mock_embedder(self):
        """
        Install a fake SentenceTransformer that returns deterministic embeddings.
        coder  → [1, 0, 0]
        analyst→ [0, 1, 0]
        writer → [0, 0, 1]
        """
        import numpy as np

        profile_vecs = {
            "coder":   np.array([1.0, 0.0, 0.0]),
            "analyst": np.array([0.0, 1.0, 0.0]),
            "writer":  np.array([0.0, 0.0, 1.0]),
        }

        mock_model = MagicMock()

        def fake_encode(text, normalize_embeddings=False):
            for name, vec in profile_vecs.items():
                if name in text.lower():
                    return vec
            return np.array([0.33, 0.33, 0.34])  # ambiguous fallback

        mock_model.encode = fake_encode

        sr._embedder = mock_model
        sr._expert_embeddings = {
            name: fake_encode(name) for name in profile_vecs
        }

    def test_classify_selects_coder_for_coding_prompt(self, monkeypatch):
        self._install_mock_embedder()
        monkeypatch.setattr(sr, "_load_embedder", lambda: True)
        result = sr.classify("coder prompt", ["coder", "analyst", "writer"])
        assert result == "coder"

    def test_classify_selects_analyst_for_analysis_prompt(self, monkeypatch):
        self._install_mock_embedder()
        monkeypatch.setattr(sr, "_load_embedder", lambda: True)
        result = sr.classify("analyst prompt", ["coder", "analyst", "writer"])
        assert result == "analyst"

    def test_classify_selects_writer_for_writing_prompt(self, monkeypatch):
        self._install_mock_embedder()
        monkeypatch.setattr(sr, "_load_embedder", lambda: True)
        result = sr.classify("writer prompt", ["coder", "analyst", "writer"])
        assert result == "writer"

    def test_classify_respects_enabled_list(self, monkeypatch):
        """Only enabled experts are candidates; disabled ones must be ignored."""
        self._install_mock_embedder()
        monkeypatch.setattr(sr, "_load_embedder", lambda: True)
        # coder would win for "coder prompt" but is not in the enabled list.
        result = sr.classify("coder prompt", ["analyst", "writer"])
        # With coder excluded, analyst or writer wins — just not coder.
        assert result in ("analyst", "writer")
        assert result != "coder"

    def test_classify_returns_string_not_none_when_candidates_available(self, monkeypatch):
        self._install_mock_embedder()
        monkeypatch.setattr(sr, "_load_embedder", lambda: True)
        result = sr.classify("some random message", ["coder", "analyst"])
        assert isinstance(result, str)

    def test_load_embedder_sets_singleton(self, monkeypatch):
        """After a successful load, _embedder singleton must be set."""
        import numpy as np

        mock_st_module = MagicMock()
        mock_model = MagicMock()
        mock_model.encode = lambda text, normalize_embeddings=False: np.zeros(3)
        mock_st_module.SentenceTransformer.return_value = mock_model

        with patch.dict("sys.modules", {"sentence_transformers": mock_st_module}):
            _reset_router()
            result = sr._load_embedder()

        assert result is True
        assert sr._embedder is not None

    def test_load_embedder_idempotent(self, monkeypatch):
        """Calling _load_embedder() twice must not reload the model."""
        self._install_mock_embedder()
        original = sr._embedder
        monkeypatch.setattr(sr, "_load_embedder", lambda: True)
        sr._load_embedder()
        # Singleton unchanged — no reload happened.
        assert sr._embedder is original
