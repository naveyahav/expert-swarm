"""
tests/test_router_route.py
Unit tests for router.route() — routing logic with all heavy dependencies mocked.

No model weights are loaded. _get_model, _infer, validate_expert, and
adapter_loader functions are all monkeypatched to fast stubs.

Run with:
    pytest tests/test_router_route.py -v
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import router
import core.adapter_loader as adapter_loader  # imported so monkeypatch can target it

# ---------------------------------------------------------------------------
# Reusable fake objects
# ---------------------------------------------------------------------------

_FAKE_MODEL = MagicMock(name="base_model")
_FAKE_TOKENIZER = MagicMock(name="tokenizer")
_FAKE_PEFT_MODEL = MagicMock(name="peft_model")


# ---------------------------------------------------------------------------
# Fixture manifests
# ---------------------------------------------------------------------------

_BASE_MANIFEST = {
    "experts": {
        "base": {
            "description": "Base model, no adapter.",
            "adapter_path": None,
            "sha256": None,
            "enabled": True,
        }
    }
}

_ADAPTER_MANIFEST = {
    "experts": {
        "coder": {
            "description": "Code expert.",
            "adapter_path": "experts/coder",
            "sha256": "a" * 64,
            "enabled": True,
        }
    }
}

_DISABLED_MANIFEST = {
    "experts": {
        "base": {
            "description": "Disabled expert.",
            "adapter_path": None,
            "sha256": None,
            "enabled": False,
        }
    }
}

_MULTI_EXPERT_MANIFEST = {
    "experts": {
        "coder": {
            "description": "Coder.",
            "adapter_path": None,
            "sha256": None,
            "enabled": True,
        },
        "analyst": {
            "description": "Analyst.",
            "adapter_path": None,
            "sha256": None,
            "enabled": True,
        },
    }
}


# ---------------------------------------------------------------------------
# Base expert (no adapter) tests
# ---------------------------------------------------------------------------

class TestRouteBaseExpert:

    def test_returns_infer_output(self, monkeypatch):
        monkeypatch.setattr(router, "load_manifest", lambda: _BASE_MANIFEST)
        monkeypatch.setattr(router, "_get_model", lambda mid: (_FAKE_MODEL, _FAKE_TOKENIZER))
        monkeypatch.setattr(router, "_infer", lambda m, t, p: "base response")
        assert router.route("hello", expert="base") == "base response"

    def test_skips_validate_expert(self, monkeypatch):
        """validate_expert must NOT be called when adapter_path is None."""
        called = []
        monkeypatch.setattr(router, "load_manifest", lambda: _BASE_MANIFEST)
        monkeypatch.setattr(router, "_get_model", lambda mid: (_FAKE_MODEL, _FAKE_TOKENIZER))
        monkeypatch.setattr(router, "_infer", lambda m, t, p: "ok")
        monkeypatch.setattr(router, "validate_expert", lambda path, h: called.append(1) or True)
        router.route("hello", expert="base")
        assert called == []

    def test_passes_prompt_to_infer(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(router, "load_manifest", lambda: _BASE_MANIFEST)
        monkeypatch.setattr(router, "_get_model", lambda mid: (_FAKE_MODEL, _FAKE_TOKENIZER))
        monkeypatch.setattr(router, "_infer", lambda m, t, p: captured.update({"prompt": p}) or "ok")
        router.route("my exact prompt", expert="base")
        assert captured["prompt"] == "my exact prompt"


# ---------------------------------------------------------------------------
# Error path tests
# ---------------------------------------------------------------------------

class TestRouteErrorPaths:

    def test_unknown_expert_returns_error(self, monkeypatch):
        monkeypatch.setattr(router, "load_manifest", lambda: _BASE_MANIFEST)
        result = router.route("hello", expert="ghost")
        assert "Unknown expert" in result
        assert "ghost" in result

    def test_disabled_expert_returns_error(self, monkeypatch):
        monkeypatch.setattr(router, "load_manifest", lambda: _DISABLED_MANIFEST)
        result = router.route("hello", expert="base")
        assert "disabled" in result.lower()

    def test_no_enabled_expert_returns_error(self, monkeypatch):
        monkeypatch.setattr(router, "load_manifest", lambda: {"experts": {}})
        result = router.route("hello")
        assert "No enabled expert" in result

    def test_hash_failure_returns_security_error(self, monkeypatch):
        monkeypatch.setattr(router, "load_manifest", lambda: _ADAPTER_MANIFEST)
        monkeypatch.setattr(router, "_get_model", lambda mid: (_FAKE_MODEL, _FAKE_TOKENIZER))
        monkeypatch.setattr(router, "validate_expert", lambda path, h: False)
        result = router.route("hello", expert="coder")
        assert "Security check failed" in result
        assert "coder" in result


# ---------------------------------------------------------------------------
# Adapter (LoRA) path tests
# ---------------------------------------------------------------------------

class TestRouteAdapterPath:

    def test_happy_path_returns_response(self, monkeypatch):
        monkeypatch.setattr(router, "load_manifest", lambda: _ADAPTER_MANIFEST)
        monkeypatch.setattr(router, "_get_model", lambda mid: (_FAKE_MODEL, _FAKE_TOKENIZER))
        monkeypatch.setattr(router, "validate_expert", lambda path, h: True)
        monkeypatch.setattr(adapter_loader, "load_adapter", lambda base, path: _FAKE_PEFT_MODEL)
        monkeypatch.setattr(adapter_loader, "unload_adapter", lambda m: None)
        monkeypatch.setattr(router, "_infer", lambda m, t, p: "coder response")
        assert router.route("write a function", expert="coder") == "coder response"

    def test_load_adapter_called_with_correct_path(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(router, "load_manifest", lambda: _ADAPTER_MANIFEST)
        monkeypatch.setattr(router, "_get_model", lambda mid: (_FAKE_MODEL, _FAKE_TOKENIZER))
        monkeypatch.setattr(router, "validate_expert", lambda path, h: True)
        monkeypatch.setattr(adapter_loader, "load_adapter",
                            lambda base, path: captured.update({"path": path}) or _FAKE_PEFT_MODEL)
        monkeypatch.setattr(adapter_loader, "unload_adapter", lambda m: None)
        monkeypatch.setattr(router, "_infer", lambda m, t, p: "ok")
        router.route("hello", expert="coder")
        assert Path(captured["path"]) == router.PROJECT_ROOT / "experts" / "coder"

    def test_unload_adapter_called_after_infer(self, monkeypatch):
        """Adapter must be unloaded even after a successful inference."""
        unloaded = []
        monkeypatch.setattr(router, "load_manifest", lambda: _ADAPTER_MANIFEST)
        monkeypatch.setattr(router, "_get_model", lambda mid: (_FAKE_MODEL, _FAKE_TOKENIZER))
        monkeypatch.setattr(router, "validate_expert", lambda path, h: True)
        monkeypatch.setattr(adapter_loader, "load_adapter", lambda base, path: _FAKE_PEFT_MODEL)
        monkeypatch.setattr(adapter_loader, "unload_adapter", lambda m: unloaded.append(m))
        monkeypatch.setattr(router, "_infer", lambda m, t, p: "ok")
        router.route("hello", expert="coder")
        assert len(unloaded) == 1
        assert unloaded[0] is _FAKE_PEFT_MODEL

    def test_infer_receives_peft_model_not_base(self, monkeypatch):
        """_infer must be called with the PeftModel, not the base model."""
        received = {}
        monkeypatch.setattr(router, "load_manifest", lambda: _ADAPTER_MANIFEST)
        monkeypatch.setattr(router, "_get_model", lambda mid: (_FAKE_MODEL, _FAKE_TOKENIZER))
        monkeypatch.setattr(router, "validate_expert", lambda path, h: True)
        monkeypatch.setattr(adapter_loader, "load_adapter", lambda base, path: _FAKE_PEFT_MODEL)
        monkeypatch.setattr(adapter_loader, "unload_adapter", lambda m: None)
        monkeypatch.setattr(router, "_infer",
                            lambda m, t, p: received.update({"model": m}) or "ok")
        router.route("hello", expert="coder")
        assert received["model"] is _FAKE_PEFT_MODEL


# ---------------------------------------------------------------------------
# Intent classifier (auto-routing) tests
# ---------------------------------------------------------------------------

class TestRouteAutoSelect:

    def test_keyword_routes_to_coder(self, monkeypatch):
        monkeypatch.setattr(router, "load_manifest", lambda: _MULTI_EXPERT_MANIFEST)
        monkeypatch.setattr(router, "_get_model", lambda mid: (_FAKE_MODEL, _FAKE_TOKENIZER))
        captured = {}
        monkeypatch.setattr(router, "_infer",
                            lambda m, t, p: captured.update({"expert_called": True}) or "ok")
        result = router.route("write a python script")
        # Keyword "python" → coder; must not fall through to an error
        assert result == "ok"

    def test_no_expert_kwarg_uses_classifier(self, monkeypatch):
        monkeypatch.setattr(router, "load_manifest", lambda: _MULTI_EXPERT_MANIFEST)
        monkeypatch.setattr(router, "_get_model", lambda mid: (_FAKE_MODEL, _FAKE_TOKENIZER))
        monkeypatch.setattr(router, "_infer", lambda m, t, p: "classified response")
        result = router.route("analyse this dataset")
        assert result == "classified response"

    def test_fallback_to_first_enabled_expert(self, monkeypatch):
        """No keyword match → fall back to first enabled expert rather than erroring."""
        monkeypatch.setattr(router, "load_manifest", lambda: _BASE_MANIFEST)
        monkeypatch.setattr(router, "_get_model", lambda mid: (_FAKE_MODEL, _FAKE_TOKENIZER))
        monkeypatch.setattr(router, "_infer", lambda m, t, p: "fallback response")
        result = router.route("tell me something interesting")
        assert result == "fallback response"
