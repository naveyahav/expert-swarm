"""
router.py — ExpertSwarm core orchestrator.

Responsibilities:
  1. Load and parse the adapter manifest.
  2. Validate every adapter via SHA-256 before it touches the model (Zero Trust).
  3. Route an incoming prompt to the correct expert and return a response.

Security contract:
  - validate_expert() MUST return True before any adapter is loaded.
  - Adapter paths are resolved and checked against the project root to
    prevent directory-traversal attacks.
  - No network calls are made; this is a fully local-first system.
"""

import hashlib
import hmac
import json
import logging
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# Project root is the directory that contains this file.
PROJECT_ROOT = Path(__file__).parent.resolve()
MANIFEST_PATH = PROJECT_ROOT / "experts" / "manifest.json"

# ---------------------------------------------------------------------------
# Model singleton cache — loaded once per process, reused across calls.
# Keys: model_id (str) → {"model": ..., "tokenizer": ...}
# ---------------------------------------------------------------------------
_MODEL_CACHE: dict = {}


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def load_manifest() -> dict:
    """Parse and return the expert manifest."""
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")
    with MANIFEST_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Security — SHA-256 validation (Zero Trust core)
# ---------------------------------------------------------------------------

def _hash_directory(directory: Path) -> str:
    """
    Compute a deterministic SHA-256 hash over every file in *directory*.

    Files are sorted by relative path so the hash is stable regardless of
    filesystem enumeration order.  Only regular files are hashed; directories
    themselves are not included.

    Returns:
        Hex-encoded SHA-256 digest string.
    """
    sha = hashlib.sha256()
    for file_path in sorted(directory.rglob("*")):
        if file_path.is_file():
            # Include the relative path in the hash so renaming a file
            # invalidates the digest even if its contents are unchanged.
            sha.update(file_path.relative_to(directory).as_posix().encode())
            with file_path.open("rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    sha.update(chunk)
    return sha.hexdigest()


def validate_expert(expert_path: str, expected_hash: str) -> bool:
    """
    Verify the SHA-256 hash of a LoRA adapter directory.

    Args:
        expert_path: Path to the adapter directory (relative or absolute).
        expected_hash: The SHA-256 hex digest recorded in manifest.json.

    Returns:
        True  — hash matches; adapter is safe to load.
        False — hash mismatch or path violation; adapter must NOT be loaded.

    Security notes:
        - Resolves symlinks and checks the path stays within PROJECT_ROOT to
          prevent directory-traversal attacks.
        - A placeholder hash ("REPLACE_WITH_ACTUAL_SHA256_*") is always
          rejected so unconfigured entries can never be loaded.
    """
    # --- Path safety check --------------------------------------------------
    resolved = Path(expert_path).resolve()
    try:
        resolved.relative_to(PROJECT_ROOT)
    except ValueError:
        log.error(
            "Directory traversal blocked: '%s' is outside project root.", expert_path
        )
        return False

    if not resolved.is_dir():
        log.error("Adapter path does not exist or is not a directory: '%s'", resolved)
        return False

    # --- Reject placeholder hashes ------------------------------------------
    if not expected_hash or expected_hash.upper().startswith("REPLACE_WITH"):
        log.error("Adapter '%s' has a placeholder hash — refusing to load.", expert_path)
        return False

    # --- Hash comparison (constant-time via hmac to prevent timing attacks) --
    actual_hash = _hash_directory(resolved)
    if not hmac.compare_digest(actual_hash, expected_hash.lower()):
        log.error(
            "SHA-256 mismatch for '%s'.\n  expected: %s\n  actual:   %s",
            expert_path,
            expected_hash,
            actual_hash,
        )
        return False

    log.info("Adapter validated: '%s'", expert_path)
    return True


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def _select_expert(prompt: str, manifest: dict) -> str | None:
    """
    Two-stage intent classifier:
      1. Semantic matching via MiniLM cosine similarity (core/semantic_router.py).
         Falls back silently if sentence-transformers is unavailable.
      2. Keyword rules — fast deterministic fallback.
      3. First enabled expert — last resort.
    """
    experts = manifest.get("experts", {})
    enabled = [n for n, e in experts.items() if e.get("enabled", False)]

    # --- Stage 1: Semantic routing -------------------------------------------
    from core.semantic_router import classify
    best = classify(prompt, enabled)
    if best is not None:
        return best

    # --- Stage 2: Keyword fallback -------------------------------------------
    prompt_lower = prompt.lower()
    rules = {
        "coder":   ["code", "function", "bug", "debug", "script", "sql", "python", "javascript"],
        "analyst": ["analyse", "analyze", "summarise", "summarize", "data", "trend", "report"],
        "writer":  ["write", "essay", "draft", "edit", "rewrite", "story"],
    }
    for expert_name, keywords in rules.items():
        entry = experts.get(expert_name, {})
        if not entry.get("enabled", False):
            continue
        if any(kw in prompt_lower for kw in keywords):
            return expert_name

    # --- Stage 3: First enabled expert ---------------------------------------
    for name, entry in experts.items():
        if entry.get("enabled", False):
            return name

    return None


def _get_model(model_id: str):
    """
    Return a cached (base_model, tokenizer) pair, loading on first call.
    The cache lives for the lifetime of the process — in Streamlit this means
    the model is loaded once and reused across all chat turns and reruns.
    """
    if model_id not in _MODEL_CACHE:
        from core.base_model import load_base_model
        log.info("Loading base model '%s' into cache…", model_id)
        model, tokenizer = load_base_model(model_id)
        _MODEL_CACHE[model_id] = {"model": model, "tokenizer": tokenizer}
    return _MODEL_CACHE[model_id]["model"], _MODEL_CACHE[model_id]["tokenizer"]


def _infer_stream(model, tokenizer, prompt: str):
    """
    Like _infer() but yields tokens one-by-one using TextIteratorStreamer.
    Generation runs in a background thread so the caller can iterate tokens
    in real time — used by route_stream() for streaming UI responses.
    """
    import threading
    from transformers import TextIteratorStreamer

    formatted = f"Q: {prompt}\nA:"
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )
    thread = threading.Thread(
        target=model.generate,
        kwargs=dict(
            **inputs,
            max_new_tokens=512,
            streamer=streamer,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
        ),
        daemon=True,
    )
    thread.start()
    for token in streamer:
        yield token
    thread.join()


def _infer(model, tokenizer, prompt: str) -> str:
    """Run a forward pass and return only the newly generated tokens.

    Applies the phi-2 instruct prefix so the model responds as an assistant
    rather than continuing the prompt as a multiple-choice question.

    Slices output_ids by input length instead of stripping decoded text, so
    the result is always clean regardless of how the model echoes the prompt.
    """
    formatted = f"Q: {prompt}\nA:"
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    input_length = inputs["input_ids"].shape[1]
    output_ids = model.generate(
        **inputs,
        max_new_tokens=512,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False,
    )
    # Decode only the tokens generated *after* the input prompt.
    new_tokens = output_ids[0][input_length:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def route(prompt: str, expert: Optional[str] = None) -> str:
    """
    Public entry point.  Select an expert, validate it, load it, and infer.

    Args:
        prompt: Raw user input string.
        expert: If provided, skip intent classification and use this expert
                directly.  The name must match a key in manifest.json.

    Returns:
        Model response string, or an error message if routing fails.
    """
    from core.base_model import DEFAULT_MODEL_ID

    manifest = load_manifest()

    # Resolve which expert to use.
    if expert is not None:
        if expert not in manifest.get("experts", {}):
            return f"Unknown expert '{expert}'. Check manifest.json."
        expert_name = expert
    else:
        expert_name = _select_expert(prompt, manifest)

    if expert_name is None:
        return "No enabled expert found for this prompt."

    expert_meta = manifest["experts"][expert_name]

    if not expert_meta.get("enabled", False):
        return f"Expert '{expert_name}' is disabled in manifest.json."

    log.info("Routing to expert: '%s'", expert_name)

    base_model, tokenizer = _get_model(DEFAULT_MODEL_ID)

    # --- Base mode: no adapter, infer directly on the base model. -----------
    if expert_meta.get("adapter_path") is None:
        return _infer(base_model, tokenizer, prompt)

    # --- Adapter mode: validate hash, load LoRA, infer, then unload. --------
    from core.adapter_loader import load_adapter, unload_adapter

    adapter_path = str(PROJECT_ROOT / expert_meta["adapter_path"])
    expected_hash = expert_meta["sha256"]

    if not validate_expert(adapter_path, expected_hash):
        return (
            f"Security check failed for expert '{expert_name}'. "
            "Adapter not loaded."
        )

    model = load_adapter(base_model, adapter_path)
    response = _infer(model, tokenizer, prompt)
    unload_adapter(model)
    return response


# ---------------------------------------------------------------------------
# Streaming public entry point
# ---------------------------------------------------------------------------

def route_stream(prompt: str, expert: Optional[str] = None):
    """
    Like route() but yields tokens incrementally via TextIteratorStreamer.
    Use this for streaming UIs (e.g. st.write_stream in web_app.py).

    Yields:
        str — individual decoded token strings as they are generated.
              On error yields a single error message string and returns.
    """
    from core.base_model import DEFAULT_MODEL_ID

    manifest = load_manifest()

    if expert is not None:
        if expert not in manifest.get("experts", {}):
            yield f"Unknown expert '{expert}'. Check manifest.json."
            return
        expert_name = expert
    else:
        expert_name = _select_expert(prompt, manifest)

    if expert_name is None:
        yield "No enabled expert found for this prompt."
        return

    expert_meta = manifest["experts"][expert_name]

    if not expert_meta.get("enabled", False):
        yield f"Expert '{expert_name}' is disabled in manifest.json."
        return

    log.info("Streaming route to expert: '%s'", expert_name)

    base_model, tokenizer = _get_model(DEFAULT_MODEL_ID)

    # Base mode — stream directly from base model.
    if expert_meta.get("adapter_path") is None:
        yield from _infer_stream(base_model, tokenizer, prompt)
        return

    # Adapter mode — validate, load, stream, unload.
    from core.adapter_loader import load_adapter, unload_adapter

    adapter_path = str(PROJECT_ROOT / expert_meta["adapter_path"])
    expected_hash = expert_meta["sha256"]

    if not validate_expert(adapter_path, expected_hash):
        yield f"Security check failed for expert '{expert_name}'. Adapter not loaded."
        return

    model = load_adapter(base_model, adapter_path)
    yield from _infer_stream(model, tokenizer, prompt)
    unload_adapter(model)


# ---------------------------------------------------------------------------
# CLI entry point (for quick local testing)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    user_prompt = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Hello, swarm."
    print(route(user_prompt))
