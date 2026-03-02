"""
core/semantic_router.py
Embedding-based expert classifier using sentence-transformers (all-MiniLM-L6-v2).

Design:
  - Loads lazily on first call; startup is never blocked.
  - If sentence-transformers is not installed or the model fails to load,
    classify() returns None and the caller falls back to keyword matching.
  - Expert profiles are short natural-language descriptions used as the
    reference embeddings; cosine similarity selects the best match.
  - Only experts that appear in both the manifest AND _EXPERT_PROFILES are
    considered; unknown experts (e.g. "base") are skipped cleanly.
"""

import logging
from typing import Optional

log = logging.getLogger(__name__)

_MODEL_NAME = "all-MiniLM-L6-v2"

# Reference description for each domain expert.
# Keep these aligned with manifest descriptions for predictable routing.
_EXPERT_PROFILES: dict[str, str] = {
    "coder": (
        "Write code, debug programs, generate Python functions, fix bugs, "
        "SQL queries, JavaScript, software development, scripting, algorithms"
    ),
    "analyst": (
        "Data analysis, summarise reports, analyse trends, structured "
        "reasoning, statistics, business intelligence, financial data, insights"
    ),
    "writer": (
        "Long-form writing, creative writing, essays, editing, storytelling, "
        "drafting, rewriting, content creation, copywriting, narrative"
    ),
}

# Module-level singletons — populated on first successful load.
_embedder = None
_expert_embeddings: dict = {}


def _load_embedder() -> bool:
    """
    Load the SentenceTransformer model once.  Thread-safe for reads after the
    first load because module-level assignment is the GIL unit in CPython.

    Returns True on success, False if the library is unavailable.
    """
    global _embedder, _expert_embeddings

    if _embedder is not None:
        return True

    try:
        from sentence_transformers import SentenceTransformer

        log.info("Loading semantic router model '%s'…", _MODEL_NAME)
        model = SentenceTransformer(_MODEL_NAME)

        embeddings = {
            name: model.encode(desc, normalize_embeddings=True)
            for name, desc in _EXPERT_PROFILES.items()
        }

        # Commit atomically — other threads see either old (None) or fully loaded.
        _expert_embeddings = embeddings
        _embedder = model
        log.info("Semantic router ready.")
        return True

    except Exception as exc:
        log.warning(
            "Semantic router unavailable (%s). Falling back to keyword matching.", exc
        )
        return False


def classify(prompt: str, enabled_experts: list[str]) -> Optional[str]:
    """
    Return the best-matching enabled expert name for *prompt* using cosine
    similarity over MiniLM embeddings.

    Args:
        prompt:          Raw user input string.
        enabled_experts: Names of currently enabled experts from the manifest.

    Returns:
        Expert name string, or None if:
          - sentence-transformers is unavailable
          - none of the enabled experts have a profile entry
    """
    if not _load_embedder():
        return None

    # Only score experts that have a profile AND are enabled in the manifest.
    candidates = [e for e in enabled_experts if e in _EXPERT_PROFILES]
    if not candidates:
        return None

    import numpy as np

    prompt_emb = _embedder.encode(prompt, normalize_embeddings=True)

    scores = {
        name: float(np.dot(prompt_emb, _expert_embeddings[name]))
        for name in candidates
    }
    best = max(scores, key=scores.get)
    log.debug("Semantic scores: %s → '%s'", scores, best)
    return best
