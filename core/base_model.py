"""
core/base_model.py
Loads the base model with automatic GPU/CPU detection and CPU threading optimisation.

  - CUDA GPU present  → 4-bit NF4 quantization via bitsandbytes (fast, low VRAM)
  - CPU / Intel GPU   → float32, all physical cores used via PyTorch thread pool
"""

import logging
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

log = logging.getLogger(__name__)

# Default model identifier — override via env var or config in production.
DEFAULT_MODEL_ID = os.environ.get("EXPERTSWARM_MODEL_ID", "microsoft/phi-2")


def _cuda_available() -> bool:
    return torch.cuda.is_available()


def _configure_cpu_threads() -> None:
    """
    Maximise CPU inference throughput by dedicating all logical cores to
    PyTorch's intra-op thread pool.  Called once at model load time.
    """
    n = os.cpu_count() or 1
    torch.set_num_threads(n)
    torch.set_num_interop_threads(max(1, n // 2))
    log.info("CPU thread pool: intra=%d  interop=%d", n, max(1, n // 2))


def load_base_model(model_id: str = DEFAULT_MODEL_ID):
    """
    Load the base model and tokenizer with automatic hardware detection.

    GPU path  : 4-bit NF4 quantization via bitsandbytes. Requires NVIDIA CUDA.
    CPU path  : Full float32 weights, all cores utilised via thread-pool tuning.

    Returns:
        tuple[PreTrainedModel, PreTrainedTokenizer]
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

    # Ensure pad token exists (required for clean generation without warnings).
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if _cuda_available():
        log.info("CUDA detected — loading '%s' in 4-bit NF4 mode.", model_id)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=False,
        )
    else:
        _configure_cpu_threads()
        log.warning(
            "No CUDA GPU found — loading '%s' on CPU in float32. "
            "All %d logical cores allocated to inference.",
            model_id,
            os.cpu_count() or 1,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=False,
        )

    return model, tokenizer
