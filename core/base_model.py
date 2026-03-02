"""
core/base_model.py
Loads the base model with automatic GPU/CPU detection.

  - CUDA GPU present  → 4-bit NF4 quantization via bitsandbytes (fast, low VRAM)
  - CPU / Intel GPU   → float32, no quantization (slow but functional)
"""

import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

log = logging.getLogger(__name__)

# Default model identifier — override via env var or config in production.
DEFAULT_MODEL_ID = "microsoft/phi-2"


def _cuda_available() -> bool:
    return torch.cuda.is_available()


def load_base_model(model_id: str = DEFAULT_MODEL_ID):
    """
    Load the base model and tokenizer with automatic hardware detection.

    GPU path  : 4-bit NF4 quantization via bitsandbytes. Requires NVIDIA CUDA.
    CPU path  : Full float32 weights, no quantization. Works on any hardware
                but is significantly slower and uses more RAM.

    Returns:
        tuple[PreTrainedModel, PreTrainedTokenizer]
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

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
        log.warning(
            "No CUDA GPU found — loading '%s' on CPU in float32. "
            "Inference will be slow. Consider a smaller model for local use.",
            model_id,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=False,
        )

    return model, tokenizer
