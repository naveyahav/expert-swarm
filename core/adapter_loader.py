"""
core/adapter_loader.py
Merges a validated LoRA adapter onto the base model using PEFT.
Called exclusively by router.py AFTER validate_expert() has passed.
"""

from peft import PeftModel


def load_adapter(base_model, adapter_path: str) -> PeftModel:
    """
    Attach a LoRA adapter to the base model.

    Args:
        base_model: A loaded HuggingFace base model.
        adapter_path: Absolute path to the adapter directory.
                      Must be pre-validated by router.validate_expert().

    Returns:
        PeftModel with the adapter merged.
    """
    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        is_trainable=False,  # Inference only — no weight updates allowed.
    )
    return model


def unload_adapter(peft_model: PeftModel):
    """
    Merge adapter weights back into the base model and discard the adapter.
    Frees VRAM between expert switches.

    Returns:
        The merged base model (no longer a PeftModel).
    """
    return peft_model.merge_and_unload()
