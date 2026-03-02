"""
scripts/train_adapter.py
ExpertSwarm LoRA fine-tuning pipeline.

Uses HuggingFace Trainer + PEFT to fine-tune phi-2 on domain datasets,
then auto-computes the SHA-256 hash and updates manifest.json so the new
adapter is immediately usable without any manual steps.

Usage:
    python scripts/train_adapter.py --expert coder
    python scripts/train_adapter.py --expert analyst --epochs 3
    python scripts/train_adapter.py --expert writer --max_samples 2000 --lr 1e-4

Outputs:
    experts/<expert>/          ← saved adapter weights
    experts/manifest.json      ← sha256 + adapter_path updated automatically

Datasets (all public, Apache/CC-licensed):
    coder   → HuggingFaceH4/CodeAlpaca_20K   (Python instruction-following)
    analyst → gbharti/finance-alpaca          (finance instruction-following)
    writer  → tatsu-lab/alpaca                (filtered to writing prompts)

Hardware note:
    CPU training is functional but slow (~1 h per 1 000 samples on phi-2).
    For production fine-tuning use a GPU instance; the script auto-detects
    CUDA and enables mixed-precision training accordingly.
"""

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

# ---------------------------------------------------------------------------
# Dataset configuration per expert
# ---------------------------------------------------------------------------

_DATASET_CONFIG: dict[str, dict] = {
    "coder": {
        "hf_path":        "HuggingFaceH4/CodeAlpaca_20K",
        "split":          "train",
        "instruction_col": "instruction",
        "output_col":     "output",
    },
    "analyst": {
        "hf_path":        "gbharti/finance-alpaca",
        "split":          "train",
        "instruction_col": "instruction",
        "output_col":     "output",
    },
    "writer": {
        "hf_path":        "tatsu-lab/alpaca",
        "split":          "train",
        "instruction_col": "instruction",
        "output_col":     "output",
        # Only keep samples that are clearly writing tasks.
        "filter_keywords": [
            "write", "essay", "story", "poem", "creative", "draft",
            "narrative", "fiction", "rewrite", "compose",
        ],
    },
}

# LoRA hyper-parameters — match adapter_config.json in existing adapters.
_LORA_R          = 16
_LORA_ALPHA      = 32
_LORA_DROPOUT    = 0.05
_TARGET_MODULES  = ["q_proj", "k_proj", "v_proj", "dense"]
_MAX_SEQ_LENGTH  = 512


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _format_sample(instruction: str, output: str) -> str:
    """phi-2 Q/A prompt format — must match router._infer() format."""
    return f"Q: {instruction.strip()}\nA: {output.strip()}"


def load_and_prepare_dataset(expert: str, max_samples: int):
    """Download, filter, shuffle, cap, and format the training dataset."""
    try:
        from datasets import load_dataset
    except ImportError:
        log.error("'datasets' package not found. Install it: pip install datasets>=2.19.0")
        sys.exit(1)

    cfg = _DATASET_CONFIG[expert]
    log.info("Loading dataset '%s' (split=%s)…", cfg["hf_path"], cfg["split"])
    ds = load_dataset(cfg["hf_path"], split=cfg["split"])

    # Filter to domain-relevant samples for the writer expert.
    if "filter_keywords" in cfg:
        kws = cfg["filter_keywords"]
        ds = ds.filter(
            lambda x: any(kw in x[cfg["instruction_col"]].lower() for kw in kws)
        )
        log.info("After keyword filter: %d samples", len(ds))

    # Reproducible shuffle + cap.
    ds = ds.shuffle(seed=42).select(range(min(max_samples, len(ds))))
    log.info("Training on %d samples.", len(ds))

    # Format as Q/A text.
    ds = ds.map(
        lambda x: {"text": _format_sample(x[cfg["instruction_col"]], x[cfg["output_col"]])},
        remove_columns=ds.column_names,
    )
    return ds


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(expert: str, epochs: int, max_samples: int, learning_rate: float) -> Path:
    """
    Fine-tune phi-2 with LoRA on the specified expert dataset.

    Returns the path to the saved adapter directory.
    """
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    from core.base_model import DEFAULT_MODEL_ID

    adapter_dir = PROJECT_ROOT / "experts" / expert
    adapter_dir.mkdir(parents=True, exist_ok=True)

    use_cuda = torch.cuda.is_available()
    log.info(
        "Expert: %s | Model: %s | Device: %s | Epochs: %d | Samples: %d | LR: %s",
        expert, DEFAULT_MODEL_ID, "cuda" if use_cuda else "cpu",
        epochs, max_samples, learning_rate,
    )

    # --- Base model ----------------------------------------------------------
    log.info("Loading base model '%s'…", DEFAULT_MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_ID, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = dict(trust_remote_code=False)
    if use_cuda:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["dtype"] = torch.float32
        model_kwargs["device_map"] = "cpu"

    model = AutoModelForCausalLM.from_pretrained(DEFAULT_MODEL_ID, **model_kwargs)

    # --- LoRA ----------------------------------------------------------------
    lora_cfg = LoraConfig(
        r=_LORA_R,
        lora_alpha=_LORA_ALPHA,
        lora_dropout=_LORA_DROPOUT,
        bias="none",
        target_modules=_TARGET_MODULES,
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # --- Dataset + tokenization ----------------------------------------------
    raw_ds = load_and_prepare_dataset(expert, max_samples)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=_MAX_SEQ_LENGTH,
            padding=False,
        )

    tokenized_ds = raw_ds.map(tokenize, batched=True, remove_columns=["text"])
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # --- Training arguments --------------------------------------------------
    training_args = TrainingArguments(
        output_dir=str(adapter_dir / "_checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        warmup_steps=20,
        learning_rate=learning_rate,
        fp16=use_cuda,       # half-precision only on GPU
        bf16=False,
        logging_steps=10,
        save_strategy="no",  # we save manually below
        report_to="none",    # no wandb / tensorboard
        remove_unused_columns=False,
    )

    # --- Train ---------------------------------------------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds,
        data_collator=data_collator,
    )
    log.info("Training started…")
    trainer.train()

    # --- Save adapter only (not base weights) --------------------------------
    log.info("Saving adapter to '%s'…", adapter_dir)
    model.save_pretrained(str(adapter_dir))

    return adapter_dir


# ---------------------------------------------------------------------------
# Hash + manifest update
# ---------------------------------------------------------------------------

def _hash_directory(directory: Path) -> str:
    """Mirrors router._hash_directory exactly — must stay in sync."""
    sha = hashlib.sha256()
    for file_path in sorted(directory.rglob("*")):
        if file_path.is_file():
            sha.update(file_path.relative_to(directory).as_posix().encode())
            with file_path.open("rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    sha.update(chunk)
    return sha.hexdigest()


def update_manifest(expert: str, adapter_dir: Path) -> str:
    """Compute SHA-256 of the saved adapter and write it to manifest.json."""
    digest = _hash_directory(adapter_dir)

    manifest_path = PROJECT_ROOT / "experts" / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    manifest["experts"][expert]["sha256"]       = digest
    manifest["experts"][expert]["adapter_path"] = f"experts/{expert}"
    manifest["experts"][expert]["enabled"]      = True

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    log.info("SHA-256: %s", digest)
    log.info("manifest.json updated for '%s'.", expert)
    return digest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ExpertSwarm LoRA fine-tuning pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--expert", required=True, choices=list(_DATASET_CONFIG.keys()),
        help="Which expert adapter to train.",
    )
    parser.add_argument("--epochs",      type=int,   default=2,    help="Training epochs.")
    parser.add_argument("--max_samples", type=int,   default=1000, help="Max training samples.")
    parser.add_argument("--lr",          type=float, default=2e-4, help="Learning rate.")
    args = parser.parse_args()

    adapter_dir = train(args.expert, args.epochs, args.max_samples, args.lr)
    update_manifest(args.expert, adapter_dir)

    print(f"\n✅ Done. Verify with: python scripts/hash_adapter.py experts/{args.expert} --verify")


if __name__ == "__main__":
    main()
