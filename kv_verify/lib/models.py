"""Local-first model management.

All models and tokenizers are loaded from /mnt/d/dev/models (HF_HOME).
No remote code execution. No implicit downloads during experiments.

Usage:
    from kv_verify.models import load_tokenizer, load_model, ensure_downloaded

    # Tokenizer only (fast, CPU, for prompt validation)
    tokenizer = load_tokenizer("qwen")

    # Full model (GPU, for inference)
    model, tokenizer = load_model("qwen")

    # Pre-download before experiments
    ensure_downloaded("qwen")
"""

import os
from pathlib import Path
from typing import Optional, Tuple

MODEL_CACHE_DIR = Path("/mnt/d/dev/models")

# Canonical model IDs
MODELS = {
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "tiny": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
}


def _resolve_model_id(name_or_id: str) -> str:
    """Resolve a short name to a full HuggingFace model ID."""
    return MODELS.get(name_or_id, name_or_id)


def _set_cache_dir():
    """Ensure HF_HOME points to our local cache."""
    os.environ["HF_HOME"] = str(MODEL_CACHE_DIR)


def is_downloaded(name_or_id: str) -> bool:
    """Check if a model is already in the local cache."""
    model_id = _resolve_model_id(name_or_id)
    # HF hub stores models as models--{org}--{name}
    cache_name = f"models--{model_id.replace('/', '--')}"
    model_dir = MODEL_CACHE_DIR / cache_name
    if not model_dir.exists():
        return False
    # Check for at least one snapshot
    snapshots = model_dir / "snapshots"
    if not snapshots.exists():
        return False
    return any(snapshots.iterdir())


def ensure_downloaded(name_or_id: str) -> Path:
    """Download a model to local cache if not already present.

    Returns the snapshot path.
    """
    _set_cache_dir()
    model_id = _resolve_model_id(name_or_id)

    if is_downloaded(name_or_id):
        cache_name = f"models--{model_id.replace('/', '--')}"
        snapshots = MODEL_CACHE_DIR / cache_name / "snapshots"
        return next(snapshots.iterdir())

    from huggingface_hub import snapshot_download
    path = snapshot_download(model_id, cache_dir=str(MODEL_CACHE_DIR))
    return Path(path)


def load_tokenizer(name_or_id: str = "qwen"):
    """Load a tokenizer from local cache. No remote code execution.

    Fast, CPU-only. Use for prompt validation and token counting.
    """
    _set_cache_dir()
    model_id = _resolve_model_id(name_or_id)

    if not is_downloaded(name_or_id):
        raise RuntimeError(
            f"Model '{model_id}' not in local cache at {MODEL_CACHE_DIR}. "
            f"Run ensure_downloaded('{name_or_id}') first."
        )

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(name_or_id: str = "qwen", dtype=None):
    """Load model + tokenizer from local cache. No remote code execution.

    Requires GPU. Returns (model, tokenizer).
    """
    _set_cache_dir()
    model_id = _resolve_model_id(name_or_id)

    if not is_downloaded(name_or_id):
        raise RuntimeError(
            f"Model '{model_id}' not in local cache at {MODEL_CACHE_DIR}. "
            f"Run ensure_downloaded('{name_or_id}') first."
        )

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if dtype is None:
        dtype = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


def generate_text(
    model,
    tokenizer,
    prompt: str,
    system_prompt: str = "You are a helpful assistant.",
    max_new_tokens: int = 200,
    temperature: float = 0.0,
) -> str:
    """Generate text from a prompt. Returns generated text only."""
    import torch

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    n_input = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
        )

    return tokenizer.decode(outputs[0, n_input:], skip_special_tokens=True)
