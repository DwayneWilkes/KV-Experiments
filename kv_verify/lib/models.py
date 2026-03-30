"""Local-first model management.

Models and tokenizers are loaded from a configurable cache directory.
Resolution order: explicit cache_dir parameter > KV_VERIFY_MODEL_DIR env var > default.
No remote code execution. No implicit downloads during experiments.

Usage:
    from kv_verify.models import load_tokenizer, load_model, ensure_downloaded

    # Tokenizer only (fast, CPU, for prompt validation)
    tokenizer = load_tokenizer("qwen")

    # Full model (GPU, for inference)
    model, tokenizer = load_model("qwen")

    # Pre-download before experiments
    ensure_downloaded("qwen")

    # Custom cache directory
    model, tokenizer = load_model("qwen", cache_dir="/workspace/models")
"""

import os
from pathlib import Path
from typing import Optional, Tuple

_DEFAULT_CACHE_DIR = Path("/mnt/d/dev/models")


def get_cache_dir(cache_dir: Optional[Path] = None) -> Path:
    """Resolve model cache directory.

    Resolution: explicit cache_dir > KV_VERIFY_MODEL_DIR env var > default.
    """
    if cache_dir is not None:
        return Path(cache_dir)
    env = os.environ.get("KV_VERIFY_MODEL_DIR")
    if env:
        return Path(env)
    return _DEFAULT_CACHE_DIR

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


def _set_cache_dir(cache_dir: Optional[Path] = None):
    """Ensure HF_HOME points to the resolved cache directory."""
    os.environ["HF_HOME"] = str(get_cache_dir(cache_dir))


def is_downloaded(name_or_id: str, cache_dir: Optional[Path] = None) -> bool:
    """Check if a model is already in the local cache."""
    resolved = get_cache_dir(cache_dir)
    model_id = _resolve_model_id(name_or_id)
    cache_name = f"models--{model_id.replace('/', '--')}"
    model_dir = resolved / cache_name
    if not model_dir.exists():
        return False
    snapshots = model_dir / "snapshots"
    if not snapshots.exists():
        return False
    return any(snapshots.iterdir())


def ensure_downloaded(name_or_id: str, cache_dir: Optional[Path] = None) -> Path:
    """Download a model to local cache if not already present.

    Returns the snapshot path.
    """
    resolved = get_cache_dir(cache_dir)
    _set_cache_dir(resolved)
    model_id = _resolve_model_id(name_or_id)

    if is_downloaded(name_or_id, cache_dir=resolved):
        cache_name = f"models--{model_id.replace('/', '--')}"
        snapshots = resolved / cache_name / "snapshots"
        return next(snapshots.iterdir())

    from huggingface_hub import snapshot_download
    path = snapshot_download(model_id, cache_dir=str(resolved))
    return Path(path)


def _get_snapshot_path(name_or_id: str, cache_dir: Optional[Path] = None) -> Path:
    """Get the local snapshot path for a downloaded model.

    Uses the snapshot directory directly instead of the model ID,
    avoiding HuggingFace Hub network calls for resolution.
    """
    resolved = get_cache_dir(cache_dir)
    model_id = _resolve_model_id(name_or_id)
    cache_name = f"models--{model_id.replace('/', '--')}"
    snapshots = resolved / cache_name / "snapshots"
    if not snapshots.exists():
        raise RuntimeError(f"No snapshots for {model_id} at {snapshots}")
    return next(snapshots.iterdir())


def load_tokenizer(name_or_id: str = "qwen", cache_dir: Optional[Path] = None):
    """Load a tokenizer from local cache. No remote code execution.

    Fast, CPU-only. Uses local snapshot path directly (no network).
    """
    resolved = get_cache_dir(cache_dir)
    _set_cache_dir(resolved)

    if not is_downloaded(name_or_id, cache_dir=resolved):
        model_id = _resolve_model_id(name_or_id)
        raise RuntimeError(
            f"Model '{model_id}' not in local cache at {resolved}. "
            f"Run ensure_downloaded('{name_or_id}') first."
        )

    local_path = str(_get_snapshot_path(name_or_id, cache_dir=resolved))
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(local_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(name_or_id: str = "qwen", dtype=None, cache_dir: Optional[Path] = None):
    """Load model + tokenizer from local cache. No remote code execution.

    Uses local snapshot path directly (no network calls).
    Returns (model, tokenizer).
    """
    resolved = get_cache_dir(cache_dir)
    _set_cache_dir(resolved)

    if not is_downloaded(name_or_id, cache_dir=resolved):
        model_id = _resolve_model_id(name_or_id)
        raise RuntimeError(
            f"Model '{model_id}' not in local cache at {resolved}. "
            f"Run ensure_downloaded('{name_or_id}') first."
        )

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if dtype is None:
        dtype = torch.bfloat16

    local_path = str(_get_snapshot_path(name_or_id, cache_dir=resolved))

    tokenizer = AutoTokenizer.from_pretrained(local_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        local_path,
        dtype=dtype,
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
