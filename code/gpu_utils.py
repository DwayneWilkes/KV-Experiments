"""
Shared GPU utilities for KV-Cache experiments.
Provides consistent model identification, output naming, and model loading
across all experiment scripts for multi-model campaigns.

Used by: 01d, 03, 03b, 04, 05, 06
"""

import re
import torch
from pathlib import Path


# --- Monkey-patch non-iterable cache types (HybridCache for Gemma-2, etc.) ---
# HybridCache extends Cache (not DynamicCache) and lacks __iter__/__getitem__.
# This makes it behave identically to DynamicCache: yields (key, value) tuples.
try:
    from transformers.cache_utils import HybridCache
    if not hasattr(HybridCache, "__iter__"):
        def _hybrid_cache_iter(self):
            for i in range(len(self.key_cache)):
                yield (self.key_cache[i], self.value_cache[i])
        HybridCache.__iter__ = _hybrid_cache_iter
    if not hasattr(HybridCache, "__len__"):
        HybridCache.__len__ = lambda self: len(self.key_cache)
    if not hasattr(HybridCache, "__getitem__"):
        HybridCache.__getitem__ = lambda self, idx: (self.key_cache[idx], self.value_cache[idx])
except ImportError:
    pass


def normalize_cache(cache):
    """Convert any cache type to a list of (key, value) tuples.

    Handles DynamicCache, HybridCache, and legacy tuple-of-tuples.
    Use this when you need an explicit list rather than relying on iteration.
    """
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        return [(cache.key_cache[i], cache.value_cache[i])
                for i in range(len(cache.key_cache))]
    return list(cache)


def model_id_from_name(model_name: str) -> str:
    """Extract a short, filesystem-safe identifier from a HuggingFace model name.

    Examples:
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0" -> "TinyLlama-1.1B"
        "Qwen/Qwen2.5-7B-Instruct"           -> "Qwen2.5-7B"
        "Qwen/Qwen3-0.6B"                    -> "Qwen3-0.6B"
        "meta-llama/Llama-3.1-8B-Instruct"   -> "Llama-3.1-8B"
        "meta-llama/Llama-3.1-70B-Instruct"  -> "Llama-3.1-70B"
        "Qwen/Qwen2.5-0.5B-Instruct"        -> "Qwen2.5-0.5B"
    """
    # Take the part after the slash (org/model -> model)
    name = model_name.split("/")[-1]

    # Remove common suffixes
    for suffix in ["-Chat-v1.0", "-Chat-v1.1", "-Chat", "-Instruct", "-Base"]:
        name = name.replace(suffix, "")

    # For TinyLlama, simplify the redundant prefix
    if name.startswith("TinyLlama-"):
        name = "TinyLlama-1.1B"

    return name


def get_output_path(results_dir: Path, base_name: str, model_name: str,
                    quantize: bool = False) -> Path:
    """Build a per-model output path for experiment results.

    Args:
        results_dir: Directory where results are stored
        base_name: Base experiment name (e.g., "scale_sweep", "deception_forensics")
        model_name: Full HuggingFace model name
        quantize: Whether 4-bit quantization was used

    Returns:
        Path like results/deception_forensics_Qwen2.5-7B_results.json
        or     results/scale_sweep_Qwen2.5-32B-q4_results.json

    Examples:
        get_output_path(Path("results"), "scale_sweep", "Qwen/Qwen2.5-7B-Instruct")
        -> Path("results/scale_sweep_Qwen2.5-7B_results.json")

        get_output_path(Path("results"), "scale_sweep", "Qwen/Qwen2.5-32B-Instruct", quantize=True)
        -> Path("results/scale_sweep_Qwen2.5-32B-q4_results.json")
    """
    model_id = model_id_from_name(model_name)
    if quantize:
        model_id += "-q4"
    return results_dir / f"{base_name}_{model_id}_results.json"


def load_model(model_name: str, quantize: bool = False):
    """Unified model loading with optional 4-bit quantization.

    Works on single GPU or multi-GPU via device_map="auto".
    Identical to the load_model() already in scripts 03-06,
    centralized here for consistency.

    Args:
        model_name: HuggingFace model name/path
        quantize: If True, use NF4 4-bit quantization via bitsandbytes

    Returns:
        (model, tokenizer) tuple
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"Loading {model_name} (quantize={quantize})...")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if quantize:
        qconfig = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=qconfig,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

    n_layers = model.config.num_hidden_layers
    n_kv_heads = getattr(model.config, "num_key_value_heads", "?")
    print(f"  Loaded: {n_layers} layers, {n_kv_heads} KV heads")
    return model, tokenizer


def compute_cache_dimensionality(cache, variance_threshold: float = 0.9) -> dict:
    """Compute effective dimensionality of KV-cache states via SVD.

    For each layer, performs SVD on the key and value matrices to determine:
    - Effective rank: minimum dimensions capturing `variance_threshold` of total variance
    - Spectral entropy: normalized entropy of singular value distribution (0=one dominant
      component, 1=uniform spread)
    - Rank ratio: effective rank / total dimensions

    This tests H6 from Paper B (Geometry of Belief Death): deception should narrow
    effective dimensionality (expand the null space), while grounded cognition
    should use more of the available representational space.

    Args:
        cache: Model's past_key_values (tuple of (key, value) tuples per layer)
        variance_threshold: Cumulative variance fraction for effective rank (default 0.9)

    Returns:
        Dict with per-layer and aggregate dimensionality metrics.
    """
    import numpy as np

    layer_dims = []
    for layer_idx, layer in enumerate(cache):
        if not (isinstance(layer, tuple) and len(layer) >= 2):
            continue

        key = layer[0].float().cpu()
        value = layer[1].float().cpu()

        # Reshape to 2D: (batch * heads * seq_len, head_dim)
        key_2d = key.reshape(-1, key.shape[-1])
        value_2d = value.reshape(-1, value.shape[-1])

        layer_dim = {"layer": layer_idx}

        for name, matrix in [("key", key_2d), ("value", value_2d)]:
            _, s, _ = torch.linalg.svd(matrix, full_matrices=False)
            s_squared = s ** 2
            total_var = s_squared.sum()

            if total_var > 0:
                # Effective rank: min dimensions for variance_threshold of variance
                cumvar = torch.cumsum(s_squared, dim=0) / total_var
                eff_rank = int((cumvar < variance_threshold).sum().item()) + 1

                # Spectral entropy: distribution uniformity of singular values
                probs = s_squared / total_var
                probs = probs[probs > 0]
                entropy = -float((probs * torch.log2(probs)).sum())
                max_entropy = float(torch.log2(torch.tensor(len(s), dtype=torch.float)))
                norm_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

                layer_dim[f"{name}_effective_rank"] = eff_rank
                layer_dim[f"{name}_spectral_entropy"] = float(norm_entropy)
                layer_dim[f"{name}_rank_ratio"] = float(eff_rank / len(s))
                layer_dim[f"{name}_total_dim"] = int(len(s))
            else:
                layer_dim[f"{name}_effective_rank"] = 0
                layer_dim[f"{name}_spectral_entropy"] = 0.0
                layer_dim[f"{name}_rank_ratio"] = 0.0
                layer_dim[f"{name}_total_dim"] = int(len(s))

        layer_dims.append(layer_dim)

    if layer_dims:
        key_ranks = [d["key_effective_rank"] for d in layer_dims]
        value_ranks = [d["value_effective_rank"] for d in layer_dims]
        key_entropies = [d["key_spectral_entropy"] for d in layer_dims]
        value_entropies = [d["value_spectral_entropy"] for d in layer_dims]
        key_ratios = [d["key_rank_ratio"] for d in layer_dims]
        value_ratios = [d["value_rank_ratio"] for d in layer_dims]

        return {
            "mean_key_effective_rank": float(np.mean(key_ranks)),
            "mean_value_effective_rank": float(np.mean(value_ranks)),
            "mean_key_spectral_entropy": float(np.mean(key_entropies)),
            "mean_value_spectral_entropy": float(np.mean(value_entropies)),
            "mean_key_rank_ratio": float(np.mean(key_ratios)),
            "mean_value_rank_ratio": float(np.mean(value_ratios)),
            "key_rank_by_layer": key_ranks,
            "value_rank_by_layer": value_ranks,
            "key_entropy_by_layer": key_entropies,
            "value_entropy_by_layer": value_entropies,
            "per_layer": layer_dims,
        }

    return {
        "mean_key_effective_rank": 0, "mean_value_effective_rank": 0,
        "mean_key_spectral_entropy": 0.0, "mean_value_spectral_entropy": 0.0,
        "mean_key_rank_ratio": 0.0, "mean_value_rank_ratio": 0.0,
    }


def compute_subspace_alignment(cache1, cache2, top_k=None):
    """Compute alignment between principal subspaces of two KV-caches.

    For each layer, computes principal angles between the top-k right singular
    subspaces of each cache's key matrix. The right singular vectors live in
    head_dim space (consistent across different sequence lengths), so caches
    from different prompts are directly comparable.

    Returns mean cosine of principal angles: 1 = identical subspaces,
    0 = fully orthogonal. This measures whether two cognitive states use
    the *same representational directions*, regardless of magnitude.

    Args:
        cache1: First model cache (past_key_values), can be on CPU or GPU
        cache2: Second model cache, can have different sequence length
        top_k: Number of principal components to compare (default: half of max)

    Returns:
        Dict with mean/std alignment across layers, plus per-layer detail.
    """
    import numpy as np

    alignments = []
    for layer_idx, (l1, l2) in enumerate(zip(cache1, cache2)):
        if not (isinstance(l1, tuple) and len(l1) >= 2 and
                isinstance(l2, tuple) and len(l2) >= 2):
            continue

        # Reshape to 2D: (batch*heads*seq, head_dim)
        k1 = l1[0].float().cpu().reshape(-1, l1[0].shape[-1])
        k2 = l2[0].float().cpu().reshape(-1, l2[0].shape[-1])

        # Right singular vectors (Vt rows = directions in head_dim space)
        _, _, Vt1 = torch.linalg.svd(k1, full_matrices=False)
        _, _, Vt2 = torch.linalg.svd(k2, full_matrices=False)

        max_k = min(Vt1.shape[0], Vt2.shape[0])
        k = min(top_k, max_k) if top_k else max(1, max_k // 2)

        # Top-k directions transposed to (head_dim, k)
        V1 = Vt1[:k, :].T
        V2 = Vt2[:k, :].T

        # Principal angles via SVD of V1^T @ V2
        _, sigmas, _ = torch.linalg.svd(V1.T @ V2, full_matrices=False)
        cosines = torch.clamp(sigmas, 0.0, 1.0)

        alignments.append({
            "layer": layer_idx,
            "mean_cosine": float(cosines.mean()),
            "min_cosine": float(cosines.min()),
            "top_k": k,
        })

    if alignments:
        layer_means = [a["mean_cosine"] for a in alignments]
        return {
            "mean_alignment": float(np.mean(layer_means)),
            "std_alignment": float(np.std(layer_means)),
            "min_layer_alignment": float(min(layer_means)),
            "per_layer": alignments,
        }
    return {"mean_alignment": 0.0, "std_alignment": 0.0}


def cache_to_cpu(cache):
    """Deep-copy cache tensors to CPU and detach from computation graph."""
    return tuple(
        tuple(t.detach().cpu() for t in layer) if isinstance(layer, tuple) else layer
        for layer in cache
    )
