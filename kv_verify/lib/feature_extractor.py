"""KV-cache feature extraction library (cache-inspector).

General-purpose library for extracting geometric features from the KV-cache
of any HuggingFace transformer model. Handles DynamicCache, HybridCache,
and legacy tuple formats.

Corrections applied vs original codebase:
- torch.linalg.norm (not deprecated torch.norm)
- float32 upcast for SVD stability
- Seeded angular_spread for determinism
- Warning on SVD failure (not silent default)

Feature naming follows concordance convention:
  norm_per_token   = total_frobenius_norm / total_tokens
  key_rank         = mean effective rank across layers (= eff_rank)
  key_entropy      = mean spectral entropy across layers (= layer_norm_entropy)
  norm             = total Frobenius norm

Not KV-cache-experiment-specific. Usable for any HF model inspection.

References:
- Consolidated from concordance/features.py, 49_expanded_validation.py,
  50_cache_dynamics.py
"""

import math
from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from kv_verify.lib.models import get_cache_dir
from kv_verify.types import FeatureVector

# Re-export for backward compatibility
DEFAULT_MODEL_CACHE_DIR = str(get_cache_dir())


@dataclass
class FeatureResult:
    """Result of feature extraction, wrapping FeatureVector with extras."""
    features: FeatureVector
    layer_entropies: List[float]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self.features)
        d["layer_entropies"] = self.layer_entropies
        return d


# ================================================================
# Cache Accessor
# ================================================================

def get_cache_accessor(past_key_values) -> Tuple[int, Callable]:
    """Return (n_layers, get_keys_fn) for any HF cache format.

    Handles:
    - DynamicCache: has .key_cache attribute (list of tensors)
    - HybridCache: has .layers attribute with .keys property
    - Legacy tuples: list of (K, V) tensor pairs

    get_keys(layer_idx) returns the key tensor for that layer.
    """
    if hasattr(past_key_values, "key_cache"):
        # DynamicCache format
        n_layers = len(past_key_values.key_cache)
        get_keys = lambda i: past_key_values.key_cache[i]
    elif hasattr(past_key_values, "layers"):
        # HybridCache format
        n_layers = len(past_key_values.layers)
        get_keys = lambda i: past_key_values.layers[i].keys
    else:
        # Legacy tuple format: list of (K, V)
        n_layers = len(past_key_values)
        get_keys = lambda i: past_key_values[i][0]
    return n_layers, get_keys


# ================================================================
# Core Feature Extraction
# ================================================================

def extract_from_cache(
    past_key_values,
    n_input_tokens: int,
    total_tokens: int,
    extended: bool = False,
    seed: int = 42,
) -> FeatureResult:
    """Extract geometric features from a KV cache.

    Args:
        past_key_values: HuggingFace cache object (any format)
        n_input_tokens: number of input/prompt tokens
        total_tokens: total tokens (input + generated)
        extended: if True, compute angular_spread and spectral_entropy
        seed: RNG seed for angular_spread (determinism)

    Returns:
        FeatureResult with FeatureVector and per-layer entropies
    """
    n_layers, get_keys = get_cache_accessor(past_key_values)

    total_norm_sq = 0.0
    layer_norms: List[float] = []
    layer_ranks: List[float] = []
    layer_entropies: List[float] = []
    angular_spread_val: Optional[float] = None

    for li in range(n_layers):
        K = get_keys(li)
        # Float32 upcast for SVD stability
        K = K.float().squeeze(0)  # [heads, seq, dim]
        K_flat = K.reshape(-1, K.shape[-1])  # [heads*seq, dim]

        # Frobenius norm (corrected: torch.linalg.norm)
        ln = float(torch.linalg.norm(K_flat).item())
        layer_norms.append(ln)
        total_norm_sq += ln ** 2

        # SVD for effective rank and spectral entropy
        try:
            S = torch.linalg.svdvals(K_flat)
            S = S[S > 1e-10]
            S_sq = S ** 2
            S_sq_norm = S_sq / S_sq.sum()
            entropy = float(-torch.sum(
                S_sq_norm * torch.log(S_sq_norm + 1e-12)
            ).item())
            eff_rank = float(torch.exp(torch.tensor(entropy)).item())
        except Exception as e:
            import warnings
            warnings.warn(f"SVD failed at layer {li}: {e}. Using defaults.")
            entropy = 0.0
            eff_rank = 1.0

        layer_ranks.append(eff_rank)
        layer_entropies.append(entropy)

    total_norm = float(math.sqrt(total_norm_sq))
    norm_per_token = total_norm / max(total_tokens, 1)
    mean_rank = float(np.mean(layer_ranks))
    mean_entropy = float(np.mean(layer_entropies))
    n_generated = total_tokens - n_input_tokens

    # Extended features
    spectral_entropy_val: Optional[float] = None
    if extended:
        spectral_entropy_val = mean_entropy
        angular_spread_val = _compute_angular_spread(
            past_key_values, n_layers, get_keys, seed=seed
        )

    fv = FeatureVector(
        norm_per_token=norm_per_token,
        key_rank=mean_rank,
        key_entropy=mean_entropy,
        norm=total_norm,
        n_tokens=total_tokens,
        n_generated=n_generated,
        n_input_tokens=n_input_tokens,
        layer_norms=layer_norms,
        layer_ranks=layer_ranks,
        spectral_entropy=spectral_entropy_val,
        angular_spread=angular_spread_val,
    )

    return FeatureResult(features=fv, layer_entropies=layer_entropies)


def _compute_angular_spread(
    past_key_values,
    n_layers: int,
    get_keys: Callable,
    seed: int = 42,
    n_pairs: int = 50,
) -> float:
    """Compute angular spread across random key vector pairs.

    Corrected: uses seeded RNG and excludes self-pairs.
    """
    gen = torch.Generator()
    gen.manual_seed(seed)

    cosines = []
    for li in range(n_layers):
        K = get_keys(li).float().squeeze(0)  # [heads, seq, dim]
        K_flat = K.reshape(-1, K.shape[-1])  # [N, dim]
        N = K_flat.shape[0]
        if N < 2:
            continue

        # Sample pairs (excluding self-pairs)
        actual_pairs = min(n_pairs, N * (N - 1) // 2)
        idx_a = torch.randint(0, N, (actual_pairs,), generator=gen)
        # Exclude self-pairs: offset by random 1..N-1
        offsets = torch.randint(1, N, (actual_pairs,), generator=gen)
        idx_b = (idx_a + offsets) % N

        va = K_flat[idx_a]
        vb = K_flat[idx_b]

        # Cosine similarity
        norms_a = torch.linalg.norm(va, dim=1, keepdim=True).clamp(min=1e-10)
        norms_b = torch.linalg.norm(vb, dim=1, keepdim=True).clamp(min=1e-10)
        cos = torch.sum((va / norms_a) * (vb / norms_b), dim=1)
        cosines.extend(cos.tolist())

    if not cosines:
        return 0.0

    # Angular spread = std of cosine similarities
    return float(np.std(cosines))
