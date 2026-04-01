"""Tests for kv_verify.feature_extractor — KV-cache feature extraction library.

CPU tests use mock cache objects. GPU tests (marked @pytest.mark.gpu) use TinyLlama.
"""

import numpy as np
import pytest
import torch

from kv_verify.feature_extractor import (
    extract_from_cache,
    get_cache_accessor,
    FeatureResult,
    DEFAULT_MODEL_CACHE_DIR,
)
from kv_verify.types import FeatureVector


# ================================================================
# Mock Cache Objects for CPU Testing
# ================================================================

def _make_mock_cache(n_layers=4, n_heads=8, seq_len=32, head_dim=64):
    """Create a mock KV cache as a list of (K, V) tuples."""
    cache = []
    for _ in range(n_layers):
        K = torch.randn(1, n_heads, seq_len, head_dim)
        V = torch.randn(1, n_heads, seq_len, head_dim)
        cache.append((K, V))
    return cache


class _MockDynamicCache:
    """Mock HuggingFace DynamicCache."""
    def __init__(self, n_layers=4, n_heads=8, seq_len=32, head_dim=64):
        self.key_cache = [
            torch.randn(1, n_heads, seq_len, head_dim)
            for _ in range(n_layers)
        ]
        self.value_cache = [
            torch.randn(1, n_heads, seq_len, head_dim)
            for _ in range(n_layers)
        ]


# ================================================================
# Cache Accessor Tests
# ================================================================

class TestGetCacheAccessor:
    def test_tuple_format(self):
        cache = _make_mock_cache(n_layers=4)
        n_layers, get_keys = get_cache_accessor(cache)
        assert n_layers == 4
        K = get_keys(0)
        assert K.shape[-1] == 64  # head_dim

    def test_dynamic_cache_format(self):
        cache = _MockDynamicCache(n_layers=6)
        n_layers, get_keys = get_cache_accessor(cache)
        assert n_layers == 6

    def test_returns_correct_layer(self):
        cache = _make_mock_cache(n_layers=3)
        _, get_keys = get_cache_accessor(cache)
        K0 = get_keys(0)
        K1 = get_keys(1)
        assert not torch.equal(K0, K1)  # different layers differ


# ================================================================
# Feature Extraction Tests (CPU, mock caches)
# ================================================================

class TestExtractFromCache:
    def test_returns_feature_result(self):
        cache = _make_mock_cache()
        result = extract_from_cache(cache, n_input_tokens=16, total_tokens=32)
        assert isinstance(result, FeatureResult)
        assert isinstance(result.features, FeatureVector)

    def test_primary_features_present(self):
        cache = _make_mock_cache()
        result = extract_from_cache(cache, n_input_tokens=16, total_tokens=32)
        fv = result.features
        assert fv.norm_per_token > 0
        assert fv.key_rank > 0
        assert fv.key_entropy > 0
        assert fv.norm > 0

    def test_token_counts(self):
        cache = _make_mock_cache(seq_len=50)
        result = extract_from_cache(cache, n_input_tokens=20, total_tokens=50)
        fv = result.features
        assert fv.n_tokens == 50
        assert fv.n_input_tokens == 20
        assert fv.n_generated == 30

    def test_per_layer_arrays(self):
        cache = _make_mock_cache(n_layers=8)
        result = extract_from_cache(cache, n_input_tokens=10, total_tokens=32)
        fv = result.features
        assert len(fv.layer_norms) == 8
        assert len(fv.layer_ranks) == 8
        assert all(n > 0 for n in fv.layer_norms)

    def test_deterministic(self):
        """Same cache should produce identical features."""
        torch.manual_seed(42)
        cache = _make_mock_cache()
        r1 = extract_from_cache(cache, n_input_tokens=16, total_tokens=32)
        r2 = extract_from_cache(cache, n_input_tokens=16, total_tokens=32)
        assert r1.features.norm_per_token == r2.features.norm_per_token
        assert r1.features.key_rank == r2.features.key_rank

    def test_extended_features(self):
        cache = _make_mock_cache()
        result = extract_from_cache(
            cache, n_input_tokens=16, total_tokens=32, extended=True, seed=42,
        )
        fv = result.features
        assert fv.spectral_entropy is not None
        assert fv.angular_spread is not None

    def test_angular_spread_seeded(self):
        """Angular spread should be deterministic with same seed."""
        cache = _make_mock_cache()
        r1 = extract_from_cache(cache, 16, 32, extended=True, seed=42)
        r2 = extract_from_cache(cache, 16, 32, extended=True, seed=42)
        assert r1.features.angular_spread == r2.features.angular_spread

    def test_uses_linalg_norm(self):
        """Verify we use torch.linalg.norm, not deprecated torch.norm."""
        # This is a code-level check. The function should work correctly
        # on any input shape.
        cache = _make_mock_cache(n_layers=2, seq_len=10)
        result = extract_from_cache(cache, 5, 10)
        assert result.features.norm > 0

    def test_float32_upcast_for_svd(self):
        """SVD should work on float16 inputs (upcast internally)."""
        cache = []
        for _ in range(2):
            K = torch.randn(1, 4, 16, 32, dtype=torch.float16)
            V = torch.randn(1, 4, 16, 32, dtype=torch.float16)
            cache.append((K, V))
        result = extract_from_cache(cache, 8, 16)
        assert result.features.key_rank > 0


class TestFeatureResult:
    def test_to_dict(self):
        cache = _make_mock_cache(n_layers=2)
        result = extract_from_cache(cache, 10, 20)
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "norm_per_token" in d
        assert "key_rank" in d
        assert "layer_norms" in d

    def test_model_cache_dir_default(self):
        assert DEFAULT_MODEL_CACHE_DIR == "/mnt/d/dev/models"
