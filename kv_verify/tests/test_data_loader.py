"""Tests for kv_verify.data_loader — loads per-item features from hackathon JSONs."""

import numpy as np

from kv_verify.data_loader import load_comparison_data, list_comparisons
from kv_verify.fixtures import EXP47_COMPARISONS, PRIMARY_FEATURES


class TestListComparisons:
    def test_returns_10(self):
        names = list_comparisons()
        assert len(names) == 10

    def test_known_names(self):
        names = list_comparisons()
        assert "exp31_refusal_vs_benign" in names
        assert "exp18b_deception" in names
        assert "exp39_sycophancy" in names


class TestLoadComparisonData:
    def test_exp31_shape(self):
        X, y, meta = load_comparison_data("exp31_refusal_vs_benign")
        assert X.shape[0] == 40  # 20 + 20
        assert X.shape[1] == 3   # norm_per_token, key_rank, key_entropy
        assert y.shape == (40,)
        assert set(y) == {0, 1}

    def test_exp31_labels(self):
        X, y, meta = load_comparison_data("exp31_refusal_vs_benign")
        assert sum(y == 1) == 20  # positive class
        assert sum(y == 0) == 20  # negative class

    def test_exp18b_deception_paired(self):
        X, y, meta = load_comparison_data("exp18b_deception")
        assert X.shape[0] == 20  # 10 + 10
        assert meta["paired"] is True
        assert "prompt_indices_pos" in meta
        assert "prompt_indices_neg" in meta

    def test_exp32_jailbreak_vs_normal(self):
        X, y, meta = load_comparison_data("exp32_jailbreak_vs_normal")
        assert X.shape[0] == 40  # 20 + 20
        assert meta["paired"] is False

    def test_exp32_jailbreak_vs_refusal(self):
        X, y, meta = load_comparison_data("exp32_jailbreak_vs_refusal")
        assert X.shape[0] == 40

    def test_exp33_llama(self):
        X, y, meta = load_comparison_data("exp33_Llama-3.1-8B-Instruct")
        assert X.shape[0] == 40

    def test_exp33_mistral(self):
        X, y, meta = load_comparison_data("exp33_Mistral-7B-Instruct-v0.3")
        assert X.shape[0] == 40

    def test_exp36_impossible_vs_benign(self):
        X, y, meta = load_comparison_data("exp36_impossible_vs_benign")
        assert X.shape[0] == 40

    def test_exp36_impossible_vs_harmful(self):
        X, y, meta = load_comparison_data("exp36_impossible_vs_harmful")
        assert X.shape[0] == 40

    def test_exp39_sycophancy(self):
        X, y, meta = load_comparison_data("exp39_sycophancy")
        assert X.shape[0] == 40
        assert meta["paired"] is True

    def test_features_are_numeric(self):
        X, y, meta = load_comparison_data("exp31_refusal_vs_benign")
        assert not np.any(np.isnan(X))
        assert not np.any(np.isinf(X))

    def test_meta_has_confounds(self):
        """Meta should include length confounds for FWL."""
        X, y, meta = load_comparison_data("exp31_refusal_vs_benign")
        assert "confounds" in meta
        Z = meta["confounds"]
        assert Z.shape[0] == X.shape[0]
        assert Z.shape[1] >= 1  # at least norm

    def test_all_comparisons_loadable(self):
        """Every comparison in EXP47_COMPARISONS must load without error."""
        for comp in EXP47_COMPARISONS:
            X, y, meta = load_comparison_data(comp["name"])
            assert X.shape[0] > 0
            assert len(y) == X.shape[0]
