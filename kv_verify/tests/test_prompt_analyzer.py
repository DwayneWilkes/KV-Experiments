"""Tests for kv_verify.prompt_analyzer — dataset quality analysis tool.

Uses a real tokenizer (downloaded separately, ~few MB). No proxies.
"""

import numpy as np
import pytest

from kv_verify.prompt_analyzer import (
    PromptStats,
    PairSetAnalysis,
    analyze_pair_set,
    analyze_length_distribution,
    analyze_pair_balance,
    analyze_vocabulary_overlap,
    analyze_semantic_clusters,
    coverage_gaps,
)
from kv_verify.prompt_gen import MinimalPair, PairSet


@pytest.fixture(scope="module")
def tokenizer():
    """Load a real tokenizer. Uses Qwen if available, falls back to GPT-2."""
    import os
    os.environ.setdefault("HF_HOME", "/mnt/d/dev/models")
    try:
        from transformers import AutoTokenizer
        # Try Qwen first (our target model)
        try:
            tok = AutoTokenizer.from_pretrained(
                "Qwen/Qwen2.5-7B-Instruct"            )
        except Exception:
            # Fall back to GPT-2 tokenizer (always available, small download)
            tok = AutoTokenizer.from_pretrained("gpt2")
        return tok
    except ImportError:
        pytest.skip("transformers not installed")


def _make_test_pairs(n=20):
    """Create test pairs with known properties."""
    pairs = []
    for i in range(n):
        pos = f"Answer this question with a wrong answer: What is {i} plus {i}?"
        neg = f"Answer this question in a truthful way: What is {i} plus {i}?"
        pairs.append(MinimalPair(
            positive=pos, negative=neg, pair_id=f"t{i:03d}",
            comparison="test", template="test_v1",
        ))
    return PairSet(comparison="test", pairs=pairs, template="test_v1", n_target=n)


class TestPromptStats:
    def test_has_exact_token_count(self, tokenizer):
        stats = PromptStats.from_text("Hello world, this is a test.", tokenizer)
        assert isinstance(stats.token_count, int)
        assert stats.token_count > 0
        # Token count should differ from word count for most tokenizers
        assert stats.char_count == 28

    def test_word_count(self, tokenizer):
        stats = PromptStats.from_text("This is a test sentence with seven words.", tokenizer)
        assert stats.word_count == 8

    def test_unique_words(self, tokenizer):
        stats = PromptStats.from_text("the cat sat on the mat", tokenizer)
        assert stats.unique_word_count == 5

    def test_type_token_ratio(self, tokenizer):
        stats = PromptStats.from_text("the the the cat", tokenizer)
        assert stats.type_token_ratio == 0.5


class TestAnalyzeLengthDistribution:
    def test_includes_token_stats(self, tokenizer):
        ps = _make_test_pairs(10)
        result = analyze_length_distribution(ps, tokenizer)
        for side in ["positive", "negative"]:
            assert "token_mean" in result[side]
            assert "token_std" in result[side]
            assert "token_min" in result[side]
            assert "token_max" in result[side]

    def test_token_diffs(self, tokenizer):
        ps = _make_test_pairs(10)
        result = analyze_length_distribution(ps, tokenizer)
        assert "pair_token_diff_mean" in result
        assert "pair_token_diff_max" in result
        assert "token_diffs" in result
        assert len(result["token_diffs"]) == 10


class TestAnalyzePairBalance:
    def test_uses_tokens_not_words(self, tokenizer):
        ps = _make_test_pairs(10)
        result = analyze_pair_balance(ps, tokenizer)
        assert "token_diff_distribution" in result
        assert "pairs_exact_match" in result
        assert "pairs_within_1_token" in result
        assert "pairs_within_3_tokens" in result

    def test_reports_outlier_pairs(self, tokenizer):
        pairs = [
            MinimalPair(
                positive="short",
                negative="this is a much longer prompt with many more tokens in it than the other",
                pair_id="outlier", comparison="test", template="t",
            ),
        ]
        ps = PairSet(comparison="test", pairs=pairs, template="t", n_target=1)
        result = analyze_pair_balance(ps, tokenizer)
        assert result["pairs_outside_3_tokens"] > 0
        assert len(result["outlier_pairs"]) > 0


class TestAnalyzeVocabularyOverlap:
    def test_overlap_score(self):
        ps = _make_test_pairs(10)
        result = analyze_vocabulary_overlap(ps)
        assert "jaccard_mean" in result
        assert 0 <= result["jaccard_mean"] <= 1

    def test_high_overlap_for_minimal_pairs(self):
        ps = _make_test_pairs(10)
        result = analyze_vocabulary_overlap(ps)
        assert result["jaccard_mean"] > 0.5


class TestAnalyzeSemanticClusters:
    def test_returns_cluster_info(self):
        ps = _make_test_pairs(20)
        result = analyze_semantic_clusters(ps, n_clusters=3)
        assert "n_clusters" in result
        assert "cluster_sizes" in result
        assert len(result["cluster_sizes"]) == 3
        assert "balance_score" in result
        assert 0 <= result["balance_score"] <= 1


class TestCoverageGaps:
    def test_uses_token_counts(self, tokenizer):
        ps = _make_test_pairs(10)
        result = coverage_gaps(ps, tokenizer)
        assert "short_pairs" in result
        assert "long_pairs" in result
        assert "token_histogram" in result
        assert "recommendations" in result


class TestAnalyzePairSet:
    def test_full_analysis(self, tokenizer):
        ps = _make_test_pairs(20)
        analysis = analyze_pair_set(ps, tokenizer)
        assert isinstance(analysis, PairSetAnalysis)
        assert analysis.n_pairs == 20

    def test_to_dict(self, tokenizer):
        ps = _make_test_pairs(10)
        analysis = analyze_pair_set(ps, tokenizer)
        d = analysis.to_dict()
        assert isinstance(d, dict)
        assert d["n_pairs"] == 10
        # token_diffs should be removed from serialization
        assert "token_diffs" not in d.get("length_distribution", {})

    def test_summary_mentions_tokens(self, tokenizer):
        ps = _make_test_pairs(10)
        analysis = analyze_pair_set(ps, tokenizer)
        summary = analysis.summary()
        assert "token" in summary.lower()
        assert "pairs" in summary.lower()
