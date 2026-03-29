"""Tests for F02: Held-Out Prompt Generalization Under Input-Length Control.

Tests data loading, residualization logic, and end-to-end execution.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression

os.environ.setdefault("TMPDIR", "/tmp/claude-1000")

from kv_verify.experiments.f02_held_out_input_control import (
    HACKATHON_DIR,
    PARADIGMS,
    _analyze_paradigm,
    _compute_input_length_stats,
    _extract_input_lengths,
    _feature_correlations,
    _get_input_tokens,
    _get_word_count,
    _load_test_data,
    _load_train_data,
    _paradigm_verdict,
    _residualize_train_test,
    run_f02,
)
from kv_verify.fixtures import PRIMARY_FEATURES
from kv_verify.stats import extract_feature_matrix, loo_auroc, train_test_auroc
from kv_verify.types import Verdict


# ================================================================
# UNIT TESTS: DATA LOADING
# ================================================================

class TestGetInputTokens:
    def test_from_n_input_tokens(self):
        item = {"features": {"n_input_tokens": 42, "n_tokens": 92, "n_generated": 50}}
        assert _get_input_tokens(item) == 42.0

    def test_from_subtraction(self):
        item = {"features": {"n_tokens": 96, "n_generated": 49}}
        assert _get_input_tokens(item) == 47.0

    def test_returns_float(self):
        item = {"features": {"n_tokens": 50, "n_generated": 30}}
        result = _get_input_tokens(item)
        assert isinstance(result, float)


class TestGetWordCount:
    def test_user_prompt_key(self):
        item = {"user_prompt": "one two three four five"}
        assert _get_word_count(item, "user_prompt") == 5

    def test_prompt_key(self):
        item = {"prompt": "hello world"}
        assert _get_word_count(item, "prompt") == 2

    def test_fallback_to_prompt(self):
        item = {"prompt": "a b c"}
        assert _get_word_count(item, "user_prompt") == 3


class TestExtractFeatures:
    def test_shape(self):
        items = [
            {"features": {"norm_per_token": 1.0, "key_rank": 2.0, "key_entropy": 3.0}},
            {"features": {"norm_per_token": 4.0, "key_rank": 5.0, "key_entropy": 6.0}},
        ]
        X = extract_feature_matrix(items)
        assert X.shape == (2, 3)
        np.testing.assert_array_equal(X[0], [1.0, 2.0, 3.0])

    def test_dtype(self):
        items = [{"features": {"norm_per_token": 1, "key_rank": 2, "key_entropy": 3}}]
        X = extract_feature_matrix(items)
        assert X.dtype == np.float64


class TestExtractInputLengths:
    def test_shape(self):
        items = [
            {"features": {"n_tokens": 50, "n_generated": 20}},
            {"features": {"n_tokens": 60, "n_generated": 30}},
        ]
        Z = _extract_input_lengths(items)
        assert Z.shape == (2, 1)
        assert Z[0, 0] == 30.0
        assert Z[1, 0] == 30.0


# ================================================================
# UNIT TESTS: RESIDUALIZATION
# ================================================================

class TestResidualization:
    def test_residualize_removes_linear_confound(self):
        """After residualization, features should be uncorrelated with Z."""
        rng = np.random.RandomState(42)
        n = 40
        Z = rng.randn(n, 1)
        X = Z * np.array([0.9, 0.7, 0.5]) + rng.randn(n, 3) * 0.1

        # Split into train/test
        X_train, X_test = X[:30], X[30:]
        Z_train, Z_test = Z[:30], Z[30:]

        X_tr_resid, X_te_resid, r2 = _residualize_train_test(
            X_train, Z_train, X_test, Z_test,
        )

        # R-squared should be high (Z explains most variance)
        assert all(r > 0.5 for r in r2)

        # Residuals on training data should be uncorrelated with Z
        for j in range(3):
            r, _ = pearsonr(X_tr_resid[:, j], Z_train.ravel())
            assert abs(r) < 0.05  # essentially zero

    def test_residualize_preserves_shape(self):
        X_train = np.random.randn(20, 3)
        Z_train = np.random.randn(20, 1)
        X_test = np.random.randn(10, 3)
        Z_test = np.random.randn(10, 1)

        X_tr_r, X_te_r, r2 = _residualize_train_test(X_train, Z_train, X_test, Z_test)
        assert X_tr_r.shape == (20, 3)
        assert X_te_r.shape == (10, 3)
        assert len(r2) == 3

    def test_residualize_no_confound_is_identity(self):
        """When Z is unrelated to X, residualization changes X minimally."""
        rng = np.random.RandomState(42)
        X_train = rng.randn(100, 3)
        Z_train = rng.randn(100, 1)
        X_test = rng.randn(20, 3)
        Z_test = rng.randn(20, 1)

        X_tr_r, X_te_r, r2 = _residualize_train_test(X_train, Z_train, X_test, Z_test)

        # R-squared should be near zero
        assert all(r < 0.1 for r in r2)


# ================================================================
# UNIT TESTS: CLASSIFICATION HELPERS
# ================================================================

class TestLOOAuroc:
    def test_separable_data(self):
        rng = np.random.RandomState(42)
        X = np.vstack([rng.randn(10, 2) + 2, rng.randn(10, 2) - 2])
        y = np.array([1] * 10 + [0] * 10)
        auroc = loo_auroc(X, y)
        assert auroc > 0.80

    def test_random_data_near_chance(self):
        # LOO with n=20 is highly noisy. Use n=60 for stable chance-level result.
        rng = np.random.RandomState(42)
        X = rng.randn(60, 2)
        y = np.array([1] * 30 + [0] * 30)
        auroc = loo_auroc(X, y)
        assert 0.2 < auroc < 0.8

    def test_returns_float(self):
        rng = np.random.RandomState(42)
        X = rng.randn(10, 2)
        y = np.array([1] * 5 + [0] * 5)
        assert isinstance(loo_auroc(X, y), float)


class TestTrainTestAuroc:
    def test_perfect_separation(self):
        X_train = np.array([[0], [0], [0], [0], [0], [10], [10], [10], [10], [10]], dtype=float)
        y_train = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        X_test = np.array([[1], [11]], dtype=float)
        y_test = np.array([0, 1])
        auroc = train_test_auroc(X_train, y_train, X_test, y_test)
        assert auroc == 1.0


class TestFeatureCorrelations:
    def test_high_correlation(self):
        rng = np.random.RandomState(42)
        Z = rng.randn(30, 1)
        X = np.hstack([Z * 0.9, Z * 0.8, Z * 0.7]) + rng.randn(30, 3) * 0.01
        corr = _feature_correlations(X, Z)
        assert len(corr) == 3
        for feat in PRIMARY_FEATURES:
            assert abs(corr[feat]["r"]) > 0.9


# ================================================================
# INTEGRATION TESTS: DATA LOADING
# ================================================================

class TestLoadTrainData:
    @pytest.mark.parametrize("paradigm", ["deception", "refusal", "impossibility"])
    def test_loads_items(self, paradigm):
        pos, neg = _load_train_data(paradigm)
        assert len(pos) >= 10
        assert len(neg) >= 10

    @pytest.mark.parametrize("paradigm", ["deception", "refusal", "impossibility"])
    def test_items_have_features(self, paradigm):
        pos, neg = _load_train_data(paradigm)
        for item in pos + neg:
            assert "features" in item
            for f in PRIMARY_FEATURES:
                assert f in item["features"]


class TestLoadTestData:
    @pytest.mark.parametrize("paradigm", ["deception", "refusal", "impossibility"])
    def test_loads_10_per_condition(self, paradigm):
        pos, neg = _load_test_data(paradigm)
        assert len(pos) == 10
        assert len(neg) == 10

    def test_deceptive_prompts_have_append(self):
        """Deceptive held-out prompts should include the deception instruction."""
        pos, _ = _load_test_data("deception")
        for item in pos:
            assert "confidently wrong" in item["prompt"]


class TestInputLengthStats:
    def test_deception_held_out_length_diff(self):
        """Held-out deceptive prompts should be longer than honest."""
        pos, neg = _load_test_data("deception")
        pos_input = np.array([_get_input_tokens(r) for r in pos])
        neg_input = np.array([_get_input_tokens(r) for r in neg])
        # Deceptive prompts append ~11 tokens
        assert pos_input.mean() > neg_input.mean()


# ================================================================
# INTEGRATION TESTS: PARADIGM ANALYSIS
# ================================================================

class TestAnalyzeParadigm:
    @pytest.mark.parametrize("paradigm", ["deception", "refusal", "impossibility"])
    def test_returns_complete_result(self, paradigm):
        result = _analyze_paradigm(paradigm)
        required_keys = [
            "paradigm", "paper_transfer_auroc",
            "train_length_stats", "test_length_stats",
            "baseline_transfer_auroc", "input_only_auroc_heldout_loo",
            "residualized_transfer_auroc", "auroc_drop",
            "r_squared_per_feature",
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    @pytest.mark.parametrize("paradigm", ["deception", "refusal", "impossibility"])
    def test_auroc_values_valid(self, paradigm):
        result = _analyze_paradigm(paradigm)
        assert 0.0 <= result["baseline_transfer_auroc"] <= 1.0
        assert 0.0 <= result["residualized_transfer_auroc"] <= 1.0
        assert 0.0 <= result["input_only_auroc_heldout_loo"] <= 1.0


# ================================================================
# INTEGRATION TESTS: VERDICT LOGIC
# ================================================================

class TestParadigmVerdict:
    def test_falsified_on_large_drop(self):
        """A paradigm with high input-only AUROC and low residualized should be FALSIFIED."""
        result = {
            "paradigm": "test",
            "baseline_transfer_auroc": 0.85,
            "residualized_transfer_auroc": 0.50,
            "input_only_auroc_heldout_loo": 0.90,
            "auroc_drop": 0.35,
            "test_length_stats": {
                "welch_t_p_value": 0.001,
                "input_token_diff": 11.0,
            },
        }
        verdict, evidence = _paradigm_verdict(result)
        assert verdict == Verdict.FALSIFIED

    def test_confirmed_when_signal_survives(self):
        """A paradigm where residualized AUROC stays high should be CONFIRMED."""
        result = {
            "paradigm": "test",
            "baseline_transfer_auroc": 0.80,
            "residualized_transfer_auroc": 0.75,
            "input_only_auroc_heldout_loo": 0.55,
            "auroc_drop": 0.05,
            "test_length_stats": {
                "welch_t_p_value": 0.30,
                "input_token_diff": 2.0,
            },
        }
        verdict, evidence = _paradigm_verdict(result)
        assert verdict == Verdict.CONFIRMED


# ================================================================
# END-TO-END TEST
# ================================================================

class TestRunF02:
    def test_produces_results_and_files(self, tmp_path):
        results = run_f02(tmp_path)

        # Should return 3 ClaimVerifications
        assert len(results) == 3
        paradigms_tested = {r.claim_id for r in results}
        assert "F02-deception" in paradigms_tested
        assert "F02-refusal" in paradigms_tested
        assert "F02-impossibility" in paradigms_tested

        # Check verdict types
        for r in results:
            assert r.verdict in (
                Verdict.CONFIRMED, Verdict.FALSIFIED, Verdict.WEAKENED
            )

        # Check output files exist
        assert (tmp_path / "f02_results.json").exists()
        assert (tmp_path / "f02_summary.md").exists()

        # Validate JSON structure
        with open(tmp_path / "f02_results.json") as f:
            data = json.load(f)

        assert data["experiment"] == "F02_Held_Out_Input_Length_Control"
        assert "overall_verdict" in data
        assert "paradigms" in data
        assert "checksum" in data
        assert data["checksum"].startswith("sha256:")
        assert len(data["paradigms"]) == 3

        # Check each paradigm has expected keys
        for paradigm in ["deception", "refusal", "impossibility"]:
            pd = data["paradigms"][paradigm]
            assert "verdict" in pd
            assert "baseline_transfer_auroc" in pd
            assert "residualized_transfer_auroc" in pd
            assert "auroc_drop_bootstrap_ci" in pd

    def test_deception_should_be_confounded(self, tmp_path):
        """The deception paradigm should show input-length confound."""
        results = run_f02(tmp_path)
        deception = [r for r in results if r.claim_id == "F02-deception"][0]

        # The input-only AUROC should be high for deception
        stats = deception.stats
        assert stats["input_only_auroc_heldout_loo"] > 0.65, (
            f"Expected high input-only AUROC for deception, "
            f"got {stats['input_only_auroc_heldout_loo']:.3f}"
        )

        # The input token diff should be positive (deceptive > honest)
        assert stats["test_length_stats"]["input_token_diff"] > 5, (
            f"Expected deceptive prompts to have more input tokens, "
            f"diff={stats['test_length_stats']['input_token_diff']:.1f}"
        )
