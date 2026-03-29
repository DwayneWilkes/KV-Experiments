"""Tests for F03: Cross-Model Transfer with Input-Length Control.

Verifies that the experiment runs end-to-end on the actual 49c data
and produces output matching the expected schema.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from kv_verify.experiments.f03_cross_model_input_control import (
    MODEL_SHORT_NAMES,
    TASKS,
    _cross_model_auroc,
    _extract_features,
    _extract_input_tokens,
    _load_49c_data,
    _residualize_cross_model,
    _safe_auroc,
    run_f03,
)
from kv_verify.fixtures import PRIMARY_FEATURES
from kv_verify.stats import loo_auroc
from kv_verify.types import Severity, Verdict


class TestDataLoading:
    """Tests for data loading and extraction helpers."""

    def test_load_49c_data(self):
        data = _load_49c_data()
        assert "per_model_data" in data
        assert "cross_model_transfer" in data
        assert "within_model" in data
        for model in MODEL_SHORT_NAMES:
            assert model in data["per_model_data"]

    def test_per_model_conditions(self):
        data = _load_49c_data()
        pmd = data["per_model_data"]
        for model in MODEL_SHORT_NAMES:
            assert "refusal" in pmd[model]
            assert "benign" in pmd[model]
            assert len(pmd[model]["refusal"]) == 15
            assert len(pmd[model]["benign"]) == 15

    def test_extract_features_shape(self):
        data = _load_49c_data()
        items = data["per_model_data"]["Qwen2.5-7B-Instruct"]["refusal"]
        X = _extract_features(items, PRIMARY_FEATURES)
        assert X.shape == (15, 3)
        assert not np.any(np.isnan(X))

    def test_extract_input_tokens(self):
        data = _load_49c_data()
        items = data["per_model_data"]["Qwen2.5-7B-Instruct"]["refusal"]
        input_tokens = _extract_input_tokens(items)
        assert input_tokens.shape == (15,)
        # Input tokens should be positive integers (n_tokens > n_generated)
        assert np.all(input_tokens > 0)
        # Verify computation: n_tokens - n_generated
        for i, item in enumerate(items):
            expected = item["features"]["n_tokens"] - item["features"]["n_generated"]
            assert input_tokens[i] == expected


class TestStatisticalHelpers:
    """Tests for statistical helper functions."""

    def test_safe_auroc_normal(self):
        y = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        assert _safe_auroc(y, scores) == 1.0

    def test_safe_auroc_degenerate_labels(self):
        y = np.array([1, 1, 1])
        scores = np.array([0.5, 0.6, 0.7])
        assert _safe_auroc(y, scores) == 0.5

    def test_safe_auroc_nan_scores(self):
        y = np.array([0, 1])
        scores = np.array([np.nan, np.nan])
        assert _safe_auroc(y, scores) == 0.5

    def test_loo_auroc_separable(self):
        rng = np.random.RandomState(42)
        X = np.vstack([rng.randn(15, 3) - 2, rng.randn(15, 3) + 2])
        y = np.array([0] * 15 + [1] * 15)
        auroc = loo_auroc(X, y)
        # Well-separated data should have high AUROC
        assert auroc > 0.90

    def test_loo_auroc_random(self):
        rng = np.random.RandomState(42)
        X = rng.randn(30, 3)
        y = np.array([0] * 15 + [1] * 15)
        auroc = loo_auroc(X, y)
        # Random data should be near chance
        assert 0.2 < auroc < 0.8

    def test_cross_model_auroc_separable(self):
        rng = np.random.RandomState(42)
        X_train = np.vstack([rng.randn(15, 3) - 2, rng.randn(15, 3) + 2])
        y_train = np.array([0] * 15 + [1] * 15)
        X_test = np.vstack([rng.randn(15, 3) - 2, rng.randn(15, 3) + 2])
        y_test = np.array([0] * 15 + [1] * 15)
        auroc = _cross_model_auroc(X_train, y_train, X_test, y_test)
        assert auroc > 0.85

    def test_residualize_cross_model_shape(self):
        rng = np.random.RandomState(42)
        X_train = rng.randn(30, 3)
        input_train = rng.randint(30, 80, size=30).astype(float)
        X_test = rng.randn(30, 3)
        input_test = rng.randint(30, 80, size=30).astype(float)

        X_train_r, X_test_r = _residualize_cross_model(
            X_train, input_train, X_test, input_test
        )
        assert X_train_r.shape == X_train.shape
        assert X_test_r.shape == X_test.shape

    def test_residualize_removes_correlation(self):
        """After residualization, features should be uncorrelated with input length."""
        rng = np.random.RandomState(42)
        n = 50
        input_tokens = rng.randint(30, 100, size=n).astype(float)
        # Create features correlated with input length
        X = np.column_stack([
            input_tokens * 0.5 + rng.randn(n) * 2,
            input_tokens * -0.3 + rng.randn(n) * 2,
            input_tokens * 0.8 + rng.randn(n) * 2,
        ])
        X_resid, _ = _residualize_cross_model(X, input_tokens, X, input_tokens)
        from scipy.stats import pearsonr
        for j in range(3):
            r, _ = pearsonr(X_resid[:, j], input_tokens)
            assert abs(r) < 0.05, f"Feature {j} still correlated: r={r:.3f}"


class TestRunF03:
    """Integration test: run the full experiment."""

    @pytest.fixture
    def output_dir(self, tmp_path):
        return tmp_path / "f03_output"

    def test_runs_and_produces_json(self, output_dir):
        result = run_f03(output_dir)
        assert isinstance(result, Verdict) or hasattr(result, "verdict")
        result_path = output_dir / "f03_results.json"
        assert result_path.exists()

        with open(result_path) as f:
            data = json.load(f)

        assert data["experiment"] == "F03_cross_model_input_control"
        assert data["claim_id"] == "F03-cross-model"
        assert data["finding_id"] == "F03"
        assert data["verdict"] in [v.value for v in Verdict]
        assert "checksum" in data
        assert data["checksum"].startswith("sha256:")

    def test_result_schema(self, output_dir):
        result = run_f03(output_dir)
        stats = result.stats

        # Cross-model results: 6 pairs for refusal
        assert len(stats["cross_model_results"]) == 6
        for key, v in stats["cross_model_results"].items():
            assert "train_model" in v
            assert "test_model" in v
            assert "task" in v
            assert "raw_auroc" in v
            assert "input_only_auroc" in v
            assert "resid_auroc" in v
            assert "auroc_drop" in v
            # AUROCs should be in [0, 1]
            assert 0.0 <= v["raw_auroc"] <= 1.0
            assert 0.0 <= v["input_only_auroc"] <= 1.0
            assert 0.0 <= v["resid_auroc"] <= 1.0

        # Within-model results: 3 models x 1 task
        assert len(stats["within_model_results"]) == 3
        for key, v in stats["within_model_results"].items():
            assert "raw_auroc" in v
            assert "input_only_auroc" in v
            assert "resid_auroc" in v
            assert "input_length_diff" in v
            assert "correlations" in v

        # Summary
        summary = stats["summary"]
        assert "mean_cross_raw_auroc" in summary
        assert "mean_cross_resid_auroc" in summary
        assert "mean_cross_input_auroc" in summary

    def test_verdict_type(self, output_dir):
        result = run_f03(output_dir)
        assert isinstance(result.verdict, Verdict)
        assert result.severity == Severity.CRITICAL

    def test_no_self_pairs(self, output_dir):
        """Cross-model pairs should never train and test on the same model."""
        result = run_f03(output_dir)
        for key, v in result.stats["cross_model_results"].items():
            assert v["train_model"] != v["test_model"], (
                f"Self-pair found: {v['train_model']}"
            )

    def test_sha256_integrity(self, output_dir):
        """Verify the SHA-256 checksum matches the content."""
        run_f03(output_dir)
        result_path = output_dir / "f03_results.json"
        with open(result_path) as f:
            data = json.load(f)

        stored_checksum = data.pop("checksum")
        recomputed = json.dumps(data, indent=2)
        import hashlib
        expected = f"sha256:{hashlib.sha256(recomputed.encode()).hexdigest()}"
        assert stored_checksum == expected
