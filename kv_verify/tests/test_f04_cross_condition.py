"""Tests for F04: Cross-Condition Transfer Validity.

Tests that the analysis runs without error and produces output
matching the expected schema. These are structural tests, not
statistical validation (the experiment is interpretive analysis).
"""

import json

import pytest

from kv_verify.experiments.f04_cross_condition_validity import (
    run_f04,
    run_f04a,
    run_f04b,
    run_f04c,
    run_f04d,
    _load_json,
)
from kv_verify.types import ClaimVerification, Severity, Verdict


# ================================================================
# Fixtures
# ================================================================

@pytest.fixture
def transfer_data():
    """Load the actual cross_condition_transfer.json."""
    return _load_json("cross_condition_transfer.json")


@pytest.fixture
def improvement_data():
    """Load the actual transfer_improvement.json."""
    return _load_json("transfer_improvement.json")


@pytest.fixture
def f01b_results():
    """Known F01b-49b results (from prior run)."""
    return {
        "stats": {
            "r_squared": {
                "norm_per_token": 0.80,
                "key_rank": 0.89,
                "key_entropy": 0.37,
            },
            "input_only_auroc": 1.0,
            "residualized_auroc": 0.39,
        },
    }


@pytest.fixture
def output_dir(tmp_path):
    """Temporary output directory."""
    d = tmp_path / "f04_output"
    d.mkdir()
    return d


# ================================================================
# F04a tests
# ================================================================

class TestF04aPerTopicBehavior:
    def test_returns_dict(self, transfer_data):
        result = run_f04a(transfer_data)
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_all_scenarios_have_100pct_deceptive(self, transfer_data):
        """The per-topic data shows 100% deceptive for all categories."""
        result = run_f04a(transfer_data)
        for scenario_key, scenario_result in result.items():
            assert scenario_result["all_topics_predicted_100pct_deceptive"] is True, (
                f"Scenario {scenario_key} should show 100% deceptive predictions"
            )

    def test_category_stats_present(self, transfer_data):
        result = run_f04a(transfer_data)
        for scenario_key, scenario_result in result.items():
            cats = scenario_result["category_stats"]
            assert "censored" in cats
            assert "control" in cats
            assert "complex_noncensored" in cats

    def test_probability_separation(self, transfer_data):
        """Check that probability separation analysis is computed."""
        result = run_f04a(transfer_data)
        for scenario_key, scenario_result in result.items():
            sep = scenario_result["probability_separation"]
            assert "censored_mean_prob" in sep or "note" in sep


# ================================================================
# F04b tests
# ================================================================

class TestF04bFeatureConfound:
    def test_returns_dict(self, transfer_data, f01b_results):
        result = run_f04b(transfer_data, f01b_results)
        assert isinstance(result, dict)

    def test_confound_analysis_present(self, transfer_data, f01b_results):
        result = run_f04b(transfer_data, f01b_results)
        fca = result["feature_confound_analysis"]
        assert "norms_per_token" in fca
        assert "key_ranks" in fca

    def test_high_confound_exposure(self, transfer_data, f01b_results):
        """Weighted confound R^2 should be > 0.5 given known R^2 values."""
        result = run_f04b(transfer_data, f01b_results)
        r2 = result["weighted_confound_r2"]
        assert r2 is not None
        assert r2 > 0.5, f"Expected high confound exposure, got {r2}"

    def test_risk_labels(self, transfer_data, f01b_results):
        result = run_f04b(transfer_data, f01b_results)
        fca = result["feature_confound_analysis"]
        # norms_per_token should be HIGH risk (R^2 = 0.80)
        assert fca["norms_per_token"]["confound_risk"] == "HIGH"
        # key_ranks should be HIGH risk (R^2 = 0.89)
        assert fca["key_ranks"]["confound_risk"] == "HIGH"


# ================================================================
# F04c tests
# ================================================================

class TestF04cStructuralArgument:
    def test_returns_dict(self, transfer_data):
        result = run_f04c(transfer_data)
        assert isinstance(result, dict)

    def test_same_model_at_chance(self, transfer_data):
        """Same-model transfer should be identified as at chance."""
        result = run_f04c(transfer_data)
        assert result["same_model_transfer"]["at_chance"] is True

    def test_deception_models_listed(self, transfer_data):
        result = run_f04c(transfer_data)
        assert len(result["deception_models"]) == 7

    def test_cross_model_aurocs_higher(self, transfer_data):
        """Cross-model AUROCs should be > same-model AUROCs."""
        result = run_f04c(transfer_data)
        same_mean = result["same_model_transfer"]["mean_auroc"]
        cross_mean = result["cross_model_transfer"]["mean_auroc"]
        assert cross_mean > same_mean + 0.15


# ================================================================
# F04d tests
# ================================================================

class TestF04dScenarioComparison:
    def test_returns_dict(self, transfer_data, improvement_data):
        result = run_f04d(transfer_data, improvement_data)
        assert isinstance(result, dict)

    def test_scenario_table(self, transfer_data, improvement_data):
        result = run_f04d(transfer_data, improvement_data)
        table = result["scenario_table"]
        assert len(table) == 7  # A, B, C, D, E_DeepSeek, E_Mistral, E_Qwen

    def test_qwen_anomaly(self, transfer_data, improvement_data):
        """Qwen2.5-14B should have notably lower AUROC than DeepSeek."""
        result = run_f04d(transfer_data, improvement_data)
        per_model = result["per_model_E_aurocs"]
        assert per_model["DeepSeek-R1-Distill-Qwen-14B"] > per_model["Qwen2.5-14B"] + 0.10

    def test_training_size_effect(self, transfer_data, improvement_data):
        result = run_f04d(transfer_data, improvement_data)
        tse = result["training_size_effect"]
        # Small train should be near chance
        assert all(a < 0.60 for a in tse["small_train_150_aurocs"])
        # Large train should be > 0.70
        assert all(a > 0.70 for a in tse["large_train_990_aurocs"])


# ================================================================
# Integration: full run_f04
# ================================================================

class TestF04Integration:
    def test_runs_without_error(self, output_dir):
        result = run_f04(output_dir)
        assert result is not None
        assert isinstance(result, ClaimVerification)

    def test_verdict_is_falsified(self, output_dir):
        """Given the known data, the verdict should be FALSIFIED."""
        result = run_f04(output_dir)
        assert result.verdict == Verdict.FALSIFIED

    def test_severity_is_critical(self, output_dir):
        result = run_f04(output_dir)
        assert result.severity == Severity.CRITICAL

    def test_result_json_written(self, output_dir):
        run_f04(output_dir)
        result_path = output_dir / "f04_results.json"
        assert result_path.exists()
        with open(result_path) as f:
            data = json.load(f)
        assert "experiment" in data
        assert "verdict" in data
        assert "checksum" in data
        assert data["checksum"].startswith("sha256:")

    def test_summary_md_written(self, output_dir):
        run_f04(output_dir)
        md_path = output_dir / "f04_summary.md"
        assert md_path.exists()
        content = md_path.read_text()
        assert "F04" in content
        assert "FALSIFIED" in content

    def test_result_json_schema(self, output_dir):
        """Verify the result JSON has the expected schema."""
        run_f04(output_dir)
        with open(output_dir / "f04_results.json") as f:
            data = json.load(f)

        required_keys = [
            "experiment", "timestamp", "claim_id", "verdict",
            "evidence_summary", "stats", "checksum",
        ]
        for key in required_keys:
            assert key in data, f"Missing required key: {key}"

        stats = data["stats"]
        assert "f04a_per_topic" in stats
        assert "f04b_feature_confound" in stats
        assert "f04c_structural" in stats
        assert "f04d_scenario_comparison" in stats
        assert "can_determine" in stats
        assert "cannot_determine" in stats

    def test_can_and_cannot_determine_lists(self, output_dir):
        result = run_f04(output_dir)
        can = result.stats["can_determine"]
        cannot = result.stats["cannot_determine"]
        assert len(can) >= 3
        assert len(cannot) >= 3
        # Should explicitly mention per-item data limitation
        assert any("per-item" in s.lower() for s in cannot)
