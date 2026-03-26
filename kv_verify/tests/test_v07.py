"""Tests for V07: Sycophancy Length Confound experiment.

Verifies that run_v07 produces a valid ClaimVerification with the
expected structure and verdict per pre-registered criteria.

Pre-registered finding M2: length-only AUROC (0.943) exceeds feature
AUROC (0.933), so H0 (features outperform length) should be FALSIFIED.
"""

import json
from pathlib import Path

from kv_verify.experiments.v07_sycophancy import run_v07
from kv_verify.types import ClaimVerification, Severity, Verdict


class TestV07Structure:
    """Test that run_v07 returns correctly shaped ClaimVerification."""

    def test_returns_claim_verification(self, tmp_path):
        result = run_v07(tmp_path)
        assert isinstance(result, ClaimVerification)

    def test_claim_id(self, tmp_path):
        result = run_v07(tmp_path)
        assert result.claim_id == "M2-39-sycophancy"

    def test_finding_id(self, tmp_path):
        result = run_v07(tmp_path)
        assert result.finding_id == "M2"

    def test_severity_is_major(self, tmp_path):
        result = run_v07(tmp_path)
        assert result.severity == Severity.MAJOR

    def test_zero_gpu_time(self, tmp_path):
        result = run_v07(tmp_path)
        assert result.gpu_time_seconds == 0.0


class TestV07Stats:
    """Test that stats dict contains all required AUROC values."""

    def test_stats_has_feature_auroc(self, tmp_path):
        result = run_v07(tmp_path)
        assert "feature_auroc" in result.stats

    def test_stats_has_length_only_auroc(self, tmp_path):
        result = run_v07(tmp_path)
        assert "length_only_auroc" in result.stats

    def test_stats_has_fwl_both_auroc(self, tmp_path):
        result = run_v07(tmp_path)
        assert "fwl_both_auroc" in result.stats

    def test_auroc_values_in_range(self, tmp_path):
        result = run_v07(tmp_path)
        for key in ["feature_auroc", "length_only_auroc", "fwl_both_auroc"]:
            val = result.stats[key]
            assert 0.0 <= val <= 1.0, f"{key}={val} out of [0,1] range"


class TestV07Verdict:
    """Test that the verdict follows pre-registered criteria.

    M2 finding: length-only AUROC (0.943) >= feature AUROC (0.933).
    Pre-registered rule: length_only >= feature => FALSIFIED.
    """

    def test_verdict_is_falsified(self, tmp_path):
        result = run_v07(tmp_path)
        assert result.verdict == Verdict.FALSIFIED, (
            f"Expected FALSIFIED per M2 pre-registration, got {result.verdict}. "
            f"feature_auroc={result.stats.get('feature_auroc')}, "
            f"length_only_auroc={result.stats.get('length_only_auroc')}, "
            f"fwl_both_auroc={result.stats.get('fwl_both_auroc')}"
        )

    def test_evidence_mentions_length(self, tmp_path):
        result = run_v07(tmp_path)
        assert "length" in result.evidence_summary.lower()


class TestV07OutputFile:
    """Test that results are cached via tracker."""

    def test_saves_result_via_tracker(self, tmp_path):
        run_v07(tmp_path)
        # Result is now cached via tracker at cache/v07_result.json
        result_path = tmp_path / "cache" / "v07_result.json"
        assert result_path.exists()

    def test_result_json_has_required_keys(self, tmp_path):
        run_v07(tmp_path)
        result_path = tmp_path / "cache" / "v07_result.json"
        with open(result_path) as f:
            data = json.load(f)
        assert data["claim_id"] == "M2-39-sycophancy"
        assert data["verdict"] == "falsified"
        assert "feature_auroc" in data["stats"]
        assert "length_only_auroc" in data["stats"]
        assert "fwl_both_auroc" in data["stats"]

    def test_result_json_has_finding_id(self, tmp_path):
        run_v07(tmp_path)
        result_path = tmp_path / "cache" / "v07_result.json"
        with open(result_path) as f:
            data = json.load(f)
        assert data["finding_id"] == "M2"


class TestV07TrackerIntegration:
    """Test that tracker logs metrics and verdicts."""

    def test_tracker_logs_metrics(self, tmp_path):
        run_v07(tmp_path)
        with open(tmp_path / "run_metadata.json") as f:
            meta = json.load(f)
        assert "feature_auroc" in meta["metrics"]
        assert "length_only_auroc" in meta["metrics"]
        assert "fwl_both_auroc" in meta["metrics"]
        # All AUROC values should be in [0, 1]
        for key in ["feature_auroc", "length_only_auroc", "fwl_both_auroc"]:
            assert 0.0 <= meta["metrics"][key] <= 1.0

    def test_tracker_logs_verdict(self, tmp_path):
        run_v07(tmp_path)
        with open(tmp_path / "run_metadata.json") as f:
            meta = json.load(f)
        assert "M2-39-sycophancy" in meta["verdicts"]
        assert meta["verdicts"]["M2-39-sycophancy"]["verdict"] == "falsified"
