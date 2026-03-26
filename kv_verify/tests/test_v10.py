"""Tests for V10: Power analysis for all comparisons."""

import json
from pathlib import Path

import pytest

from kv_verify.experiments.v10_power_analysis import run_v10
from kv_verify.types import ClaimVerification, Severity, Verdict


@pytest.fixture(scope="module")
def v10_result(tmp_path_factory):
    """Run V10 once, share across all tests."""
    output_dir = tmp_path_factory.mktemp("v10")
    return run_v10(output_dir, n_sim=500)


class TestV10Structure:
    def test_returns_claim_verification(self, v10_result):
        assert isinstance(v10_result, ClaimVerification)

    def test_claim_id(self, v10_result):
        assert v10_result.claim_id == "M7-power"

    def test_finding_id(self, v10_result):
        assert v10_result.finding_id == "M7"

    def test_severity(self, v10_result):
        assert v10_result.severity == Severity.MAJOR

    def test_zero_gpu_time(self, v10_result):
        assert v10_result.gpu_time_seconds == 0.0


class TestV10Stats:
    def test_has_power_table(self, v10_result):
        assert "power_table" in v10_result.stats

    def test_power_table_has_all_comparisons(self, v10_result):
        table = v10_result.stats["power_table"]
        assert len(table) == 10

    def test_each_entry_has_required_fields(self, v10_result):
        for entry in v10_result.stats["power_table"]:
            assert "name" in entry
            assert "n_per_group" in entry
            assert "observed_auroc" in entry
            assert "achieved_power" in entry
            assert "required_n" in entry

    def test_power_values_in_range(self, v10_result):
        for entry in v10_result.stats["power_table"]:
            assert 0 <= entry["achieved_power"] <= 1

    def test_underpowered_count(self, v10_result):
        assert "n_underpowered" in v10_result.stats
        assert isinstance(v10_result.stats["n_underpowered"], int)


class TestV10Verdict:
    def test_verdict_is_weakened(self, v10_result):
        # With N=10-20 and moderate effects, some comparisons must be underpowered
        assert v10_result.verdict == Verdict.WEAKENED

    def test_linchpin_underpowered(self, v10_result):
        """exp36_impossible_vs_harmful at AUROC=0.65, N=20 should be underpowered."""
        table = v10_result.stats["power_table"]
        linchpin = [e for e in table if "impossible_vs_harmful" in e["name"]]
        assert len(linchpin) == 1
        assert linchpin[0]["achieved_power"] < 0.50


class TestV10OutputFile:
    def test_saves_json(self, v10_result, tmp_path_factory):
        # Re-run to check output (the fixture already ran once)
        output_dir = tmp_path_factory.mktemp("v10_output")
        run_v10(output_dir, n_sim=200)
        assert (output_dir / "v10_results.json").exists()

    def test_json_valid(self, tmp_path):
        run_v10(tmp_path, n_sim=200)
        with open(tmp_path / "v10_results.json") as f:
            data = json.load(f)
        assert data["claim_id"] == "M7-power"
        assert "power_table" in data["stats"]
