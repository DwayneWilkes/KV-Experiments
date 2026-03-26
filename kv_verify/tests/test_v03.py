"""Tests for V03: FWL Leakage Test.

Verifies that run_v03() produces correct ClaimVerification results
for the FWL leakage analysis across all 10 Exp 47 comparisons.

Tests written FIRST (TDD) before implementation.
"""

import json
from pathlib import Path

import numpy as np

from kv_verify.experiments.v03_fwl_leakage import run_v03, EXCLUDED_COMPARISONS
from kv_verify.fixtures import EXP47_COMPARISONS
from kv_verify.types import ClaimVerification, Severity, Verdict


class TestV03ReturnType:
    """run_v03 must return a list of ClaimVerification objects."""

    def test_returns_list(self, tmp_path):
        results = run_v03(tmp_path)
        assert isinstance(results, list)

    def test_returns_claim_verifications(self, tmp_path):
        results = run_v03(tmp_path)
        for r in results:
            assert isinstance(r, ClaimVerification)

    def test_returns_one_per_comparison(self, tmp_path):
        results = run_v03(tmp_path)
        assert len(results) == len(EXP47_COMPARISONS)


class TestV03ClaimIds:
    """Each result must have a claim_id starting with C4-."""

    def test_claim_ids_start_with_c4(self, tmp_path):
        results = run_v03(tmp_path)
        for r in results:
            assert r.claim_id.startswith("C4-"), (
                f"Expected claim_id starting with 'C4-', got '{r.claim_id}'"
            )

    def test_claim_ids_unique(self, tmp_path):
        results = run_v03(tmp_path)
        ids = [r.claim_id for r in results]
        assert len(ids) == len(set(ids)), f"Duplicate claim_ids: {ids}"

    def test_finding_id_is_c4(self, tmp_path):
        results = run_v03(tmp_path)
        for r in results:
            assert r.finding_id == "C4"


class TestV03StatsSchema:
    """Each result.stats must contain the required AUROC and R-squared keys."""

    REQUIRED_KEYS = [
        "auroc_fwl_full",
        "auroc_fwl_within",
        "auroc_poly2",
        "auroc_poly3",
        "r_squared",
    ]

    def test_stats_contains_required_keys(self, tmp_path):
        results = run_v03(tmp_path)
        for r in results:
            for key in self.REQUIRED_KEYS:
                assert key in r.stats, (
                    f"Missing key '{key}' in stats for {r.claim_id}"
                )

    def test_auroc_values_are_numeric(self, tmp_path):
        results = run_v03(tmp_path)
        for r in results:
            for key in ["auroc_fwl_full", "auroc_fwl_within",
                        "auroc_poly2", "auroc_poly3"]:
                val = r.stats[key]
                assert isinstance(val, (int, float)), (
                    f"Expected numeric for {key}, got {type(val)}"
                )

    def test_auroc_values_in_range(self, tmp_path):
        results = run_v03(tmp_path)
        for r in results:
            for key in ["auroc_fwl_full", "auroc_fwl_within",
                        "auroc_poly2", "auroc_poly3"]:
                val = r.stats[key]
                if not np.isnan(val):
                    assert 0.0 <= val <= 1.0, (
                        f"{key}={val} out of [0,1] for {r.claim_id}"
                    )

    def test_r_squared_is_dict(self, tmp_path):
        results = run_v03(tmp_path)
        for r in results:
            assert isinstance(r.stats["r_squared"], dict), (
                f"r_squared should be dict for {r.claim_id}"
            )

    def test_r_squared_contains_poly_degrees(self, tmp_path):
        results = run_v03(tmp_path)
        for r in results:
            r2 = r.stats["r_squared"]
            assert "linear" in r2
            assert "poly2" in r2
            assert "poly3" in r2


class TestV03ResultJSON:
    """Results must be cached via tracker."""

    def test_saves_result_via_tracker(self, tmp_path):
        run_v03(tmp_path)
        # Result is now cached via tracker at cache/v03_result.json
        result_path = tmp_path / "cache" / "v03_result.json"
        assert result_path.exists()

    def test_json_is_valid(self, tmp_path):
        run_v03(tmp_path)
        result_path = tmp_path / "cache" / "v03_result.json"
        with open(result_path) as f:
            data = json.load(f)
        assert isinstance(data, dict)

    def test_json_has_comparisons(self, tmp_path):
        run_v03(tmp_path)
        with open(tmp_path / "cache" / "v03_result.json") as f:
            data = json.load(f)
        assert "comparisons" in data
        assert len(data["comparisons"]) == len(EXP47_COMPARISONS)

    def test_json_has_overall_verdict(self, tmp_path):
        run_v03(tmp_path)
        with open(tmp_path / "cache" / "v03_result.json") as f:
            data = json.load(f)
        assert "leakage_verdict" in data
        assert "polynomial_verdict" in data

    def test_json_comparison_schema(self, tmp_path):
        run_v03(tmp_path)
        with open(tmp_path / "cache" / "v03_result.json") as f:
            data = json.load(f)
        for comp in data["comparisons"]:
            assert "name" in comp
            assert "claim_id" in comp
            assert "auroc_fwl_full" in comp
            assert "auroc_fwl_within" in comp
            assert "auroc_poly2" in comp
            assert "auroc_poly3" in comp
            assert "leakage_detected" in comp


class TestV03Verdicts:
    """Verdict logic must follow the pre-registered criteria."""

    def test_verdict_is_valid_enum(self, tmp_path):
        results = run_v03(tmp_path)
        valid = {Verdict.CONFIRMED, Verdict.FALSIFIED, Verdict.WEAKENED,
                 Verdict.STRENGTHENED, Verdict.INDETERMINATE}
        for r in results:
            assert r.verdict in valid, (
                f"Invalid verdict {r.verdict} for {r.claim_id}"
            )

    def test_leakage_detected_when_auroc_delta_large(self, tmp_path):
        results = run_v03(tmp_path)
        for r in results:
            delta = abs(r.stats["auroc_fwl_full"] - r.stats["auroc_fwl_within"])
            leakage = r.stats.get("leakage_detected", False)
            if delta > 0.05:
                assert leakage is True, (
                    f"delta={delta:.3f} > 0.05 but leakage_detected=False "
                    f"for {r.claim_id}"
                )

    def test_severity_is_critical(self, tmp_path):
        results = run_v03(tmp_path)
        for r in results:
            assert r.severity == Severity.CRITICAL


class TestV03ExcludedComparisons:
    """Excluded comparisons must be correctly identified."""

    def test_excluded_set(self):
        assert "exp39_sycophancy" in EXCLUDED_COMPARISONS
        assert "exp18b_deception" in EXCLUDED_COMPARISONS
        assert "exp32_jailbreak_vs_refusal" in EXCLUDED_COMPARISONS
        assert len(EXCLUDED_COMPARISONS) == 3

    def test_surviving_count(self, tmp_path):
        all_names = [c["name"] for c in EXP47_COMPARISONS]
        surviving = [n for n in all_names if n not in EXCLUDED_COMPARISONS]
        assert len(surviving) == 7


class TestV03OverallVerdict:
    """The overall polynomial verdict must follow pre-registered logic."""

    def test_json_polynomial_verdict_value(self, tmp_path):
        run_v03(tmp_path)
        with open(tmp_path / "cache" / "v03_result.json") as f:
            data = json.load(f)
        valid_verdicts = ["falsified", "strengthened", "weakened"]
        assert data["polynomial_verdict"] in valid_verdicts

    def test_json_has_surviving_count(self, tmp_path):
        run_v03(tmp_path)
        with open(tmp_path / "cache" / "v03_result.json") as f:
            data = json.load(f)
        assert "n_surviving" in data
        assert data["n_surviving"] == 7

    def test_json_has_collapsed_count(self, tmp_path):
        run_v03(tmp_path)
        with open(tmp_path / "cache" / "v03_result.json") as f:
            data = json.load(f)
        assert "n_collapsed_poly2" in data
        assert isinstance(data["n_collapsed_poly2"], int)


class TestV03ZeroGpuTime:
    """CPU-only experiment: gpu_time_seconds must be 0."""

    def test_zero_gpu_time(self, tmp_path):
        results = run_v03(tmp_path)
        for r in results:
            assert r.gpu_time_seconds == 0.0


class TestV03TrackerIntegration:
    """Test that tracker logs metrics and verdicts."""

    def test_tracker_logs_metrics(self, tmp_path):
        run_v03(tmp_path)
        with open(tmp_path / "run_metadata.json") as f:
            meta = json.load(f)
        assert "n_leakage_detected" in meta["metrics"]
        assert "n_collapsed_poly2" in meta["metrics"]
        assert "n_surviving" in meta["metrics"]
        assert meta["metrics"]["n_surviving"] == 7

    def test_tracker_logs_verdicts(self, tmp_path):
        run_v03(tmp_path)
        with open(tmp_path / "run_metadata.json") as f:
            meta = json.load(f)
        # Should have one verdict per comparison
        assert len(meta["verdicts"]) == len(EXP47_COMPARISONS)
        for claim_id, verdict_data in meta["verdicts"].items():
            assert claim_id.startswith("C4-")
            assert verdict_data["verdict"] in (
                "confirmed", "falsified", "weakened",
                "strengthened", "indeterminate",
            )
