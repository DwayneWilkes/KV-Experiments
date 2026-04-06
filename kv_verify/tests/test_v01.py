"""Tests for V1: GroupKFold Bug Detection experiment.

Tests written FIRST per TDD. The experiment must:
1. Return a List[ClaimVerification] (one per comparison with data)
2. Each result has claim_id starting with "C2-"
3. Each result's stats dict contains required keys
4. Overall verdict uses pre-registered criteria
5. Results JSON saved to output_dir
"""

import json
from pathlib import Path
from typing import List

import numpy as np
import pytest

from kv_verify.experiments.v01_groupkfold import run_v01
from kv_verify.fixtures import EXP47_COMPARISONS
from kv_verify.types import ClaimVerification, Severity, Verdict


# Use 200 permutations in tests for speed
N_PERMS_TEST = 200


@pytest.fixture(scope="module")
def v01_output(tmp_path_factory):
    """Run V01 once for all tests in this module."""
    out = tmp_path_factory.mktemp("v01")
    results = run_v01(out, n_permutations=N_PERMS_TEST)
    return results, out


@pytest.fixture(scope="module")
def v01_results(v01_output):
    """The List[ClaimVerification] from run_v01."""
    return v01_output[0]


@pytest.fixture(scope="module")
def v01_json(v01_output):
    """Parsed JSON from cache/v01_result.json (via tracker)."""
    _, out_dir = v01_output
    with open(out_dir / "cache" / "v01_result.json") as f:
        return json.load(f)


class TestV01ReturnType:
    def test_returns_list_of_claim_verifications(self, v01_results):
        assert isinstance(v01_results, list)
        assert len(v01_results) > 0
        for r in v01_results:
            assert isinstance(r, ClaimVerification)

    def test_returns_one_per_comparison(self, v01_results):
        assert len(v01_results) == len(EXP47_COMPARISONS)


class TestV01ClaimIDs:
    def test_all_claim_ids_start_with_c2(self, v01_results):
        for r in v01_results:
            assert r.claim_id.startswith("C2-"), (
                f"Expected claim_id starting with 'C2-', got '{r.claim_id}'"
            )

    def test_claim_ids_are_unique(self, v01_results):
        ids = [r.claim_id for r in v01_results]
        assert len(ids) == len(set(ids)), "Duplicate claim IDs found"

    def test_finding_id_is_c2(self, v01_results):
        for r in v01_results:
            assert r.finding_id == "C2"

    def test_severity_is_critical(self, v01_results):
        for r in v01_results:
            assert r.severity == Severity.CRITICAL


class TestV01Stats:
    def test_stats_contain_required_keys(self, v01_results):
        required_keys = {
            "auroc_buggy", "auroc_fixed", "auroc_delta",
            "p_value_buggy", "p_value_fixed", "significance_flipped",
        }
        for r in v01_results:
            missing = required_keys - set(r.stats.keys())
            assert not missing, (
                f"Missing stats keys for {r.claim_id}: {missing}"
            )

    def test_auroc_values_in_range(self, v01_results):
        for r in v01_results:
            assert 0.0 <= r.stats["auroc_buggy"] <= 1.0
            assert 0.0 <= r.stats["auroc_fixed"] <= 1.0

    def test_auroc_delta_is_difference(self, v01_results):
        for r in v01_results:
            expected = r.stats["auroc_fixed"] - r.stats["auroc_buggy"]
            assert abs(r.stats["auroc_delta"] - expected) < 1e-10, (
                f"auroc_delta should be fixed - buggy for {r.claim_id}"
            )

    def test_p_values_in_range(self, v01_results):
        for r in v01_results:
            assert 0.0 < r.stats["p_value_buggy"] <= 1.0
            assert 0.0 < r.stats["p_value_fixed"] <= 1.0

    def test_significance_flipped_is_bool(self, v01_results):
        for r in v01_results:
            assert isinstance(r.stats["significance_flipped"], bool)

    def test_stats_has_corrected_p_values(self, v01_results):
        """Holm-Bonferroni corrected p-values must be present."""
        for r in v01_results:
            assert "p_value_fixed_corrected" in r.stats


class TestV01VerdictLogic:
    def test_verdicts_are_confirmed_or_weakened(self, v01_results):
        for r in v01_results:
            assert r.verdict in (Verdict.CONFIRMED, Verdict.WEAKENED), (
                f"Unexpected verdict {r.verdict} for {r.claim_id}"
            )

    def test_weakened_if_delta_exceeds_threshold(self, v01_results):
        """Any comparison with abs(delta) > 0.05 must be WEAKENED."""
        for r in v01_results:
            if abs(r.stats["auroc_delta"]) > 0.05:
                assert r.verdict == Verdict.WEAKENED, (
                    f"{r.claim_id} has delta={r.stats['auroc_delta']:.4f} "
                    f"but verdict is {r.verdict}"
                )

    def test_weakened_if_significance_flipped(self, v01_results):
        """Any comparison where significance flipped must be WEAKENED."""
        for r in v01_results:
            if r.stats["significance_flipped"]:
                assert r.verdict == Verdict.WEAKENED, (
                    f"{r.claim_id} has flipped significance "
                    f"but verdict is {r.verdict}"
                )


class TestV01OutputFile:
    def test_saves_result_via_tracker(self, v01_output):
        _, out_dir = v01_output
        # Result is now cached via tracker at cache/v01_result.json
        assert (out_dir / "cache" / "v01_result.json").exists()

    def test_result_json_is_valid(self, v01_json):
        assert "comparisons" in v01_json
        assert "overall_verdict" in v01_json
        assert len(v01_json["comparisons"]) == len(EXP47_COMPARISONS)

    def test_result_json_has_overall_verdict(self, v01_json):
        assert v01_json["overall_verdict"] in ("confirmed", "weakened")

    def test_result_json_comparisons_have_names(self, v01_json):
        names = {c["name"] for c in v01_json["comparisons"]}
        expected = {c["name"] for c in EXP47_COMPARISONS}
        assert names == expected

    def test_result_json_has_n_permutations(self, v01_json):
        assert v01_json["parameters"]["n_permutations"] == N_PERMS_TEST


class TestV01BuggyGroups:
    def test_buggy_groups_have_overlap(self, v01_json):
        """Verify we reproduce the original bug: overlapping group IDs."""
        for comp in v01_json["comparisons"]:
            if not comp.get("paired", False):
                assert comp["n_groups_buggy"] < comp["n_pos"] + comp["n_neg"], (
                    f"{comp['name']}: buggy groups should overlap "
                    f"({comp['n_groups_buggy']} groups for "
                    f"{comp['n_pos']}+{comp['n_neg']} samples)"
                )

    def test_fixed_groups_no_overlap_unpaired(self, v01_json):
        """Fixed non-paired groups must have unique IDs."""
        for comp in v01_json["comparisons"]:
            if not comp.get("paired", False):
                assert comp["n_groups_fixed"] == comp["n_pos"] + comp["n_neg"]


class TestV01TrackerIntegration:
    def test_tracker_logs_metrics(self, v01_output):
        _, out_dir = v01_output
        with open(out_dir / "run_metadata.json") as f:
            meta = json.load(f)
        assert "n_weakened" in meta["metrics"]
        assert "n_confirmed" in meta["metrics"]
        assert "n_comparisons" in meta["metrics"]
        assert meta["metrics"]["n_comparisons"] == len(EXP47_COMPARISONS)

    def test_tracker_logs_verdicts(self, v01_output):
        _, out_dir = v01_output
        with open(out_dir / "run_metadata.json") as f:
            meta = json.load(f)
        # Should have one verdict per comparison
        assert len(meta["verdicts"]) == len(EXP47_COMPARISONS)
        for claim_id, verdict_data in meta["verdicts"].items():
            assert claim_id.startswith("C2-")
            assert verdict_data["verdict"] in ("confirmed", "weakened")
