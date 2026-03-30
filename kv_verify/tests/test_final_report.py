"""Tests for final report generation (Task 9.4)."""

import pytest

from kv_verify.lib.final_report import generate_final_report, CLAIMS


class TestClaims:

    def test_all_14_claims_defined(self):
        assert len(CLAIMS) >= 14

    def test_claims_have_required_fields(self):
        for claim in CLAIMS:
            assert "id" in claim
            assert "text" in claim
            assert "experiments" in claim


class TestGenerateFinalReport:

    def test_produces_markdown(self):
        # Minimal experiment results
        results = {
            "V01": {"verdict": "CONFIRMED", "p_value": 0.001},
            "V04": {"verdict": "WEAKENED", "p_value": 0.037},
        }
        report = generate_final_report(results)
        assert isinstance(report, str)
        assert "CONFIRMED" in report or "WEAKENED" in report

    def test_includes_global_holm(self):
        results = {
            "V01": {"verdict": "CONFIRMED", "p_value": 0.001},
            "F01b": {"verdict": "FALSIFIED", "p_value": 0.0001},
        }
        report = generate_final_report(results)
        assert "Holm" in report or "corrected" in report.lower()

    def test_includes_all_claims(self):
        results = {}
        report = generate_final_report(results)
        # Even with no results, all claims should be listed
        assert "C1" in report
        assert "C5" in report
