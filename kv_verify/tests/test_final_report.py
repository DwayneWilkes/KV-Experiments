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

    def test_produces_markdown_with_verdicts(self):
        results = {
            "V01": {"verdict": "CONFIRMED", "p_value": 0.001},
            "V04": {"verdict": "WEAKENED", "p_value": 0.037},
        }
        report = generate_final_report(results)
        assert isinstance(report, str)
        # Both experiments should appear in the verdict table
        assert "V01" in report
        assert "V04" in report

    def test_includes_global_holm_table(self):
        results = {
            "V01": {"verdict": "CONFIRMED", "p_value": 0.001},
            "F01b": {"verdict": "FALSIFIED", "p_value": 0.0001},
        }
        report = generate_final_report(results)
        # The report must contain the Holm correction table header
        assert "Holm-Bonferroni" in report
        assert "Corrected p" in report

    def test_includes_all_14_claims(self):
        results = {}
        report = generate_final_report(results)
        for claim in CLAIMS:
            assert claim["id"] in report, f"Missing claim {claim['id']} from report"
