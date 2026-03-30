"""Tests for global Holm-Bonferroni across experiments (Task 9.1)."""

import pytest

from kv_verify.lib.stats import global_holm_bonferroni


class TestGlobalHolm:

    def test_all_significant_stay_significant(self):
        """Strong p-values survive correction."""
        experiments = [
            ("V01", 0.001),
            ("V03", 0.002),
            ("V04", 0.003),
        ]
        result = global_holm_bonferroni(experiments)
        assert all(r["reject"] for r in result)

    def test_weak_p_value_loses_significance(self):
        """Marginal p-value fails after Holm correction with enough tests."""
        experiments = [
            ("V01", 0.001),
            ("V03", 0.002),
            ("V04", 0.003),
            ("V07", 0.004),
            ("V10_linchpin", 0.04),  # marginal: 0.04 * 1 = 0.04 < 0.05, but
            # Holm step-down: at rank 5, adjusted_alpha = 0.05/1 = 0.05
            # Still passes. Need to push it above.
            ("F01", 0.005),
            ("F02", 0.006),
            ("F03", 0.045),  # this one: corrected_p = 0.045 * 3 = 0.135 > 0.05
        ]
        result = global_holm_bonferroni(experiments)
        f03 = next(r for r in result if r["name"] == "F03")
        assert not f03["reject"]

    def test_returns_corrected_p_values(self):
        experiments = [("A", 0.01), ("B", 0.04)]
        result = global_holm_bonferroni(experiments)
        assert all("corrected_p" in r for r in result)
        assert all("original_p" in r for r in result)

    def test_preserves_experiment_names(self):
        experiments = [("exp_A", 0.01), ("exp_B", 0.02), ("exp_C", 0.03)]
        result = global_holm_bonferroni(experiments)
        names = {r["name"] for r in result}
        assert names == {"exp_A", "exp_B", "exp_C"}

    def test_single_experiment(self):
        result = global_holm_bonferroni([("only", 0.03)])
        assert len(result) == 1
        assert result[0]["corrected_p"] == 0.03  # no correction needed

    def test_custom_alpha(self):
        experiments = [("A", 0.02), ("B", 0.03)]
        result_strict = global_holm_bonferroni(experiments, alpha=0.01)
        result_lenient = global_holm_bonferroni(experiments, alpha=0.05)
        # Strict should reject fewer
        strict_rejects = sum(r["reject"] for r in result_strict)
        lenient_rejects = sum(r["reject"] for r in result_lenient)
        assert strict_rejects <= lenient_rejects
