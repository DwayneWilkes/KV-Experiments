"""
Tests for independent statistical recomputation module.

These tests verify our implementations against known values and scipy references.
This module MUST NOT import from KV-Cache_Experiments/code/stats_utils.py.
"""
import math
import numpy as np
import pytest
from scipy.stats import norm, spearmanr as scipy_spearmanr

# Import the module under test
from independent_stats import (
    spearman_rho,
    kendall_w,
    cohens_d,
    hedges_g,
    conservative_p,
    tost,
    ancova_f,
    effective_rank,
    d_to_auroc,
    power_analysis,
)


# ================================================================
# spearman_rho
# ================================================================

class TestSpearmanRho:
    def test_perfect_positive(self):
        """Perfect monotonic relationship → rho = 1.0"""
        rho, p = spearman_rho([1, 2, 3, 4, 5], [10, 20, 30, 40, 50])
        assert rho == pytest.approx(1.0, abs=1e-10)

    def test_perfect_negative(self):
        """Perfect inverse relationship → rho = -1.0"""
        rho, p = spearman_rho([1, 2, 3, 4, 5], [50, 40, 30, 20, 10])
        assert rho == pytest.approx(-1.0, abs=1e-10)

    def test_no_correlation(self):
        """Known uncorrelated data → rho near 0"""
        # Use a specific shuffled sequence that gives rho ~ 0
        rho, _ = spearman_rho([1, 2, 3, 4, 5], [3, 5, 1, 4, 2])
        assert abs(rho) < 0.5  # Not strongly correlated

    def test_matches_scipy(self):
        """Our implementation matches scipy.stats.spearmanr"""
        x = [1.5, 2.3, 3.1, 4.7, 5.0, 6.2, 7.8]
        y = [2.1, 3.5, 2.9, 5.1, 4.8, 6.0, 7.2]
        our_rho, our_p = spearman_rho(x, y)
        ref_rho, ref_p = scipy_spearmanr(x, y)
        assert our_rho == pytest.approx(ref_rho, abs=1e-10)
        assert our_p == pytest.approx(ref_p, abs=1e-6)

    def test_returns_tuple(self):
        rho, p = spearman_rho([1, 2, 3], [3, 2, 1])
        assert isinstance(rho, float)
        assert isinstance(p, float)


# ================================================================
# kendall_w
# ================================================================

class TestKendallW:
    def test_perfect_agreement(self):
        """All judges rank identically → W = 1.0"""
        # 3 judges, 4 items, all rank the same
        rankings = np.array([
            [1, 2, 3, 4],
            [1, 2, 3, 4],
            [1, 2, 3, 4],
        ])
        w = kendall_w(rankings)
        assert w == pytest.approx(1.0, abs=1e-10)

    def test_no_agreement(self):
        """Random disagreement → W near 0"""
        # Designed to produce low W
        rankings = np.array([
            [1, 2, 3, 4],
            [4, 3, 2, 1],
            [2, 4, 1, 3],
        ])
        w = kendall_w(rankings)
        assert w < 0.3

    def test_known_value(self):
        """
        Known computation:
        3 judges, 3 items
        Rankings: [1,2,3], [1,3,2], [1,2,3]
        Rank sums: [3, 7, 8] → R_bar = 6
        S = (3-6)^2 + (7-6)^2 + (8-6)^2 = 9 + 1 + 4 = 14
        W = 12*S / (k^2 * (n^3 - n)) = 12*14 / (9 * 24) = 168/216 = 0.7778
        """
        rankings = np.array([
            [1, 2, 3],
            [1, 3, 2],
            [1, 2, 3],
        ])
        w = kendall_w(rankings)
        assert w == pytest.approx(168 / 216, abs=1e-4)

    def test_two_judges(self):
        """2 judges — W should still work"""
        rankings = np.array([
            [1, 2, 3, 4],
            [1, 2, 4, 3],
        ])
        w = kendall_w(rankings)
        assert 0 <= w <= 1


# ================================================================
# cohens_d
# ================================================================

class TestCohensD:
    def test_identical_groups(self):
        """Same data → d = 0"""
        d = cohens_d([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
        assert d == pytest.approx(0.0, abs=1e-10)

    def test_known_value(self):
        """
        group1 = [10, 10, 10], group2 = [0, 0, 0]
        mean1 = 10, mean2 = 0, pooled_sd = 0 → but let's use real variance
        group1 = [8, 10, 12], group2 = [2, 4, 6]
        mean1=10, mean2=4, var1=4, var2=4, pooled_sd=2
        d = (10-4)/2 = 3.0
        """
        d = cohens_d([8, 10, 12], [2, 4, 6])
        assert d == pytest.approx(3.0, abs=0.01)

    def test_sign_convention(self):
        """Positive d means group1 > group2"""
        d = cohens_d([10, 11, 12], [1, 2, 3])
        assert d > 0
        d_rev = cohens_d([1, 2, 3], [10, 11, 12])
        assert d_rev < 0

    def test_symmetry(self):
        """d(A,B) = -d(B,A)"""
        g1, g2 = [5, 6, 7, 8], [1, 2, 3, 4]
        assert cohens_d(g1, g2) == pytest.approx(-cohens_d(g2, g1), abs=1e-10)


# ================================================================
# hedges_g
# ================================================================

class TestHedgesG:
    def test_correction_factor(self):
        """Hedges' g should be smaller than Cohen's d (bias correction)"""
        g1, g2 = [8, 10, 12, 14, 16], [2, 4, 6, 8, 10]
        d = cohens_d(g1, g2)
        g = hedges_g(g1, g2)
        assert abs(g) < abs(d)
        assert g > 0  # Same sign

    def test_known_correction(self):
        """
        For n1=n2=5, df=8, J = 1 - 3/(4*8 - 1) = 1 - 3/31 = 0.90323
        So g = d * 0.90323
        """
        g1, g2 = [8, 10, 12, 14, 16], [2, 4, 6, 8, 10]
        d = cohens_d(g1, g2)
        g = hedges_g(g1, g2)
        j_expected = 1 - 3 / (4 * 8 - 1)
        assert g == pytest.approx(d * j_expected, abs=1e-10)

    def test_large_sample_converges(self):
        """With large n, g ≈ d"""
        rng = np.random.RandomState(42)
        g1 = rng.normal(10, 2, 1000)
        g2 = rng.normal(8, 2, 1000)
        d = cohens_d(g1, g2)
        g = hedges_g(g1, g2)
        assert abs(d - g) < 0.01  # Nearly identical for large n


# ================================================================
# conservative_p
# ================================================================

class TestConservativeP:
    def test_returns_max(self):
        """Conservative p = max(welch_p, mann_whitney_p)"""
        g1 = [10, 11, 12, 13, 14]
        g2 = [1, 2, 3, 4, 5]
        p = conservative_p(g1, g2)
        from scipy.stats import ttest_ind, mannwhitneyu
        _, p_welch = ttest_ind(g1, g2, equal_var=False)
        _, p_mw = mannwhitneyu(g1, g2, alternative='two-sided')
        assert p == pytest.approx(max(p_welch, p_mw), abs=1e-10)

    def test_significant_groups(self):
        """Clearly different groups → small p"""
        p = conservative_p([100, 101, 102, 103, 104], [1, 2, 3, 4, 5])
        assert p < 0.05

    def test_identical_groups(self):
        """Same data → large p"""
        p = conservative_p([5, 5, 5, 5, 5], [5, 5, 5, 5, 5])
        assert p >= 0.05


# ================================================================
# tost
# ================================================================

class TestTOST:
    def test_equivalent_groups(self):
        """Very similar groups with delta=0.3 → equivalence confirmed"""
        rng = np.random.RandomState(42)
        # Large n to ensure observed d is well within [-0.3, 0.3]
        g1 = rng.normal(10, 1, 200)
        g2 = rng.normal(10, 1, 200)
        p_lower, p_upper, equivalent = tost(g1, g2, delta=0.3)
        assert equivalent is True
        assert max(p_lower, p_upper) < 0.05

    def test_different_groups(self):
        """Very different groups → NOT equivalent"""
        g1 = np.array([10, 11, 12, 13, 14], dtype=float)
        g2 = np.array([1, 2, 3, 4, 5], dtype=float)
        p_lower, p_upper, equivalent = tost(g1, g2, delta=0.3)
        assert equivalent is False

    def test_returns_three_values(self):
        """Returns (p_lower, p_upper, equivalent)"""
        result = tost([1, 2, 3, 4, 5], [1.1, 2.1, 3.1, 4.1, 5.1])
        assert len(result) == 3
        p_lower, p_upper, equivalent = result
        assert isinstance(p_lower, float)
        assert isinstance(p_upper, float)
        assert isinstance(equivalent, bool)

    def test_delta_in_d_units(self):
        """Delta is in Cohen's d units, not raw units"""
        # Large n with very small true difference → d near 0 → passes delta=0.3
        rng = np.random.RandomState(123)
        g1 = rng.normal(10, 1, 200)
        g2 = rng.normal(10.05, 1, 200)  # tiny shift, d ≈ 0.05
        _, _, equiv = tost(g1, g2, delta=0.3)
        assert equiv is True


# ================================================================
# ancova_f
# ================================================================

class TestAncovaF:
    def test_strong_category_effect(self):
        """Clear category differences → large F, small p"""
        # 3 categories with distinct means, 2 registers
        data = []
        rng = np.random.RandomState(42)
        for cat, base in [("coding", 20), ("creative", 15), ("rote", 10)]:
            for reg in ["formal", "colloquial"]:
                for _ in range(10):
                    data.append({
                        "category": cat,
                        "register": reg,
                        "metric": base + rng.normal(0, 1),
                    })
        F, p = ancova_f(data, "category", "register", "metric")
        assert F > 5
        assert p < 0.01

    def test_no_category_effect(self):
        """Same distribution across categories → small F"""
        data = []
        rng = np.random.RandomState(42)
        for cat in ["coding", "creative", "rote"]:
            for reg in ["formal", "colloquial"]:
                for _ in range(20):
                    data.append({
                        "category": cat,
                        "register": reg,
                        "metric": rng.normal(10, 1),
                    })
        F, p = ancova_f(data, "category", "register", "metric")
        assert p > 0.05

    def test_returns_f_and_p(self):
        data = [
            {"category": "a", "register": "x", "metric": 10},
            {"category": "a", "register": "y", "metric": 11},
            {"category": "b", "register": "x", "metric": 20},
            {"category": "b", "register": "y", "metric": 21},
            {"category": "a", "register": "x", "metric": 12},
            {"category": "b", "register": "y", "metric": 19},
        ]
        F, p = ancova_f(data, "category", "register", "metric")
        assert isinstance(F, float)
        assert isinstance(p, float)
        assert F >= 0
        assert 0 <= p <= 1


# ================================================================
# effective_rank
# ================================================================

class TestEffectiveRank:
    def test_all_equal(self):
        """All singular values equal → effective rank = n (all needed for 90%)"""
        # 5 equal SVs: each contributes 20%. Need 5 for >= 90%.
        svs = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        rank = effective_rank(svs, threshold=0.9)
        assert rank == 5

    def test_single_dominant(self):
        """One dominant SV → effective rank = 1"""
        # First SV dominates: 100^2 = 10000, rest = 1 each = 4.
        # Total = 10004. First = 10000/10004 > 0.99 > 0.90
        svs = np.array([100.0, 1.0, 1.0, 1.0, 1.0])
        rank = effective_rank(svs, threshold=0.9)
        assert rank == 1

    def test_known_value(self):
        """
        SVs = [3, 2, 1]
        SV^2 = [9, 4, 1], total = 14
        Cumulative: 9/14 = 0.643, (9+4)/14 = 0.929 > 0.9
        So effective rank = 2
        """
        svs = np.array([3.0, 2.0, 1.0])
        rank = effective_rank(svs, threshold=0.9)
        assert rank == 2

    def test_threshold_boundary(self):
        """Exact boundary case"""
        # SVs = [sqrt(0.9), sqrt(0.1)] → SV^2 = [0.9, 0.1]
        # Cumulative: 0.9/1.0 = 0.9 — exactly at threshold
        svs = np.array([math.sqrt(0.9), math.sqrt(0.1)])
        rank = effective_rank(svs, threshold=0.9)
        assert rank == 1  # First SV captures exactly 90%


# ================================================================
# d_to_auroc
# ================================================================

class TestDToAUROC:
    def test_zero(self):
        """d=0 → AUROC = 0.5 (random classifier)"""
        assert d_to_auroc(0) == pytest.approx(0.5, abs=1e-10)

    def test_large_positive(self):
        """Large d → AUROC near 1.0"""
        assert d_to_auroc(5.0) > 0.99

    def test_known_values(self):
        """Known d → AUROC conversions"""
        # AUROC = Phi(d / sqrt(2))
        # d=1: Phi(1/1.414) = Phi(0.707) ≈ 0.7602
        assert d_to_auroc(1.0) == pytest.approx(norm.cdf(1.0 / math.sqrt(2)), abs=1e-6)
        # d=2: Phi(2/1.414) = Phi(1.414) ≈ 0.9214
        assert d_to_auroc(2.0) == pytest.approx(norm.cdf(2.0 / math.sqrt(2)), abs=1e-6)

    def test_negative_d(self):
        """Negative d → AUROC < 0.5 (using absolute value is caller's responsibility)"""
        auroc = d_to_auroc(-1.0)
        assert auroc < 0.5

    def test_paper_conversion_table(self):
        """Verify the d-to-AUROC table used in the paper assessment"""
        assert d_to_auroc(0.5) == pytest.approx(0.638, abs=0.002)
        assert d_to_auroc(1.5) == pytest.approx(0.856, abs=0.002)
        assert d_to_auroc(2.0) == pytest.approx(0.921, abs=0.002)
        assert d_to_auroc(3.0) == pytest.approx(0.983, abs=0.002)


# ================================================================
# power_analysis
# ================================================================

class TestPowerAnalysis:
    def test_n25_paper_claim(self):
        """Paper claims: n=25, alpha=0.05, 80% power → d >= 0.81"""
        min_d = power_analysis(n=25, alpha=0.05, power=0.8)
        assert min_d == pytest.approx(0.81, abs=0.02)

    def test_larger_n_smaller_d(self):
        """More samples → can detect smaller effects"""
        d_small = power_analysis(n=100, alpha=0.05, power=0.8)
        d_large = power_analysis(n=25, alpha=0.05, power=0.8)
        assert d_small < d_large

    def test_higher_power_larger_d(self):
        """Higher power requirement → need larger effect or more samples"""
        d_80 = power_analysis(n=25, alpha=0.05, power=0.8)
        d_95 = power_analysis(n=25, alpha=0.05, power=0.95)
        assert d_95 > d_80

    def test_n5_abliteration(self):
        """At n=5 (abliteration sample size), minimum detectable d is very large"""
        min_d = power_analysis(n=5, alpha=0.05, power=0.8)
        assert min_d > 1.5  # Need huge effects at n=5
