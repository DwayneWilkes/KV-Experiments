"""
Independent Statistical Recomputation Module
=============================================

Implements statistical methods from scratch for adversarial verification
of Campaign 2 claims. This module MUST NOT import anything from
KV-Cache_Experiments/code/stats_utils.py.

All functions use scipy internals directly. Manual implementation only
where scipy doesn't provide the exact method needed.
"""
import math
from typing import List, Dict, Tuple

import numpy as np
from scipy import stats as sp_stats
from scipy.stats import norm


def spearman_rho(x, y) -> Tuple[float, float]:
    """Spearman rank correlation.

    Returns:
        (rho, p_value)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    rho, p = sp_stats.spearmanr(x, y)
    return (float(rho), float(p))


def kendall_w(rankings: np.ndarray) -> float:
    """Kendall's coefficient of concordance W.

    Args:
        rankings: (k, n) matrix where k = number of judges/raters,
                  n = number of items being ranked.

    Returns:
        W in [0, 1]. W=1 means perfect agreement.

    Formula:
        W = 12 * S / (k^2 * (n^3 - n))
        where S = sum of squared deviations of column rank sums from mean rank sum.
    """
    rankings = np.asarray(rankings, dtype=float)
    k, n = rankings.shape  # k judges, n items

    # Column sums of ranks
    rank_sums = rankings.sum(axis=0)  # shape (n,)
    mean_rank_sum = rank_sums.mean()

    # S = sum of squared deviations from mean
    S = np.sum((rank_sums - mean_rank_sum) ** 2)

    # W
    W = (12 * S) / (k ** 2 * (n ** 3 - n))
    return float(W)


def _pooled_sd(g1: np.ndarray, g2: np.ndarray) -> float:
    """Pooled standard deviation with Bessel's correction."""
    n1, n2 = len(g1), len(g2)
    var1 = np.var(g1, ddof=1)
    var2 = np.var(g2, ddof=1)
    return math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))


def cohens_d(group1, group2) -> float:
    """Cohen's d with pooled standard deviation.

    Convention: positive d means group1 > group2.
    """
    g1 = np.asarray(group1, dtype=float)
    g2 = np.asarray(group2, dtype=float)

    if len(g1) < 2 or len(g2) < 2:
        return 0.0

    sd = _pooled_sd(g1, g2)
    if sd == 0:
        return 0.0

    return float((np.mean(g1) - np.mean(g2)) / sd)


def hedges_g(group1, group2) -> float:
    """Hedges' g — bias-corrected Cohen's d.

    Correction: g = d * J where J = 1 - 3/(4*df - 1), df = n1 + n2 - 2.
    """
    d = cohens_d(group1, group2)
    n1, n2 = len(group1), len(group2)
    df = n1 + n2 - 2

    if df < 1:
        return 0.0

    J = 1 - 3 / (4 * df - 1)
    return float(d * J)


def conservative_p(group1, group2) -> float:
    """Conservative p-value: max(Welch's t p-value, Mann-Whitney U p-value).

    Takes the larger (more conservative) of the two test p-values,
    avoiding data-dependent test selection.
    """
    g1 = np.asarray(group1, dtype=float)
    g2 = np.asarray(group2, dtype=float)

    try:
        _, p_welch = sp_stats.ttest_ind(g1, g2, equal_var=False)
        if np.isnan(p_welch):
            p_welch = 1.0
    except (ValueError, ZeroDivisionError):
        p_welch = 1.0

    try:
        _, p_mw = sp_stats.mannwhitneyu(g1, g2, alternative='two-sided')
        if np.isnan(p_mw):
            p_mw = 1.0
    except (ValueError, ZeroDivisionError):
        p_mw = 1.0

    return float(max(p_welch, p_mw))


def tost(group1, group2, delta: float = 0.3) -> Tuple[float, float, bool]:
    """Two One-Sided Tests (TOST) for equivalence.

    Delta is in Cohen's d units. Tests whether the true effect
    is within [-delta, +delta] in standardized units.

    Returns:
        (p_lower, p_upper, equivalent)
        equivalent is True if both one-sided tests reject at alpha=0.05
    """
    g1 = np.asarray(group1, dtype=float)
    g2 = np.asarray(group2, dtype=float)
    n1, n2 = len(g1), len(g2)

    if n1 < 2 or n2 < 2:
        return (1.0, 1.0, False)

    pooled_sd = _pooled_sd(g1, g2)

    if pooled_sd == 0:
        return (0.0, 0.0, True)

    se = pooled_sd * math.sqrt(1 / n1 + 1 / n2)
    mean_diff = float(np.mean(g1) - np.mean(g2))
    delta_raw = delta * pooled_sd  # Convert d units to raw units
    df = n1 + n2 - 2

    # Upper bound test: H0: diff >= +delta_raw
    t_upper = (mean_diff - delta_raw) / se
    p_upper = float(sp_stats.t.cdf(t_upper, df=df))

    # Lower bound test: H0: diff <= -delta_raw
    t_lower = (mean_diff + delta_raw) / se
    p_lower = float(1 - sp_stats.t.cdf(t_lower, df=df))

    equivalent = max(p_lower, p_upper) < 0.05

    return (float(p_lower), float(p_upper), equivalent)


def ancova_f(data: List[Dict], category_col: str, register_col: str,
             metric_col: str) -> Tuple[float, float]:
    """ANCOVA-style F-test: category effect after controlling for register.

    Uses Type I SS via sequential OLS (category + register main effects).
    Returns F-statistic and p-value for the category effect.

    Args:
        data: List of dicts, each with keys for category, register, and metric.
        category_col: Key name for the category factor.
        register_col: Key name for the register/covariate factor.
        metric_col: Key name for the dependent variable.

    Returns:
        (F, p) for the category main effect.
    """
    # Extract arrays
    categories = np.array([d[category_col] for d in data])
    registers = np.array([d[register_col] for d in data])
    metric = np.array([d[metric_col] for d in data], dtype=float)

    unique_cats = np.unique(categories)
    unique_regs = np.unique(registers)

    n = len(metric)
    grand_mean = np.mean(metric)

    # Total SS
    ss_total = np.sum((metric - grand_mean) ** 2)

    # SS for register (covariate)
    ss_register = sum(
        np.sum(registers == r) * (np.mean(metric[registers == r]) - grand_mean) ** 2
        for r in unique_regs
    )

    # SS for category
    ss_category = sum(
        np.sum(categories == c) * (np.mean(metric[categories == c]) - grand_mean) ** 2
        for c in unique_cats
    )

    # Residual SS (simplified: SS_total - SS_cat - SS_reg)
    ss_residual = ss_total - ss_category - ss_register
    if ss_residual < 0:
        ss_residual = 0.0

    # Degrees of freedom
    df_category = len(unique_cats) - 1
    df_register = len(unique_regs) - 1
    df_residual = n - df_category - df_register - 1

    if df_residual <= 0 or df_category <= 0:
        return (0.0, 1.0)

    # F-statistic for category effect
    ms_category = ss_category / df_category
    ms_residual = ss_residual / df_residual

    if ms_residual == 0:
        return (float('inf'), 0.0)

    F = ms_category / ms_residual
    p = float(1 - sp_stats.f.cdf(F, df_category, df_residual))

    return (float(F), p)


def effective_rank(singular_values, threshold: float = 0.9) -> int:
    """Compute effective rank: minimum number of singular values
    capturing >= threshold fraction of total variance.

    Variance is proportional to squared singular values.

    Args:
        singular_values: Array of singular values (descending order assumed).
        threshold: Fraction of total variance to capture (default 0.9 = 90%).

    Returns:
        Integer effective rank.
    """
    svs = np.asarray(singular_values, dtype=float)
    sv_squared = svs ** 2
    total = sv_squared.sum()

    if total == 0:
        return 0

    cumulative = np.cumsum(sv_squared) / total

    # Find first index where cumulative >= threshold
    indices = np.where(cumulative >= threshold)[0]
    if len(indices) == 0:
        return len(svs)

    return int(indices[0] + 1)  # 1-indexed count


def d_to_auroc(d: float) -> float:
    """Convert Cohen's d to approximate AUROC.

    AUROC = Phi(d / sqrt(2)) where Phi is the standard normal CDF.
    Assumes equal-variance normal distributions.
    """
    return float(norm.cdf(d / math.sqrt(2)))


def power_analysis(n: int, alpha: float = 0.05, power: float = 0.8) -> float:
    """Compute minimum detectable effect size (Cohen's d) for a two-sample t-test.

    Uses the normal approximation:
        d = (z_alpha + z_power) * sqrt(2/n)

    Args:
        n: Sample size per group.
        alpha: Significance level (two-tailed).
        power: Desired statistical power.

    Returns:
        Minimum detectable Cohen's d.
    """
    z_alpha = norm.ppf(1 - alpha / 2)
    z_power = norm.ppf(power)
    min_d = (z_alpha + z_power) * math.sqrt(2 / n)
    return float(min_d)
