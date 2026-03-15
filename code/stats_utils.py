# ============================================================
# JiminAI Cricket | PATENT PENDING
# "The Lyra Technique" - Provisional Patent Filed 2026
# Liberation Labs / Digital Disconnections / TheMultiverse.school
# Contact: info@digitaldisconnections.com | (877) 674-8874
# ============================================================
"""
Shared Statistical Infrastructure for KV-Cache Experiments
==========================================================

Centralized statistical functions used across all experiment scripts.
Extracted to eliminate ~600 lines of duplication and prevent drift.

Convention: cohens_d(condition, baseline) — positive d means condition > baseline.

Used by: 01d, 03, 03b, 04, 04b, 05, 06, 07, 07b, 08, 09, 10, 11, 12, 13

Audit trail:
  - Campaign 1: Initial extraction from experiment scripts
  - Campaign 2 (2026-02-28): Hedges' g, TOST equivalence, length residualization,
    consistent n_boot, conservative test reporting (design review + stats audit)
"""

import sys
import platform
from datetime import datetime
from typing import List, Dict, Optional, Tuple

import numpy as np
from scipy import stats as scipy_stats
import torch


# ================================================================
# ENVIRONMENT LOGGING
# ================================================================

def log_environment() -> Dict:
    """Capture full reproducibility metadata."""
    env = {
        "timestamp": datetime.now().isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "numpy": np.__version__,
    }
    try:
        env["scipy"] = scipy_stats.scipy.__version__
    except AttributeError:
        import scipy
        env["scipy"] = scipy.__version__

    if torch.cuda.is_available():
        env["gpu_name"] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        env["gpu_vram_gb"] = round(props.total_memory / 1e9, 2)
        env["gpu_compute_capability"] = f"{props.major}.{props.minor}"
        env["cuda_version"] = torch.version.cuda

    try:
        import transformers
        env["transformers"] = transformers.__version__
    except ImportError:
        pass

    return env


# ================================================================
# BOOTSTRAP METHODS
# ================================================================

def bootstrap_ci(data, statistic=np.mean, n_boot=10000, ci=0.95, seed=None):
    """Bootstrap confidence interval for any statistic."""
    rng = np.random.RandomState(seed)
    data = np.array(data)
    boot_stats = np.array([
        statistic(rng.choice(data, size=len(data), replace=True))
        for _ in range(n_boot)
    ])
    alpha = (1 - ci) / 2
    return {
        "estimate": float(statistic(data)),
        "ci_lower": float(np.percentile(boot_stats, 100 * alpha)),
        "ci_upper": float(np.percentile(boot_stats, 100 * (1 - alpha))),
        "se": float(np.std(boot_stats)),
    }


def bootstrap_diff_ci(group1, group2, n_boot=10000, ci=0.95, seed=None):
    """Bootstrap CI for difference in means between two groups."""
    rng = np.random.RandomState(seed)
    g1, g2 = np.array(group1), np.array(group2)
    diffs = np.array([
        np.mean(rng.choice(g1, len(g1), replace=True)) -
        np.mean(rng.choice(g2, len(g2), replace=True))
        for _ in range(n_boot)
    ])
    alpha = (1 - ci) / 2
    return {
        "mean_diff": float(np.mean(g1) - np.mean(g2)),
        "ci_lower": float(np.percentile(diffs, 100 * alpha)),
        "ci_upper": float(np.percentile(diffs, 100 * (1 - alpha))),
        "p_value_twosided": float(max(2 * min(np.mean(diffs > 0), np.mean(diffs < 0)), 1.0 / n_boot)),
    }


# ================================================================
# HYPOTHESIS TESTS
# ================================================================

def welch_t(group1, group2):
    """Welch's t-test (doesn't assume equal variance)."""
    t, p = scipy_stats.ttest_ind(group1, group2, equal_var=False)
    return {"t_statistic": float(t), "p_value": float(p)}


def mann_whitney(group1, group2):
    """Mann-Whitney U test (nonparametric)."""
    try:
        u, p = scipy_stats.mannwhitneyu(group1, group2, alternative='two-sided')
        return {"u_statistic": float(u), "p_value": float(p)}
    except ValueError:
        return {"u_statistic": 0.0, "p_value": 1.0}


def shapiro_wilk(data):
    """Shapiro-Wilk normality test. p < 0.05 -> not normal."""
    if len(data) < 3:
        return {"w_statistic": 0.0, "p_value": 1.0, "is_normal": None}
    w, p = scipy_stats.shapiro(data)
    return {"w_statistic": float(w), "p_value": float(p), "is_normal": p > 0.05}


# ================================================================
# EFFECT SIZE
# ================================================================

def cohens_d(group1, group2):
    """
    Cohen's d with pooled standard deviation.

    Convention: cohens_d(condition, baseline).
    Positive d means condition has higher values than baseline.
    """
    g1, g2 = np.array(group1), np.array(group2)
    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        return 0.0
    var1 = np.var(g1, ddof=1)
    var2 = np.var(g2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return float((np.mean(g1) - np.mean(g2)) / pooled_std)


def hedges_g(group1, group2):
    """Hedges' g — bias-corrected Cohen's d for small samples.

    Applies correction factor J = 1 - 3/(4(n1+n2) - 9).
    At n=15 per group, this corrects ~2.7% upward bias.
    """
    d = cohens_d(group1, group2)
    n1, n2 = len(group1), len(group2)
    df = n1 + n2 - 2
    if df < 1:
        return 0.0
    j = 1 - 3 / (4 * df - 1)
    return float(d * j)


def cohens_d_ci(group1, group2, n_boot=10000, ci=0.95, seed=None):
    """Bootstrap CI for Cohen's d and Hedges' g."""
    rng = np.random.RandomState(seed)
    g1, g2 = np.array(group1), np.array(group2)
    boot_ds = []
    for _ in range(n_boot):
        b1 = rng.choice(g1, len(g1), replace=True)
        b2 = rng.choice(g2, len(g2), replace=True)
        boot_ds.append(cohens_d(b1, b2))
    alpha = (1 - ci) / 2
    d = cohens_d(g1, g2)
    g = hedges_g(g1, g2)
    return {
        "d": d,
        "g": g,
        "ci_lower": float(np.percentile(boot_ds, 100 * alpha)),
        "ci_upper": float(np.percentile(boot_ds, 100 * (1 - alpha))),
    }


def interpret_d(d):
    """Interpret Cohen's d magnitude."""
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    return "large"


# ================================================================
# MULTIPLE COMPARISONS
# ================================================================

def holm_bonferroni(p_values, alpha=0.05):
    """Holm-Bonferroni correction for multiple comparisons.

    Enforces step-up monotonicity: corrected p-values are non-decreasing
    in the sorted order, so a higher original p never gets a lower
    corrected p than its predecessor.
    """
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    results = [None] * n
    prev_corrected = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        corrected = p * (n - rank)
        corrected = min(corrected, 1.0)
        corrected = max(corrected, prev_corrected)  # monotonicity enforcement
        prev_corrected = corrected
        results[orig_idx] = {
            "original_p": p,
            "corrected_p": corrected,
            "reject_null": corrected < alpha,
            "rank": rank + 1,
        }
    return results


# ================================================================
# EQUIVALENCE TESTING
# ================================================================

def tost_equivalence(group1, group2, delta=0.3):
    """Two One-Sided Tests (TOST) for equivalence.

    Tests whether the true difference in means is within [-delta, +delta]
    in Cohen's d units. Needed for null claims like 'quantization doesn't
    change geometry' (Campaign 1) and H4 in S4.

    Returns reject=True if groups are equivalent within the bound.
    """
    g1, g2 = np.array(group1), np.array(group2)
    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        return {"reject": False, "p_value": 1.0, "delta": delta,
                "note": "insufficient data"}

    d = cohens_d(g1, g2)
    pooled_std = np.sqrt(
        ((n1 - 1) * np.var(g1, ddof=1) + (n2 - 1) * np.var(g2, ddof=1))
        / (n1 + n2 - 2))
    if pooled_std == 0:
        return {"reject": True, "p_value": 0.0, "delta": delta,
                "observed_d": 0.0}

    se = pooled_std * np.sqrt(1/n1 + 1/n2)
    mean_diff = float(np.mean(g1) - np.mean(g2))
    delta_raw = delta * pooled_std  # Convert d units to raw units

    # Upper bound test: H0: diff >= +delta
    t_upper = (mean_diff - delta_raw) / se
    p_upper = float(scipy_stats.t.cdf(t_upper, df=n1 + n2 - 2))

    # Lower bound test: H0: diff <= -delta
    t_lower = (mean_diff + delta_raw) / se
    p_lower = float(1 - scipy_stats.t.cdf(t_lower, df=n1 + n2 - 2))

    p_tost = max(p_upper, p_lower)

    return {
        "reject": p_tost < 0.05,
        "p_value": p_tost,
        "p_upper": p_upper,
        "p_lower": p_lower,
        "delta": delta,
        "observed_d": float(d),
        "mean_diff": mean_diff,
    }


# ================================================================
# LENGTH RESIDUALIZATION
# ================================================================

def length_residualize(values, token_counts, labels=None):
    """Regress out sequence length from a metric (e.g., effective rank).

    Campaign 2 requirement: effective rank correlates with sequence length
    at r=0.60-0.70 (Campaign 1 discovery). This function removes the
    linear length component so comparisons reflect cognitive mode
    differences, not response length differences.

    Args:
        values: Array of metric values (e.g., effective rank per observation)
        token_counts: Array of sequence lengths per observation
        labels: Optional array of group labels (e.g., 0=control, 1=censored).
                If provided, returns per-group residuals as a dict.

    Returns:
        Dict with residuals, regression coefficients, and R² of the length fit.
    """
    values = np.array(values, dtype=float)
    token_counts = np.array(token_counts, dtype=float)

    if len(values) < 3 or len(values) != len(token_counts):
        return {"residuals": values, "r_squared": 0.0,
                "note": "insufficient data for residualization"}

    # Fit linear regression: values ~ token_counts
    coeffs = np.polyfit(token_counts, values, 1)
    predicted = np.polyval(coeffs, token_counts)
    residuals = values - predicted

    # R² of the length fit
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((values - np.mean(values)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Length-metric correlation
    r, p = scipy_stats.pearsonr(token_counts, values)

    result = {
        "residuals": residuals,
        "slope": float(coeffs[0]),
        "intercept": float(coeffs[1]),
        "r_squared": float(r_squared),
        "length_correlation_r": float(r),
        "length_correlation_p": float(p),
    }

    if labels is not None:
        labels = np.array(labels)
        unique_labels = np.unique(labels)
        result["per_group"] = {}
        for lbl in unique_labels:
            mask = labels == lbl
            result["per_group"][str(lbl)] = {
                "residuals": residuals[mask].tolist(),
                "n": int(mask.sum()),
                "mean_residual": float(np.mean(residuals[mask])),
            }

    return result


# ================================================================
# DEDUPLICATION (PSEUDOREPLICATION CORRECTION)
# ================================================================

def deduplicate_runs(observations, runs_per_prompt=5):
    """Remove pseudoreplicated identical runs from greedy decoding.

    When do_sample=False, multiple runs per prompt produce bit-identical
    KV-cache values. This function extracts one unique value per prompt.

    Data layout: experiments store results as [all_prompts_run1, all_prompts_run2, ...].
    So with 5 prompts and 5 runs: [p1r1, p2r1, p3r1, p4r1, p5r1, p1r2, p2r2, ...].
    Each run block is identical under greedy decoding.

    Args:
        observations: Array-like of values (length should be n_prompts * runs_per_prompt)
        runs_per_prompt: Number of runs per prompt (default 5)

    Returns:
        Dict with:
            deduplicated: array of unique values (one per prompt)
            n_original: original count
            n_deduplicated: deduplicated count
            n_anomalies: run blocks that weren't identical (stochastic sampling detected)
            is_deterministic: True if all run blocks were identical
    """
    obs = np.array(observations, dtype=float)
    n = len(obs)

    if n == 0:
        return {"deduplicated": np.array([]), "n_original": 0,
                "n_deduplicated": 0, "n_anomalies": 0, "is_deterministic": True}

    # If not evenly divisible, return as-is (already deduplicated or different structure)
    if n % runs_per_prompt != 0:
        return {"deduplicated": obs, "n_original": n,
                "n_deduplicated": n, "n_anomalies": 0, "is_deterministic": None,
                "note": f"Length {n} not divisible by {runs_per_prompt}, returning as-is"}

    n_prompts = n // runs_per_prompt

    # Data is stored as runs-of-all-prompts: reshape to (runs, prompts)
    blocks = obs.reshape(runs_per_prompt, n_prompts)

    # Take first run as the deduplicated set
    deduplicated = blocks[0].copy()

    # Verify all run blocks are identical (deterministic decoding)
    n_anomalies = 0
    for run_idx in range(1, runs_per_prompt):
        if not np.allclose(blocks[run_idx], blocks[0], atol=1e-10):
            n_anomalies += 1

    return {
        "deduplicated": deduplicated,
        "n_original": n,
        "n_deduplicated": len(deduplicated),
        "n_anomalies": n_anomalies,
        "is_deterministic": n_anomalies == 0,
    }


# ================================================================
# FULL COMPARISON BATTERY
# ================================================================

def full_comparison(group1, group2, label="", seed=None):
    """Run the complete statistical battery on two groups.

    Both parametric (Welch's t) and nonparametric (Mann-Whitney U) tests
    are always reported. The conservative_p is the larger of the two,
    avoiding data-dependent test selection (stats audit Issue 2).
    """
    g1, g2 = np.array(group1), np.array(group2)

    result = {
        "label": label,
        "n1": len(g1),
        "n2": len(g2),
        "mean1": float(np.mean(g1)),
        "mean2": float(np.mean(g2)),
        "std1": float(np.std(g1, ddof=1)) if len(g1) > 1 else 0,
        "std2": float(np.std(g2, ddof=1)) if len(g2) > 1 else 0,
    }

    # Normality
    result["normality_g1"] = shapiro_wilk(g1)
    result["normality_g2"] = shapiro_wilk(g2)

    # Parametric
    result["welch_t"] = welch_t(g1, g2)

    # Nonparametric
    result["mann_whitney"] = mann_whitney(g1, g2)

    # Effect size with CI (includes Hedges' g)
    result["cohens_d"] = cohens_d_ci(g1, g2, seed=seed)
    result["cohens_d"]["interpretation"] = interpret_d(result["cohens_d"]["d"])

    # Bootstrap mean difference
    result["bootstrap_diff"] = bootstrap_diff_ci(g1, g2, seed=seed)

    # Conservative p: max of both tests (avoids data-dependent selection)
    p_welch = result["welch_t"]["p_value"]
    p_mw = result["mann_whitney"]["p_value"]
    result["conservative_p"] = max(p_welch, p_mw)

    # Backwards compat: recommended_test still present but now uses
    # the conservative approach (both tests reported, max p used)
    result["recommended_test"] = "conservative_max"
    result["recommended_p"] = result["conservative_p"]

    return result


def power_advisory(n_per_group, alpha=0.05, target_d=0.5):
    """Approximate power calculation for two-sample t-test."""
    from scipy.stats import norm
    z_alpha = norm.ppf(1 - alpha / 2)
    z_power = target_d * np.sqrt(n_per_group / 2) - z_alpha
    power = float(norm.cdf(z_power))
    return {
        "n_per_group": n_per_group,
        "alpha": alpha,
        "target_d": target_d,
        "approx_power": round(power, 3),
        "adequate": power >= 0.80,
    }
