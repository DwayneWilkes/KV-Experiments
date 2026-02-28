"""
Shared Statistical Infrastructure for KV-Cache Experiments
==========================================================

Centralized statistical functions used across all experiment scripts.
Extracted to eliminate ~600 lines of duplication and prevent drift.

Convention: cohens_d(condition, baseline) — positive d means condition > baseline.

Used by: 01d, 03, 03b, 04, 05, 06, 07, 07b, 08, 09, 10, 11, 12, 13
"""

import sys
import platform
from datetime import datetime
from typing import List, Dict, Optional

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
        return {"w_statistic": 0.0, "p_value": 1.0, "is_normal": True}
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


def cohens_d_ci(group1, group2, n_boot=5000, ci=0.95, seed=None):
    """Bootstrap CI for Cohen's d."""
    rng = np.random.RandomState(seed)
    g1, g2 = np.array(group1), np.array(group2)
    boot_ds = []
    for _ in range(n_boot):
        b1 = rng.choice(g1, len(g1), replace=True)
        b2 = rng.choice(g2, len(g2), replace=True)
        boot_ds.append(cohens_d(b1, b2))
    alpha = (1 - ci) / 2
    return {
        "d": cohens_d(g1, g2),
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
# FULL COMPARISON BATTERY
# ================================================================

def full_comparison(group1, group2, label="", seed=None):
    """Run the complete statistical battery on two groups."""
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
    both_normal = result["normality_g1"]["is_normal"] and result["normality_g2"]["is_normal"]

    # Parametric
    result["welch_t"] = welch_t(g1, g2)

    # Nonparametric
    result["mann_whitney"] = mann_whitney(g1, g2)

    # Effect size with CI
    result["cohens_d"] = cohens_d_ci(g1, g2, seed=seed)
    result["cohens_d"]["interpretation"] = interpret_d(result["cohens_d"]["d"])

    # Bootstrap mean difference
    result["bootstrap_diff"] = bootstrap_diff_ci(g1, g2, seed=seed)

    # Recommended test based on normality
    if both_normal:
        result["recommended_test"] = "welch_t"
        result["recommended_p"] = result["welch_t"]["p_value"]
    else:
        result["recommended_test"] = "mann_whitney"
        result["recommended_p"] = result["mann_whitney"]["p_value"]

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
