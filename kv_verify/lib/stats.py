"""Statistical testing library for kv_verify.

Adapted from claim-review/stats/independent_stats.py and code/stats_utils.py.
All functions return structured results, not raw floats.

Bug fixes embedded:
  C2: assign_groups() — correct GroupKFold group assignment
  C4: groupkfold_auroc() — within-fold FWL residualization
  C5: holm_bonferroni() — applied (was never called in original code)
  M6: permutation_test() — group-level permutation
"""

import math
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as sp_stats
from scipy.stats import norm
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from kv_verify.types import ClassificationResult


# ================================================================
# EFFECT SIZES
# ================================================================

def _pooled_sd(g1: np.ndarray, g2: np.ndarray) -> float:
    """Pooled standard deviation with Bessel's correction."""
    n1, n2 = len(g1), len(g2)
    var1 = np.var(g1, ddof=1)
    var2 = np.var(g2, ddof=1)
    return math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))


def cohens_d(group1, group2) -> float:
    """Cohen's d with pooled standard deviation.

    Convention: positive d means group1 > group2.
    Returns 0.0 for degenerate cases (n < 2 or zero variance).
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
    """Hedges' g: bias-corrected Cohen's d.

    Correction: g = d * J where J = 1 - 3/(4*df - 1), df = n1 + n2 - 2.
    """
    d = cohens_d(group1, group2)
    n1, n2 = len(group1), len(group2)
    df = n1 + n2 - 2

    if df < 1:
        return 0.0

    J = 1 - 3 / (4 * df - 1)
    return float(d * J)


# ================================================================
# P-VALUES
# ================================================================

def conservative_p(group1, group2) -> float:
    """Conservative p-value: max(Welch's t, Mann-Whitney U).

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
        _, p_mw = sp_stats.mannwhitneyu(g1, g2, alternative="two-sided")
        if np.isnan(p_mw):
            p_mw = 1.0
    except (ValueError, ZeroDivisionError):
        p_mw = 1.0

    return float(max(p_welch, p_mw))


# ================================================================
# EQUIVALENCE TESTING
# ================================================================

def tost(group1, group2, delta: float = 0.3) -> Tuple[float, float, bool]:
    """Two One-Sided Tests (TOST) for equivalence.

    Delta is in Cohen's d units. Tests whether the true effect
    is within [-delta, +delta] in standardized units.

    Returns:
        (p_lower, p_upper, equivalent)
        equivalent is True if both one-sided tests reject at alpha=0.05.
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
    delta_raw = delta * pooled_sd
    df = n1 + n2 - 2

    # Upper bound test: H0: diff >= +delta_raw
    t_upper = (mean_diff - delta_raw) / se
    p_upper = float(sp_stats.t.cdf(t_upper, df=df))

    # Lower bound test: H0: diff <= -delta_raw
    t_lower = (mean_diff + delta_raw) / se
    p_lower = float(1 - sp_stats.t.cdf(t_lower, df=df))

    equivalent = max(p_lower, p_upper) < 0.05

    return (float(p_lower), float(p_upper), equivalent)


# ================================================================
# AUROC CONVERSION
# ================================================================

def d_to_auroc(d: float) -> float:
    """Convert Cohen's d to approximate AUROC.

    AUROC = Phi(d / sqrt(2)) where Phi is the standard normal CDF.
    Assumes equal-variance normal distributions.
    """
    return float(norm.cdf(d / math.sqrt(2)))


# ================================================================
# MULTIPLE COMPARISONS
# ================================================================

def holm_bonferroni(p_values: List[float], alpha: float = 0.05) -> List[Dict]:
    """Holm-Bonferroni correction for multiple comparisons.

    Enforces step-up monotonicity: corrected p-values are non-decreasing
    in the sorted order, so a higher original p never gets a lower
    corrected p than its predecessor.

    Adapted from code/stats_utils.py:210-232.

    Returns:
        List of dicts with: original_p, corrected_p, reject_null, rank.
        Order matches input p_values order.
    """
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    results: List[Optional[Dict]] = [None] * n
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
# BOOTSTRAP
# ================================================================

def bootstrap_ci(
    data,
    statistic: Callable = np.mean,
    n_boot: int = 10000,
    ci: float = 0.95,
    seed: Optional[int] = None,
) -> Dict:
    """Bootstrap confidence interval for any statistic.

    Adapted from code/stats_utils.py:74-88.

    Returns:
        Dict with: estimate, ci_lower, ci_upper, se.
    """
    rng = np.random.RandomState(seed)
    data = np.asarray(data)
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


# ================================================================
# GROUP ASSIGNMENT (C2 fix)
# ================================================================

def assign_groups(
    n_pos: int,
    n_neg: int,
    paired: bool = False,
    prompt_indices_pos: Optional[np.ndarray] = None,
    prompt_indices_neg: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Assign GroupKFold groups correctly.

    FIX for C2 (49_expanded_validation.py:773): original code used
    groups = list(range(n_pos)) + list(range(n_neg)), creating overlapping
    group IDs across classes.

    For non-paired experiments: unique group per sample.
    For paired experiments: same prompt shares group across conditions.
    For stochastic runs: all runs of same prompt share one group.
    """
    if not paired:
        return np.arange(n_pos + n_neg)

    if prompt_indices_pos is None or prompt_indices_neg is None:
        raise ValueError("paired=True requires prompt_indices_pos and prompt_indices_neg")

    pos_idx = np.asarray(prompt_indices_pos)
    neg_idx = np.asarray(prompt_indices_neg)

    if len(pos_idx) != n_pos or len(neg_idx) != n_neg:
        raise ValueError(
            f"prompt_indices length mismatch: pos={len(pos_idx)} vs n_pos={n_pos}, "
            f"neg={len(neg_idx)} vs n_neg={n_neg}"
        )

    # Map prompt indices to shared group IDs.
    # pos prompt 0 and neg prompt 0 get the same group.
    # Use prompt index directly as the group ID.
    groups = np.concatenate([pos_idx, neg_idx])
    return groups


# ================================================================
# CLASSIFIER FACTORY
# ================================================================

def make_classifier():
    """Standard classifier pipeline used across all experiments.

    Centralizes the repeated make_pipeline(StandardScaler(), LogisticRegression(...))
    pattern. Change here to update all experiments.
    """
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, solver="lbfgs"))


# ================================================================
# SHARED AUROC UTILITIES (extracted from experiment DRY violations)
# ================================================================

def stratified_auroc(X, y, n_splits=5, seed=42):
    """Quick AUROC via StratifiedKFold. For null experiments and baselines."""
    from sklearn.model_selection import StratifiedKFold
    clf = make_classifier()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    y_proba = np.full(len(y), np.nan)
    for train_idx, test_idx in skf.split(X, y):
        if len(np.unique(y[train_idx])) < 2:
            continue
        clf.fit(X[train_idx], y[train_idx])
        y_proba[test_idx] = clf.predict_proba(X[test_idx])[:, 1]
    valid = ~np.isnan(y_proba)
    if valid.sum() < 4 or len(np.unique(y[valid])) < 2:
        return 0.5
    return float(roc_auc_score(y[valid], y_proba[valid]))


def loo_auroc(X, y):
    """Leave-one-out AUROC for small samples."""
    from sklearn.model_selection import LeaveOneOut
    clf = make_classifier()
    loo = LeaveOneOut()
    y_proba = np.zeros(len(y))
    for train_idx, test_idx in loo.split(X):
        if len(np.unique(y[train_idx])) < 2:
            y_proba[test_idx] = 0.5
            continue
        clf.fit(X[train_idx], y[train_idx])
        y_proba[test_idx] = clf.predict_proba(X[test_idx])[:, 1]
    return float(roc_auc_score(y, y_proba))


def train_test_auroc(X_train, y_train, X_test, y_test):
    """Train on one set, test on another. Return AUROC."""
    clf = make_classifier()
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return 0.5
    clf.fit(X_train, y_train)
    y_proba = clf.predict_proba(X_test)[:, 1]
    return float(roc_auc_score(y_test, y_proba))


def extract_feature_matrix(items, feature_names=None):
    """Extract feature matrix from a list of result items with 'features' dict."""
    if feature_names is None:
        feature_names = ["norm_per_token", "key_rank", "key_entropy"]
    return np.array([[float(item["features"][f]) for f in feature_names] for item in items])


# ================================================================
# GROUPKFOLD AUROC (C4 fix for within-fold FWL)
# ================================================================

def groupkfold_auroc(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    fwl_confounds: Optional[np.ndarray] = None,
    fwl_within_fold: bool = True,
    feature_names: Optional[List[str]] = None,
) -> ClassificationResult:
    """GroupKFold AUROC with optional within-fold FWL residualization.

    Adapted from 49_expanded_validation.py:623-663.

    FIX C4: When fwl_confounds is provided and fwl_within_fold=True,
    residualization is done INSIDE each fold (train/test split).
    """
    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)
    actual_splits = min(n_splits, n_groups)

    if n_groups < 2:
        raise ValueError(f"Need at least 2 groups, got {n_groups}")

    if feature_names is None:
        feature_names = [f"f{i}" for i in range(X.shape[1])]

    # Handle full-dataset FWL (buggy mode for comparison)
    X_work = X.copy()
    if fwl_confounds is not None and not fwl_within_fold:
        X_work, _ = fwl_residualize(X_work, fwl_confounds, within_fold=False)

    clf = make_classifier()
    gkf = GroupKFold(n_splits=actual_splits)

    y_proba = np.full(len(y), np.nan)
    for train_idx, test_idx in gkf.split(X_work, y, groups):
        X_train, X_test = X_work[train_idx], X_work[test_idx]
        y_train = y[train_idx]

        # Within-fold FWL (C4 fix)
        if fwl_confounds is not None and fwl_within_fold:
            Z = fwl_confounds
            X_train, _ = fwl_residualize(
                X[train_idx], Z[train_idx], within_fold=False,
            )
            # Apply training-fold regression to test data
            reg = LinearRegression()
            reg.fit(Z[train_idx], X[train_idx])
            X_test = X[test_idx] - reg.predict(Z[test_idx])

        if len(np.unique(y_train)) < 2:
            continue

        clf.fit(X_train, y_train)
        y_proba[test_idx] = clf.predict_proba(X_test)[:, 1]

    valid = ~np.isnan(y_proba)
    if valid.sum() < 4 or len(np.unique(y[valid])) < 2:
        auroc = float("nan")
    else:
        auroc = float(roc_auc_score(y[valid], y_proba[valid]))

    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))

    return ClassificationResult(
        auroc=auroc,
        auroc_ci_lower=0.0,  # filled by bootstrap_auroc_ci
        auroc_ci_upper=1.0,
        p_value=1.0,  # filled by permutation_test
        p_value_corrected=None,
        null_mean=0.5,
        null_std=0.0,
        n_positive=n_pos,
        n_negative=n_neg,
        n_groups=n_groups,
        effect_sizes={},
        cv_method=f"GroupKFold-{actual_splits}",
        group_scheme="paired" if n_groups < n_pos + n_neg else "unique",
        bootstrap_n=0,
        permutation_n=0,
        features_used=feature_names,
    )


# ================================================================
# FWL RESIDUALIZATION (C4 fix)
# ================================================================

def fwl_residualize(
    X: np.ndarray,
    Z: np.ndarray,
    within_fold: bool = True,
    train_idx: Optional[np.ndarray] = None,
    test_idx: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, List[float]]:
    """FWL residualization with optional within-fold constraint.

    FIX C4: When within_fold=True, fits OLS on train_idx only,
    applies to both train and test. When within_fold=False, fits
    on all data (original buggy behavior for comparison).

    Frisch, R. & Waugh, F.V. (1933). Lovell, M.C. (1963).

    Returns (X_residualized, r_squared_per_feature).
    """
    Z = np.asarray(Z)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)

    X_resid = X.copy()
    r_squared = []

    if within_fold and train_idx is not None and test_idx is not None:
        reg = LinearRegression()
        for j in range(X.shape[1]):
            reg.fit(Z[train_idx], X[train_idx, j])
            r2 = reg.score(Z[train_idx], X[train_idx, j])
            r_squared.append(float(r2))
            X_resid[train_idx, j] = X[train_idx, j] - reg.predict(Z[train_idx])
            X_resid[test_idx, j] = X[test_idx, j] - reg.predict(Z[test_idx])
    else:
        reg = LinearRegression()
        for j in range(X.shape[1]):
            reg.fit(Z, X[:, j])
            r2 = reg.score(Z, X[:, j])
            r_squared.append(float(r2))
            X_resid[:, j] = X[:, j] - reg.predict(Z)

    return X_resid, r_squared


def fwl_nonlinear(
    X: np.ndarray,
    Z: np.ndarray,
    degree: int = 2,
) -> Tuple[np.ndarray, List[float]]:
    """Polynomial FWL residualization.

    Expands Z to polynomial features up to given degree, then
    residualizes X against the expanded basis. Tests whether
    nonlinear confounds explain the signal.
    """
    Z = np.asarray(Z)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)

    poly = PolynomialFeatures(degree=degree, include_bias=False)
    Z_poly = poly.fit_transform(Z)

    return fwl_residualize(X, Z_poly, within_fold=False)


# ================================================================
# PERMUTATION TEST (M6 fix: group-level)
# ================================================================

def _fast_cv_auroc(X, y, splits, fwl_confounds=None, fwl_within_fold=True):
    """Fast inner loop for CV AUROC — pre-computed splits, reused pipeline."""
    clf = make_classifier()
    y_proba = np.full(len(y), np.nan)

    X_work = X
    if fwl_confounds is not None and not fwl_within_fold:
        X_work, _ = fwl_residualize(X, fwl_confounds, within_fold=False)

    for train_idx, test_idx in splits:
        X_train, X_test = X_work[train_idx], X_work[test_idx]
        y_train = y[train_idx]

        if fwl_confounds is not None and fwl_within_fold:
            Z = fwl_confounds
            X_train, _ = fwl_residualize(X[train_idx], Z[train_idx], within_fold=False)
            reg = LinearRegression()
            reg.fit(Z[train_idx], X[train_idx])
            X_test = X[test_idx] - reg.predict(Z[test_idx])

        if len(np.unique(y_train)) < 2:
            continue
        clf.fit(X_train, y_train)
        y_proba[test_idx] = clf.predict_proba(X_test)[:, 1]

    valid = ~np.isnan(y_proba)
    if valid.sum() < 4 or len(np.unique(y[valid])) < 2:
        return float("nan")
    return float(roc_auc_score(y[valid], y_proba[valid]))


def permutation_test(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_permutations: int = 10000,
    group_level: bool = True,
    seed: int = 42,
    fwl_confounds: Optional[np.ndarray] = None,
    fwl_within_fold: bool = True,
) -> Dict:
    """Permutation test for AUROC significance.

    FIX M6: When group_level=True, permutes group-to-label mapping
    (Winkler et al. 2014), not individual labels. p-value uses
    Phipson & Smyth (2010) correction: (count + 1) / (n_perm + 1).

    Optimized: pre-computes fold splits once (splits depend on groups,
    not y). Pre-computes group membership indices. Reuses pipeline object.
    """
    rng = np.random.RandomState(seed)

    # Pre-compute fold splits once (GroupKFold depends on groups, not y)
    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)
    n_splits = min(5, n_groups)
    gkf = GroupKFold(n_splits=n_splits)
    splits = list(gkf.split(X, y, groups))

    # Observed AUROC
    observed_auroc = _fast_cv_auroc(X, y, splits, fwl_confounds, fwl_within_fold)

    # Pre-compute group membership indices (avoid repeated boolean scans)
    group_indices = {g: np.where(groups == g)[0] for g in unique_groups}

    null_dist = np.zeros(n_permutations)

    if group_level:
        group_labels = np.array([
            int(np.median(y[group_indices[g]] > 0.5))
            for g in unique_groups
        ])

        for i in range(n_permutations):
            perm_labels = rng.permutation(group_labels)
            y_perm = np.empty_like(y)
            for gi, g in enumerate(unique_groups):
                y_perm[group_indices[g]] = perm_labels[gi]
            null_dist[i] = _fast_cv_auroc(X, y_perm, splits, fwl_confounds, fwl_within_fold)
    else:
        for i in range(n_permutations):
            y_perm = rng.permutation(y)
            null_dist[i] = _fast_cv_auroc(X, y_perm, splits, fwl_confounds, fwl_within_fold)

    # Phipson & Smyth (2010): p = (count + 1) / (n_perm + 1)
    count_ge = np.sum(null_dist >= observed_auroc)
    p_value = (count_ge + 1) / (n_permutations + 1)

    return {
        "observed_auroc": float(observed_auroc),
        "p_value": float(p_value),
        "null_distribution": null_dist,
        "null_mean": float(np.mean(null_dist)),
        "null_std": float(np.std(null_dist)),
        "n_permutations": n_permutations,
    }


# ================================================================
# BOOTSTRAP AUROC CI
# ================================================================

def bootstrap_auroc_ci(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_bootstrap: int = 10000,
    method: str = "bca",
    ci: float = 0.95,
    seed: int = 42,
    fwl_confounds: Optional[np.ndarray] = None,
    fwl_within_fold: bool = True,
) -> Dict:
    """Bootstrap CI for AUROC with group-level resampling.

    Supports both percentile and BCa methods.
    BCa: Efron, B. (1987). "Better bootstrap confidence intervals."

    10K resamples (was 1K in original). BCa method (was percentile).
    """
    rng = np.random.RandomState(seed)

    # Observed AUROC
    observed = groupkfold_auroc(
        X, y, groups,
        fwl_confounds=fwl_confounds,
        fwl_within_fold=fwl_within_fold,
    )
    observed_auroc = observed.auroc

    unique_groups = np.unique(groups)
    n_groups_total = len(unique_groups)

    boot_aurocs = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        # Group-level resampling
        sampled_groups = rng.choice(unique_groups, size=n_groups_total, replace=True)

        indices = []
        new_groups = []
        for new_gid, orig_gid in enumerate(sampled_groups):
            mask = groups == orig_gid
            idx = np.where(mask)[0]
            indices.extend(idx.tolist())
            new_groups.extend([new_gid] * len(idx))

        indices = np.array(indices)
        new_groups = np.array(new_groups)

        X_boot = X[indices]
        y_boot = y[indices]
        Z_boot = fwl_confounds[indices] if fwl_confounds is not None else None

        if len(np.unique(y_boot)) < 2:
            boot_aurocs[i] = 0.5
            continue

        result = groupkfold_auroc(
            X_boot, y_boot, new_groups,
            fwl_confounds=Z_boot,
            fwl_within_fold=fwl_within_fold,
        )
        boot_aurocs[i] = result.auroc if not np.isnan(result.auroc) else 0.5

    alpha = (1 - ci) / 2

    if method == "bca":
        # Bias correction
        z0 = norm.ppf(np.mean(boot_aurocs < observed_auroc))
        if np.isinf(z0):
            z0 = 0.0

        # Acceleration (jackknife)
        jack_aurocs = np.zeros(n_groups_total)
        for j, g in enumerate(unique_groups):
            mask = groups != g
            if np.sum(y[mask] == 1) < 2 or np.sum(y[mask] == 0) < 2:
                jack_aurocs[j] = observed_auroc
                continue
            result = groupkfold_auroc(
                X[mask], y[mask], groups[mask],
                fwl_confounds=fwl_confounds[mask] if fwl_confounds is not None else None,
                fwl_within_fold=fwl_within_fold,
            )
            jack_aurocs[j] = result.auroc if not np.isnan(result.auroc) else 0.5

        jack_mean = np.mean(jack_aurocs)
        num = np.sum((jack_mean - jack_aurocs) ** 3)
        denom = 6 * (np.sum((jack_mean - jack_aurocs) ** 2) ** 1.5)
        a = num / denom if denom != 0 else 0.0

        # Adjusted percentiles
        z_alpha = norm.ppf(alpha)
        z_1alpha = norm.ppf(1 - alpha)

        p_lower = norm.cdf(z0 + (z0 + z_alpha) / (1 - a * (z0 + z_alpha)))
        p_upper = norm.cdf(z0 + (z0 + z_1alpha) / (1 - a * (z0 + z_1alpha)))

        ci_lower = float(np.percentile(boot_aurocs, 100 * p_lower))
        ci_upper = float(np.percentile(boot_aurocs, 100 * p_upper))
    else:
        # Percentile method
        ci_lower = float(np.percentile(boot_aurocs, 100 * alpha))
        ci_upper = float(np.percentile(boot_aurocs, 100 * (1 - alpha)))

    return {
        "observed_auroc": float(observed_auroc),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "se": float(np.std(boot_aurocs)),
        "n_bootstrap": n_bootstrap,
        "method": method,
    }


# ================================================================
# POWER ANALYSIS
# ================================================================

def _analytical_power(n: int, d: float, alpha: float = 0.05) -> float:
    """Analytical power for two-sample t-test via non-central t-distribution.

    Replaces simulation (~80K LogReg fits) with exact computation.
    Assumes equal-variance Gaussian, which is the model underlying
    the AUROC-to-d conversion.
    """
    from scipy.stats import nct
    df = 2 * n - 2
    se = math.sqrt(2.0 / max(n, 2))
    noncentrality = d / se
    t_crit = float(sp_stats.t.ppf(1 - alpha / 2, df=df))
    # Power = P(|T| > t_crit) under non-central t
    power = 1.0 - nct.cdf(t_crit, df=df, nc=noncentrality) + nct.cdf(-t_crit, df=df, nc=noncentrality)
    return float(np.clip(power, 0.0, 1.0))


def power_analysis(
    n_per_group: int,
    observed_auroc: float,
    alpha: float = 0.05,
    **kwargs,
) -> Dict:
    """Compute achieved power for observed effect size.

    Uses analytical non-central t-distribution (not simulation).
    Cohen, J. (1988). Chapter 2.

    Returns dict with: achieved_power, min_detectable_auroc, required_n.
    """
    # Convert AUROC to Cohen's d
    observed_auroc_clipped = np.clip(observed_auroc, 0.501, 0.999)
    observed_d = math.sqrt(2) * float(norm.ppf(observed_auroc_clipped))

    n = n_per_group

    # Achieved power (analytical)
    achieved_power = _analytical_power(n, observed_d, alpha)

    # Minimum detectable AUROC at 80% power (binary search, analytical)
    min_detectable_auroc = _find_min_detectable_auroc(n, alpha)

    # Required N for 80% power at observed effect (binary search, analytical)
    required_n = _find_required_n(observed_d, alpha)

    return {
        "achieved_power": achieved_power,
        "min_detectable_auroc": min_detectable_auroc,
        "required_n": required_n,
        "observed_d": float(observed_d),
        "null_critical": float(sp_stats.t.ppf(1 - alpha / 2, df=2 * n - 2)),
    }


def _find_min_detectable_auroc(
    n: int, alpha: float,
    target_power: float = 0.80,
) -> float:
    """Binary search for minimum AUROC detectable at target power (analytical)."""
    lo, hi = 0.51, 0.99
    for _ in range(30):
        mid = (lo + hi) / 2
        d = math.sqrt(2) * float(norm.ppf(mid))
        power = _analytical_power(n, d, alpha)
        if power < target_power:
            lo = mid
        else:
            hi = mid
    return round((lo + hi) / 2, 3)


def _find_required_n(
    d: float, alpha: float, target_power: float = 0.80,
) -> int:
    """Binary search for required N per group at target power (analytical)."""
    if d <= 0.01:
        return 9999
    lo, hi = 5, 500
    for _ in range(30):
        mid = (lo + hi) // 2
        power = _analytical_power(mid, d, alpha)
        if power < target_power:
            lo = mid + 1
        else:
            hi = mid
    return hi


# ================================================================
# HANLEY-MCNEIL SE (analytical AUROC standard error)
# ================================================================

def hanley_mcneil_se(auroc: float, n_pos: int, n_neg: int) -> float:
    """Analytical standard error for AUROC via Hanley & McNeil (1982).

    SE = sqrt((A(1-A) + (n_pos-1)(Q1-A^2) + (n_neg-1)(Q2-A^2)) / (n_pos*n_neg))
    where Q1 = A/(2-A), Q2 = 2A^2/(1+A).
    """
    A = float(np.clip(auroc, 0.001, 0.999))
    Q1 = A / (2 - A)
    Q2 = 2 * A ** 2 / (1 + A)
    numerator = A * (1 - A) + (n_pos - 1) * (Q1 - A ** 2) + (n_neg - 1) * (Q2 - A ** 2)
    se = math.sqrt(max(numerator, 0) / max(n_pos * n_neg, 1))
    return float(se)


# ================================================================
# REPEATED CV (variance estimation)
# ================================================================

def repeated_cv_auroc(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_repeats: int = 20,
    n_splits: int = 5,
    seed: int = 42,
    fwl_confounds: Optional[np.ndarray] = None,
    fwl_within_fold: bool = True,
) -> Dict:
    """Run GroupKFold CV multiple times with shuffled group ordering.

    Returns distribution of AUROCs for variance estimation.
    Varoquaux (2018): single CV gives ±15-20% error at N=40.
    Repeated CV estimates this variance directly.
    """
    rng = np.random.RandomState(seed)
    aurocs = []

    for rep in range(n_repeats):
        # Shuffle samples within each group to get different fold assignments
        # (GroupKFold is deterministic given group order, so we permute group IDs)
        perm = rng.permutation(len(y))
        X_shuf = X[perm]
        y_shuf = y[perm]
        groups_shuf = groups[perm]
        Z_shuf = fwl_confounds[perm] if fwl_confounds is not None else None

        result = groupkfold_auroc(
            X_shuf, y_shuf, groups_shuf,
            n_splits=n_splits,
            fwl_confounds=Z_shuf,
            fwl_within_fold=fwl_within_fold,
        )
        if not np.isnan(result.auroc):
            aurocs.append(result.auroc)

    if not aurocs:
        return {"mean": 0.5, "std": 0.0, "ci_lower": 0.5, "ci_upper": 0.5, "aurocs": []}

    aurocs_arr = np.array(aurocs)
    return {
        "mean": round(float(np.mean(aurocs_arr)), 4),
        "std": round(float(np.std(aurocs_arr)), 4),
        "ci_lower": round(float(np.percentile(aurocs_arr, 2.5)), 4),
        "ci_upper": round(float(np.percentile(aurocs_arr, 97.5)), 4),
        "n_repeats": len(aurocs),
        "aurocs": [round(float(a), 4) for a in aurocs],
    }


# ================================================================
# FULL VALIDATION (all tiers in one call)
# ================================================================

def full_validation(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_permutations: int = 10000,
    n_bootstrap: int = 10000,
    n_repeats: int = 20,
    seed: int = 42,
    fwl_confounds: Optional[np.ndarray] = None,
    fwl_within_fold: bool = True,
) -> Dict:
    """Complete validation: point estimate + CIs + repeated CV + power.

    Tier 1: AUROC point estimate + permutation p-value
    Tier 2: BCa bootstrap CI + repeated CV + Hanley-McNeil SE
    Tier 3: Power analysis

    The verdict_auroc field uses the bootstrap CI lower bound for
    conservative threshold comparison (not the point estimate).
    """
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))

    # Tier 1: Point estimate
    result = groupkfold_auroc(X, y, groups, fwl_confounds=fwl_confounds, fwl_within_fold=fwl_within_fold)
    auroc = result.auroc

    perm = permutation_test(X, y, groups, n_permutations=n_permutations, seed=seed,
                            fwl_confounds=fwl_confounds, fwl_within_fold=fwl_within_fold)

    # Tier 2: Uncertainty quantification
    boot = bootstrap_auroc_ci(X, y, groups, n_bootstrap=n_bootstrap, seed=seed,
                              fwl_confounds=fwl_confounds, fwl_within_fold=fwl_within_fold)

    hm_se = hanley_mcneil_se(auroc, n_pos, n_neg)

    repeated = repeated_cv_auroc(X, y, groups, n_repeats=n_repeats, seed=seed,
                                 fwl_confounds=fwl_confounds, fwl_within_fold=fwl_within_fold)

    # Tier 3: Power
    pwr = power_analysis(n_per_group=min(n_pos, n_neg), observed_auroc=auroc)

    # Conservative verdict value: use bootstrap CI lower bound
    verdict_auroc = boot["ci_lower"]

    return {
        # Tier 1
        "auroc": auroc,
        "p_value": perm["p_value"],
        "n_positive": n_pos,
        "n_negative": n_neg,
        # Tier 2
        "auroc_ci_lower": boot["ci_lower"],
        "auroc_ci_upper": boot["ci_upper"],
        "auroc_std": repeated["std"],
        "hanley_mcneil_se": hm_se,
        "repeated_cv": repeated,
        "bootstrap": boot,
        # Tier 3
        "power": pwr["achieved_power"],
        "required_n": pwr["required_n"],
        # For verdict thresholds: use CI lower bound (conservative)
        "verdict_auroc": verdict_auroc,
    }
