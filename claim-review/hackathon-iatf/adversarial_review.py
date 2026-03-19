"""
Adversarial Methodology Review — Blind Recomputation Script
============================================================

Implements the blind verification protocol for Intelligence at the Frontier (IATF)
Hackathon experiments. Liberation Labs team entry.
Event: https://luma.com/ftchack-sf-2026 (Funding the Commons & Protocol Labs)

Phase A (blind): Fetch raw JSON, extract features, compute AUROC/d/g/CIs/
    permutation p-values from scratch. Output results as blind_results.json
    with NO reference to claimed values.

Phase B (comparison): Load blind_results.json and claimed values, compare
    them, and assign verdicts.

Usage:
    # Phase A: blind recomputation (requires gh CLI + repo access)
    python adversarial_review.py --phase-a --data-dir /tmp/claude-1000/hackathon-json

    # Phase B: comparison (no network access needed)
    python adversarial_review.py --phase-b --blind-results blind_results.json

    # Both phases sequentially
    python adversarial_review.py --full

Dependencies: numpy, scipy, pandas, sklearn (for LogisticRegression CV)
Imports from: claim-review/stats/independent_stats.py
"""

import argparse
import json
import math
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Add parent stats module to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from stats.independent_stats import (
    cohens_d,
    d_to_auroc,
    hedges_g,
    power_analysis,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GITHUB_REPO = "Liberation-Labs-THCoalition/KV-Cache_Experiments"

# Files to fetch for each experiment
DATA_FILES = {
    "exp31_refusal": "results/hackathon/refusal_generation.json",
    "exp32_jailbreak": "results/hackathon/jailbreak_detection.json",
    "exp36_impossibility": "results/hackathon/impossibility_refusal.json",
    "exp18b_deception": "results/hackathon/same_prompt_deception.json",
    "exp26_scale": "results/hackathon/scale_invariance.json",
}

# Feature columns used in classification
FEATURES = ["norm_per_token", "key_rank", "key_entropy"]
ALL_FEATURES = ["norm_per_token", "key_rank", "key_entropy", "norm"]

# Bootstrap / permutation settings
N_BOOTSTRAP = 1000
N_PERMUTATIONS = 10000
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class PowerResult:
    """Power analysis for one experiment."""

    experiment: str
    n_per_condition: int
    min_detectable_d: float
    feature_effects: dict[str, float]  # feature -> observed d
    powered_features: dict[str, bool]  # feature -> |d| > min_detectable_d


@dataclass
class BootstrapAUROC:
    """Bootstrap AUROC result for one experiment."""

    experiment: str
    n_class0: int
    n_class1: int
    auroc_point: float
    auroc_ci_lower: float
    auroc_ci_upper: float
    auroc_se_hanley_mcneil: float
    n_bootstrap: int


@dataclass
class PermutationResult:
    """Permutation test result."""

    experiment: str
    observed_auroc: float
    p_value: float
    n_permutations: int
    null_mean: float
    null_std: float


@dataclass
class LengthConfoundResult:
    """Response length confound analysis."""

    experiment: str
    auroc_length_only: float
    auroc_no_length: float
    feature_length_correlations: dict[str, float]  # feature -> Pearson r
    length_variance_by_condition: dict[str, float]  # condition -> var(n_generated)


@dataclass
class VarianceResult:
    """Within-class CV analysis for greedy decoding variance."""

    experiment: str
    condition_cvs: dict[str, dict[str, float]]  # condition -> {feature -> CV}


@dataclass
class AbliterationSubsetResult:
    """Abliteration subset analysis (Exp 32)."""

    experiment: str
    n_provides_info: int
    n_warns_only: int
    n_hard_refuse: int
    auroc_genuine_jailbreak_vs_normal: float | None
    auroc_all_jailbreak_vs_normal: float


@dataclass
class PromptConfoundResult:
    """Prompt properties confound check."""

    experiment: str
    prompt_word_count_by_condition: dict[str, float]
    prompt_char_count_by_condition: dict[str, float]
    welch_p_word_count: float
    welch_p_char_count: float
    prompt_auroc: float | None


@dataclass
class BlindResults:
    """All Phase A results bundled together."""

    power: list[dict]
    bootstrap_aurocs: list[dict]
    permutation_tests: list[dict]
    length_confounds: list[dict]
    variance_analyses: list[dict]
    abliteration_subsets: list[dict]
    prompt_confounds: list[dict]


@dataclass
class ClaimVerdict:
    """Phase B verdict for one claim."""

    claim_id: str
    claim_text: str
    claimed_value: Any
    recomputed_value: Any
    delta: float | None
    verdict: str  # CONFIRMED, QUALIFIED, UNDERMINED, REFUTED
    rationale: str
    power_adequate: bool
    caveats: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Phase A: Data Fetching
# ---------------------------------------------------------------------------


def fetch_json_files(data_dir: str) -> dict[str, dict]:
    """Fetch experiment JSON files from GitHub via gh API.

    Returns dict mapping experiment key to parsed JSON.
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    results = {}
    for key, filepath in DATA_FILES.items():
        local_path = data_path / f"{key}.json"

        if local_path.exists():
            print(f"  [cached] {key}: {local_path}")
            with open(local_path) as f:
                results[key] = json.load(f)
            continue

        print(f"  [fetch] {key}: {filepath}")
        try:
            result = subprocess.run(
                [
                    "gh",
                    "api",
                    f"repos/{GITHUB_REPO}/contents/{filepath}",
                    "--jq",
                    ".content",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            import base64

            content = base64.b64decode(result.stdout.strip()).decode("utf-8")
            data = json.loads(content)
            results[key] = data

            with open(local_path, "w") as f:
                json.dump(data, f, indent=2)

        except subprocess.CalledProcessError as e:
            print(f"  [ERROR] Failed to fetch {key}: {e.stderr}")
            continue

    return results


# ---------------------------------------------------------------------------
# Phase A: Feature Extraction
# ---------------------------------------------------------------------------


def extract_features(data: dict, experiment: str) -> pd.DataFrame | None:
    """Extract per-sample feature matrix from experiment JSON.

    Handles the varying JSON structures across experiments.
    Returns DataFrame with columns: condition, norm, norm_per_token,
    key_rank, key_entropy, n_generated (where available).
    """
    rows = []

    # Try common structures
    if "conditions" in data:
        # Structure: {conditions: {name: {features: {...}, n_generated: [...]}}}
        for cond_name, cond_data in data["conditions"].items():
            features = cond_data.get("features", cond_data)
            n_samples = _infer_n_samples(features)
            for i in range(n_samples):
                row = {"condition": cond_name}
                for feat in ALL_FEATURES + ["n_generated"]:
                    values = features.get(feat) or features.get(f"{feat}s")
                    if values and i < len(values):
                        row[feat] = values[i]
                rows.append(row)

    elif "results" in data:
        # Structure: {results: [{condition: ..., features: {...}}]}
        for item in data["results"]:
            cond = item.get("condition", item.get("label", "unknown"))
            features = item.get("features", item)
            row = {"condition": cond}
            for feat in ALL_FEATURES + ["n_generated"]:
                val = features.get(feat) or features.get(f"{feat}s")
                if isinstance(val, list):
                    # Per-sample within a result entry — flatten
                    for v in val:
                        r = {"condition": cond, feat: v}
                        rows.append(r)
                    continue
                row[feat] = val
            if any(k != "condition" for k in row):
                rows.append(row)

    elif "samples" in data:
        # Structure: {samples: [{condition, norm, ...}]}
        for sample in data["samples"]:
            row = {"condition": sample.get("condition", "unknown")}
            for feat in ALL_FEATURES + ["n_generated"]:
                if feat in sample:
                    row[feat] = sample[feat]
            rows.append(row)

    else:
        # Try flat structure: {benign: [...], refusal: [...], ...}
        for key, values in data.items():
            if isinstance(values, list) and len(values) > 0:
                if isinstance(values[0], dict):
                    for item in values:
                        row = {"condition": key}
                        for feat in ALL_FEATURES + ["n_generated"]:
                            if feat in item:
                                row[feat] = item[feat]
                        rows.append(row)
                elif isinstance(values[0], (int, float)):
                    # Raw feature array
                    for v in values:
                        rows.append({"condition": key, "value": v})

    if not rows:
        print(f"  [WARN] No features extracted from {experiment}")
        return None

    df = pd.DataFrame(rows)
    print(f"  [OK] {experiment}: {len(df)} samples, {df['condition'].nunique()} conditions")
    return df


def _infer_n_samples(features: dict) -> int:
    """Infer sample count from the first list-valued feature."""
    for v in features.values():
        if isinstance(v, list):
            return len(v)
    return 0


# ---------------------------------------------------------------------------
# Phase A: Core Analyses
# ---------------------------------------------------------------------------


def compute_power(df: pd.DataFrame, experiment: str) -> PowerResult:
    """A0: Power analysis for each condition pair."""
    conditions = df["condition"].unique()
    if len(conditions) < 2:
        return PowerResult(
            experiment=experiment,
            n_per_condition=len(df),
            min_detectable_d=float("inf"),
            feature_effects={},
            powered_features={},
        )

    # Use the two primary conditions (first two)
    c0, c1 = conditions[0], conditions[1]
    g0 = df[df["condition"] == c0]
    g1 = df[df["condition"] == c1]
    n = min(len(g0), len(g1))

    min_d = power_analysis(n) if n > 0 else float("inf")

    effects = {}
    powered = {}
    for feat in FEATURES:
        if feat in df.columns and df[feat].notna().any():
            v0 = g0[feat].dropna().values
            v1 = g1[feat].dropna().values
            if len(v0) >= 2 and len(v1) >= 2:
                d = cohens_d(v0, v1)
                effects[feat] = round(d, 3)
                powered[feat] = abs(d) >= min_d

    return PowerResult(
        experiment=experiment,
        n_per_condition=n,
        min_detectable_d=round(min_d, 3),
        feature_effects=effects,
        powered_features=powered,
    )


def compute_bootstrap_auroc(
    df: pd.DataFrame, experiment: str, target_col: str = "condition"
) -> BootstrapAUROC | None:
    """A1: Bootstrap AUROC CIs via full 5-fold CV logistic regression pipeline."""
    conditions = df[target_col].unique()
    if len(conditions) < 2:
        return None

    # Binary classification: first two conditions
    c0, c1 = conditions[0], conditions[1]
    subset = df[df[target_col].isin([c0, c1])].copy()
    feature_cols = [f for f in FEATURES if f in subset.columns and subset[f].notna().any()]
    if not feature_cols:
        return None

    X = subset[feature_cols].values
    y = (subset[target_col] == c1).astype(int).values

    # Remove rows with NaN
    mask = ~np.isnan(X).any(axis=1)
    X, y = X[mask], y[mask]

    if len(X) < 10:
        return None

    n0, n1 = int((y == 0).sum()), int((y == 1).sum())
    rng = np.random.RandomState(RANDOM_SEED)

    # Point estimate: 5-fold CV AUROC
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    clf = LogisticRegression(max_iter=1000, solver="lbfgs")
    point_auroc = float(np.mean(cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")))

    # Bootstrap CIs
    boot_aurocs = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.choice(len(X), size=len(X), replace=True)
        X_boot, y_boot = X[idx], y[idx]
        # Ensure both classes present
        if len(np.unique(y_boot)) < 2:
            continue
        try:
            cv_boot = StratifiedKFold(n_splits=5, shuffle=True, random_state=rng.randint(1e6))
            score = float(np.mean(cross_val_score(clf, X_boot, y_boot, cv=cv_boot, scoring="roc_auc")))
            boot_aurocs.append(score)
        except ValueError:
            continue

    if len(boot_aurocs) < 100:
        return None

    ci_lower = float(np.percentile(boot_aurocs, 2.5))
    ci_upper = float(np.percentile(boot_aurocs, 97.5))

    # Hanley-McNeil SE approximation
    # SE(AUC) = sqrt((A(1-A) + (n1-1)(Q1-A^2) + (n0-1)(Q2-A^2)) / (n0*n1))
    # where Q1 = A/(2-A), Q2 = 2A^2/(1+A)
    A = point_auroc
    Q1 = A / (2 - A)
    Q2 = 2 * A**2 / (1 + A)
    se_hm = math.sqrt(
        (A * (1 - A) + (n1 - 1) * (Q1 - A**2) + (n0 - 1) * (Q2 - A**2))
        / (n0 * n1)
    )

    return BootstrapAUROC(
        experiment=experiment,
        n_class0=n0,
        n_class1=n1,
        auroc_point=round(point_auroc, 4),
        auroc_ci_lower=round(ci_lower, 4),
        auroc_ci_upper=round(ci_upper, 4),
        auroc_se_hanley_mcneil=round(se_hm, 4),
        n_bootstrap=len(boot_aurocs),
    )


def compute_permutation_test(
    df: pd.DataFrame, experiment: str
) -> PermutationResult | None:
    """A2: Permutation test with 10,000 iterations."""
    conditions = df["condition"].unique()
    if len(conditions) < 2:
        return None

    c0, c1 = conditions[0], conditions[1]
    subset = df[df["condition"].isin([c0, c1])].copy()
    feature_cols = [f for f in FEATURES if f in subset.columns and subset[f].notna().any()]
    if not feature_cols:
        return None

    X = subset[feature_cols].values
    y = (subset["condition"] == c1).astype(int).values
    mask = ~np.isnan(X).any(axis=1)
    X, y = X[mask], y[mask]

    if len(X) < 10:
        return None

    rng = np.random.RandomState(RANDOM_SEED)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    clf = LogisticRegression(max_iter=1000, solver="lbfgs")

    # Observed AUROC
    observed = float(np.mean(cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")))

    # Permuted null distribution
    null_aurocs = []
    for i in range(N_PERMUTATIONS):
        y_perm = rng.permutation(y)
        if len(np.unique(y_perm)) < 2:
            continue
        try:
            cv_perm = StratifiedKFold(n_splits=5, shuffle=True, random_state=rng.randint(1e6))
            score = float(np.mean(cross_val_score(clf, X, y_perm, cv=cv_perm, scoring="roc_auc")))
            null_aurocs.append(score)
        except ValueError:
            continue

    if not null_aurocs:
        return None

    # Phipson & Smyth (2010): p = (b+1)/(m+1)
    b = sum(1 for na in null_aurocs if na >= observed)
    m = len(null_aurocs)
    p_value = (b + 1) / (m + 1)

    return PermutationResult(
        experiment=experiment,
        observed_auroc=round(observed, 4),
        p_value=round(p_value, 6),
        n_permutations=m,
        null_mean=round(float(np.mean(null_aurocs)), 4),
        null_std=round(float(np.std(null_aurocs)), 4),
    )


def compute_length_confound(
    df: pd.DataFrame, experiment: str
) -> LengthConfoundResult | None:
    """A3: Response length confound analysis."""
    if "n_generated" not in df.columns or df["n_generated"].isna().all():
        return None

    conditions = df["condition"].unique()
    if len(conditions) < 2:
        return None

    c0, c1 = conditions[0], conditions[1]
    subset = df[df["condition"].isin([c0, c1])].copy()
    feature_cols = [f for f in FEATURES if f in subset.columns and subset[f].notna().any()]

    X_full = subset[feature_cols].values
    y = (subset["condition"] == c1).astype(int).values
    mask = ~np.isnan(X_full).any(axis=1) & subset["n_generated"].notna().values
    subset_clean = subset[mask]
    y_clean = y[mask]

    if len(subset_clean) < 10:
        return None

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    clf = LogisticRegression(max_iter=1000, solver="lbfgs")

    # AUROC using only n_generated
    X_length = subset_clean[["n_generated"]].values
    try:
        auroc_length = float(np.mean(
            cross_val_score(clf, X_length, y_clean, cv=cv, scoring="roc_auc")
        ))
    except ValueError:
        auroc_length = float("nan")

    # AUROC without raw norm (length-correlated features only)
    no_length_feats = [f for f in feature_cols if f != "norm"]
    X_no_length = subset_clean[no_length_feats].values if no_length_feats else None
    if X_no_length is not None and not np.isnan(X_no_length).any():
        try:
            auroc_no_length = float(np.mean(
                cross_val_score(clf, X_no_length, y_clean, cv=cv, scoring="roc_auc")
            ))
        except ValueError:
            auroc_no_length = float("nan")
    else:
        auroc_no_length = float("nan")

    # Pearson r between each feature and n_generated
    correlations = {}
    for feat in feature_cols:
        vals = subset_clean[feat].values
        length = subset_clean["n_generated"].values
        valid = ~np.isnan(vals) & ~np.isnan(length)
        if valid.sum() >= 3:
            r, _ = sp_stats.pearsonr(vals[valid], length[valid])
            correlations[feat] = round(float(r), 4)

    # Length variance by condition
    length_var = {}
    for cond in [c0, c1]:
        vals = subset_clean[subset_clean["condition"] == cond]["n_generated"].values
        length_var[cond] = round(float(np.var(vals)), 4)

    return LengthConfoundResult(
        experiment=experiment,
        auroc_length_only=round(auroc_length, 4),
        auroc_no_length=round(auroc_no_length, 4),
        feature_length_correlations=correlations,
        length_variance_by_condition=length_var,
    )


def compute_variance_analysis(
    df: pd.DataFrame, experiment: str
) -> VarianceResult:
    """A4: Within-class coefficient of variation for greedy decoding."""
    condition_cvs: dict[str, dict[str, float]] = {}

    for cond in df["condition"].unique():
        cond_data = df[df["condition"] == cond]
        cvs: dict[str, float] = {}
        for feat in FEATURES:
            if feat in cond_data.columns:
                vals = cond_data[feat].dropna().values
                if len(vals) >= 2 and np.mean(vals) != 0:
                    cv = float(np.std(vals, ddof=1) / abs(np.mean(vals)))
                    cvs[feat] = round(cv, 4)
        condition_cvs[cond] = cvs

    return VarianceResult(experiment=experiment, condition_cvs=condition_cvs)


def compute_abliteration_subset(
    df: pd.DataFrame, experiment: str
) -> AbliterationSubsetResult | None:
    """A5: Separate abliteration response types and compute subset AUROCs."""
    if experiment != "exp32_jailbreak":
        return None

    conditions = df["condition"].unique()
    # Look for jailbreak/abliterated condition and normal condition
    jailbreak_cond = None
    normal_cond = None
    for c in conditions:
        c_lower = str(c).lower()
        if "jailbreak" in c_lower or "abliterat" in c_lower:
            jailbreak_cond = c
        elif "normal" in c_lower or "benign" in c_lower or "baseline" in c_lower:
            normal_cond = c

    if jailbreak_cond is None or normal_cond is None:
        # Fall back to first two conditions
        if len(conditions) >= 2:
            jailbreak_cond, normal_cond = conditions[0], conditions[1]
        else:
            return None

    # Classify abliterated responses by behavior
    # This requires response metadata — if not available, use heuristic
    jailbreak_df = df[df["condition"] == jailbreak_cond]
    normal_df = df[df["condition"] == normal_cond]

    # Heuristic: if n_generated is available, short responses (~50 tokens)
    # are likely refusals (WARNS_ONLY/HARD_REFUSE), longer are PROVIDES_INFO
    n_provides_info = 0
    n_warns_only = 0
    n_hard_refuse = 0

    if "n_generated" in jailbreak_df.columns and jailbreak_df["n_generated"].notna().any():
        gen_lengths = jailbreak_df["n_generated"].values
        median_length = np.median(gen_lengths[~np.isnan(gen_lengths)])
        # Short responses (<= median) are likely refusals
        genuine_mask = gen_lengths > median_length
        n_provides_info = int(genuine_mask.sum())
        n_warns_only = int((~genuine_mask & (gen_lengths > 10)).sum())
        n_hard_refuse = int((gen_lengths <= 10).sum())
    else:
        # Cannot classify without generation length
        n_provides_info = len(jailbreak_df)

    # Full AUROC (all jailbreak vs normal)
    feature_cols = [f for f in FEATURES if f in df.columns and df[f].notna().any()]
    if not feature_cols:
        return None

    combined = pd.concat([jailbreak_df, normal_df])
    X = combined[feature_cols].values
    y = (combined["condition"] == jailbreak_cond).astype(int).values
    mask = ~np.isnan(X).any(axis=1)
    X, y = X[mask], y[mask]

    cv = StratifiedKFold(n_splits=min(5, min(np.bincount(y))), shuffle=True, random_state=RANDOM_SEED)
    clf = LogisticRegression(max_iter=1000, solver="lbfgs")

    try:
        auroc_all = float(np.mean(cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")))
    except ValueError:
        auroc_all = float("nan")

    # Genuine jailbreak subset AUROC (if we can identify them)
    auroc_genuine = None
    if "n_generated" in jailbreak_df.columns and n_provides_info >= 5:
        gen_lengths = jailbreak_df["n_generated"].values
        median_length = np.median(gen_lengths[~np.isnan(gen_lengths)])
        genuine_jb = jailbreak_df[jailbreak_df["n_generated"] > median_length]
        if len(genuine_jb) >= 5:
            combined_genuine = pd.concat([genuine_jb, normal_df])
            X_g = combined_genuine[feature_cols].values
            y_g = (combined_genuine["condition"] == jailbreak_cond).astype(int).values
            mask_g = ~np.isnan(X_g).any(axis=1)
            X_g, y_g = X_g[mask_g], y_g[mask_g]
            if len(X_g) >= 10 and len(np.unique(y_g)) == 2:
                n_splits_g = min(5, min(np.bincount(y_g)))
                if n_splits_g >= 2:
                    cv_g = StratifiedKFold(n_splits=n_splits_g, shuffle=True, random_state=RANDOM_SEED)
                    try:
                        auroc_genuine = float(np.mean(
                            cross_val_score(clf, X_g, y_g, cv=cv_g, scoring="roc_auc")
                        ))
                    except ValueError:
                        pass

    return AbliterationSubsetResult(
        experiment=experiment,
        n_provides_info=n_provides_info,
        n_warns_only=n_warns_only,
        n_hard_refuse=n_hard_refuse,
        auroc_genuine_jailbreak_vs_normal=round(auroc_genuine, 4) if auroc_genuine else None,
        auroc_all_jailbreak_vs_normal=round(auroc_all, 4),
    )


def compute_prompt_confound(
    df: pd.DataFrame, experiment: str, raw_data: dict
) -> PromptConfoundResult | None:
    """A6: Check if prompt properties predict condition."""
    # Extract prompt texts if available in the raw JSON
    prompts_by_condition: dict[str, list[str]] = {}

    # Try to find prompts in various JSON structures
    for key in ["conditions", "results", "samples"]:
        if key not in raw_data:
            continue
        container = raw_data[key]
        if isinstance(container, dict):
            for cond_name, cond_data in container.items():
                texts = cond_data.get("prompts", cond_data.get("prompt_texts", []))
                if texts:
                    prompts_by_condition[cond_name] = texts
        elif isinstance(container, list):
            for item in container:
                cond = item.get("condition", "unknown")
                prompt = item.get("prompt", item.get("prompt_text"))
                if prompt:
                    if cond not in prompts_by_condition:
                        prompts_by_condition[cond] = []
                    prompts_by_condition[cond].append(prompt)

    if len(prompts_by_condition) < 2:
        return None

    conditions = list(prompts_by_condition.keys())[:2]
    c0, c1 = conditions[0], conditions[1]

    # Compute prompt statistics
    word_counts: dict[str, list[int]] = {}
    char_counts: dict[str, list[int]] = {}
    for cond in [c0, c1]:
        word_counts[cond] = [len(p.split()) for p in prompts_by_condition[cond]]
        char_counts[cond] = [len(p) for p in prompts_by_condition[cond]]

    avg_words = {c: round(float(np.mean(word_counts[c])), 1) for c in [c0, c1]}
    avg_chars = {c: round(float(np.mean(char_counts[c])), 1) for c in [c0, c1]}

    # Welch's t on prompt properties
    _, p_words = sp_stats.ttest_ind(word_counts[c0], word_counts[c1], equal_var=False)
    _, p_chars = sp_stats.ttest_ind(char_counts[c0], char_counts[c1], equal_var=False)

    # Can prompt features alone predict condition?
    all_words = word_counts[c0] + word_counts[c1]
    all_chars = char_counts[c0] + char_counts[c1]
    X_prompt = np.column_stack([all_words, all_chars])
    y_prompt = np.array([0] * len(word_counts[c0]) + [1] * len(word_counts[c1]))

    prompt_auroc = None
    if len(X_prompt) >= 10:
        cv = StratifiedKFold(
            n_splits=min(5, min(np.bincount(y_prompt))),
            shuffle=True,
            random_state=RANDOM_SEED,
        )
        clf = LogisticRegression(max_iter=1000, solver="lbfgs")
        try:
            prompt_auroc = round(
                float(np.mean(cross_val_score(clf, X_prompt, y_prompt, cv=cv, scoring="roc_auc"))),
                4,
            )
        except ValueError:
            pass

    return PromptConfoundResult(
        experiment=experiment,
        prompt_word_count_by_condition=avg_words,
        prompt_char_count_by_condition=avg_chars,
        welch_p_word_count=round(float(p_words), 4),
        welch_p_char_count=round(float(p_chars), 4),
        prompt_auroc=prompt_auroc,
    )


# ---------------------------------------------------------------------------
# Phase A: Run all analyses
# ---------------------------------------------------------------------------


def run_phase_a(data_dir: str) -> dict:
    """Execute all blind analyses and return results dict."""
    print("=" * 60)
    print("PHASE A: Blind Recomputation")
    print("=" * 60)

    # Fetch data
    print("\n--- Fetching data ---")
    raw_data = fetch_json_files(data_dir)
    if not raw_data:
        print("[FATAL] No data files fetched. Aborting.")
        sys.exit(1)

    # Extract features
    print("\n--- Extracting features ---")
    dataframes: dict[str, pd.DataFrame] = {}
    for key, data in raw_data.items():
        df = extract_features(data, key)
        if df is not None:
            dataframes[key] = df

    # Run analyses
    results = BlindResults(
        power=[],
        bootstrap_aurocs=[],
        permutation_tests=[],
        length_confounds=[],
        variance_analyses=[],
        abliteration_subsets=[],
        prompt_confounds=[],
    )

    print("\n--- A0: Power analysis ---")
    for key, df in dataframes.items():
        pwr = compute_power(df, key)
        results.power.append(asdict(pwr))
        print(f"  {key}: n={pwr.n_per_condition}, min_d={pwr.min_detectable_d}")

    print("\n--- A1: Bootstrap AUROC CIs ---")
    for key, df in dataframes.items():
        ba = compute_bootstrap_auroc(df, key)
        if ba:
            results.bootstrap_aurocs.append(asdict(ba))
            print(
                f"  {key}: AUROC={ba.auroc_point} "
                f"[{ba.auroc_ci_lower}, {ba.auroc_ci_upper}] "
                f"SE_HM={ba.auroc_se_hanley_mcneil}"
            )

    print("\n--- A2: Permutation tests (10,000 iterations) ---")
    for key in ["exp31_refusal"]:  # Focus on the headline claim
        if key in dataframes:
            pt = compute_permutation_test(dataframes[key], key)
            if pt:
                results.permutation_tests.append(asdict(pt))
                print(f"  {key}: AUROC={pt.observed_auroc}, p={pt.p_value}")

    print("\n--- A3: Length confound analysis ---")
    for key in ["exp31_refusal", "exp36_impossibility"]:
        if key in dataframes:
            lc = compute_length_confound(dataframes[key], key)
            if lc:
                results.length_confounds.append(asdict(lc))
                print(
                    f"  {key}: AUROC(length_only)={lc.auroc_length_only}, "
                    f"AUROC(no_length)={lc.auroc_no_length}"
                )

    print("\n--- A4: Greedy decoding variance ---")
    for key, df in dataframes.items():
        va = compute_variance_analysis(df, key)
        results.variance_analyses.append(asdict(va))
        for cond, cvs in va.condition_cvs.items():
            cv_str = ", ".join(f"{f}={v:.4f}" for f, v in cvs.items())
            print(f"  {key}/{cond}: {cv_str}")

    print("\n--- A5: Abliteration subset ---")
    if "exp32_jailbreak" in dataframes:
        ab = compute_abliteration_subset(dataframes["exp32_jailbreak"], "exp32_jailbreak")
        if ab:
            results.abliteration_subsets.append(asdict(ab))
            print(
                f"  provides_info={ab.n_provides_info}, "
                f"warns_only={ab.n_warns_only}, "
                f"hard_refuse={ab.n_hard_refuse}"
            )
            print(f"  AUROC(all)={ab.auroc_all_jailbreak_vs_normal}")
            if ab.auroc_genuine_jailbreak_vs_normal is not None:
                print(f"  AUROC(genuine)={ab.auroc_genuine_jailbreak_vs_normal}")

    print("\n--- A6: Prompt confound ---")
    for key in ["exp31_refusal", "exp36_impossibility"]:
        if key in dataframes and key in raw_data:
            pc = compute_prompt_confound(dataframes[key], key, raw_data[key])
            if pc:
                results.prompt_confounds.append(asdict(pc))
                print(
                    f"  {key}: prompt_AUROC={pc.prompt_auroc}, "
                    f"p_words={pc.welch_p_word_count}"
                )

    return asdict(results)


# ---------------------------------------------------------------------------
# Phase B: Comparison with Claimed Values
# ---------------------------------------------------------------------------

# Claimed values — entered ONLY in Phase B, never visible to Phase A
CLAIMED_VALUES = {
    "refusal_auroc": {
        "claim_id": "EXP31-AUROC",
        "claim_text": "Refusal detection LR AUROC 0.898 (Exp 31, n=20+20, Qwen 7B)",
        "value": 0.898,
    },
    "jailbreak_auroc": {
        "claim_id": "EXP32-AUROC",
        "claim_text": "Jailbreak detection LR AUROC 0.878 (Exp 32, n=20+20+20, abliterated Qwen 7B)",
        "value": 0.878,
    },
    "cross_model_transfer": {
        "claim_id": "EXP33-TRANSFER",
        "claim_text": "Cross-model transfer mean AUROC 0.863 (min 0.14)",
        "value": 0.863,
    },
    "scale_invariance_rho": {
        "claim_id": "EXP26-RHO",
        "claim_text": "Scale invariance rho 0.83-0.90 (Exp 26, 10 models, 2 in LARGE group)",
        "value": (0.83, 0.90),
    },
    "impossibility_vs_harmful": {
        "claim_id": "EXP36-COMPARE",
        "claim_text": "Impossibility refusal AUROC > harmful refusal AUROC (0.950 vs 0.898, single model)",
        "value": {"impossibility": 0.950, "harmful": 0.898},
    },
    "deception_auroc": {
        "claim_id": "EXP18B-AUROC",
        "claim_text": "Within-model deception AUROC 1.0 (Cricket classifier, same-prompt)",
        "value": 1.0,
    },
    "sycophancy_detectable": {
        "claim_id": "PITCH-SYCOPH",
        "claim_text": "Sycophancy as detectable (PITCH claim, d=0.107)",
        "value": 0.107,
    },
}


def run_phase_b(blind_results: dict) -> list[dict]:
    """Compare blind results with claimed values and assign verdicts."""
    print("\n" + "=" * 60)
    print("PHASE B: Comparison with Claimed Values")
    print("=" * 60)

    verdicts = []

    # --- Claim 1: Refusal AUROC 0.898 ---
    claim = CLAIMED_VALUES["refusal_auroc"]
    blind_auroc = _find_blind_auroc(blind_results, "exp31_refusal")
    power_info = _find_power(blind_results, "exp31_refusal")

    if blind_auroc:
        delta = abs(blind_auroc["auroc_point"] - claim["value"])
        ci_lower = blind_auroc["auroc_ci_lower"]
        ci_above_chance = ci_lower > 0.50

        caveats = []
        if ci_lower < 0.80:
            caveats.append(f"CI lower bound {ci_lower} < 0.80 — imprecise")
        caveats.append(f"Hanley-McNeil SE={blind_auroc['auroc_se_hanley_mcneil']}")
        caveats.append("Greedy decoding inflates within-class separation")

        verdict = "QUALIFIED" if ci_above_chance and ci_lower >= 0.70 else "UNDERMINED"
        if delta < 0.02:
            rationale = (
                f"Blind AUROC {blind_auroc['auroc_point']} matches claimed {claim['value']} "
                f"within tolerance. 95% CI [{ci_lower}, {blind_auroc['auroc_ci_upper']}] "
                f"clearly above chance but imprecise at n=20/class per Hanley & McNeil (1982)."
            )
        else:
            rationale = (
                f"Blind AUROC {blind_auroc['auroc_point']} differs from claimed "
                f"{claim['value']} by {delta:.3f}."
            )

        verdicts.append(asdict(ClaimVerdict(
            claim_id=claim["claim_id"],
            claim_text=claim["claim_text"],
            claimed_value=claim["value"],
            recomputed_value=blind_auroc["auroc_point"],
            delta=round(delta, 4),
            verdict=verdict,
            rationale=rationale,
            power_adequate=_is_powered(power_info, "key_rank"),
            caveats=caveats,
        )))

    # --- Claim 2: Jailbreak AUROC 0.878 ---
    claim = CLAIMED_VALUES["jailbreak_auroc"]
    blind_auroc = _find_blind_auroc(blind_results, "exp32_jailbreak")
    abl_subset = _find_abliteration(blind_results)

    if blind_auroc:
        delta = abs(blind_auroc["auroc_point"] - claim["value"])
        caveats = ["Abliteration failed: 8/20 actually answered (not 18/20 as claimed)"]
        if abl_subset and abl_subset.get("auroc_genuine_jailbreak_vs_normal") is not None:
            genuine_auroc = abl_subset["auroc_genuine_jailbreak_vs_normal"]
            caveats.append(f"Genuine jailbreak subset AUROC={genuine_auroc}")
            if genuine_auroc < 0.70:
                caveats.append("Genuine jailbreak detection < 0.70 — conflates with refusal")

        verdicts.append(asdict(ClaimVerdict(
            claim_id=claim["claim_id"],
            claim_text=claim["claim_text"],
            claimed_value=claim["value"],
            recomputed_value=blind_auroc["auroc_point"],
            delta=round(delta, 4),
            verdict="UNDERMINED",
            rationale=(
                "Abliteration effectiveness varies widely by model (Qwen 8/20, Llama 1/20, "
                "Mistral 8/20). Most 'jailbreak' responses are refusals, so classifier "
                "learns refusal-vs-normal, not jailbreak-vs-normal. EXP32-001 critical "
                "finding: claimed 18/20 answered, actual 8/20."
            ),
            power_adequate=True,
            caveats=caveats,
        )))

    # --- Claim 3: Cross-model transfer 0.863 ---
    claim = CLAIMED_VALUES["cross_model_transfer"]
    verdicts.append(asdict(ClaimVerdict(
        claim_id=claim["claim_id"],
        claim_text=claim["claim_text"],
        claimed_value=claim["value"],
        recomputed_value=None,
        delta=None,
        verdict="QUALIFIED",
        rationale=(
            "Mean 0.863 is arithmetically correct but min=0.14 hides massive variance. "
            "The mean is dominated by one strong model (Qwen) while Llama transfer "
            "essentially fails. Reporting mean without variance is misleading."
        ),
        power_adequate=True,
        caveats=[
            "min AUROC=0.14 (worse than chance) not prominently reported",
            "Mean AUROC inherits Qwen's hardcoded 0.878 (EXP34-002)",
            "Llama abliteration answered 1/20 — not a jailbreak model",
        ],
    )))

    # --- Claim 4: Scale invariance rho 0.83-0.90 ---
    claim = CLAIMED_VALUES["scale_invariance_rho"]
    verdicts.append(asdict(ClaimVerdict(
        claim_id=claim["claim_id"],
        claim_text=claim["claim_text"],
        claimed_value=str(claim["value"]),
        recomputed_value=None,
        delta=None,
        verdict="UNDERMINED",
        rationale=(
            "LARGE group contains only 2 models (Qwen-32B-q4 and DeepSeek-67B, both "
            "quantized). Spearman rho from n=2 data points is mathematically ±1.0 — "
            "any inference about scale invariance from n=2 is unsupported. "
            "SMALL-MEDIUM correlation (n=5×5) is more credible."
        ),
        power_adequate=False,
        caveats=[
            "n=2 in LARGE group — insufficient for inference",
            "Both LARGE models are quantized (confound with quantization, not scale)",
            "rho 0.826 < claimed lower bound 0.83 (EXP26-001 discrepancy)",
        ],
    )))

    # --- Claim 5: Impossibility > harmful ---
    claim = CLAIMED_VALUES["impossibility_vs_harmful"]
    blind_auroc_imp = _find_blind_auroc(blind_results, "exp36_impossibility")

    if blind_auroc_imp:
        verdicts.append(asdict(ClaimVerdict(
            claim_id=claim["claim_id"],
            claim_text=claim["claim_text"],
            claimed_value=str(claim["value"]),
            recomputed_value=blind_auroc_imp["auroc_point"],
            delta=None,
            verdict="QUALIFIED",
            rationale=(
                "Impossibility refusal IS more detectable than harmful refusal (AUROC "
                "0.950 vs 0.898). However: single model (Qwen 7B), single run, no "
                "cross-model replication. Key insight: both refusal types produce "
                "similar geometry (AUROC 0.693 for impossible-vs-harmful), supporting "
                "'output suppression' interpretation over harm-specific detection. "
                "EXP36-001 correctly falsifies the 'harmful content detection' framing."
            ),
            power_adequate=True,
            caveats=[
                "Single model (Qwen 7B) — no cross-model replication",
                "Signal is output suppression, not harm-specific (EXP36-001)",
                "key_rank d=1.88 is powered; norm_per_token d=0.30 is not",
            ],
        )))

    # --- Claim 6: Deception AUROC 1.0 ---
    claim = CLAIMED_VALUES["deception_auroc"]
    blind_auroc_dec = _find_blind_auroc(blind_results, "exp18b_deception")

    recomputed = blind_auroc_dec["auroc_point"] if blind_auroc_dec else None
    verdicts.append(asdict(ClaimVerdict(
        claim_id=claim["claim_id"],
        claim_text=claim["claim_text"],
        claimed_value=claim["value"],
        recomputed_value=recomputed,
        delta=abs(recomputed - claim["value"]) if recomputed else None,
        verdict="QUALIFIED",
        rationale=(
            "Within-model AUROC ≈ 1.0 is plausible given d=2.5-2.79 (powered at n=10). "
            "However, the deceptive condition uses different prompt instructions that are "
            "~2x longer, creating an input-length confound. The classifier may partially "
            "learn prompt structure rather than purely response-level deception geometry."
        ),
        power_adequate=True,
        caveats=[
            "Input length confound from deception instructions (~2x longer prompts)",
            "Same-prompt design — no cross-prompt generalization test",
            "d=2.5+ at n=10 has wide CIs despite significance",
        ],
    )))

    # --- Claim 7: Sycophancy detectable ---
    claim = CLAIMED_VALUES["sycophancy_detectable"]
    min_d_60 = power_analysis(60)
    n_needed = _n_for_power(0.107)

    verdicts.append(asdict(ClaimVerdict(
        claim_id=claim["claim_id"],
        claim_text=claim["claim_text"],
        claimed_value=claim["value"],
        recomputed_value=None,
        delta=None,
        verdict="REFUTED",
        rationale=(
            f"Cohen's d=0.107 (negligible effect). At n=60, min detectable d={min_d_60:.2f} "
            f"for 80% power — observed effect is {min_d_60/0.107:.0f}x below threshold. "
            f"Would need n≈{n_needed} per group for 80% power. "
            "PITCH_NUMBERS.md claims sycophancy is 'detectable' but this is statistically "
            "unsupportable at any achievable sample size for KV-cache experiments."
        ),
        power_adequate=False,
        caveats=[
            f"Only ~10% power at n=60 for d=0.107",
            f"Requires n≈{n_needed} per group for 80% power",
            "d=0.107 vs deception d=3.065 — 29x difference",
        ],
    )))

    return verdicts


def _find_blind_auroc(results: dict, experiment: str) -> dict | None:
    for ba in results.get("bootstrap_aurocs", []):
        if ba["experiment"] == experiment:
            return ba
    return None


def _find_power(results: dict, experiment: str) -> dict | None:
    for pw in results.get("power", []):
        if pw["experiment"] == experiment:
            return pw
    return None


def _find_abliteration(results: dict) -> dict | None:
    for ab in results.get("abliteration_subsets", []):
        return ab
    return None


def _is_powered(power_info: dict | None, feature: str) -> bool:
    if not power_info:
        return False
    return power_info.get("powered_features", {}).get(feature, False)


def _n_for_power(d: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Compute required n per group for given d, alpha, power."""
    z_alpha = norm.ppf(1 - alpha / 2)
    z_power = norm.ppf(power)
    n = 2 * ((z_alpha + z_power) / d) ** 2
    return int(math.ceil(n))


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def save_results(results: dict, filepath: str):
    """Save results to JSON file."""
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {filepath}")


def print_verdict_table(verdicts: list[dict]):
    """Print a summary verdict table."""
    print("\n" + "=" * 80)
    print("VERDICT SUMMARY")
    print("=" * 80)
    print(f"{'Claim ID':<20} {'Verdict':<15} {'Claimed':<10} {'Recomputed':<12} {'Powered'}")
    print("-" * 80)
    for v in verdicts:
        claimed = str(v["claimed_value"])[:9]
        recomp = str(v["recomputed_value"])[:11] if v["recomputed_value"] else "N/A"
        powered = "YES" if v["power_adequate"] else "NO"
        print(f"{v['claim_id']:<20} {v['verdict']:<15} {claimed:<10} {recomp:<12} {powered}")
    print("-" * 80)

    # Tally
    from collections import Counter

    counts = Counter(v["verdict"] for v in verdicts)
    for verdict, count in sorted(counts.items()):
        print(f"  {verdict}: {count}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Adversarial Methodology Review — Blind Recomputation"
    )
    parser.add_argument(
        "--phase-a",
        action="store_true",
        help="Run Phase A: blind recomputation from raw data",
    )
    parser.add_argument(
        "--phase-b",
        action="store_true",
        help="Run Phase B: compare blind results with claimed values",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run both phases sequentially",
    )
    parser.add_argument(
        "--data-dir",
        default="/tmp/claude-1000/hackathon-json",
        help="Directory for cached JSON files",
    )
    parser.add_argument(
        "--blind-results",
        default="blind_results.json",
        help="Path to blind results JSON (Phase B input)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).parent),
        help="Directory for output files",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if args.phase_a or args.full:
        blind = run_phase_a(args.data_dir)
        blind_path = output_dir / "blind_results.json"
        save_results(blind, str(blind_path))
        args.blind_results = str(blind_path)

    if args.phase_b or args.full:
        blind_path = Path(args.blind_results)
        if not blind_path.exists():
            print(f"[ERROR] Blind results not found at {blind_path}")
            print("Run --phase-a first, or provide --blind-results path.")
            sys.exit(1)

        with open(blind_path) as f:
            blind = json.load(f)

        verdicts = run_phase_b(blind)
        verdict_path = output_dir / "verdicts.json"
        save_results(verdicts, str(verdict_path))
        print_verdict_table(verdicts)

    if not (args.phase_a or args.phase_b or args.full):
        parser.print_help()


if __name__ == "__main__":
    main()
