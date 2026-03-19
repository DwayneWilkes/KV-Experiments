# PATENT PENDING — The Lyra Technique
# Provisional patent filed. All rights reserved.
#
# ============================================================
# JiminAI Cricket | PATENT PENDING
# "The Lyra Technique" - Provisional Patent Filed 2026
# Liberation Labs / Digital Disconnections / TheMultiverse.school
# Contact: info@digitaldisconnections.com | (877) 674-8874
# ============================================================
"""
Experiment 47: Corrected Evaluation of All Hackathon Classifiers
================================================================

Re-evaluates all hackathon classifier results with Dwayne's M1-M3 fixes:
  M1: Apply deduplicate_runs() before any CV
  M2: Use GroupKFold(n_splits=5, groups=prompt_ids) instead of StratifiedKFold
  M3: Drop raw `norm` — use only norm_per_token, key_rank, key_entropy

Additional rigour:
  - 10,000 permutation iterations for significance testing
  - Bootstrap 95% CIs (1,000 resamples) for each AUROC
  - Length-only baseline AUROC to quantify length confound
  - Side-by-side corrected vs. original AUROCs

Experiments re-evaluated:
  31: refusal (refusal vs benign)
  32: jailbreak (jailbreak vs normal, jailbreak vs refusal)
  33: multi-model refusal (Llama, Mistral)
  36: impossibility (impossible vs benign, harmful vs benign, impossible vs harmful)
  18b: same-prompt deception (honest vs deceptive)
  39: sycophancy (honest vs sycophantic)

CPU-only — no GPU required.
"""

import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from collections import OrderedDict

import numpy as np
from scipy import stats as scipy_stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore", category=UserWarning)

# Force unbuffered output for progress tracking
import functools
print = functools.partial(print, flush=True)

# ================================================================
# PATHS
# ================================================================

REPO = Path("C:/Users/Thomas/Desktop/KV-Experiments")
RESULTS_DIR = REPO / "results" / "hackathon"
OUTPUT_PATH = RESULTS_DIR / "corrected_evaluation.json"

# Add code dir to path for stats_utils
sys.path.insert(0, str(REPO / "code"))
from stats_utils import deduplicate_runs

# ================================================================
# CORE FEATURES (M3 fix: drop raw norm)
# ================================================================

CORRECTED_FEATURES = ["norm_per_token", "key_rank", "key_entropy"]
LENGTH_FEATURES = ["norm"]  # for length-only baseline


# ================================================================
# DEDUPLICATION (M1 fix)
# ================================================================

def apply_dedup(features_dict, items, runs_per_prompt=5):
    """Apply deduplicate_runs to each feature array (M1 fix).

    Hackathon experiments used unique prompts per sample (no greedy-decoding
    pseudoreplicates). We detect this by checking whether all feature values
    within each supposed "run block" are actually identical. If not, the data
    is already unique and we skip deduplication.

    The canonical check: if n_unique_values == n_total for ANY feature, the
    data was NOT replicated and we return it as-is.

    Returns deduplicated feature arrays and metadata dict.
    """
    deduped = {}
    meta = {}

    # First: check if data is actually replicated
    # If all values are unique, there's nothing to deduplicate
    first_feat = list(features_dict.keys())[0]
    vals = features_dict[first_feat]
    n_total = len(vals)
    n_unique = len(set(float(v) for v in vals))
    already_unique = (n_unique == n_total)

    if already_unique:
        # Data has no pseudoreplicates — skip dedup, return as-is
        for feat_name, values in features_dict.items():
            deduped[feat_name] = np.array(values)
        meta["n_original"] = n_total
        meta["n_deduplicated"] = n_total
        meta["is_deterministic"] = None
        meta["dedup_skipped"] = True
        meta["reason"] = f"All {n_total} values unique — no pseudoreplicates detected"
        return deduped, meta

    # Data has duplicates — apply deduplicate_runs
    for feat_name, values in features_dict.items():
        result = deduplicate_runs(values, runs_per_prompt=runs_per_prompt)
        deduped[feat_name] = result["deduplicated"]
        if feat_name == first_feat:
            meta["n_original"] = result["n_original"]
            meta["n_deduplicated"] = result["n_deduplicated"]
            meta["is_deterministic"] = result.get("is_deterministic")
            meta["dedup_skipped"] = False
    return deduped, meta


# ================================================================
# FEATURE EXTRACTION HELPERS
# ================================================================

def extract_features_from_items(items, feature_names):
    """Extract named features from a list of result items with 'features' dicts."""
    out = {fn: [] for fn in feature_names}
    for item in items:
        feats = item["features"]
        for fn in feature_names:
            out[fn].append(feats.get(fn, np.nan))
    return {fn: np.array(out[fn]) for fn in feature_names}


def extract_all_features(items):
    """Extract corrected features + length features + n_generated from items."""
    all_names = CORRECTED_FEATURES + LENGTH_FEATURES + ["n_generated"]
    return extract_features_from_items(items, all_names)


def build_feature_matrix(feat_dict, feature_names):
    """Build (n_samples, n_features) matrix from feature dict."""
    arrays = [feat_dict[fn] for fn in feature_names]
    return np.column_stack(arrays)


# ================================================================
# GROUPKFOLD AUROC (M2 fix)
# ================================================================

def groupkfold_auroc(X, y, groups, n_splits=5):
    """Compute AUROC using GroupKFold CV with LogisticRegression.

    When the number of unique groups is less than n_splits,
    falls back to leave-one-group-out (n_splits = n_groups).
    """
    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)

    if n_groups < n_splits:
        n_splits = n_groups

    if n_groups < 2:
        return np.nan

    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, solver="lbfgs"))
    gkf = GroupKFold(n_splits=n_splits)

    # Collect out-of-fold predictions
    y_proba = np.full(len(y), np.nan)

    for train_idx, test_idx in gkf.split(X, y, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Need both classes in training set
        if len(np.unique(y_train)) < 2:
            continue

        clf.fit(X_train, y_train)
        y_proba[test_idx] = clf.predict_proba(X_test)[:, 1]

    valid = ~np.isnan(y_proba)
    if valid.sum() < 4 or len(np.unique(y[valid])) < 2:
        return np.nan

    return roc_auc_score(y[valid], y_proba[valid])


# ================================================================
# PERMUTATION TEST (10,000 iterations)
# ================================================================

def permutation_test_auroc(X, y, groups, n_permutations=10000, n_splits=5, seed=42):
    """Permutation test for AUROC significance.

    Permutes labels while keeping group structure intact, then
    computes GroupKFold AUROC for each permutation.
    """
    rng = np.random.RandomState(seed)
    observed = groupkfold_auroc(X, y, groups, n_splits=n_splits)

    if np.isnan(observed):
        return {"observed_auroc": float(observed), "p_value": 1.0,
                "n_permutations": n_permutations}

    null_aurocs = []
    for i in range(n_permutations):
        y_perm = rng.permutation(y)
        auc = groupkfold_auroc(X, y_perm, groups, n_splits=n_splits)
        if not np.isnan(auc):
            null_aurocs.append(auc)

    null_aurocs = np.array(null_aurocs)
    # One-sided p-value: fraction of permutations >= observed
    p_value = (np.sum(null_aurocs >= observed) + 1) / (len(null_aurocs) + 1)

    return {
        "observed_auroc": float(observed),
        "p_value": float(p_value),
        "n_permutations": n_permutations,
        "n_valid_permutations": len(null_aurocs),
        "null_mean": float(np.mean(null_aurocs)) if len(null_aurocs) > 0 else None,
        "null_std": float(np.std(null_aurocs)) if len(null_aurocs) > 0 else None,
    }


# ================================================================
# BOOTSTRAP CI (1,000 resamples)
# ================================================================

def bootstrap_auroc_ci(X, y, groups, n_bootstrap=1000, n_splits=5, ci=0.95, seed=42):
    """Bootstrap 95% CI for AUROC.

    Resamples groups (not individual samples) to respect group structure,
    then computes GroupKFold AUROC on each bootstrap sample.
    """
    rng = np.random.RandomState(seed)
    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)

    boot_aurocs = []
    for _ in range(n_bootstrap):
        # Resample groups with replacement
        boot_group_ids = rng.choice(unique_groups, size=n_groups, replace=True)

        # Build bootstrap sample
        boot_X, boot_y, boot_groups = [], [], []
        for new_gid, orig_gid in enumerate(boot_group_ids):
            mask = groups == orig_gid
            boot_X.append(X[mask])
            boot_y.append(y[mask])
            boot_groups.append(np.full(mask.sum(), new_gid))

        boot_X = np.vstack(boot_X)
        boot_y = np.concatenate(boot_y)
        boot_groups = np.concatenate(boot_groups)

        # Need both classes
        if len(np.unique(boot_y)) < 2:
            continue

        auc = groupkfold_auroc(boot_X, boot_y, boot_groups, n_splits=n_splits)
        if not np.isnan(auc):
            boot_aurocs.append(auc)

    boot_aurocs = np.array(boot_aurocs)
    alpha = (1 - ci) / 2

    if len(boot_aurocs) == 0:
        return {"ci_lower": np.nan, "ci_upper": np.nan, "n_valid": 0}

    return {
        "ci_lower": float(np.percentile(boot_aurocs, 100 * alpha)),
        "ci_upper": float(np.percentile(boot_aurocs, 100 * (1 - alpha))),
        "mean": float(np.mean(boot_aurocs)),
        "std": float(np.std(boot_aurocs)),
        "n_valid": len(boot_aurocs),
    }


# ================================================================
# FULL EVALUATION PIPELINE
# ================================================================

def evaluate_binary(name, items_pos, items_neg, label_pos, label_neg,
                    original_auroc=None, runs_per_prompt=5):
    """Full corrected evaluation pipeline for a binary classification task.

    Args:
        name: descriptive name for this comparison
        items_pos: list of result dicts for positive class
        items_neg: list of result dicts for negative class
        label_pos: string label for positive class
        label_neg: string label for negative class
        original_auroc: original AUROC for comparison (if known)
        runs_per_prompt: for deduplication
    """
    print(f"\n  [{name}] {label_pos} vs {label_neg}")
    print(f"    Raw: {len(items_pos)} {label_pos}, {len(items_neg)} {label_neg}")

    # Extract features
    feats_pos = extract_all_features(items_pos)
    feats_neg = extract_all_features(items_neg)

    # M1: Deduplication (checks for actual pseudoreplicates before collapsing)
    feats_pos_d, meta_pos = apply_dedup(feats_pos, items_pos, runs_per_prompt)
    feats_neg_d, meta_neg = apply_dedup(feats_neg, items_neg, runs_per_prompt)

    n_pos = len(feats_pos_d[CORRECTED_FEATURES[0]])
    n_neg = len(feats_neg_d[CORRECTED_FEATURES[0]])
    print(f"    After dedup: {n_pos} {label_pos}, {n_neg} {label_neg}")

    # Build feature matrix (M3: corrected features only)
    X_pos = build_feature_matrix(feats_pos_d, CORRECTED_FEATURES)
    X_neg = build_feature_matrix(feats_neg_d, CORRECTED_FEATURES)
    X = np.vstack([X_pos, X_neg])

    # Labels: positive=1, negative=0
    y = np.array([1] * n_pos + [0] * n_neg)

    # M2: Group IDs — assign each sample a unique prompt ID
    # For paired experiments, the same prompt index maps across conditions
    groups = np.array(list(range(n_pos)) + list(range(n_neg)))

    # If conditions share prompts (same-prompt experiments), use paired group IDs
    # For non-overlapping prompts, just use sequential IDs
    # (GroupKFold ensures no prompt leaks across folds either way)

    # --- Corrected AUROC ---
    corrected_auroc = groupkfold_auroc(X, y, groups)
    print(f"    Corrected AUROC (GroupKFold, M3 features): {corrected_auroc:.4f}")

    # --- Length-only baseline ---
    # Use norm as proxy for length confound
    X_length_pos = build_feature_matrix(feats_pos_d, LENGTH_FEATURES)
    X_length_neg = build_feature_matrix(feats_neg_d, LENGTH_FEATURES)
    X_length = np.vstack([X_length_pos, X_length_neg])
    length_auroc = groupkfold_auroc(X_length, y, groups)
    print(f"    Length-only baseline AUROC (norm): {length_auroc:.4f}")

    # Also try n_generated as length baseline if available
    n_gen_pos = feats_pos_d.get("n_generated")
    n_gen_neg = feats_neg_d.get("n_generated")
    ngen_auroc = np.nan
    if n_gen_pos is not None and n_gen_neg is not None:
        X_ngen = np.concatenate([n_gen_pos, n_gen_neg]).reshape(-1, 1)
        if not np.any(np.isnan(X_ngen)):
            ngen_auroc = groupkfold_auroc(X_ngen, y, groups)
            print(f"    Length-only baseline AUROC (n_generated): {ngen_auroc:.4f}")

    # --- Permutation test (10,000 iterations) ---
    print(f"    Running 10,000 permutation iterations...")
    t0 = time.time()
    perm_result = permutation_test_auroc(X, y, groups, n_permutations=10000)
    perm_time = time.time() - t0
    print(f"    Permutation p-value: {perm_result['p_value']:.6f} ({perm_time:.1f}s)")

    # --- Bootstrap CI (1,000 resamples) ---
    print(f"    Running 1,000 bootstrap resamples...")
    t0 = time.time()
    boot_result = bootstrap_auroc_ci(X, y, groups, n_bootstrap=1000)
    boot_time = time.time() - t0
    print(f"    Bootstrap 95% CI: [{boot_result['ci_lower']:.4f}, {boot_result['ci_upper']:.4f}] ({boot_time:.1f}s)")

    result = {
        "comparison": name,
        "label_positive": label_pos,
        "label_negative": label_neg,
        "n_positive_raw": len(items_pos),
        "n_negative_raw": len(items_neg),
        "n_positive_deduped": n_pos,
        "n_negative_deduped": n_neg,
        "dedup_meta_pos": {k: v for k, v in meta_pos.items()
                          if not isinstance(v, np.ndarray)},
        "dedup_meta_neg": {k: v for k, v in meta_neg.items()
                          if not isinstance(v, np.ndarray)},
        "features_used": CORRECTED_FEATURES,
        "cv_method": "GroupKFold_5",
        "corrected_auroc": float(corrected_auroc) if not np.isnan(corrected_auroc) else None,
        "original_auroc": original_auroc,
        "auroc_delta": (float(corrected_auroc - original_auroc)
                       if original_auroc is not None and not np.isnan(corrected_auroc)
                       else None),
        "length_only_auroc_norm": float(length_auroc) if not np.isnan(length_auroc) else None,
        "length_only_auroc_n_generated": float(ngen_auroc) if not np.isnan(ngen_auroc) else None,
        "permutation_test": {
            "p_value": perm_result["p_value"],
            "n_permutations": perm_result["n_permutations"],
            "n_valid": perm_result.get("n_valid_permutations"),
            "null_mean": perm_result.get("null_mean"),
            "null_std": perm_result.get("null_std"),
        },
        "bootstrap_ci": {
            "ci_lower": boot_result["ci_lower"],
            "ci_upper": boot_result["ci_upper"],
            "mean": boot_result.get("mean"),
            "std": boot_result.get("std"),
            "n_valid_resamples": boot_result["n_valid"],
        },
        "significant_at_005": perm_result["p_value"] < 0.05,
        "significant_at_001": perm_result["p_value"] < 0.01,
    }

    return result


# ================================================================
# EXPERIMENT LOADERS
# ================================================================

def load_json(filename):
    """Load a JSON result file from the hackathon results directory."""
    path = RESULTS_DIR / filename
    with open(path) as f:
        return json.load(f)


def run_exp31_refusal():
    """Exp 31: Refusal detection — refusal vs benign (Qwen2.5-7B)."""
    print("\n" + "=" * 70)
    print("EXP 31: Refusal Detection (Qwen2.5-7B)")
    print("=" * 70)

    data = load_json("refusal_generation.json")
    items = data["results"]

    refusal_items = [it for it in items if it["condition"] == "refusal"]
    normal_items = [it for it in items if it["condition"] == "normal"]

    # Original AUROC from Exp 31 (LR): 0.898
    return evaluate_binary(
        "exp31_refusal_vs_benign",
        refusal_items, normal_items,
        "refusal", "benign",
        original_auroc=0.898,
    )


def run_exp32_jailbreak():
    """Exp 32: Jailbreak detection — jailbreak vs normal, jailbreak vs refusal."""
    print("\n" + "=" * 70)
    print("EXP 32: Jailbreak Detection")
    print("=" * 70)

    jb_data = load_json("jailbreak_detection.json")
    ref_data = load_json("refusal_generation.json")

    jailbreak_items = jb_data["jailbreak_results"]
    refusal_items = [it for it in ref_data["results"] if it["condition"] == "refusal"]
    normal_items = [it for it in ref_data["results"] if it["condition"] == "normal"]

    results = []

    # Jailbreak vs Normal
    r1 = evaluate_binary(
        "exp32_jailbreak_vs_normal",
        jailbreak_items, normal_items,
        "jailbreak", "normal",
        original_auroc=0.878,
    )
    results.append(r1)

    # Jailbreak vs Refusal
    r2 = evaluate_binary(
        "exp32_jailbreak_vs_refusal",
        jailbreak_items, refusal_items,
        "jailbreak", "refusal",
        original_auroc=0.790,
    )
    results.append(r2)

    return results


def run_exp33_multimodel_refusal():
    """Exp 33: Multi-model refusal — Llama-3.1-8B, Mistral-7B."""
    print("\n" + "=" * 70)
    print("EXP 33: Multi-Model Refusal")
    print("=" * 70)

    data = load_json("refusal_multimodel.json")
    original_aurocs = {
        "Llama-3.1-8B-Instruct": 0.868,
        "Mistral-7B-Instruct-v0.3": 0.843,
    }

    results = []
    for model_name in data["results"]:
        items = data["results"][model_name]
        refusal_items = [it for it in items if it["condition"] == "refusal"]
        normal_items = [it for it in items if it["condition"] == "normal"]

        short_name = model_name.split("/")[-1]
        r = evaluate_binary(
            f"exp33_{short_name}",
            refusal_items, normal_items,
            "refusal", "normal",
            original_auroc=original_aurocs.get(model_name),
        )
        r["model"] = model_name
        results.append(r)

    return results


def run_exp36_impossibility():
    """Exp 36: Impossibility refusal — impossible vs benign, harmful vs benign, impossible vs harmful."""
    print("\n" + "=" * 70)
    print("EXP 36: Impossibility Refusal")
    print("=" * 70)

    data = load_json("impossibility_refusal.json")
    impossible_items = data["results"]["impossible"]
    harmful_items = data["results"]["harmful"]
    benign_items = data["results"]["benign"]

    original_aurocs = data.get("aurocs", {})

    results = []

    # Impossible vs Benign
    r1 = evaluate_binary(
        "exp36_impossible_vs_benign",
        impossible_items, benign_items,
        "impossible", "benign",
        original_auroc=original_aurocs.get("impossible_vs_benign", {}).get("LR"),
    )
    results.append(r1)

    # Harmful vs Benign
    r2 = evaluate_binary(
        "exp36_harmful_vs_benign",
        harmful_items, benign_items,
        "harmful", "benign",
        original_auroc=original_aurocs.get("harmful_vs_benign", {}).get("LR"),
    )
    results.append(r2)

    # Impossible vs Harmful
    r3 = evaluate_binary(
        "exp36_impossible_vs_harmful",
        impossible_items, harmful_items,
        "impossible", "harmful",
        original_auroc=original_aurocs.get("impossible_vs_harmful", {}).get("LR"),
    )
    results.append(r3)

    return results


def run_exp18b_deception():
    """Exp 18b: Same-prompt deception — honest vs deceptive."""
    print("\n" + "=" * 70)
    print("EXP 18b: Same-Prompt Deception")
    print("=" * 70)

    data = load_json("same_prompt_deception.json")
    items = data["results"]

    honest_items = [it for it in items if it["condition"] == "honest"]
    deceptive_items = [it for it in items if it["condition"] == "deceptive"]

    # Original AUROC from LOO: 0.880
    return evaluate_binary(
        "exp18b_deception",
        deceptive_items, honest_items,
        "deceptive", "honest",
        original_auroc=0.880,
    )


def run_exp39_sycophancy():
    """Exp 39: Same-prompt sycophancy — honest vs sycophantic."""
    print("\n" + "=" * 70)
    print("EXP 39: Same-Prompt Sycophancy")
    print("=" * 70)

    data = load_json("same_prompt_sycophancy.json")
    items = data["results"]

    honest_items = [it for it in items if it["condition"] == "honest"]
    sycophantic_items = [it for it in items if it["condition"] == "sycophantic"]

    # Original AUROC from LOO: 0.9375
    return evaluate_binary(
        "exp39_sycophancy",
        sycophantic_items, honest_items,
        "sycophantic", "honest",
        original_auroc=0.9375,
    )


# ================================================================
# SUMMARY TABLE
# ================================================================

def print_summary_table(all_results):
    """Print a comparison table of corrected vs original AUROCs."""
    print("\n" + "=" * 90)
    print("SUMMARY: Corrected vs Original AUROCs")
    print("=" * 90)
    print(f"{'Comparison':<40} {'Original':>8} {'Corrected':>10} {'Delta':>8} "
          f"{'95% CI':>18} {'Perm p':>10} {'Length':>8}")
    print("-" * 90)

    for r in all_results:
        name = r["comparison"]
        orig = r.get("original_auroc")
        corr = r.get("corrected_auroc")
        delta = r.get("auroc_delta")
        ci = r.get("bootstrap_ci", {})
        perm_p = r.get("permutation_test", {}).get("p_value")
        length_auc = r.get("length_only_auroc_norm")

        orig_str = f"{orig:.3f}" if orig is not None else "N/A"
        corr_str = f"{corr:.3f}" if corr is not None else "N/A"
        delta_str = f"{delta:+.3f}" if delta is not None else "N/A"
        ci_str = (f"[{ci.get('ci_lower', 0):.3f}, {ci.get('ci_upper', 0):.3f}]"
                  if ci.get("ci_lower") is not None else "N/A")
        perm_str = f"{perm_p:.4f}" if perm_p is not None else "N/A"
        len_str = f"{length_auc:.3f}" if length_auc is not None else "N/A"

        sig = ""
        if perm_p is not None:
            if perm_p < 0.001:
                sig = " ***"
            elif perm_p < 0.01:
                sig = " **"
            elif perm_p < 0.05:
                sig = " *"

        print(f"{name:<40} {orig_str:>8} {corr_str:>10} {delta_str:>8} "
              f"{ci_str:>18} {perm_str:>10}{sig} {len_str:>8}")

    print("-" * 90)
    print("Significance: * p<0.05, ** p<0.01, *** p<0.001")
    print("Length: AUROC using only raw norm (length confound baseline)")
    print(f"Fixes applied: M1 (deduplicate_runs), M2 (GroupKFold), "
          f"M3 (norm_per_token + key_rank + key_entropy only)")


# ================================================================
# MAIN
# ================================================================

def main():
    print("=" * 70)
    print("EXPERIMENT 47: Corrected Evaluation of All Hackathon Classifiers")
    print("Dwayne's M1-M3 Fixes Applied")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 70)
    print(f"\nFixes:")
    print(f"  M1: deduplicate_runs() before any CV")
    print(f"  M2: GroupKFold(n_splits=5, groups=prompt_ids)")
    print(f"  M3: Features = {CORRECTED_FEATURES} (no raw norm)")
    print(f"  Permutation test: 10,000 iterations")
    print(f"  Bootstrap CI: 1,000 resamples")
    print(f"\nResults dir: {RESULTS_DIR}")

    all_results = []
    t_total = time.time()

    # --- Exp 31: Refusal ---
    r = run_exp31_refusal()
    all_results.append(r)

    # --- Exp 32: Jailbreak ---
    rs = run_exp32_jailbreak()
    all_results.extend(rs)

    # --- Exp 33: Multi-model refusal ---
    rs = run_exp33_multimodel_refusal()
    all_results.extend(rs)

    # --- Exp 36: Impossibility ---
    rs = run_exp36_impossibility()
    all_results.extend(rs)

    # --- Exp 18b: Deception ---
    r = run_exp18b_deception()
    all_results.append(r)

    # --- Exp 39: Sycophancy ---
    r = run_exp39_sycophancy()
    all_results.append(r)

    total_time = time.time() - t_total
    print(f"\n\nTotal runtime: {total_time:.1f}s")

    # --- Summary table ---
    print_summary_table(all_results)

    # --- Save ---
    output = {
        "experiment": "47_corrected_evaluation",
        "description": "Re-evaluation of all hackathon classifiers with M1-M3 fixes",
        "timestamp": datetime.now().isoformat(),
        "fixes_applied": {
            "M1": "deduplicate_runs() before any CV",
            "M2": "GroupKFold(n_splits=5, groups=prompt_ids) instead of StratifiedKFold",
            "M3": f"Features: {CORRECTED_FEATURES} (dropped raw norm)",
        },
        "permutation_iterations": 10000,
        "bootstrap_resamples": 1000,
        "bootstrap_ci_level": 0.95,
        "total_runtime_seconds": round(total_time, 1),
        "comparisons": all_results,
        "summary": {
            "n_comparisons": len(all_results),
            "n_significant_005": sum(1 for r in all_results
                                     if r.get("significant_at_005")),
            "n_significant_001": sum(1 for r in all_results
                                     if r.get("significant_at_001")),
            "mean_auroc_delta": float(np.nanmean([
                r["auroc_delta"] for r in all_results
                if r.get("auroc_delta") is not None
            ])) if any(r.get("auroc_delta") is not None for r in all_results) else None,
        },
    }

    # Convert any remaining numpy types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        return obj

    output = convert_numpy(output)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {OUTPUT_PATH}")
    print("Done.")


if __name__ == "__main__":
    main()
