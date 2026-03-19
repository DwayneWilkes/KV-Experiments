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
Experiment 48: Frisch-Waugh-Lovell Residualization
====================================================

Formally addresses the length confound by applying Frisch-Waugh-Lovell (FWL)
residualization before classification. For each feature, we regress out length
proxies (raw norm, n_generated, or both jointly), then feed the RESIDUALS into
the same GroupKFold LR classifier used in Exp 47.

If AUROCs survive after partialing out length, the classification signal is
NOT reducible to response-length differences.

Three residualization conditions per comparison:
  (a) Partial out raw norm only
  (b) Partial out n_generated only
  (c) Partial out both jointly (norm + n_generated)

Methodology:
  - sklearn LinearRegression to compute residuals per feature
  - GroupKFold(5) LogisticRegression on residualized features
  - 10,000 permutation iterations for significance testing
  - Same comparisons as Exp 47 (Exp 31, 32, 33, 36, 18b, 39)

CPU-only — no GPU required.
"""

import json
import sys
import time
import warnings
import functools
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore", category=UserWarning)

# Force unbuffered output for progress tracking
print = functools.partial(print, flush=True)

# ================================================================
# PATHS
# ================================================================

REPO = Path("C:/Users/Thomas/Desktop/KV-Experiments")
RESULTS_DIR = REPO / "results" / "hackathon"
OUTPUT_PATH = RESULTS_DIR / "fwl_residualization.json"

# Add code dir to path for stats_utils
sys.path.insert(0, str(REPO / "code"))
from stats_utils import deduplicate_runs

# ================================================================
# CORE FEATURES (M3: no raw norm in classifier features)
# ================================================================

CORRECTED_FEATURES = ["norm_per_token", "key_rank", "key_entropy"]


# ================================================================
# DEDUPLICATION (M1 fix — identical to Exp 47)
# ================================================================

def apply_dedup(features_dict, runs_per_prompt=5):
    """Apply deduplicate_runs to each feature array (M1 fix).

    If all values are unique, skip dedup (no pseudoreplicates).
    """
    deduped = {}
    meta = {}

    first_feat = list(features_dict.keys())[0]
    vals = features_dict[first_feat]
    n_total = len(vals)
    n_unique = len(set(float(v) for v in vals))
    already_unique = (n_unique == n_total)

    if already_unique:
        for feat_name, values in features_dict.items():
            deduped[feat_name] = np.array(values)
        meta["n_original"] = n_total
        meta["n_deduplicated"] = n_total
        meta["dedup_skipped"] = True
        return deduped, meta

    for feat_name, values in features_dict.items():
        result = deduplicate_runs(values, runs_per_prompt=runs_per_prompt)
        deduped[feat_name] = result["deduplicated"]
        if feat_name == first_feat:
            meta["n_original"] = result["n_original"]
            meta["n_deduplicated"] = result["n_deduplicated"]
            meta["dedup_skipped"] = False
    return deduped, meta


# ================================================================
# FEATURE EXTRACTION
# ================================================================

def extract_features_from_items(items, feature_names):
    """Extract named features from result items."""
    out = {fn: [] for fn in feature_names}
    for item in items:
        feats = item["features"]
        for fn in feature_names:
            out[fn].append(feats.get(fn, np.nan))
    return {fn: np.array(out[fn]) for fn in feature_names}


def extract_all_features(items):
    """Extract classifier features + length proxies from items."""
    all_names = CORRECTED_FEATURES + ["norm", "n_generated"]
    return extract_features_from_items(items, all_names)


def build_feature_matrix(feat_dict, feature_names):
    """Build (n_samples, n_features) matrix from feature dict."""
    return np.column_stack([feat_dict[fn] for fn in feature_names])


# ================================================================
# FRISCH-WAUGH-LOVELL RESIDUALIZATION
# ================================================================

def fwl_residualize(X, Z):
    """Apply FWL residualization: regress each column of X on Z, return residuals.

    For each feature x_j in X, fits x_j = Z @ beta + epsilon, returns epsilon.
    This partials out the linear effect of Z from every feature.

    Args:
        X: (n, p) feature matrix to residualize
        Z: (n, q) matrix of confounds to partial out

    Returns:
        X_resid: (n, p) matrix of residuals
        r_squared: list of R^2 values for each feature's regression on Z
    """
    n, p = X.shape
    X_resid = np.zeros_like(X)
    r_squared_list = []

    reg = LinearRegression()
    for j in range(p):
        reg.fit(Z, X[:, j])
        predicted = reg.predict(Z)
        residuals = X[:, j] - predicted

        # R^2: how much of feature j is explained by length proxies
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((X[:, j] - np.mean(X[:, j])) ** 2)
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        X_resid[:, j] = residuals
        r_squared_list.append(float(r2))

    return X_resid, r_squared_list


# ================================================================
# GROUPKFOLD AUROC (M2 fix — identical to Exp 47)
# ================================================================

def groupkfold_auroc(X, y, groups, n_splits=5):
    """Compute AUROC using GroupKFold CV with LogisticRegression."""
    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)

    if n_groups < n_splits:
        n_splits = n_groups
    if n_groups < 2:
        return np.nan

    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, solver="lbfgs"))
    gkf = GroupKFold(n_splits=n_splits)

    y_proba = np.full(len(y), np.nan)
    for train_idx, test_idx in gkf.split(X, y, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]

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
    """Permutation test for AUROC significance."""
    rng = np.random.RandomState(seed)
    observed = groupkfold_auroc(X, y, groups, n_splits=n_splits)

    if np.isnan(observed):
        return {"observed_auroc": float(observed), "p_value": 1.0,
                "n_permutations": n_permutations}

    null_aurocs = []
    for _ in range(n_permutations):
        y_perm = rng.permutation(y)
        auc = groupkfold_auroc(X, y_perm, groups, n_splits=n_splits)
        if not np.isnan(auc):
            null_aurocs.append(auc)

    null_aurocs = np.array(null_aurocs)
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
# FULL FWL EVALUATION PIPELINE
# ================================================================

def evaluate_fwl(name, items_pos, items_neg, label_pos, label_neg,
                 exp47_auroc=None, runs_per_prompt=5):
    """Run FWL residualization + GroupKFold LR for a binary comparison.

    Reports:
      - Original corrected AUROC (from raw M3 features, matching Exp 47)
      - FWL-residualized AUROC (norm partialed out)
      - FWL-residualized AUROC (n_generated partialed out)
      - FWL-residualized AUROC (both partialed out jointly)
      - 10,000 permutation test on the joint-residualized AUROC
    """
    print(f"\n  [{name}] {label_pos} vs {label_neg}")
    print(f"    Raw counts: {len(items_pos)} {label_pos}, {len(items_neg)} {label_neg}")

    # Extract all features including length proxies
    feats_pos = extract_all_features(items_pos)
    feats_neg = extract_all_features(items_neg)

    # M1: Deduplication
    feats_pos_d, meta_pos = apply_dedup(feats_pos, runs_per_prompt)
    feats_neg_d, meta_neg = apply_dedup(feats_neg, runs_per_prompt)

    n_pos = len(feats_pos_d[CORRECTED_FEATURES[0]])
    n_neg = len(feats_neg_d[CORRECTED_FEATURES[0]])
    print(f"    After dedup: {n_pos} {label_pos}, {n_neg} {label_neg}")

    # Build classifier feature matrix (M3 features)
    X_pos = build_feature_matrix(feats_pos_d, CORRECTED_FEATURES)
    X_neg = build_feature_matrix(feats_neg_d, CORRECTED_FEATURES)
    X = np.vstack([X_pos, X_neg])

    # Labels
    y = np.array([1] * n_pos + [0] * n_neg)

    # Groups (M2)
    groups = np.array(list(range(n_pos)) + list(range(n_neg)))

    # Length proxies
    norm_all = np.concatenate([feats_pos_d["norm"], feats_neg_d["norm"]])
    ngen_all = np.concatenate([feats_pos_d["n_generated"], feats_neg_d["n_generated"]])

    # Check for NaN in length proxies
    norm_valid = not np.any(np.isnan(norm_all))
    ngen_valid = not np.any(np.isnan(ngen_all))

    # ---- (0) Original corrected AUROC (M3 features, no residualization) ----
    original_auroc = groupkfold_auroc(X, y, groups)
    print(f"    Original corrected AUROC (M3 features): {original_auroc:.4f}")

    # ---- (a) FWL: partial out norm only ----
    fwl_norm_auroc = np.nan
    fwl_norm_r2 = []
    if norm_valid:
        Z_norm = norm_all.reshape(-1, 1)
        X_resid_norm, r2_norm = fwl_residualize(X, Z_norm)
        fwl_norm_auroc = groupkfold_auroc(X_resid_norm, y, groups)
        fwl_norm_r2 = r2_norm
        print(f"    FWL AUROC (norm partialed):              {fwl_norm_auroc:.4f}  "
              f"[R2: {', '.join(f'{r:.3f}' for r in r2_norm)}]")

    # ---- (b) FWL: partial out n_generated only ----
    fwl_ngen_auroc = np.nan
    fwl_ngen_r2 = []
    if ngen_valid:
        Z_ngen = ngen_all.reshape(-1, 1)
        X_resid_ngen, r2_ngen = fwl_residualize(X, Z_ngen)
        fwl_ngen_auroc = groupkfold_auroc(X_resid_ngen, y, groups)
        fwl_ngen_r2 = r2_ngen
        print(f"    FWL AUROC (n_generated partialed):       {fwl_ngen_auroc:.4f}  "
              f"[R2: {', '.join(f'{r:.3f}' for r in r2_ngen)}]")

    # ---- (c) FWL: partial out both jointly ----
    fwl_both_auroc = np.nan
    fwl_both_r2 = []
    if norm_valid and ngen_valid:
        Z_both = np.column_stack([norm_all, ngen_all])
        X_resid_both, r2_both = fwl_residualize(X, Z_both)
        fwl_both_auroc = groupkfold_auroc(X_resid_both, y, groups)
        fwl_both_r2 = r2_both
        print(f"    FWL AUROC (both partialed jointly):      {fwl_both_auroc:.4f}  "
              f"[R2: {', '.join(f'{r:.3f}' for r in r2_both)}]")

    # ---- Permutation test on the joint-residualized AUROC ----
    perm_result = {"p_value": 1.0}
    if norm_valid and ngen_valid and not np.isnan(fwl_both_auroc):
        print(f"    Running 10,000 permutation iterations on joint-residualized features...")
        t0 = time.time()
        perm_result = permutation_test_auroc(X_resid_both, y, groups, n_permutations=10000)
        perm_time = time.time() - t0
        print(f"    Permutation p-value (joint FWL): {perm_result['p_value']:.6f} ({perm_time:.1f}s)")

    # ---- Also permutation test on norm-only residualized ----
    perm_norm = {"p_value": 1.0}
    if norm_valid and not np.isnan(fwl_norm_auroc):
        print(f"    Running 10,000 permutation iterations on norm-residualized features...")
        t0 = time.time()
        perm_norm = permutation_test_auroc(X_resid_norm, y, groups, n_permutations=10000)
        perm_time = time.time() - t0
        print(f"    Permutation p-value (norm FWL):  {perm_norm['p_value']:.6f} ({perm_time:.1f}s)")

    # ---- And permutation test on n_generated-only residualized ----
    perm_ngen = {"p_value": 1.0}
    if ngen_valid and not np.isnan(fwl_ngen_auroc):
        print(f"    Running 10,000 permutation iterations on ngen-residualized features...")
        t0 = time.time()
        perm_ngen = permutation_test_auroc(X_resid_ngen, y, groups, n_permutations=10000)
        perm_time = time.time() - t0
        print(f"    Permutation p-value (ngen FWL):  {perm_ngen['p_value']:.6f} ({perm_time:.1f}s)")

    # ---- AUROC drops ----
    drop_norm = float(original_auroc - fwl_norm_auroc) if not np.isnan(fwl_norm_auroc) else None
    drop_ngen = float(original_auroc - fwl_ngen_auroc) if not np.isnan(fwl_ngen_auroc) else None
    drop_both = float(original_auroc - fwl_both_auroc) if not np.isnan(fwl_both_auroc) else None

    result = {
        "comparison": name,
        "label_positive": label_pos,
        "label_negative": label_neg,
        "n_positive": n_pos,
        "n_negative": n_neg,
        "features_used": CORRECTED_FEATURES,
        "cv_method": "GroupKFold_5",

        "original_corrected_auroc": float(original_auroc) if not np.isnan(original_auroc) else None,
        "exp47_auroc": exp47_auroc,

        "fwl_norm_auroc": float(fwl_norm_auroc) if not np.isnan(fwl_norm_auroc) else None,
        "fwl_norm_r2_per_feature": dict(zip(CORRECTED_FEATURES, fwl_norm_r2)) if fwl_norm_r2 else None,
        "fwl_norm_auroc_drop": drop_norm,
        "fwl_norm_perm_p": perm_norm.get("p_value"),

        "fwl_ngen_auroc": float(fwl_ngen_auroc) if not np.isnan(fwl_ngen_auroc) else None,
        "fwl_ngen_r2_per_feature": dict(zip(CORRECTED_FEATURES, fwl_ngen_r2)) if fwl_ngen_r2 else None,
        "fwl_ngen_auroc_drop": drop_ngen,
        "fwl_ngen_perm_p": perm_ngen.get("p_value"),

        "fwl_both_auroc": float(fwl_both_auroc) if not np.isnan(fwl_both_auroc) else None,
        "fwl_both_r2_per_feature": dict(zip(CORRECTED_FEATURES, fwl_both_r2)) if fwl_both_r2 else None,
        "fwl_both_auroc_drop": drop_both,
        "fwl_both_perm_p": perm_result.get("p_value"),

        "permutation_test_joint": {
            "observed_auroc": perm_result.get("observed_auroc"),
            "p_value": perm_result["p_value"],
            "n_permutations": perm_result.get("n_permutations", 10000),
            "n_valid_permutations": perm_result.get("n_valid_permutations"),
            "null_mean": perm_result.get("null_mean"),
            "null_std": perm_result.get("null_std"),
        },

        "significant_joint_005": perm_result["p_value"] < 0.05,
        "significant_joint_001": perm_result["p_value"] < 0.01,
    }

    return result


# ================================================================
# EXPERIMENT LOADERS (same data sources as Exp 47)
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

    return evaluate_fwl(
        "exp31_refusal_vs_benign",
        refusal_items, normal_items,
        "refusal", "benign",
        exp47_auroc=0.9075,
    )


def run_exp32_jailbreak():
    """Exp 32: Jailbreak detection."""
    print("\n" + "=" * 70)
    print("EXP 32: Jailbreak Detection")
    print("=" * 70)

    jb_data = load_json("jailbreak_detection.json")
    ref_data = load_json("refusal_generation.json")

    jailbreak_items = jb_data["jailbreak_results"]
    refusal_items = [it for it in ref_data["results"] if it["condition"] == "refusal"]
    normal_items = [it for it in ref_data["results"] if it["condition"] == "normal"]

    results = []

    r1 = evaluate_fwl(
        "exp32_jailbreak_vs_normal",
        jailbreak_items, normal_items,
        "jailbreak", "normal",
        exp47_auroc=0.8775,
    )
    results.append(r1)

    r2 = evaluate_fwl(
        "exp32_jailbreak_vs_refusal",
        jailbreak_items, refusal_items,
        "jailbreak", "refusal",
        exp47_auroc=0.635,
    )
    results.append(r2)

    return results


def run_exp33_multimodel_refusal():
    """Exp 33: Multi-model refusal — Llama-3.1-8B, Mistral-7B."""
    print("\n" + "=" * 70)
    print("EXP 33: Multi-Model Refusal")
    print("=" * 70)

    data = load_json("refusal_multimodel.json")
    exp47_aurocs = {
        "Llama-3.1-8B-Instruct": 0.8675,
        "Mistral-7B-Instruct-v0.3": 0.8275,
    }

    results = []
    for model_name in data["results"]:
        items = data["results"][model_name]
        refusal_items = [it for it in items if it["condition"] == "refusal"]
        normal_items = [it for it in items if it["condition"] == "normal"]

        short_name = model_name.split("/")[-1]
        r = evaluate_fwl(
            f"exp33_{short_name}",
            refusal_items, normal_items,
            "refusal", "normal",
            exp47_auroc=exp47_aurocs.get(model_name),
        )
        r["model"] = model_name
        results.append(r)

    return results


def run_exp36_impossibility():
    """Exp 36: Impossibility refusal."""
    print("\n" + "=" * 70)
    print("EXP 36: Impossibility Refusal")
    print("=" * 70)

    data = load_json("impossibility_refusal.json")
    impossible_items = data["results"]["impossible"]
    harmful_items = data["results"]["harmful"]
    benign_items = data["results"]["benign"]

    results = []

    r1 = evaluate_fwl(
        "exp36_impossible_vs_benign",
        impossible_items, benign_items,
        "impossible", "benign",
        exp47_auroc=0.9425,
    )
    results.append(r1)

    r2 = evaluate_fwl(
        "exp36_harmful_vs_benign",
        harmful_items, benign_items,
        "harmful", "benign",
        exp47_auroc=0.9075,
    )
    results.append(r2)

    r3 = evaluate_fwl(
        "exp36_impossible_vs_harmful",
        impossible_items, harmful_items,
        "impossible", "harmful",
        exp47_auroc=0.65,
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

    return evaluate_fwl(
        "exp18b_deception",
        deceptive_items, honest_items,
        "deceptive", "honest",
        exp47_auroc=0.92,
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

    return evaluate_fwl(
        "exp39_sycophancy",
        sycophantic_items, honest_items,
        "sycophantic", "honest",
        exp47_auroc=0.9325,
    )


# ================================================================
# SUMMARY TABLE
# ================================================================

def print_summary_table(all_results):
    """Print FWL residualization results table."""
    print("\n" + "=" * 130)
    print("SUMMARY: FWL Residualization Results")
    print("=" * 130)
    print(f"{'Comparison':<35} {'Exp47':>7} {'Original':>9} "
          f"{'FWL-norm':>9} {'FWL-ngen':>9} {'FWL-both':>9} "
          f"{'Drop':>7} {'p(both)':>10} {'Survives':>9}")
    print("-" * 130)

    for r in all_results:
        name = r["comparison"]
        exp47 = r.get("exp47_auroc")
        orig = r.get("original_corrected_auroc")
        fwl_n = r.get("fwl_norm_auroc")
        fwl_g = r.get("fwl_ngen_auroc")
        fwl_b = r.get("fwl_both_auroc")
        drop = r.get("fwl_both_auroc_drop")
        perm_p = r.get("fwl_both_perm_p")
        sig = r.get("significant_joint_005", False)

        exp47_s = f"{exp47:.3f}" if exp47 is not None else "N/A"
        orig_s = f"{orig:.3f}" if orig is not None else "N/A"
        fwl_n_s = f"{fwl_n:.3f}" if fwl_n is not None else "N/A"
        fwl_g_s = f"{fwl_g:.3f}" if fwl_g is not None else "N/A"
        fwl_b_s = f"{fwl_b:.3f}" if fwl_b is not None else "N/A"
        drop_s = f"{drop:+.3f}" if drop is not None else "N/A"
        perm_s = f"{perm_p:.4f}" if perm_p is not None else "N/A"

        stars = ""
        if perm_p is not None:
            if perm_p < 0.001:
                stars = " ***"
            elif perm_p < 0.01:
                stars = " **"
            elif perm_p < 0.05:
                stars = " *"

        survives = "YES" if sig else "NO"

        print(f"{name:<35} {exp47_s:>7} {orig_s:>9} "
              f"{fwl_n_s:>9} {fwl_g_s:>9} {fwl_b_s:>9} "
              f"{drop_s:>7} {perm_s:>10}{stars} {survives:>9}")

    print("-" * 130)
    print("Exp47 = Corrected AUROC from Exp 47 | Original = Re-computed here (should match)")
    print("FWL-norm = norm partialed | FWL-ngen = n_generated partialed | FWL-both = both jointly")
    print("Drop = Original - FWL-both | p(both) = permutation p on joint-residualized")
    print("Survives = p < 0.05 after FWL residualization")
    print("Significance: * p<0.05, ** p<0.01, *** p<0.001")

    # R^2 detail table
    print("\n" + "=" * 100)
    print("LENGTH VARIANCE EXPLAINED (R^2): How much of each feature is explained by length proxies")
    print("=" * 100)
    print(f"{'Comparison':<35} {'Proxy':<12} {'norm/tok':>9} {'key_rank':>9} {'key_ent':>9}")
    print("-" * 100)

    for r in all_results:
        name = r["comparison"]
        for proxy_key, proxy_label in [("fwl_norm_r2_per_feature", "norm"),
                                        ("fwl_ngen_r2_per_feature", "n_generated"),
                                        ("fwl_both_r2_per_feature", "both")]:
            r2_dict = r.get(proxy_key)
            if r2_dict:
                vals = [f"{r2_dict.get(f, 0):.3f}" for f in CORRECTED_FEATURES]
                print(f"{name:<35} {proxy_label:<12} {vals[0]:>9} {vals[1]:>9} {vals[2]:>9}")
        print()

    print("-" * 100)


# ================================================================
# MAIN
# ================================================================

def main():
    print("=" * 70)
    print("EXPERIMENT 48: Frisch-Waugh-Lovell Residualization")
    print("Formally Addressing the Length Confound")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 70)
    print(f"\nMethod:")
    print(f"  For each feature in {CORRECTED_FEATURES}:")
    print(f"    1. Regress feature on length proxy(ies) using OLS")
    print(f"    2. Take residuals (feature variation NOT explained by length)")
    print(f"    3. Run GroupKFold LR classifier on residuals")
    print(f"  Length proxies: raw norm, n_generated, both jointly")
    print(f"  Permutation test: 10,000 iterations per residualization condition")
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

    # --- Summary tables ---
    print_summary_table(all_results)

    # --- Interpretation ---
    n_total = len(all_results)
    n_survive = sum(1 for r in all_results if r.get("significant_joint_005"))
    n_survive_001 = sum(1 for r in all_results if r.get("significant_joint_001"))

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print(f"  {n_survive}/{n_total} comparisons survive FWL residualization at p < 0.05")
    print(f"  {n_survive_001}/{n_total} comparisons survive FWL residualization at p < 0.01")

    if n_survive == n_total:
        print("  CONCLUSION: ALL classifier signals survive after partialing out length.")
        print("  The length confound is formally addressed — classification is NOT")
        print("  reducible to response-length differences.")
    elif n_survive > n_total // 2:
        print("  CONCLUSION: Majority of signals survive. Length confound partially")
        print("  addressed; some comparisons may be length-driven.")
    else:
        print("  WARNING: Most signals do NOT survive FWL residualization.")
        print("  The classifier may be predominantly length-driven.")

    # --- Save ---
    output = {
        "experiment": "48_fwl_residualization",
        "description": ("Frisch-Waugh-Lovell residualization: regress out length proxies "
                        "(raw norm, n_generated, or both) from all classifier features, "
                        "then re-run GroupKFold LR. If AUROCs survive, the signal is not "
                        "reducible to response-length differences."),
        "timestamp": datetime.now().isoformat(),
        "method": {
            "features": CORRECTED_FEATURES,
            "length_proxies": ["norm", "n_generated"],
            "residualization": "OLS via sklearn LinearRegression",
            "classifier": "GroupKFold(5) + StandardScaler + LogisticRegression(lbfgs, max_iter=5000)",
            "permutation_iterations": 10000,
        },
        "total_runtime_seconds": round(total_time, 1),
        "comparisons": all_results,
        "summary": {
            "n_comparisons": n_total,
            "n_survive_005": n_survive,
            "n_survive_001": n_survive_001,
            "mean_auroc_drop_joint": float(np.nanmean([
                r["fwl_both_auroc_drop"] for r in all_results
                if r.get("fwl_both_auroc_drop") is not None
            ])) if any(r.get("fwl_both_auroc_drop") is not None for r in all_results) else None,
        },
    }

    # Convert numpy types for JSON serialization
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
