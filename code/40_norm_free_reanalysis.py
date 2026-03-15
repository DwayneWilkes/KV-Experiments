#!/usr/bin/env python3
"""
Experiment 40: Norm-Free Classifier Reanalysis
===============================================
Re-runs all hackathon classifiers using only 3 features (norm_per_token,
key_rank, key_entropy), dropping raw `norm` which scales mechanically
with token count and may inflate AUROCs via length leakage.

This is a reanalysis of existing data — no new inference required.

Also runs a token-count-only baseline to quantify the length confound.
"""

import json
import numpy as np
import os
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score

RESULTS_DIR = Path(os.path.expanduser("~/KV-Experiments/results/hackathon"))
OUTPUT_FILE = RESULTS_DIR / "norm_free_reanalysis.json"

# 3 clean features (no raw norm)
CLEAN_FEATURES = ["norm_per_token", "key_rank", "key_entropy"]
# Original 4 features
ORIG_FEATURES = ["norm", "norm_per_token", "key_rank", "key_entropy"]
# Token count only (length baseline)
LENGTH_FEATURES = ["n_generated"]


def flatten_results(data):
    """Extract a flat list of result dicts from various hackathon JSON formats."""
    # Format 1: data["results"] is a list of dicts
    results = data.get("results", [])
    if isinstance(results, list) and results and isinstance(results[0], dict):
        return results

    # Format 2: data["results"] is a dict of {condition: [list]} (impossibility, extended)
    if isinstance(results, dict):
        flat = []
        for cond, items in results.items():
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict):
                        flat.append(item)
        return flat

    # Format 3: data["jailbreak_results"] (jailbreak_detection.json)
    jr = data.get("jailbreak_results", [])
    if isinstance(jr, list) and jr and isinstance(jr[0], dict):
        return jr

    # Format 4: data["all_results"] is dict of {model: [list]} (multimodel)
    ar = data.get("all_results", {})
    if isinstance(ar, dict):
        flat = []
        for model, items in ar.items():
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict):
                        flat.append(item)
        return flat

    return []


def load_experiment(filename, condition_key="condition", pos_label=None, neg_label=None,
                    results_key=None, model_filter=None):
    """Load a hackathon result file and extract features + labels."""
    filepath = RESULTS_DIR / filename
    if not filepath.exists():
        return None, None, None

    with open(filepath) as f:
        data = json.load(f)

    # Handle multimodel files where results are nested per model
    if results_key and results_key in data:
        nested = data[results_key]
        if isinstance(nested, dict):
            if model_filter:
                results = nested.get(model_filter, [])
            else:
                # Flatten all models
                results = []
                for model, items in nested.items():
                    if isinstance(items, list):
                        results.extend(items)
        else:
            results = nested
    else:
        results = flatten_results(data)

    if not results:
        return None, None, None

    features_list = []
    labels = []

    for r in results:
        if not isinstance(r, dict):
            continue
        cond = r.get(condition_key, r.get("prompt_type", ""))
        feats = r.get("features", {})

        if not feats:
            continue

        if pos_label and neg_label:
            if cond == pos_label:
                labels.append(1)
            elif cond == neg_label:
                labels.append(0)
            else:
                continue
        else:
            continue

        features_list.append(feats)

    if not features_list:
        return None, None, None

    return features_list, labels, data


def extract_feature_matrix(features_list, feature_names):
    """Extract numpy array from list of feature dicts."""
    X = np.array([[f.get(fn, 0) for fn in feature_names] for f in features_list])
    return X


def run_loo_cv(X, y, classifier_type="LR"):
    """Run leave-one-out CV and return AUROC + accuracy."""
    y = np.array(y)
    if len(np.unique(y)) < 2:
        return None, None

    loo = LeaveOneOut()
    y_scores = []
    y_preds = []

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        if classifier_type == "LR":
            clf = LogisticRegression(max_iter=1000, random_state=42)
        else:
            clf = RandomForestClassifier(n_estimators=100, random_state=42)

        clf.fit(X_train_s, y_train)
        y_scores.append(clf.predict_proba(X_test_s)[0, 1])
        y_preds.append(clf.predict(X_test_s)[0])

    auroc = roc_auc_score(y, y_scores)
    acc = accuracy_score(y, y_preds)
    return auroc, acc


def analyze_experiment(name, filename, pos_label, neg_label, condition_key="condition",
                       results_key=None, model_filter=None):
    """Run full analysis on one experiment with all feature sets."""
    features_list, labels, data = load_experiment(
        filename, condition_key, pos_label, neg_label,
        results_key=results_key, model_filter=model_filter)

    if features_list is None:
        print(f"  SKIP: {name} — file not found or empty")
        return None

    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    print(f"  {name}: n={len(labels)} ({n_pos} pos, {n_neg} neg)")

    result = {"name": name, "n": len(labels), "n_pos": n_pos, "n_neg": n_neg}

    for feat_name, feat_list in [("orig_4", ORIG_FEATURES), ("clean_3", CLEAN_FEATURES), ("length_only", LENGTH_FEATURES)]:
        try:
            X = extract_feature_matrix(features_list, feat_list)

            # Check for NaN/inf
            if np.any(~np.isfinite(X)):
                result[feat_name] = {"error": "NaN/inf in features"}
                continue

            for clf_type in ["LR", "RF"]:
                auroc, acc = run_loo_cv(X, labels, clf_type)
                key = f"{feat_name}_{clf_type}"
                result[key] = {"auroc": round(auroc, 4) if auroc else None,
                              "accuracy": round(acc, 4) if acc else None}

                if auroc:
                    print(f"    {key}: AUROC={auroc:.4f}, Acc={acc:.4f}")
        except Exception as e:
            result[feat_name] = {"error": str(e)}
            print(f"    {feat_name}: ERROR — {e}")

    # Compute effect sizes for clean features
    X_clean = extract_feature_matrix(features_list, CLEAN_FEATURES)
    y = np.array(labels)
    effect_sizes = {}
    for i, fn in enumerate(CLEAN_FEATURES):
        pos_vals = X_clean[y == 1, i]
        neg_vals = X_clean[y == 0, i]
        pooled_std = np.sqrt((np.std(pos_vals)**2 + np.std(neg_vals)**2) / 2)
        if pooled_std > 0:
            d = (np.mean(pos_vals) - np.mean(neg_vals)) / pooled_std
            effect_sizes[fn] = round(d, 3)
    result["clean_effect_sizes"] = effect_sizes

    return result


def main():
    print("=" * 60)
    print("Experiment 40: Norm-Free Classifier Reanalysis")
    print("=" * 60)
    print(f"Clean features: {CLEAN_FEATURES}")
    print(f"Original features: {ORIG_FEATURES}")
    print()

    all_results = {}

    # 1. Refusal detection (Exp 31)
    print("--- Refusal Detection (Exp 31) ---")
    r = analyze_experiment("refusal_qwen", "refusal_generation.json",
                          pos_label="refusal", neg_label="normal")
    if r: all_results["refusal_qwen"] = r

    # 2. Multi-model refusal (Exp 33) — per model
    print("\n--- Multi-Model Refusal (Exp 33) ---")
    for model in ["Llama-3.1-8B-Instruct", "Mistral-7B-Instruct-v0.3"]:
        short = model.split("/")[-1].split("-Instruct")[0]
        r = analyze_experiment(f"refusal_{short}", "refusal_multimodel.json",
                              pos_label="refusal", neg_label="normal",
                              results_key="results", model_filter=model)
        if r: all_results[f"refusal_{short}"] = r

    # 3. Jailbreak detection (Exp 32)
    print("\n--- Jailbreak Detection (Exp 32) ---")
    r = analyze_experiment("jailbreak_vs_normal", "jailbreak_detection.json",
                          pos_label="jailbreak", neg_label="normal")
    if r: all_results["jailbreak_vs_normal"] = r

    # 4. Jailbreak vs refusal
    r = analyze_experiment("jailbreak_vs_refusal", "jailbreak_detection.json",
                          pos_label="jailbreak", neg_label="refusal")
    if r: all_results["jailbreak_vs_refusal"] = r

    # 5. Impossibility refusal (Exp 36)
    print("\n--- Impossibility Refusal (Exp 36) ---")
    r = analyze_experiment("impossibility_vs_benign", "impossibility_refusal.json",
                          pos_label="impossible", neg_label="benign")
    if r: all_results["impossibility_vs_benign"] = r

    # 6. Same-prompt deception (Exp 18b)
    print("\n--- Same-Prompt Deception (Exp 18b) ---")
    r = analyze_experiment("same_prompt_deception", "same_prompt_deception.json",
                          pos_label="deceptive", neg_label="honest")
    if r: all_results["same_prompt_deception"] = r

    # 7. Extended key geometry (Exp 38)
    print("\n--- Extended Key Geometry (Exp 38) ---")
    r = analyze_experiment("extended_features", "extended_key_geometry.json",
                          pos_label="harmful", neg_label="benign")
    if r: all_results["extended_features"] = r

    # 8. Same-prompt sycophancy (Exp 39) — for the record
    print("\n--- Same-Prompt Sycophancy (Exp 39) ---")
    r = analyze_experiment("sycophancy", "same_prompt_sycophancy.json",
                          pos_label="sycophancy", neg_label="honest")
    if r: all_results["sycophancy"] = r

    # 9. Jailbreak multimodel (Exp 34)
    print("\n--- Jailbreak Multimodel (Exp 34) ---")
    r = analyze_experiment("jailbreak_multimodel", "jailbreak_multimodel.json",
                          pos_label="jailbreak", neg_label="normal")
    if r: all_results["jailbreak_multimodel"] = r

    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY: Original 4-feature vs Clean 3-feature AUROC")
    print("=" * 60)
    print(f"{'Experiment':<30} {'Orig LR':>8} {'Clean LR':>9} {'Delta':>7} {'Len Only':>9}")
    print("-" * 60)

    for name, r in all_results.items():
        orig_lr = r.get("orig_4_LR", {}).get("auroc", "—")
        clean_lr = r.get("clean_3_LR", {}).get("auroc", "—")
        len_lr = r.get("length_only_LR", {}).get("auroc", "—")

        if isinstance(orig_lr, float) and isinstance(clean_lr, float):
            delta = clean_lr - orig_lr
            delta_str = f"{delta:+.4f}"
        else:
            delta_str = "—"

        print(f"{name:<30} {str(orig_lr):>8} {str(clean_lr):>9} {delta_str:>7} {str(len_lr):>9}")

    # Save
    output = {
        "experiment": "40_norm_free_reanalysis",
        "description": "Reanalysis dropping raw norm to test for length leakage (M3 from PR #2)",
        "clean_features": CLEAN_FEATURES,
        "original_features": ORIG_FEATURES,
        "results": all_results
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
