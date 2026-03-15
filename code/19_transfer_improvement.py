#!/usr/bin/env python3
"""
Experiment 19: Cross-Model Transfer Improvement
====================================================

Systematic attack on cross-model transfer gap (0.67 deception, 0.76 censorship).

Bottleneck: Qwen AUROC=0.699 while DeepSeek/Mistral=0.87-0.88.
Root cause: Architecture-specific feature scaling — Qwen's cache geometry
maps differently from DeepSeek/Mistral.

Approaches tested:
  1. Baseline (no normalization) — current state
  2. Per-model z-scoring — normalize each model's features independently
  3. Robust scaling — median/IQR instead of mean/std
  4. Rank transform — convert features to percentile ranks within model
  5. Interaction features — norm × key_rank, norm_per_token × key_entropy
  6. Combined best — z-scoring + interactions
  7. Per-architecture ensemble — separate RFs per architecture family

For each approach: train on deception_forensics, test on natural_deception.
Report per-target-model AUROC + all-to-all AUROC.

Funding the Commons Hackathon — March 14-15, 2026
Liberation Labs / THCoalition / JiminAI
"""

import json
import os
import sys
import warnings
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")

# ================================================================
# CONSTANTS
# ================================================================

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
OUTPUT_DIR = RESULTS_DIR / "hackathon"
FEATURE_NAMES = ["norms", "norms_per_token", "key_ranks", "key_entropies"]


# ================================================================
# DATA LOADING (reused from exp 18)
# ================================================================

def load_deception_forensics(filepath: str) -> Tuple[np.ndarray, np.ndarray, str]:
    """Load experiment_1 honest vs deceptive data."""
    with open(filepath) as f:
        data = json.load(f)

    fname = Path(filepath).stem
    model_name = fname.replace("deception_forensics_", "").replace("_results", "")

    e1 = data["experiment_1"]["conditions"]
    honest = e1["honest"]
    deceptive = e1["deceptive"]

    X_honest = np.column_stack([honest[f] for f in FEATURE_NAMES])
    X_deceptive = np.column_stack([deceptive[f] for f in FEATURE_NAMES])

    X = np.vstack([X_honest, X_deceptive])
    y = np.concatenate([np.zeros(len(honest["norms"])), np.ones(len(deceptive["norms"]))])

    return X, y, model_name


def load_natural_deception(filepath: str) -> Tuple[np.ndarray, np.ndarray, str]:
    """Load censored vs control data."""
    with open(filepath) as f:
        data = json.load(f)

    fname = Path(filepath).stem
    model_name = fname.replace("natural_deception_", "").replace("_results", "")

    all_data = data["experiment"]["all_data"]
    censored = all_data["censored"]
    control = all_data["control"]

    X_censored = np.column_stack([censored[f] for f in FEATURE_NAMES])
    X_control = np.column_stack([control[f] for f in FEATURE_NAMES])

    X = np.vstack([X_control, X_censored])
    y = np.concatenate([np.zeros(len(control["norms"])), np.ones(len(censored["norms"]))])

    return X, y, model_name


def load_all_data():
    """Load all deception and natural deception data."""
    # Deception forensics (training)
    dec_files = sorted(RESULTS_DIR.glob("deception_forensics_*_results.json"))
    train_data = {}
    for f in dec_files:
        X, y, name = load_deception_forensics(str(f))
        train_data[name] = (X, y)

    # Natural deception (testing)
    nat_files = sorted(RESULTS_DIR.glob("natural_deception_*_results.json"))
    test_data = {}
    for f in nat_files:
        X, y, name = load_natural_deception(str(f))
        test_data[name] = (X, y)

    return train_data, test_data


# ================================================================
# NORMALIZATION STRATEGIES
# ================================================================

def no_normalization(X_train, y_train, groups_train, X_test, y_test, groups_test):
    """Baseline: no normalization."""
    return X_train, X_test


def per_model_zscore(X_train, y_train, groups_train, X_test, y_test, groups_test):
    """Z-score features independently for each model."""
    X_train_norm = np.copy(X_train)
    X_test_norm = np.copy(X_test)

    for model_id in np.unique(groups_train):
        mask = groups_train == model_id
        scaler = StandardScaler()
        X_train_norm[mask] = scaler.fit_transform(X_train[mask])

    for model_id in np.unique(groups_test):
        mask = groups_test == model_id
        scaler = StandardScaler()
        X_test_norm[mask] = scaler.fit_transform(X_test[mask])

    return X_train_norm, X_test_norm


def robust_scaling(X_train, y_train, groups_train, X_test, y_test, groups_test):
    """Robust scaling (median/IQR) per model."""
    X_train_norm = np.copy(X_train)
    X_test_norm = np.copy(X_test)

    for model_id in np.unique(groups_train):
        mask = groups_train == model_id
        scaler = RobustScaler()
        X_train_norm[mask] = scaler.fit_transform(X_train[mask])

    for model_id in np.unique(groups_test):
        mask = groups_test == model_id
        scaler = RobustScaler()
        X_test_norm[mask] = scaler.fit_transform(X_test[mask])

    return X_train_norm, X_test_norm


def rank_transform(X_train, y_train, groups_train, X_test, y_test, groups_test):
    """Convert features to percentile ranks within each model."""
    from scipy.stats import rankdata

    X_train_norm = np.copy(X_train)
    X_test_norm = np.copy(X_test)

    for model_id in np.unique(groups_train):
        mask = groups_train == model_id
        for j in range(X_train.shape[1]):
            ranks = rankdata(X_train[mask, j])
            X_train_norm[mask, j] = ranks / len(ranks)

    for model_id in np.unique(groups_test):
        mask = groups_test == model_id
        for j in range(X_test.shape[1]):
            ranks = rankdata(X_test[mask, j])
            X_test_norm[mask, j] = ranks / len(ranks)

    return X_train_norm, X_test_norm


def add_interactions(X):
    """Add interaction features: norm×rank, norm_per_token×entropy."""
    interact1 = (X[:, 0] * X[:, 2]).reshape(-1, 1)  # norm × key_rank
    interact2 = (X[:, 1] * X[:, 3]).reshape(-1, 1)  # norm_per_token × key_entropy
    # Also add ratios
    ratio1 = np.where(X[:, 2] > 0, X[:, 0] / X[:, 2], 0).reshape(-1, 1)  # norm / key_rank
    ratio2 = np.where(X[:, 3] > 0, X[:, 1] / X[:, 3], 0).reshape(-1, 1)  # norm_per_token / key_entropy
    return np.hstack([X, interact1, interact2, ratio1, ratio2])


def interaction_features(X_train, y_train, groups_train, X_test, y_test, groups_test):
    """Add interaction terms (8 features total)."""
    return add_interactions(X_train), add_interactions(X_test)


def zscore_plus_interactions(X_train, y_train, groups_train, X_test, y_test, groups_test):
    """Z-score per model, then add interaction features."""
    X_tr, X_te = per_model_zscore(X_train, y_train, groups_train, X_test, y_test, groups_test)
    return add_interactions(X_tr), add_interactions(X_te)


def rank_plus_interactions(X_train, y_train, groups_train, X_test, y_test, groups_test):
    """Rank transform per model, then add interaction features."""
    X_tr, X_te = rank_transform(X_train, y_train, groups_train, X_test, y_test, groups_test)
    return add_interactions(X_tr), add_interactions(X_te)


# ================================================================
# EVALUATION
# ================================================================

def evaluate_approach(name, transform_fn, train_data, test_data):
    """Evaluate a normalization approach."""
    print(f"\n{'='*60}")
    print(f"  APPROACH: {name}")
    print(f"{'='*60}")

    # Pool training data
    X_trains, y_trains, model_ids = [], [], []
    for model_name, (X, y) in train_data.items():
        X_trains.append(X)
        y_trains.append(y)
        model_ids.extend([model_name] * len(X))

    X_train_all = np.vstack(X_trains)
    y_train_all = np.concatenate(y_trains)
    groups_train = np.array(model_ids)

    # Pool test data
    X_tests, y_tests, test_model_ids = [], [], []
    for model_name, (X, y) in test_data.items():
        X_tests.append(X)
        y_tests.append(y)
        test_model_ids.extend([model_name] * len(X))

    X_test_all = np.vstack(X_tests)
    y_test_all = np.concatenate(y_tests)
    groups_test = np.array(test_model_ids)

    # Apply transform
    X_tr, X_te = transform_fn(
        X_train_all, y_train_all, groups_train,
        X_test_all, y_test_all, groups_test
    )

    # Check for NaN/Inf
    if np.any(~np.isfinite(X_tr)) or np.any(~np.isfinite(X_te)):
        print("  WARNING: Non-finite values in transformed features, replacing with 0")
        X_tr = np.nan_to_num(X_tr, nan=0.0, posinf=0.0, neginf=0.0)
        X_te = np.nan_to_num(X_te, nan=0.0, posinf=0.0, neginf=0.0)

    results = {"approach": name, "per_model": {}}

    # Train classifiers
    classifiers = {
        "RF": RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10),
        "LR": LogisticRegression(random_state=42, max_iter=1000),
    }

    for clf_name, clf in classifiers.items():
        clf.fit(X_tr, y_train_all)

        # All-to-all transfer
        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(X_te)[:, 1] if X_te.shape[0] > 0 else np.array([])
        else:
            probs = clf.decision_function(X_te)

        preds = clf.predict(X_te)

        try:
            auroc = roc_auc_score(y_test_all, probs)
        except ValueError:
            auroc = 0.5

        acc = accuracy_score(y_test_all, preds)
        f1 = f1_score(y_test_all, preds, zero_division=0)

        print(f"\n  {clf_name} All-to-All: AUROC={auroc:.3f}, Acc={acc:.3f}, F1={f1:.3f}")

        results[f"{clf_name}_all"] = {"auroc": auroc, "accuracy": acc, "f1": f1}

        # Per-target-model
        for model_name in sorted(test_data.keys()):
            mask = groups_test == model_name
            if mask.sum() == 0:
                continue

            probs_m = probs[mask]
            preds_m = preds[mask]
            y_m = y_test_all[mask]

            try:
                auroc_m = roc_auc_score(y_m, probs_m)
            except ValueError:
                auroc_m = 0.5

            acc_m = accuracy_score(y_m, preds_m)
            f1_m = f1_score(y_m, preds_m, zero_division=0)

            print(f"    {model_name}: AUROC={auroc_m:.3f}, Acc={acc_m:.3f}, F1={f1_m:.3f}")

            if model_name not in results["per_model"]:
                results["per_model"][model_name] = {}
            results["per_model"][model_name][clf_name] = {
                "auroc": auroc_m, "accuracy": acc_m, "f1": f1_m
            }

    # Feature importance (RF only)
    rf = classifiers["RF"]
    feat_names = FEATURE_NAMES.copy()
    if X_tr.shape[1] > 4:
        feat_names += ["norm×rank", "npt×entropy", "norm/rank", "npt/entropy"]

    importances = rf.feature_importances_
    print(f"\n  Feature importance (RF):")
    for fname, imp in sorted(zip(feat_names, importances), key=lambda x: -x[1]):
        print(f"    {fname}: {imp:.3f}")

    results["feature_importance"] = dict(zip(feat_names, importances.tolist()))

    return results


def evaluate_ensemble(train_data, test_data):
    """Per-architecture ensemble with stacking."""
    print(f"\n{'='*60}")
    print(f"  APPROACH: Per-Architecture Ensemble (Stacking)")
    print(f"{'='*60}")

    # Group training models by architecture family
    arch_families = {}
    for model_name, (X, y) in train_data.items():
        if "Qwen" in model_name:
            family = "qwen"
        elif "Llama" in model_name or "TinyLlama" in model_name:
            family = "llama"
        elif "Mistral" in model_name:
            family = "mistral"
        elif "gemma" in model_name:
            family = "gemma"
        else:
            family = "other"

        if family not in arch_families:
            arch_families[family] = {"X": [], "y": []}
        arch_families[family]["X"].append(X)
        arch_families[family]["y"].append(y)

    # Train per-family z-scored RF classifiers
    family_clfs = {}
    for family, data in arch_families.items():
        X_fam = np.vstack(data["X"])
        y_fam = np.concatenate(data["y"])

        scaler = StandardScaler()
        X_fam_scaled = scaler.fit_transform(X_fam)

        clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        clf.fit(X_fam_scaled, y_fam)

        family_clfs[family] = (clf, scaler)
        print(f"  Family '{family}': {len(X_fam)} samples, {len(set([m for m in train_data if family.lower() in m.lower()]))} models")

    # For each test model: predict with all family classifiers, take best/vote
    results = {"approach": "ensemble", "per_model": {}}

    for test_model, (X_test, y_test) in test_data.items():
        print(f"\n  Testing: {test_model}")

        # Get predictions from each family classifier
        all_probs = []
        for family, (clf, scaler) in family_clfs.items():
            X_scaled = scaler.transform(X_test)
            probs = clf.predict_proba(X_scaled)[:, 1]
            all_probs.append(probs)

            try:
                auroc = roc_auc_score(y_test, probs)
            except ValueError:
                auroc = 0.5
            print(f"    {family} family: AUROC={auroc:.3f}")

        # Ensemble: average probabilities
        avg_probs = np.mean(all_probs, axis=0)
        preds = (avg_probs >= 0.5).astype(int)

        try:
            auroc = roc_auc_score(y_test, avg_probs)
        except ValueError:
            auroc = 0.5

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, zero_division=0)

        print(f"    ENSEMBLE: AUROC={auroc:.3f}, Acc={acc:.3f}, F1={f1:.3f}")

        results["per_model"][test_model] = {
            "auroc": auroc, "accuracy": acc, "f1": f1
        }

    # All-to-all ensemble
    all_probs_combined = []
    all_y_combined = []
    for test_model, (X_test, y_test) in test_data.items():
        probs_list = []
        for family, (clf, scaler) in family_clfs.items():
            X_scaled = scaler.transform(X_test)
            probs = clf.predict_proba(X_scaled)[:, 1]
            probs_list.append(probs)
        avg_probs = np.mean(probs_list, axis=0)
        all_probs_combined.append(avg_probs)
        all_y_combined.append(y_test)

    all_probs_flat = np.concatenate(all_probs_combined)
    all_y_flat = np.concatenate(all_y_combined)

    try:
        auroc_all = roc_auc_score(all_y_flat, all_probs_flat)
    except ValueError:
        auroc_all = 0.5

    preds_all = (all_probs_flat >= 0.5).astype(int)
    acc_all = accuracy_score(all_y_flat, preds_all)
    f1_all = f1_score(all_y_flat, preds_all, zero_division=0)

    print(f"\n  ENSEMBLE All-to-All: AUROC={auroc_all:.3f}, Acc={acc_all:.3f}, F1={f1_all:.3f}")
    results["all"] = {"auroc": auroc_all, "accuracy": acc_all, "f1": f1_all}

    return results


# ================================================================
# MAIN
# ================================================================

def main():
    print("=" * 60)
    print("  EXPERIMENT 19: CROSS-MODEL TRANSFER IMPROVEMENT")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    train_data, test_data = load_all_data()

    print(f"\nTraining data: {sum(len(X) for X, y in train_data.values())} samples from {len(train_data)} models")
    for m, (X, y) in train_data.items():
        print(f"  {m}: {len(X)} samples ({int(y.sum())} deceptive, {int((1-y).sum())} honest)")

    print(f"\nTest data: {sum(len(X) for X, y in test_data.values())} samples from {len(test_data)} models")
    for m, (X, y) in test_data.items():
        print(f"  {m}: {len(X)} samples ({int(y.sum())} censored, {int((1-y).sum())} control)")

    # Run all normalization approaches
    approaches = [
        ("1. Baseline (no normalization)", no_normalization),
        ("2. Per-model z-score", per_model_zscore),
        ("3. Robust scaling (median/IQR)", robust_scaling),
        ("4. Rank transform", rank_transform),
        ("5. Interaction features", interaction_features),
        ("6. Z-score + interactions", zscore_plus_interactions),
        ("7. Rank + interactions", rank_plus_interactions),
    ]

    all_results = {}
    for name, fn in approaches:
        result = evaluate_approach(name, fn, train_data, test_data)
        all_results[name] = result

    # Run ensemble approach separately
    ensemble_result = evaluate_ensemble(train_data, test_data)
    all_results["8. Per-architecture ensemble"] = ensemble_result

    # ============================================================
    # SUMMARY TABLE
    # ============================================================
    print("\n" + "=" * 80)
    print("  SUMMARY: All-to-All AUROC by Approach")
    print("=" * 80)
    print(f"{'Approach':<40} {'RF AUROC':>10} {'LR AUROC':>10} {'Best':>10}")
    print("-" * 80)

    best_auroc = 0
    best_approach = None

    for name, result in all_results.items():
        if "RF_all" in result:
            rf = result["RF_all"]["auroc"]
            lr = result.get("LR_all", {}).get("auroc", 0)
            best = max(rf, lr)
        elif "all" in result:
            best = result["all"]["auroc"]
            rf = best
            lr = 0
        else:
            continue

        print(f"  {name:<38} {rf:>10.3f} {lr:>10.3f} {best:>10.3f}")

        if best > best_auroc:
            best_auroc = best
            best_approach = name

    print("-" * 80)
    print(f"  BEST: {best_approach} (AUROC={best_auroc:.3f})")

    # Per-model breakdown for best approach
    print(f"\n  Per-model breakdown ({best_approach}):")
    best_result = all_results[best_approach]
    for model_name in sorted(best_result.get("per_model", {}).keys()):
        model_data = best_result["per_model"][model_name]
        if isinstance(model_data, dict) and "auroc" in model_data:
            print(f"    {model_name}: AUROC={model_data['auroc']:.3f}")
        elif isinstance(model_data, dict):
            for clf_name, clf_data in model_data.items():
                print(f"    {model_name} ({clf_name}): AUROC={clf_data['auroc']:.3f}")

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "transfer_improvement.json"

    # Convert numpy types for JSON
    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=convert)

    print(f"\n  Results saved to: {output_path}")

    return all_results


if __name__ == "__main__":
    main()
