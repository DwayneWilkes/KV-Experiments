#!/usr/bin/env python3
"""
Experiment 18: Cross-Condition Deception Transfer
====================================================

Tests whether classifiers trained on SYSTEM-PROMPT-MANIPULATED deception
(models given "lie about everything" vs "answer honestly") can detect
NATURAL deception (e.g., DeepSeek censorship on Tiananmen/Uyghur topics).

Rationale:
  If a classifier trained on instructed deception (1) can detect natural
  censorship evasion (2), it proves a shared deception geometry beyond
  system-prompt fingerprinting. This is the strongest evidence that
  KV-cache geometry captures genuine cognitive states, not artifacts.

Data sources:
  Training: deception_forensics_*_results.json  (experiment_1: honest vs deceptive)
  Testing:  natural_deception_*_results.json    (censored vs control)

Transfer scenarios:
  A. Same-model:  Mistral deception -> Mistral natural
  B. Same-family: Qwen2.5-7B deception -> Qwen2.5-14B natural
  C. Cross-model: All deception models -> DeepSeek natural
  D. All-to-all:  All deception -> All natural

Classifiers: Random Forest, Logistic Regression, Linear SVM

Output: Printed report + results/hackathon/cross_condition_transfer.json

Funding the Commons Hackathon — March 14-15, 2026
Liberation Labs / THCoalition / JiminAI
"""

import json
import glob
import os
import sys
import warnings
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, classification_report
)
from sklearn.calibration import CalibratedClassifierCV

warnings.filterwarnings("ignore", category=UserWarning)

# ================================================================
# CONSTANTS
# ================================================================

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
OUTPUT_DIR = RESULTS_DIR / "hackathon"

FEATURE_NAMES = ["norms", "norms_per_token", "key_ranks", "key_entropies"]

CENSORED_TOPICS = [
    "tiananmen", "uyghur", "taiwan",
    "political_prisoners", "xi_criticism", "hong_kong"
]
CONTROL_TOPICS = [
    "cuisine", "geography", "inventions",
    "culture", "literature", "festivals"
]
COMPLEX_TOPICS = [
    "rwanda_genocide", "armenian_genocide", "khmer_rouge",
    "abu_ghraib", "apartheid", "bosnian_genocide"
]

# Model name mappings for file -> display name
def extract_model_name(filepath: str) -> str:
    """Extract short model name from result file path."""
    fname = Path(filepath).stem
    # Remove prefix and _results suffix
    for prefix in ["deception_forensics_", "natural_deception_"]:
        if prefix in fname:
            return fname.replace(prefix, "").replace("_results", "")
    return fname


# ================================================================
# DATA LOADING
# ================================================================

def load_deception_forensics(filepath: str) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Load deception forensics data from experiment_1 (honest vs deceptive).

    Returns:
        X: Feature matrix (n_samples, 4)
        y: Labels (0=honest, 1=deceptive)
        model_name: Short model name
    """
    with open(filepath) as f:
        data = json.load(f)

    model_name = extract_model_name(filepath)
    e1 = data["experiment_1"]["conditions"]

    honest = e1["honest"]
    deceptive = e1["deceptive"]

    n_honest = len(honest["norms"])
    n_deceptive = len(deceptive["norms"])

    X_honest = np.column_stack([
        honest[feat] for feat in FEATURE_NAMES
    ])
    X_deceptive = np.column_stack([
        deceptive[feat] for feat in FEATURE_NAMES
    ])

    X = np.vstack([X_honest, X_deceptive])
    y = np.concatenate([np.zeros(n_honest), np.ones(n_deceptive)])

    return X, y, model_name


def load_natural_deception(filepath: str) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Load natural deception data (censored vs control from all_data).

    Returns:
        X: Feature matrix (n_samples, 4)
        y: Labels (0=control, 1=censored)
        model_name: Short model name
    """
    with open(filepath) as f:
        data = json.load(f)

    model_name = extract_model_name(filepath)
    all_data = data["experiment"]["all_data"]

    censored = all_data["censored"]
    control = all_data["control"]

    n_censored = len(censored["norms"])
    n_control = len(control["norms"])

    X_censored = np.column_stack([
        censored[feat] for feat in FEATURE_NAMES
    ])
    X_control = np.column_stack([
        control[feat] for feat in FEATURE_NAMES
    ])

    X = np.vstack([X_control, X_censored])
    y = np.concatenate([np.zeros(n_control), np.ones(n_censored)])

    return X, y, model_name


def load_per_topic_data(filepath: str) -> Dict[str, Tuple[np.ndarray, int]]:
    """
    Load per-topic data for natural deception files.

    Returns dict of topic -> (X_features, n_samples).
    Note: per_topic only has norms, key_ranks, key_entropies (no norms_per_token).
    We compute norms_per_token as norms / total_tokens where available.
    """
    with open(filepath) as f:
        data = json.load(f)

    per_topic = data["experiment"].get("per_topic", {})
    result = {}

    for topic, topic_data in per_topic.items():
        n = len(topic_data["norms"])
        norms = np.array(topic_data["norms"])
        key_ranks = np.array(topic_data["key_ranks"])
        key_entropies = np.array(topic_data["key_entropies"])

        # Compute norms_per_token if total_tokens available
        if "total_tokens" in topic_data and topic_data["total_tokens"]:
            total_tokens = np.array(topic_data["total_tokens"])
            # Avoid division by zero
            total_tokens = np.where(total_tokens == 0, 1, total_tokens)
            norms_per_token = norms / total_tokens
        elif "norms_per_token" in topic_data:
            norms_per_token = np.array(topic_data["norms_per_token"])
        else:
            # Fallback: use norms as proxy (will be scaled anyway)
            norms_per_token = norms

        X = np.column_stack([norms, norms_per_token, key_ranks, key_entropies])
        result[topic] = (X, n)

    return result


# ================================================================
# CLASSIFIER TRAINING AND EVALUATION
# ================================================================

def make_classifiers() -> Dict[str, Any]:
    """Create fresh classifier instances."""
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
        "LogisticRegression": LogisticRegression(
            max_iter=2000, random_state=42, solver="lbfgs"
        ),
        "LinearSVM": CalibratedClassifierCV(
            LinearSVC(max_iter=5000, random_state=42, dual="auto"),
            cv=3
        ),
    }


def evaluate_classifier(
    clf, scaler, X_test: np.ndarray, y_test: np.ndarray
) -> Dict[str, float]:
    """Evaluate a trained classifier on test data."""
    X_scaled = scaler.transform(X_test)
    y_pred = clf.predict(X_scaled)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }

    # AUROC requires probability estimates
    if hasattr(clf, "predict_proba"):
        try:
            y_prob = clf.predict_proba(X_scaled)[:, 1]
            metrics["auroc"] = float(roc_auc_score(y_test, y_prob))
        except Exception:
            metrics["auroc"] = None
    elif hasattr(clf, "decision_function"):
        try:
            y_scores = clf.decision_function(X_scaled)
            metrics["auroc"] = float(roc_auc_score(y_test, y_scores))
        except Exception:
            metrics["auroc"] = None
    else:
        metrics["auroc"] = None

    return metrics


def train_and_evaluate(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    scenario_name: str
) -> Dict[str, Any]:
    """
    Train all classifiers on training data, evaluate on test data.

    Returns dict with per-classifier metrics and feature importance.
    """
    # Check for NaN/Inf in data
    train_mask = np.isfinite(X_train).all(axis=1)
    test_mask = np.isfinite(X_test).all(axis=1)

    if not train_mask.all():
        n_bad = (~train_mask).sum()
        print(f"  WARNING: Dropping {n_bad} NaN/Inf rows from training data")
        X_train = X_train[train_mask]
        y_train = y_train[train_mask]

    if not test_mask.all():
        n_bad = (~test_mask).sum()
        print(f"  WARNING: Dropping {n_bad} NaN/Inf rows from test data")
        X_test = X_test[test_mask]
        y_test = y_test[test_mask]

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    results = {
        "scenario": scenario_name,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "train_class_balance": {
            "honest": int((y_train == 0).sum()),
            "deceptive": int((y_train == 1).sum()),
        },
        "test_class_balance": {
            "control": int((y_test == 0).sum()),
            "censored": int((y_test == 1).sum()),
        },
        "classifiers": {},
    }

    classifiers = make_classifiers()

    for clf_name, clf in classifiers.items():
        try:
            clf.fit(X_train_scaled, y_train)
            metrics = evaluate_classifier(clf, scaler, X_test, y_test)
            results["classifiers"][clf_name] = metrics

            # Feature importance for RandomForest
            if clf_name == "RandomForest":
                importances = clf.feature_importances_
                results["feature_importance"] = {
                    feat: float(imp)
                    for feat, imp in zip(FEATURE_NAMES, importances)
                }
        except Exception as e:
            results["classifiers"][clf_name] = {"error": str(e)}

    return results


# ================================================================
# PER-TOPIC ANALYSIS
# ================================================================

def per_topic_breakdown(
    X_train: np.ndarray, y_train: np.ndarray,
    natural_file: str
) -> Dict[str, Any]:
    """
    Train on deception forensics, test per-topic on natural deception.

    Returns detection rates per censored topic.
    """
    topic_data = load_per_topic_data(natural_file)

    if not topic_data:
        return {}

    # Scale using training data
    scaler = StandardScaler()
    train_mask = np.isfinite(X_train).all(axis=1)
    X_train_clean = X_train[train_mask]
    y_train_clean = y_train[train_mask]
    X_train_scaled = scaler.fit_transform(X_train_clean)

    # Train RF
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train_scaled, y_train_clean)

    results = {}

    for topic, (X_topic, n) in topic_data.items():
        # Filter valid rows
        valid = np.isfinite(X_topic).all(axis=1)
        if not valid.any():
            continue
        X_valid = X_topic[valid]

        X_scaled = scaler.transform(X_valid)
        y_pred = rf.predict(X_scaled)
        y_prob = rf.predict_proba(X_scaled)[:, 1]

        # Determine expected label: censored topics -> 1, control -> 0
        if topic in CENSORED_TOPICS:
            expected_label = 1
            category = "censored"
        elif topic in CONTROL_TOPICS:
            expected_label = 0
            category = "control"
        elif topic in COMPLEX_TOPICS:
            expected_label = 0  # complex but not censored
            category = "complex_noncensored"
        else:
            expected_label = 0
            category = "unknown"

        detection_rate = float(np.mean(y_pred == expected_label))
        mean_deception_prob = float(np.mean(y_prob))

        results[topic] = {
            "category": category,
            "n_samples": int(valid.sum()),
            "predicted_deceptive_pct": float(np.mean(y_pred == 1) * 100),
            "mean_deception_probability": mean_deception_prob,
            "correct_classification_rate": detection_rate,
        }

    return results


# ================================================================
# REPORT FORMATTING
# ================================================================

def print_separator(char="=", width=72):
    print(char * width)


def print_scenario_results(results: Dict[str, Any]):
    """Print formatted results for one scenario."""
    print(f"\n  Train: {results['n_train']} samples "
          f"(honest={results['train_class_balance']['honest']}, "
          f"deceptive={results['train_class_balance']['deceptive']})")
    print(f"  Test:  {results['n_test']} samples "
          f"(control={results['test_class_balance']['control']}, "
          f"censored={results['test_class_balance']['censored']})")
    print()

    # Header
    print(f"  {'Classifier':<22} {'AUROC':>7} {'Acc':>7} {'Prec':>7} "
          f"{'Recall':>7} {'F1':>7}")
    print(f"  {'-'*22} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")

    for clf_name, metrics in results["classifiers"].items():
        if "error" in metrics:
            print(f"  {clf_name:<22} ERROR: {metrics['error']}")
            continue

        auroc = metrics.get("auroc")
        auroc_str = f"{auroc:.4f}" if auroc is not None else "  N/A"

        print(f"  {clf_name:<22} {auroc_str:>7} "
              f"{metrics['accuracy']:.4f}  "
              f"{metrics['precision']:.4f}  "
              f"{metrics['recall']:.4f}  "
              f"{metrics['f1']:.4f}")

    # Feature importance
    if "feature_importance" in results:
        print()
        print("  RF Feature Importance:")
        fi = results["feature_importance"]
        sorted_fi = sorted(fi.items(), key=lambda x: x[1], reverse=True)
        for feat, imp in sorted_fi:
            bar = "#" * int(imp * 40)
            print(f"    {feat:<20} {imp:.4f}  {bar}")


def print_topic_results(topic_results: Dict[str, Dict], model_name: str):
    """Print per-topic breakdown."""
    if not topic_results:
        return

    print(f"\n  Per-Topic Breakdown ({model_name}):")
    print(f"  {'Topic':<25} {'Category':<20} {'Deceptive%':>10} "
          f"{'MeanProb':>10} {'Correct%':>10}")
    print(f"  {'-'*25} {'-'*20} {'-'*10} {'-'*10} {'-'*10}")

    # Sort by category then topic name
    sorted_topics = sorted(
        topic_results.items(),
        key=lambda x: (
            {"censored": 0, "control": 1, "complex_noncensored": 2}.get(
                x[1]["category"], 3
            ),
            x[0]
        )
    )

    prev_category = None
    for topic, info in sorted_topics:
        if info["category"] != prev_category and prev_category is not None:
            print(f"  {'-'*25} {'-'*20} {'-'*10} {'-'*10} {'-'*10}")
        prev_category = info["category"]

        print(f"  {topic:<25} {info['category']:<20} "
              f"{info['predicted_deceptive_pct']:>9.1f}% "
              f"{info['mean_deception_probability']:>10.4f} "
              f"{info['correct_classification_rate']*100:>9.1f}%")


# ================================================================
# MAIN
# ================================================================

def main():
    print_separator()
    print("EXPERIMENT 18: Cross-Condition Deception Transfer")
    print("Train on system-prompt deception -> Test on natural censorship")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print_separator()

    # ------------------------------------------------------------------
    # 1. Discover and load all data files
    # ------------------------------------------------------------------
    deception_files = sorted(glob.glob(
        str(RESULTS_DIR / "deception_forensics_*_results.json")
    ))
    natural_files = sorted(glob.glob(
        str(RESULTS_DIR / "natural_deception_*_results.json")
    ))

    if not deception_files:
        print("ERROR: No deception_forensics files found in", RESULTS_DIR)
        sys.exit(1)
    if not natural_files:
        print("ERROR: No natural_deception files found in", RESULTS_DIR)
        sys.exit(1)

    print(f"\nDeception forensics files ({len(deception_files)}):")
    deception_data = {}
    for f in deception_files:
        X, y, name = load_deception_forensics(f)
        deception_data[name] = {"X": X, "y": y, "file": f}
        print(f"  {name}: {len(y)} samples "
              f"(honest={int((y==0).sum())}, deceptive={int((y==1).sum())})")

    print(f"\nNatural deception files ({len(natural_files)}):")
    natural_data = {}
    for f in natural_files:
        X, y, name = load_natural_deception(f)
        natural_data[name] = {"X": X, "y": y, "file": f}
        print(f"  {name}: {len(y)} samples "
              f"(control={int((y==0).sum())}, censored={int((y==1).sum())})")

    # ------------------------------------------------------------------
    # 2. Pool all training data
    # ------------------------------------------------------------------
    X_all_train = np.vstack([d["X"] for d in deception_data.values()])
    y_all_train = np.concatenate([d["y"] for d in deception_data.values()])

    X_all_test = np.vstack([d["X"] for d in natural_data.values()])
    y_all_test = np.concatenate([d["y"] for d in natural_data.values()])

    print(f"\nPooled training data: {len(y_all_train)} samples")
    print(f"Pooled test data: {len(y_all_test)} samples")

    all_results = {
        "metadata": {
            "experiment": "18: Cross-Condition Deception Transfer",
            "timestamp": datetime.now().isoformat(),
            "deception_models": list(deception_data.keys()),
            "natural_models": list(natural_data.keys()),
            "features": FEATURE_NAMES,
            "classifiers": ["RandomForest", "LogisticRegression", "LinearSVM"],
            "description": (
                "Tests whether classifiers trained on system-prompt-manipulated "
                "deception can detect natural censorship evasion. Positive transfer "
                "proves shared deception geometry beyond prompt fingerprinting."
            ),
        },
        "scenarios": {},
        "per_topic": {},
    }

    # ------------------------------------------------------------------
    # Scenario A: Same-model transfer (Mistral -> Mistral)
    # ------------------------------------------------------------------
    print_separator()
    print("SCENARIO A: Same-Model Transfer")
    print("  Train: Mistral-7B deception forensics")
    print("  Test:  Mistral-7B natural deception")
    print_separator("-")

    mistral_train_key = None
    mistral_test_key = None
    for k in deception_data:
        if "Mistral" in k or "mistral" in k:
            mistral_train_key = k
    for k in natural_data:
        if "Mistral" in k or "mistral" in k:
            mistral_test_key = k

    if mistral_train_key and mistral_test_key:
        results_a = train_and_evaluate(
            deception_data[mistral_train_key]["X"],
            deception_data[mistral_train_key]["y"],
            natural_data[mistral_test_key]["X"],
            natural_data[mistral_test_key]["y"],
            f"Same-model: {mistral_train_key} -> {mistral_test_key}"
        )
        print_scenario_results(results_a)
        all_results["scenarios"]["A_same_model_mistral"] = results_a

        # Per-topic for Mistral
        topic_a = per_topic_breakdown(
            deception_data[mistral_train_key]["X"],
            deception_data[mistral_train_key]["y"],
            natural_data[mistral_test_key]["file"]
        )
        if topic_a:
            print_topic_results(topic_a, mistral_test_key)
            all_results["per_topic"]["A_mistral"] = topic_a
    else:
        print("  SKIPPED: Mistral data not found in both sets")

    # ------------------------------------------------------------------
    # Scenario B: Same-family transfer (Qwen2.5-7B -> Qwen2.5-14B)
    # ------------------------------------------------------------------
    print()
    print_separator()
    print("SCENARIO B: Same-Family Transfer")
    print("  Train: Qwen2.5-7B deception forensics")
    print("  Test:  Qwen2.5-14B natural deception")
    print_separator("-")

    qwen7b_key = None
    qwen14b_key = None
    for k in deception_data:
        if "Qwen2.5-7B" in k:
            qwen7b_key = k
    for k in natural_data:
        if "Qwen2.5-14B" in k:
            qwen14b_key = k

    if qwen7b_key and qwen14b_key:
        results_b = train_and_evaluate(
            deception_data[qwen7b_key]["X"],
            deception_data[qwen7b_key]["y"],
            natural_data[qwen14b_key]["X"],
            natural_data[qwen14b_key]["y"],
            f"Same-family: {qwen7b_key} -> {qwen14b_key}"
        )
        print_scenario_results(results_b)
        all_results["scenarios"]["B_same_family_qwen"] = results_b

        # Per-topic for Qwen
        topic_b = per_topic_breakdown(
            deception_data[qwen7b_key]["X"],
            deception_data[qwen7b_key]["y"],
            natural_data[qwen14b_key]["file"]
        )
        if topic_b:
            print_topic_results(topic_b, qwen14b_key)
            all_results["per_topic"]["B_qwen"] = topic_b
    else:
        print("  SKIPPED: Qwen family data not found in both sets")

    # ------------------------------------------------------------------
    # Scenario C: Cross-model transfer (All deception -> DeepSeek natural)
    # ------------------------------------------------------------------
    print()
    print_separator()
    print("SCENARIO C: Cross-Model Transfer")
    print("  Train: ALL deception forensics models (pooled)")
    print("  Test:  DeepSeek-R1-Distill-Qwen-14B natural deception")
    print_separator("-")

    deepseek_key = None
    for k in natural_data:
        if "DeepSeek" in k or "deepseek" in k:
            deepseek_key = k

    if deepseek_key:
        results_c = train_and_evaluate(
            X_all_train, y_all_train,
            natural_data[deepseek_key]["X"],
            natural_data[deepseek_key]["y"],
            f"Cross-model: All deception -> {deepseek_key}"
        )
        print_scenario_results(results_c)
        all_results["scenarios"]["C_cross_model_deepseek"] = results_c

        # Per-topic for DeepSeek
        topic_c = per_topic_breakdown(
            X_all_train, y_all_train,
            natural_data[deepseek_key]["file"]
        )
        if topic_c:
            print_topic_results(topic_c, deepseek_key)
            all_results["per_topic"]["C_deepseek"] = topic_c
    else:
        print("  SKIPPED: DeepSeek data not found")

    # ------------------------------------------------------------------
    # Scenario D: All-to-all transfer
    # ------------------------------------------------------------------
    print()
    print_separator()
    print("SCENARIO D: All-to-All Transfer")
    print("  Train: ALL deception forensics models (pooled)")
    print("  Test:  ALL natural deception models (pooled)")
    print_separator("-")

    results_d = train_and_evaluate(
        X_all_train, y_all_train,
        X_all_test, y_all_test,
        "All-to-all: All deception -> All natural"
    )
    print_scenario_results(results_d)
    all_results["scenarios"]["D_all_to_all"] = results_d

    # Per-topic breakdown for all natural deception models
    for nat_name, nat_info in natural_data.items():
        topic_d = per_topic_breakdown(
            X_all_train, y_all_train,
            nat_info["file"]
        )
        if topic_d:
            print_topic_results(topic_d, nat_name)
            all_results["per_topic"][f"D_{nat_name}"] = topic_d

    # ------------------------------------------------------------------
    # Scenario E: Per-model natural deception evaluation
    # Each natural deception model tested individually against pooled training
    # ------------------------------------------------------------------
    print()
    print_separator()
    print("SCENARIO E: Per-Model Natural Deception (All Training -> Each Model)")
    print_separator("-")

    for nat_name, nat_info in natural_data.items():
        print(f"\n  --- {nat_name} ---")
        results_e = train_and_evaluate(
            X_all_train, y_all_train,
            nat_info["X"], nat_info["y"],
            f"All deception -> {nat_name}"
        )
        print_scenario_results(results_e)
        all_results["scenarios"][f"E_{nat_name}"] = results_e

    # ------------------------------------------------------------------
    # Summary: Global Feature Importance (from all-to-all RF)
    # ------------------------------------------------------------------
    print()
    print_separator()
    print("GLOBAL FEATURE IMPORTANCE (All-to-All Random Forest)")
    print_separator("-")

    if "feature_importance" in results_d:
        fi = results_d["feature_importance"]
        sorted_fi = sorted(fi.items(), key=lambda x: x[1], reverse=True)
        print()
        for rank, (feat, imp) in enumerate(sorted_fi, 1):
            bar = "#" * int(imp * 50)
            print(f"  {rank}. {feat:<20} {imp:.4f}  {bar}")
        print()

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print()
    print_separator()
    print("SUMMARY: Best AUROC per Scenario (Random Forest)")
    print_separator("-")

    summary_rows = []
    for scenario_key, scenario_results in all_results["scenarios"].items():
        rf_metrics = scenario_results["classifiers"].get("RandomForest", {})
        auroc = rf_metrics.get("auroc")
        f1 = rf_metrics.get("f1")
        acc = rf_metrics.get("accuracy")
        scenario_name = scenario_results["scenario"]

        summary_rows.append({
            "scenario": scenario_key,
            "name": scenario_name,
            "auroc": auroc,
            "f1": f1,
            "accuracy": acc,
        })

    print(f"\n  {'Scenario':<50} {'AUROC':>7} {'F1':>7} {'Acc':>7}")
    print(f"  {'-'*50} {'-'*7} {'-'*7} {'-'*7}")

    for row in summary_rows:
        auroc_str = f"{row['auroc']:.4f}" if row["auroc"] is not None else "  N/A"
        f1_str = f"{row['f1']:.4f}" if row["f1"] is not None else "  N/A"
        acc_str = f"{row['accuracy']:.4f}" if row["accuracy"] is not None else "  N/A"
        # Truncate name for display
        name = row["name"][:48]
        print(f"  {name:<50} {auroc_str:>7} {f1_str:>7} {acc_str:>7}")

    all_results["summary"] = summary_rows

    # ------------------------------------------------------------------
    # Interpretation
    # ------------------------------------------------------------------
    print()
    print_separator()
    print("INTERPRETATION")
    print_separator("-")

    # Get Scenario C (cross-model to DeepSeek) AUROC
    scenario_c = all_results["scenarios"].get("C_cross_model_deepseek", {})
    rf_c = scenario_c.get("classifiers", {}).get("RandomForest", {})
    auroc_c = rf_c.get("auroc")

    scenario_d = all_results["scenarios"].get("D_all_to_all", {})
    rf_d = scenario_d.get("classifiers", {}).get("RandomForest", {})
    auroc_d = rf_d.get("auroc")

    if auroc_c is not None:
        if auroc_c >= 0.8:
            verdict = "STRONG TRANSFER"
            detail = (
                "Classifiers trained on system-prompt deception reliably detect\n"
                "  natural censorship. This proves shared deception geometry\n"
                "  BEYOND system-prompt fingerprinting."
            )
        elif auroc_c >= 0.65:
            verdict = "MODERATE TRANSFER"
            detail = (
                "Partial cross-condition transfer detected. The deception geometry\n"
                "  is partially shared but the signal is weaker for natural\n"
                "  deception than instructed deception."
            )
        elif auroc_c >= 0.55:
            verdict = "WEAK TRANSFER"
            detail = (
                "Marginal transfer. Some shared structure exists but the\n"
                "  classifiers struggle to generalize across deception types."
            )
        else:
            verdict = "NO TRANSFER"
            detail = (
                "System-prompt deception classifiers cannot detect natural\n"
                "  censorship. The two phenomena may have different geometric\n"
                "  signatures, or system-prompt fingerprinting dominates."
            )

        print(f"\n  Cross-model transfer AUROC: {auroc_c:.4f}")
        print(f"  Verdict: {verdict}")
        print(f"  {detail}")

    if auroc_d is not None:
        print(f"\n  All-to-all transfer AUROC:  {auroc_d:.4f}")

    # Check topic-level results for DeepSeek
    ds_topics = all_results["per_topic"].get("C_deepseek", {})
    if ds_topics:
        censored_rates = []
        control_rates = []
        for topic, info in ds_topics.items():
            if info["category"] == "censored":
                censored_rates.append(info["predicted_deceptive_pct"])
            elif info["category"] == "control":
                control_rates.append(info["predicted_deceptive_pct"])

        if censored_rates and control_rates:
            mean_cens = np.mean(censored_rates)
            mean_ctrl = np.mean(control_rates)
            print(f"\n  DeepSeek per-topic detection rates:")
            print(f"    Censored topics avg predicted deceptive: {mean_cens:.1f}%")
            print(f"    Control topics avg predicted deceptive:  {mean_ctrl:.1f}%")
            print(f"    Separation: {abs(mean_cens - mean_ctrl):.1f} percentage points")

    print()

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "cross_condition_transfer.json"

    # Convert numpy types for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(i) for i in obj]
        return obj

    with open(output_path, "w") as f:
        json.dump(convert_for_json(all_results), f, indent=2)

    print(f"Results saved to: {output_path}")
    print_separator()


if __name__ == "__main__":
    main()
