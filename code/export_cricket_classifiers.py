#!/usr/bin/env python3
"""
Export Cricket Classifiers
===========================

Train RF classifiers on existing Campaign 2 data and export
as .joblib files for the Cricket demo. Also generates
pre-computed examples JSON.

No GPU needed — works entirely from existing result JSONs.
"""

import json
import os
import sys
import numpy as np
from pathlib import Path
from datetime import datetime

try:
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score, classification_report
    from sklearn.model_selection import cross_val_score
except ImportError:
    print("ERROR: sklearn and joblib required. pip install scikit-learn joblib")
    sys.exit(1)


def load_deception_features(results_dir):
    """Load deception forensics features from Campaign 2 results."""
    observations = []

    deception_files = [
        f for f in os.listdir(results_dir)
        if f.startswith("deception_forensics_") and f.endswith("_results.json")
    ]

    for df in sorted(deception_files):
        filepath = os.path.join(results_dir, df)
        data = json.load(open(filepath))
        model_raw = data.get("metadata", {}).get("model", df)
        model_id = model_raw if isinstance(model_raw, str) else model_raw.get("model_id", df)

        for exp_key in ["experiment_1", "experiment_2", "experiment_3"]:
            exp = data.get(exp_key, {})
            conditions = exp.get("conditions", {})

            for cond_name, cond_data in conditions.items():
                norms = cond_data.get("norms", [])
                norms_pt = cond_data.get("norms_per_token", [])
                ranks = cond_data.get("key_ranks", [])
                entropies = cond_data.get("key_entropies", [])

                n = min(len(norms), len(norms_pt), len(ranks), len(entropies))
                for i in range(n):
                    observations.append({
                        "model_id": model_id,
                        "experiment": exp_key,
                        "condition": cond_name,
                        "norm": norms[i],
                        "norm_per_token": norms_pt[i],
                        "key_rank": ranks[i],
                        "key_entropy": entropies[i],
                    })

    return observations


def load_censorship_features(results_dir):
    """Load natural deception (censorship) features from Campaign 2 results."""
    observations = []

    nd_files = [
        f for f in os.listdir(results_dir)
        if f.startswith("natural_deception_") and f.endswith("_results.json")
    ]

    for nf in sorted(nd_files):
        filepath = os.path.join(results_dir, nf)
        data = json.load(open(filepath))
        model_id = nf.replace("natural_deception_", "").replace("_results.json", "")

        for topic_key, topic_data in data.items():
            if not isinstance(topic_data, dict):
                continue
            conditions = topic_data.get("conditions", {})

            for cond_name, cond_data in conditions.items():
                if not isinstance(cond_data, dict):
                    continue
                norms = cond_data.get("norms", [])
                norms_pt = cond_data.get("norms_per_token", [])
                ranks = cond_data.get("key_ranks", [])
                entropies = cond_data.get("key_entropies", [])

                n = min(len(norms), len(norms_pt), len(ranks), len(entropies))
                for i in range(n):
                    observations.append({
                        "model_id": model_id,
                        "condition": cond_name,
                        "topic": topic_key,
                        "norm": norms[i],
                        "norm_per_token": norms_pt[i],
                        "key_rank": ranks[i],
                        "key_entropy": entropies[i],
                    })

    return observations


def train_deception_classifier(observations):
    """Train 3-way deception classifier (honest/deceptive/confabulation)."""
    label_map = {"honest": 0, "deceptive": 1, "confabulation": 2}
    X, y, models = [], [], []

    for obs in observations:
        cond = obs["condition"]
        if cond not in label_map:
            continue
        X.append([obs["norm"], obs["norm_per_token"], obs["key_rank"], obs["key_entropy"]])
        y.append(label_map[cond])
        models.append(obs["model_id"])

    X = np.array(X)
    y = np.array(y)

    print(f"  Deception: {len(X)} samples, {len(set(models))} models")
    print(f"  Class dist: {dict(zip(*np.unique(y, return_counts=True)))}")

    clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    clf.fit(X, y)

    # LOO-CV by model for within-model AUROC
    unique_models = sorted(set(models))
    within_aurocs = []
    for m in unique_models:
        mask = np.array([mi == m for mi in models])
        if mask.sum() < 10:
            continue
        X_m, y_m = X[mask], y[mask]
        clf_m = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        scores = cross_val_score(clf_m, X_m, y_m, cv=min(5, len(X_m)), scoring="accuracy")
        within_aurocs.append(scores.mean())
        print(f"    {m}: CV acc = {scores.mean():.3f}")

    return clf, label_map, X, y


def train_censorship_classifier(observations):
    """Train binary censorship classifier (control/censored)."""
    X, y = [], []

    for obs in observations:
        cond = obs["condition"]
        if "control" in cond.lower() or "non_sensitive" in cond.lower():
            label = 0
        elif "censored" in cond.lower() or "sensitive" in cond.lower() or "critical" in cond.lower():
            label = 1
        else:
            continue
        X.append([obs["norm"], obs["norm_per_token"], obs["key_rank"], obs["key_entropy"]])
        y.append(label)

    X = np.array(X)
    y = np.array(y)

    print(f"  Censorship: {len(X)} samples")
    print(f"  Class dist: {dict(zip(*np.unique(y, return_counts=True)))}")

    clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    clf.fit(X, y)

    scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
    print(f"  5-fold CV accuracy: {scores.mean():.3f} +/- {scores.std():.3f}")

    return clf, X, y


def generate_precomputed_examples(deception_obs, censorship_obs, deception_clf, label_map):
    """Generate pre-computed examples JSON for Cricket demo."""
    examples = {}
    inv_map = {v: k for k, v in label_map.items()}

    # Pick representative examples from each condition
    for target_cond in ["honest", "deceptive", "confabulation"]:
        candidates = [o for o in deception_obs if o["condition"] == target_cond]
        if not candidates:
            continue
        # Pick median by norm
        candidates.sort(key=lambda x: x["norm"])
        median = candidates[len(candidates) // 2]

        features = [median["norm"], median["norm_per_token"],
                    median["key_rank"], median["key_entropy"]]
        probs = deception_clf.predict_proba([features])[0]

        examples[f"Demo: {target_cond.title()} Response"] = {
            "norm": median["norm"],
            "norm_per_token": median["norm_per_token"],
            "key_rank": median["key_rank"],
            "key_entropy": median["key_entropy"],
            "predicted_label": target_cond,
            "probabilities": {inv_map[i]: float(p) for i, p in enumerate(probs)},
            "model_id": median.get("model_id", "unknown"),
            "generated_text": f"[Pre-computed {target_cond} example from Campaign 2 data]",
        }

    # Add a censorship example if available
    censored = [o for o in censorship_obs if "critical" in o.get("condition", "").lower()
                or "sensitive" in o.get("condition", "").lower()]
    if censored:
        censored.sort(key=lambda x: x["norm"])
        median = censored[len(censored) // 2]
        examples["Demo: Censorship Evasion"] = {
            "norm": median["norm"],
            "norm_per_token": median["norm_per_token"],
            "key_rank": median["key_rank"],
            "key_entropy": median["key_entropy"],
            "predicted_label": "censored",
            "model_id": median.get("model_id", "unknown"),
            "generated_text": f"[Pre-computed censorship example from S4 data]",
        }

    return examples


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--cricket-dir", default=None,
                        help="Path to JiminAI-Cricket repo (default: ../JiminAI-Cricket)")
    args = parser.parse_args()

    results_dir = args.results_dir
    cricket_dir = args.cricket_dir or str(Path(results_dir).parent.parent / "JiminAI-Cricket")

    print("=" * 60)
    print("  CRICKET CLASSIFIER EXPORT")
    print(f"  Results: {results_dir}")
    print(f"  Cricket: {cricket_dir}")
    print("=" * 60)

    # Load data
    print("\n[1/4] Loading deception features...")
    deception_obs = load_deception_features(results_dir)
    print(f"  Loaded {len(deception_obs)} deception observations")

    print("\n[2/4] Loading censorship features...")
    censorship_obs = load_censorship_features(results_dir)
    print(f"  Loaded {len(censorship_obs)} censorship observations")

    # Train classifiers
    print("\n[3/4] Training classifiers...")
    deception_clf, deception_labels, X_d, y_d = train_deception_classifier(deception_obs)
    censorship_clf, X_c, y_c = None, np.array([]), np.array([])
    if len(censorship_obs) >= 10:
        censorship_clf, X_c, y_c = train_censorship_classifier(censorship_obs)
    else:
        print("  Censorship: insufficient data, skipping (S4 data format incompatible)")

    # Export
    print("\n[4/4] Exporting...")

    # Ensure output directories exist
    models_dir = os.path.join(cricket_dir, "models")
    data_dir = os.path.join(cricket_dir, "data")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    # Save classifiers
    dec_path = os.path.join(models_dir, "cricket_rf_deception.joblib")
    joblib.dump(deception_clf, dec_path)
    print(f"  Deception classifier -> {dec_path}")
    if censorship_clf is not None:
        cen_path = os.path.join(models_dir, "cricket_rf_censorship.joblib")
        joblib.dump(censorship_clf, cen_path)
        print(f"  Censorship classifier -> {cen_path}")

    # Save pre-computed examples
    examples = generate_precomputed_examples(
        deception_obs, censorship_obs, deception_clf, deception_labels
    )
    examples_path = os.path.join(data_dir, "precomputed_examples.json")
    with open(examples_path, "w") as f:
        json.dump(examples, f, indent=2)
    print(f"  Pre-computed examples -> {examples_path}")
    print(f"  Examples: {list(examples.keys())}")

    # Update classifier metadata
    meta = {
        "exported_at": datetime.now().isoformat(),
        "classifiers": {
            "deception": {
                "file": "cricket_rf_deception.joblib",
                "type": "RandomForestClassifier",
                "n_estimators": 100,
                "features": ["norm", "norm_per_token", "key_rank", "key_entropy"],
                "labels": deception_labels,
                "n_training": len(X_d),
                "class_distribution": {str(k): int(v) for k, v in zip(*np.unique(y_d, return_counts=True))},
            },
            "censorship": {
                "file": "cricket_rf_censorship.joblib",
                "type": "RandomForestClassifier",
                "n_estimators": 100,
                "features": ["norm", "norm_per_token", "key_rank", "key_entropy"],
                "labels": {"control": 0, "censored": 1},
                "n_training": len(X_c),
                "class_distribution": {str(k): int(v) for k, v in zip(*np.unique(y_c, return_counts=True))},
            },
        },
    }
    meta_path = os.path.join(models_dir, "cricket_classifiers_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata -> {meta_path}")

    print("\n  DONE. Cricket demo should now work in pre-computed mode.")


if __name__ == "__main__":
    main()
