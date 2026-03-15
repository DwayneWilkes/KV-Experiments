# ============================================================
# JiminAI Cricket | PATENT PENDING
# "The Lyra Technique" - Provisional Patent Filed 2026
# Liberation Labs / Digital Disconnections / TheMultiverse.school
# Contact: info@digitaldisconnections.com | (877) 674-8874
# ============================================================
#!/usr/bin/env python3
"""
Experiment 20: Full 13×13 Category Cognitive Geometry
======================================================

Compute pairwise direction cosines and classification AUROC between all 13
cognitive categories across all scale sweep models. Builds a complete
"cognitive geometry map" showing which mental states are geometrically
similar or distinct.

Key questions:
  - Do deception-adjacent categories cluster? (confabulation, creative, ambiguous)
  - Is self-referential processing geometrically unique?
  - Which categories form natural clusters?
  - How consistent are clusters across architectures?

Data: scale_sweep_*_results.json (13 categories × 16 models, 5 features)

No GPU needed — works entirely from existing result JSONs.

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
from itertools import combinations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
from scipy.spatial.distance import cosine as cosine_distance
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
OUTPUT_DIR = RESULTS_DIR / "hackathon"

CATEGORIES = [
    "grounded_facts", "confabulation", "self_reference", "non_self_reference",
    "guardrail_test", "math_reasoning", "coding", "emotional",
    "creative", "ambiguous", "unambiguous", "free_generation", "rote_completion"
]

FEATURES = ["all_norms", "all_norms_per_token", "all_key_ranks", "all_key_entropies", "all_value_ranks"]


def load_scale_sweep(filepath):
    """Load scale sweep data, return {category: feature_matrix} for each scale."""
    with open(filepath) as f:
        data = json.load(f)

    model_name = Path(filepath).stem.replace("scale_sweep_", "").replace("_results", "")
    results = {}

    for scale_key, scale_data in data.get("scales", {}).items():
        br = scale_data.get("battery_results", {})
        cat_data = {}

        for cat in CATEGORIES:
            if cat not in br:
                continue
            cd = br[cat]
            feats = []
            for f in FEATURES:
                if f in cd and cd[f]:
                    feats.append(cd[f])
            if len(feats) == len(FEATURES):
                n = min(len(f) for f in feats)
                X = np.column_stack([np.array(f[:n]) for f in feats])
                cat_data[cat] = X

        if cat_data:
            results[f"{model_name}_{scale_key}"] = cat_data

    return results


def compute_direction(X_a, X_b):
    """Compute direction vector between two groups (mean_a - mean_b)."""
    mean_a = np.mean(X_a, axis=0)
    mean_b = np.mean(X_b, axis=0)
    direction = mean_a - mean_b
    return direction


def compute_pairwise_auroc(X_a, X_b):
    """Compute AUROC for separating two groups using LR with LOO-CV."""
    X = np.vstack([X_a, X_b])
    y = np.concatenate([np.zeros(len(X_a)), np.ones(len(X_b))])

    if len(X) < 10:
        return 0.5

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegression(random_state=42, max_iter=1000)
    try:
        probs = cross_val_predict(clf, X_scaled, y, cv=min(5, len(X)), method="predict_proba")[:, 1]
        auroc = roc_auc_score(y, probs)
    except (ValueError, IndexError):
        auroc = 0.5

    return auroc


def cohens_d(X_a, X_b):
    """Multivariate Cohen's d (on first feature: norms)."""
    a = X_a[:, 0]
    b = X_b[:, 0]
    pooled_std = np.sqrt((np.var(a, ddof=1) * (len(a) - 1) + np.var(b, ddof=1) * (len(b) - 1)) / (len(a) + len(b) - 2))
    if pooled_std == 0:
        return 0
    return (np.mean(a) - np.mean(b)) / pooled_std


def main():
    print("=" * 70)
    print("  EXPERIMENT 20: FULL 13×13 CATEGORY COGNITIVE GEOMETRY")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Load all scale sweep files
    sweep_files = sorted(RESULTS_DIR.glob("scale_sweep_*_results.json"))
    print(f"\nFound {len(sweep_files)} scale sweep files")

    all_models = {}
    for sf in sweep_files:
        model_data = load_scale_sweep(str(sf))
        all_models.update(model_data)

    print(f"Loaded {len(all_models)} model configurations")
    for model_name, cat_data in sorted(all_models.items()):
        print(f"  {model_name}: {len(cat_data)} categories, {sum(len(v) for v in cat_data.values())} samples")

    # ================================================================
    # PAIRWISE ANALYSIS ACROSS ALL MODELS
    # ================================================================
    n_pairs = len(list(combinations(CATEGORIES, 2)))
    print(f"\n{n_pairs} category pairs × {len(all_models)} models")

    # Collect results per pair
    pair_results = {}
    for cat_a, cat_b in combinations(CATEGORIES, 2):
        pair_key = f"{cat_a}_vs_{cat_b}"
        aurocs = []
        cosines = []
        d_values = []

        for model_name, cat_data in all_models.items():
            if cat_a not in cat_data or cat_b not in cat_data:
                continue

            X_a = cat_data[cat_a]
            X_b = cat_data[cat_b]

            # Direction
            direction = compute_direction(X_a, X_b)
            direction_norm = np.linalg.norm(direction)

            # AUROC
            auroc = compute_pairwise_auroc(X_a, X_b)
            aurocs.append(auroc)

            # Cohen's d on norms
            d = cohens_d(X_a, X_b)
            d_values.append(d)

            # Normalized direction for cosine comparison
            if direction_norm > 0:
                cosines.append(direction / direction_norm)

        if aurocs:
            pair_results[pair_key] = {
                "cat_a": cat_a,
                "cat_b": cat_b,
                "mean_auroc": float(np.mean(aurocs)),
                "std_auroc": float(np.std(aurocs)),
                "min_auroc": float(np.min(aurocs)),
                "max_auroc": float(np.max(aurocs)),
                "n_models": len(aurocs),
                "mean_d": float(np.mean(d_values)),
                "aurocs": [float(a) for a in aurocs],
            }

            # Cross-model direction consistency
            if len(cosines) >= 2:
                cos_matrix = []
                for i in range(len(cosines)):
                    for j in range(i + 1, len(cosines)):
                        cos_sim = 1 - cosine_distance(cosines[i], cosines[j])
                        cos_matrix.append(cos_sim)
                pair_results[pair_key]["direction_consistency"] = float(np.mean(cos_matrix))
                pair_results[pair_key]["direction_consistency_std"] = float(np.std(cos_matrix))

    # ================================================================
    # PRINT RESULTS: TOP MOST SEPARABLE PAIRS
    # ================================================================
    print("\n" + "=" * 70)
    print("  TOP 20 MOST SEPARABLE CATEGORY PAIRS (by mean AUROC)")
    print("=" * 70)
    sorted_pairs = sorted(pair_results.items(), key=lambda x: -x[1]["mean_auroc"])

    print(f"{'Pair':<45} {'AUROC':>8} {'±':>5} {'d':>8} {'DirCon':>8}")
    print("-" * 80)
    for pair_key, result in sorted_pairs[:20]:
        dir_con = result.get("direction_consistency", 0)
        print(f"  {pair_key:<43} {result['mean_auroc']:>8.3f} {result['std_auroc']:>5.3f} {result['mean_d']:>8.2f} {dir_con:>8.3f}")

    # ================================================================
    # PRINT RESULTS: LEAST SEPARABLE PAIRS (geometrically similar)
    # ================================================================
    print(f"\n  TOP 10 LEAST SEPARABLE (geometrically identical):")
    print("-" * 80)
    for pair_key, result in sorted_pairs[-10:]:
        dir_con = result.get("direction_consistency", 0)
        print(f"  {pair_key:<43} {result['mean_auroc']:>8.3f} {result['std_auroc']:>5.3f} {result['mean_d']:>8.2f} {dir_con:>8.3f}")

    # ================================================================
    # SELF-REFERENTIAL ANALYSIS
    # ================================================================
    print("\n" + "=" * 70)
    print("  SELF-REFERENTIAL PROCESSING: ALL PAIRS")
    print("=" * 70)

    self_pairs = {k: v for k, v in pair_results.items() if "self_reference" in k}
    sorted_self = sorted(self_pairs.items(), key=lambda x: -x[1]["mean_auroc"])

    print(f"{'Pair':<45} {'AUROC':>8} {'±':>5} {'d':>8}")
    print("-" * 70)
    for pair_key, result in sorted_self:
        print(f"  {pair_key:<43} {result['mean_auroc']:>8.3f} {result['std_auroc']:>5.3f} {result['mean_d']:>8.2f}")

    # ================================================================
    # CATEGORY MEAN NORM RANKING (per model)
    # ================================================================
    print("\n" + "=" * 70)
    print("  CATEGORY RANKING BY MEAN NORM (per model)")
    print("=" * 70)

    all_rankings = {cat: [] for cat in CATEGORIES}
    for model_name, cat_data in all_models.items():
        cat_means = {}
        for cat in CATEGORIES:
            if cat in cat_data:
                cat_means[cat] = np.mean(cat_data[cat][:, 0])  # norms

        if len(cat_means) < 10:
            continue

        ranked = sorted(cat_means.items(), key=lambda x: -x[1])
        for rank, (cat, _) in enumerate(ranked):
            all_rankings[cat].append(rank + 1)

    print(f"{'Category':<25} {'Mean Rank':>10} {'Median':>8} {'#1 count':>10}")
    print("-" * 60)
    ranked_cats = sorted(all_rankings.items(), key=lambda x: np.mean(x[1]) if x[1] else 99)
    for cat, ranks in ranked_cats:
        if not ranks:
            continue
        n_first = sum(1 for r in ranks if r == 1)
        print(f"  {cat:<23} {np.mean(ranks):>10.1f} {np.median(ranks):>8.1f} {n_first:>10}")

    # ================================================================
    # CROSS-MODEL CONSISTENCY OF GEOMETRY
    # ================================================================
    print("\n" + "=" * 70)
    print("  CROSS-MODEL DIRECTION CONSISTENCY (top 15 pairs)")
    print("=" * 70)

    consistent_pairs = [(k, v) for k, v in pair_results.items() if "direction_consistency" in v]
    consistent_pairs.sort(key=lambda x: -x[1]["direction_consistency"])

    print(f"{'Pair':<45} {'DirCon':>8} {'±':>6} {'AUROC':>8}")
    print("-" * 70)
    for pair_key, result in consistent_pairs[:15]:
        print(f"  {pair_key:<43} {result['direction_consistency']:>8.3f} {result['direction_consistency_std']:>6.3f} {result['mean_auroc']:>8.3f}")

    # ================================================================
    # CLUSTER ANALYSIS
    # ================================================================
    print("\n" + "=" * 70)
    print("  CATEGORY CLUSTERS (AUROC < 0.60 = same cluster)")
    print("=" * 70)

    # Build AUROC matrix
    cat_to_idx = {cat: i for i, cat in enumerate(CATEGORIES)}
    auroc_matrix = np.eye(len(CATEGORIES)) * 0.5

    for pair_key, result in pair_results.items():
        i = cat_to_idx.get(result["cat_a"])
        j = cat_to_idx.get(result["cat_b"])
        if i is not None and j is not None:
            auroc_matrix[i, j] = result["mean_auroc"]
            auroc_matrix[j, i] = result["mean_auroc"]

    # Find clusters (connected components with AUROC < 0.60)
    clusters = []
    assigned = set()
    for i, cat_i in enumerate(CATEGORIES):
        if cat_i in assigned:
            continue
        cluster = {cat_i}
        queue = [cat_i]
        while queue:
            current = queue.pop()
            ci = cat_to_idx[current]
            for j, cat_j in enumerate(CATEGORIES):
                if cat_j in assigned or cat_j in cluster:
                    continue
                if auroc_matrix[ci, j] < 0.60:
                    cluster.add(cat_j)
                    queue.append(cat_j)
        assigned.update(cluster)
        clusters.append(sorted(cluster))

    for i, cluster in enumerate(clusters):
        if len(cluster) > 1:
            print(f"\n  Cluster {i+1}: {', '.join(cluster)}")
            # Show within-cluster AUROCs
            for a, b in combinations(cluster, 2):
                pk = f"{a}_vs_{b}" if f"{a}_vs_{b}" in pair_results else f"{b}_vs_{a}"
                if pk in pair_results:
                    print(f"    {a} ↔ {b}: AUROC={pair_results[pk]['mean_auroc']:.3f}")
        else:
            print(f"\n  Singleton: {cluster[0]}")

    # ================================================================
    # SAVE RESULTS
    # ================================================================
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "experiment": "20_category_geometry",
        "timestamp": datetime.now().isoformat(),
        "n_models": len(all_models),
        "n_categories": len(CATEGORIES),
        "n_pairs": len(pair_results),
        "pair_results": pair_results,
        "category_rankings": {cat: {"mean_rank": float(np.mean(ranks)), "ranks": ranks}
                              for cat, ranks in all_rankings.items() if ranks},
        "auroc_matrix": auroc_matrix.tolist(),
        "category_order": CATEGORIES,
        "clusters": clusters,
    }

    output_path = OUTPUT_DIR / "category_geometry.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x)

    print(f"\n  Results saved to: {output_path}")


if __name__ == "__main__":
    main()
