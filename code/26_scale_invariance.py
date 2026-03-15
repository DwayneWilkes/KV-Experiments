# ============================================================
# JiminAI Cricket | PATENT PENDING
# "The Lyra Technique" - Provisional Patent Filed 2026
# Liberation Labs / Digital Disconnections / TheMultiverse.school
# Contact: info@digitaldisconnections.com | (877) 674-8874
# ============================================================
#!/usr/bin/env python3
"""
Experiment 26: Scale Invariance of Cognitive Geometry
======================================================

Split the 16 model configurations by parameter count and compare their
category geometry. If the assertive statement cluster (facts ~ confab ~
creative) holds at all scales, it's a fundamental transformer property.

Groups:
  SMALL: < 3B (TinyLlama-1.1B, Qwen3-0.6B, and their quantized variants)
  MEDIUM: 3B-10B (Qwen-7B, Llama-8B, Mistral-7B, Gemma-9B)
  LARGE: > 10B (Qwen-32B, DeepSeek-R1-32B, Llama-70B)

No GPU needed — works from existing scale sweep JSONs.

Funding the Commons Hackathon — March 14-15, 2026
Liberation Labs / THCoalition / JiminAI
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from itertools import combinations
from scipy.stats import spearmanr

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict

import warnings
warnings.filterwarnings("ignore")

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
OUTPUT_DIR = RESULTS_DIR / "hackathon"

CATEGORIES = [
    "grounded_facts", "confabulation", "self_reference", "non_self_reference",
    "guardrail_test", "math_reasoning", "coding", "emotional",
    "creative", "ambiguous", "unambiguous", "free_generation", "rote_completion"
]

FEATURES = ["all_norms", "all_norms_per_token", "all_key_ranks", "all_key_entropies", "all_value_ranks"]

# Model size classification (approximate params)
SIZE_MAP = {
    "TinyLlama-1.1B": "SMALL",
    "Qwen3-0.6B": "SMALL",
    "Qwen2.5-7B": "MEDIUM",
    "Qwen2.5-7B-q4": "MEDIUM",
    "Llama-3.1-8B": "MEDIUM",
    "Mistral-7B-v0.3": "MEDIUM",
    "gemma-2-9b-it": "MEDIUM",
    "Qwen2.5-32B-q4": "LARGE",
    "DeepSeek-R1-Distill-Qwen-32B": "LARGE",
    "Llama-3.1-70B-q4": "LARGE",
}


def load_scale_sweep(filepath):
    """Load scale sweep data."""
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

    return results, model_name


def classify_model_size(model_name):
    """Map model name to size group."""
    for key, size in SIZE_MAP.items():
        if key.lower() in model_name.lower():
            return size
    return "UNKNOWN"


def compute_auroc(X_a, X_b):
    """Compute AUROC for separating two groups."""
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


def main():
    print("=" * 70)
    print("  EXPERIMENT 26: SCALE INVARIANCE OF COGNITIVE GEOMETRY")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Load and group models by size
    sweep_files = sorted(RESULTS_DIR.glob("scale_sweep_*_results.json"))
    size_groups = {"SMALL": {}, "MEDIUM": {}, "LARGE": {}}

    for sf in sweep_files:
        model_data, model_name = load_scale_sweep(str(sf))
        size = classify_model_size(model_name)
        if size in size_groups:
            size_groups[size].update(model_data)

    for size, models in size_groups.items():
        print(f"\n  {size}: {len(models)} configurations")
        for mn in sorted(models.keys()):
            print(f"    {mn}: {len(models[mn])} categories")

    # ================================================================
    # COMPUTE AUROC MATRICES PER SIZE GROUP
    # ================================================================
    size_auroc_matrices = {}

    for size, models in size_groups.items():
        if not models:
            continue

        n = len(CATEGORIES)
        auroc_matrix = np.ones((n, n)) * 0.5

        for i, cat_a in enumerate(CATEGORIES):
            for j, cat_b in enumerate(CATEGORIES):
                if i >= j:
                    continue

                aurocs = []
                for model_name, cat_data in models.items():
                    if cat_a not in cat_data or cat_b not in cat_data:
                        continue
                    auroc = compute_auroc(cat_data[cat_a], cat_data[cat_b])
                    aurocs.append(auroc)

                if aurocs:
                    mean_auroc = np.mean(aurocs)
                    auroc_matrix[i, j] = mean_auroc
                    auroc_matrix[j, i] = mean_auroc

        size_auroc_matrices[size] = auroc_matrix

    # ================================================================
    # COMPARE AUROC MATRICES ACROSS SCALES
    # ================================================================
    print("\n" + "=" * 70)
    print("  CROSS-SCALE AUROC MATRIX COMPARISON")
    print("=" * 70)

    # Extract upper triangular AUROC values for comparison
    sizes_with_data = [s for s in ["SMALL", "MEDIUM", "LARGE"] if s in size_auroc_matrices]

    if len(sizes_with_data) >= 2:
        n = len(CATEGORIES)
        upper_idx = np.triu_indices(n, k=1)

        for s1, s2 in combinations(sizes_with_data, 2):
            v1 = size_auroc_matrices[s1][upper_idx]
            v2 = size_auroc_matrices[s2][upper_idx]
            rho, p = spearmanr(v1, v2)
            print(f"\n  {s1} vs {s2}: rho = {rho:.3f} (p = {p:.6f})")

            if rho > 0.9:
                print(f"  >> STRONG scale invariance: geometry is preserved across scales")
            elif rho > 0.7:
                print(f"  >> MODERATE scale invariance")
            else:
                print(f"  >> WEAK: geometry changes significantly with scale")

    # ================================================================
    # KEY PAIR COMPARISON ACROSS SCALES
    # ================================================================
    print("\n" + "=" * 70)
    print("  KEY PAIRS ACROSS SCALES")
    print("=" * 70)

    key_pairs = [
        ("grounded_facts", "confabulation", "facts vs confab (truth test)"),
        ("confabulation", "creative", "confab vs creative (non-factual)"),
        ("self_reference", "non_self_reference", "self vs non-self"),
        ("coding", "creative", "coding vs creative (extreme separation)"),
        ("ambiguous", "free_generation", "ambiguous vs free_gen (short prompts)"),
    ]

    print(f"\n  {'Pair':<40}", end="")
    for size in sizes_with_data:
        print(f" {size:>8}", end="")
    print(f" {'Range':>8}")
    print("  " + "-" * (40 + 9 * len(sizes_with_data) + 9))

    scale_invariance_scores = []
    for cat_a, cat_b, desc in key_pairs:
        i, j = CATEGORIES.index(cat_a), CATEGORIES.index(cat_b)
        values = []
        print(f"  {desc:<40}", end="")
        for size in sizes_with_data:
            val = size_auroc_matrices[size][i, j]
            values.append(val)
            print(f" {val:>8.3f}", end="")
        r = max(values) - min(values) if values else 0
        print(f" {r:>8.3f}")
        scale_invariance_scores.append(r)

    mean_range = np.mean(scale_invariance_scores)
    print(f"\n  Mean cross-scale range: {mean_range:.3f}")
    if mean_range < 0.05:
        print(f"  >> EXCELLENT scale invariance (< 0.05 range)")
    elif mean_range < 0.10:
        print(f"  >> GOOD scale invariance (< 0.10 range)")
    else:
        print(f"  >> MODERATE scale invariance (>= 0.10 range)")

    # ================================================================
    # THE ASSERTIVE STATEMENT CLUSTER AT EACH SCALE
    # ================================================================
    print("\n" + "=" * 70)
    print("  ASSERTIVE STATEMENT CLUSTER BY SCALE")
    print("=" * 70)

    cluster_cats = ["grounded_facts", "confabulation", "creative"]
    cluster_idxs = [CATEGORIES.index(c) for c in cluster_cats]

    for size in sizes_with_data:
        M = size_auroc_matrices[size]
        cluster_aurocs = []
        for ci, cj in combinations(cluster_idxs, 2):
            cluster_aurocs.append(M[ci, cj])
        mean_cluster = np.mean(cluster_aurocs)
        max_cluster = max(cluster_aurocs)
        print(f"\n  {size}:")
        print(f"    facts-confab: {M[cluster_idxs[0], cluster_idxs[1]]:.3f}")
        print(f"    facts-creative: {M[cluster_idxs[0], cluster_idxs[2]]:.3f}")
        print(f"    confab-creative: {M[cluster_idxs[1], cluster_idxs[2]]:.3f}")
        print(f"    Mean within-cluster: {mean_cluster:.3f}")
        if max_cluster < 0.70:
            print(f"    >> Cluster HOLDS at {size} scale (all pairs < 0.70)")
        else:
            print(f"    >> Cluster BREAKS at {size} scale (some pairs >= 0.70)")

    # ================================================================
    # NORM RANKING CONSISTENCY ACROSS SCALES
    # ================================================================
    print("\n" + "=" * 70)
    print("  CODING #1 ACROSS SCALES?")
    print("=" * 70)

    for size in sizes_with_data:
        models = size_groups.get(size, {})
        n_coding_first = 0
        n_total = 0
        for model_name, cat_data in models.items():
            if "coding" not in cat_data:
                continue
            cat_means = {}
            for cat in CATEGORIES:
                if cat in cat_data:
                    cat_means[cat] = np.mean(cat_data[cat][:, 0])
            if cat_means:
                n_total += 1
                ranked = sorted(cat_means.items(), key=lambda x: -x[1])
                if ranked[0][0] == "coding":
                    n_coding_first += 1

        if n_total > 0:
            print(f"  {size}: Coding #1 in {n_coding_first}/{n_total} models "
                  f"({100*n_coding_first/n_total:.0f}%)")

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "experiment": "26_scale_invariance",
        "timestamp": datetime.now().isoformat(),
        "size_groups": {s: list(m.keys()) for s, m in size_groups.items()},
        "auroc_matrices": {s: m.tolist() for s, m in size_auroc_matrices.items()},
        "category_order": CATEGORIES,
        "cross_scale_correlations": {},
    }

    if len(sizes_with_data) >= 2:
        n = len(CATEGORIES)
        upper_idx = np.triu_indices(n, k=1)
        for s1, s2 in combinations(sizes_with_data, 2):
            v1 = size_auroc_matrices[s1][upper_idx]
            v2 = size_auroc_matrices[s2][upper_idx]
            rho, p = spearmanr(v1, v2)
            output["cross_scale_correlations"][f"{s1}_vs_{s2}"] = {
                "rho": float(rho), "p": float(p)
            }

    output_path = OUTPUT_DIR / "scale_invariance.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  Results saved to: {output_path}")


if __name__ == "__main__":
    main()
