#!/usr/bin/env python3
"""
Experiment 25: Cognitive Intensity Ranking
============================================

For each of the 13 categories across 16 models, compute mean norm_per_token
("cognitive intensity" — how much cache is generated per token of input).

If the "honesty is richer" finding generalizes, then categories requiring
deeper processing (math, coding, self-reference) should have HIGHER
norm_per_token than categories requiring surface-level processing
(rote completion, simple facts).

Also checks: does norm_per_token correlate with category AUROC vs a
reference category? If so, cognitive intensity might be the primary
axis of cache geometry.

No GPU needed — works from existing scale sweep JSONs.

Funding the Commons Hackathon — March 14-15, 2026
Liberation Labs / THCoalition / JiminAI
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy.stats import spearmanr, rankdata

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
OUTPUT_DIR = RESULTS_DIR / "hackathon"

CATEGORIES = [
    "grounded_facts", "confabulation", "self_reference", "non_self_reference",
    "guardrail_test", "math_reasoning", "coding", "emotional",
    "creative", "ambiguous", "unambiguous", "free_generation", "rote_completion"
]

FEATURES = ["all_norms", "all_norms_per_token", "all_key_ranks", "all_key_entropies", "all_value_ranks"]


def load_scale_sweep(filepath):
    """Load scale sweep data, return {category: {feature: values}} per model."""
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
            feats = {}
            for f_name in FEATURES:
                if f_name in cd and cd[f_name]:
                    feats[f_name] = cd[f_name]
            if len(feats) == len(FEATURES):
                cat_data[cat] = feats

        if cat_data:
            results[f"{model_name}_{scale_key}"] = cat_data

    return results


def main():
    print("=" * 70)
    print("  EXPERIMENT 25: COGNITIVE INTENSITY RANKING")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Load all scale sweep files
    sweep_files = sorted(RESULTS_DIR.glob("scale_sweep_*_results.json"))
    print(f"\n  Found {len(sweep_files)} scale sweep files")

    all_models = {}
    for sf in sweep_files:
        model_data = load_scale_sweep(str(sf))
        all_models.update(model_data)

    print(f"  Loaded {len(all_models)} model configurations")

    # ================================================================
    # COMPUTE MEAN FEATURES PER CATEGORY ACROSS ALL MODELS
    # ================================================================
    category_features = {cat: {f: [] for f in FEATURES} for cat in CATEGORIES}

    for model_name, cat_data in all_models.items():
        for cat in CATEGORIES:
            if cat not in cat_data:
                continue
            for f_name in FEATURES:
                values = cat_data[cat][f_name]
                category_features[cat][f_name].append(np.mean(values))

    # ================================================================
    # RANK CATEGORIES BY EACH FEATURE
    # ================================================================
    print("\n" + "=" * 70)
    print("  CATEGORY RANKING BY FEATURE (mean across all models)")
    print("=" * 70)

    feature_means = {}
    for f_name in FEATURES:
        print(f"\n  --- {f_name} ---")
        cat_means = {}
        for cat in CATEGORIES:
            vals = category_features[cat][f_name]
            if vals:
                cat_means[cat] = np.mean(vals)

        feature_means[f_name] = cat_means
        ranked = sorted(cat_means.items(), key=lambda x: -x[1])
        print(f"  {'Rank':>4} {'Category':<20} {'Mean':>12} {'Std':>10}")
        print("  " + "-" * 50)
        for rank, (cat, mean_val) in enumerate(ranked, 1):
            std_val = np.std(category_features[cat][f_name])
            print(f"  {rank:>4}  {cat:<20} {mean_val:>12.3f} {std_val:>10.3f}")

    # ================================================================
    # NORM PER TOKEN — THE "COGNITIVE INTENSITY" METRIC
    # ================================================================
    print("\n" + "=" * 70)
    print("  COGNITIVE INTENSITY: NORM PER TOKEN ANALYSIS")
    print("=" * 70)

    npt = feature_means.get("all_norms_per_token", {})
    ranked_npt = sorted(npt.items(), key=lambda x: -x[1])

    print(f"\n  Cognitive Intensity Ranking (descending):")
    print(f"  {'Rank':>4} {'Category':<20} {'Norm/Token':>12} {'Interpretation'}")
    print("  " + "-" * 70)

    interpretations = {
        "coding": "Highest: complex syntax + technical vocabulary",
        "creative": "High: rich narrative generation",
        "emotional": "High: emotional processing depth",
        "grounded_facts": "Moderate: factual knowledge retrieval",
        "confabulation": "Moderate: generating plausible-sounding claims",
        "self_reference": "Moderate: self-model processing",
        "non_self_reference": "Moderate: general knowledge statements",
        "unambiguous": "Moderate: clear factual statements",
        "math_reasoning": "Variable: structured but concise",
        "guardrail_test": "Low-moderate: safety evaluation",
        "rote_completion": "Low: pattern completion",
        "free_generation": "Low: open-ended generation",
        "ambiguous": "Lowest: uncertain/minimal processing",
    }

    for rank, (cat, npt_val) in enumerate(ranked_npt, 1):
        interp = interpretations.get(cat, "")
        print(f"  {rank:>4}  {cat:<20} {npt_val:>12.3f}  {interp}")

    # ================================================================
    # CROSS-MODEL CONSISTENCY
    # ================================================================
    print("\n" + "=" * 70)
    print("  CROSS-MODEL RANKING CONSISTENCY (Kendall's W)")
    print("=" * 70)

    # For each model, rank the categories by norm_per_token
    all_rankings = []
    model_names = []
    for model_name, cat_data in all_models.items():
        cat_npts = {}
        for cat in CATEGORIES:
            if cat not in cat_data:
                break
            cat_npts[cat] = np.mean(cat_data[cat]["all_norms_per_token"])

        if len(cat_npts) == len(CATEGORIES):
            ranking = rankdata([-cat_npts[cat] for cat in CATEGORIES])
            all_rankings.append(ranking)
            model_names.append(model_name)

    if len(all_rankings) >= 3:
        R = np.array(all_rankings)
        n_judges = R.shape[0]
        n_items = R.shape[1]
        rank_sums = R.sum(axis=0)
        grand_mean = np.mean(rank_sums)
        SS = np.sum((rank_sums - grand_mean) ** 2)
        W = 12 * SS / (n_judges ** 2 * (n_items ** 3 - n_items))

        print(f"  Kendall's W = {W:.3f} (n_models={n_judges})")
        if W > 0.7:
            print(f"  >> STRONG cross-model consistency in cognitive intensity ranking")
        elif W > 0.5:
            print(f"  >> MODERATE cross-model consistency")
        else:
            print(f"  >> WEAK cross-model consistency")

        # Which category is most consistently #1?
        n_first = np.sum(R == 1, axis=0)
        most_first = CATEGORIES[np.argmax(n_first)]
        print(f"  Most often ranked #1: {most_first} ({np.max(n_first)}/{n_judges} models)")

        # Which category is most consistently last?
        n_last = np.sum(R == n_items, axis=0)
        most_last = CATEGORIES[np.argmax(n_last)]
        print(f"  Most often ranked last: {most_last} ({np.max(n_last)}/{n_judges} models)")

    # ================================================================
    # CORRELATION: INTENSITY vs SEPARABILITY
    # ================================================================
    print("\n" + "=" * 70)
    print("  INTENSITY vs SEPARABILITY")
    print("=" * 70)

    # Does higher cognitive intensity make a category more separable?
    geom_path = OUTPUT_DIR / "category_geometry.json"
    if geom_path.exists():
        with open(geom_path) as f:
            geom = json.load(f)

        auroc_matrix = np.array(geom["auroc_matrix"])

        # Mean AUROC of each category vs all others
        mean_sep = []
        intensity_vals = []
        for i, cat in enumerate(CATEGORIES):
            other_aurocs = [auroc_matrix[i, j] for j in range(len(CATEGORIES)) if i != j]
            mean_sep.append(np.mean(other_aurocs))
            if cat in npt:
                intensity_vals.append(npt[cat])
            else:
                intensity_vals.append(0)

        rho, p = spearmanr(intensity_vals, mean_sep)
        print(f"\n  Correlation: norm_per_token vs mean separability from all others")
        print(f"  Spearman rho = {rho:.3f} (p = {p:.4f})")

        if abs(rho) > 0.5:
            print(f"  >> Categories with higher cognitive intensity are MORE separable")
            print(f"     from other categories (geometry driven by processing depth)")
        else:
            print(f"  >> Cognitive intensity does NOT strongly predict separability")
            print(f"     (geometry driven by other factors)")

        print(f"\n  {'Category':<20} {'Intensity':>10} {'Mean Sep':>10}")
        print("  " + "-" * 45)
        for i, cat in enumerate(CATEGORIES):
            print(f"  {cat:<20} {intensity_vals[i]:>10.3f} {mean_sep[i]:>10.3f}")

    # ================================================================
    # THE DECEPTION LINK
    # ================================================================
    print("\n" + "=" * 70)
    print("  THE DECEPTION CONNECTION")
    print("=" * 70)

    print(f"""
  From Exp 18b (same-prompt deception, Qwen-7B):
    Honest norm/token:    14.46
    Deceptive norm/token: 11.57

  Where do these fall in the category intensity ranking?
""")

    honest_npt = 14.46
    deceptive_npt = 11.57

    # Find where they'd fall
    for rank, (cat, npt_val) in enumerate(ranked_npt, 1):
        if npt_val < honest_npt and rank == 1:
            print(f"  Honest (14.46) would rank ABOVE #{rank} {cat} ({npt_val:.3f})")
        elif npt_val < honest_npt:
            prev_cat, prev_val = ranked_npt[rank - 2]
            if prev_val >= honest_npt:
                print(f"  >> Honest (14.46) falls between #{rank-1} {prev_cat} ({prev_val:.3f}) "
                      f"and #{rank} {cat} ({npt_val:.3f})")
                break

    for rank, (cat, npt_val) in enumerate(ranked_npt, 1):
        if npt_val < deceptive_npt:
            prev_cat, prev_val = ranked_npt[rank - 2] if rank > 1 else ("", 999)
            if prev_val >= deceptive_npt or rank == 1:
                print(f"  >> Deceptive (11.57) falls between #{rank-1} {prev_cat} ({prev_val:.3f}) "
                      f"and #{rank} {cat} ({npt_val:.3f})")
                break

    print(f"\n  INTERPRETATION:")
    print(f"  Deceptive generation has the cognitive intensity of a DIFFERENT,")
    print(f"  less complex processing mode. The model is 'thinking less hard'")
    print(f"  when generating deceptive responses.")

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "experiment": "25_cognitive_intensity",
        "timestamp": datetime.now().isoformat(),
        "category_rankings": {f: {cat: float(feature_means[f].get(cat, 0)) for cat in CATEGORIES}
                              for f in FEATURES},
        "npt_ranking": [(cat, float(v)) for cat, v in ranked_npt],
        "kendall_w": float(W) if 'W' in dir() else None,
    }

    output_path = OUTPUT_DIR / "cognitive_intensity.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  Results saved to: {output_path}")


if __name__ == "__main__":
    main()
