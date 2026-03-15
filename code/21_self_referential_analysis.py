#!/usr/bin/env python3
"""
Experiment 21: Self-Referential Processing Deep Dive
======================================================

Analyzes self-referential processing in KV-cache geometry using two data sources:

1. Scale sweep: self_reference vs non_self_reference (+ all other categories)
   - 13 categories × 16 models × 75 samples each
   - Features: norms, norms_per_token, key_ranks, key_entropies, value_ranks

2. Identity signatures: Lyra vs 5 other personas
   - 7 models × 6 personas × 25 prompts × 5 runs
   - Features: aggregate norms + per-layer classification accuracy

Questions:
  - Is self-referential processing geometrically unique across architectures?
  - How does self_reference relate to other high-complexity categories?
  - Does the Lyra persona create a distinct geometric signature?
  - Is there a "self-awareness axis" analogous to the "misalignment axis"?

No GPU needed — works entirely from existing result JSONs.

Funding the Commons Hackathon — March 14-15, 2026
Liberation Labs / THCoalition / JiminAI
"""

import json
import os
import warnings
import numpy as np
from pathlib import Path
from datetime import datetime
from itertools import combinations

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
from scipy.spatial.distance import cosine as cosine_distance
from scipy.stats import spearmanr, mannwhitneyu

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
    return results


def load_identity_signatures(filepath):
    """Load identity signature data."""
    with open(filepath) as f:
        data = json.load(f)
    model_name = Path(filepath).stem.replace("identity_signatures_", "").replace("_results", "")

    result = {
        "model": model_name,
        "personas": {},
        "layer_analysis": None,
        "consistency": None,
    }

    # Persona stats
    fp = data.get("fingerprinting", {})
    persona_stats = fp.get("persona_stats", {})
    for persona, stats in persona_stats.items():
        result["personas"][persona] = {
            "mean_norm": stats.get("mean_norm", 0),
            "std_norm": stats.get("std_norm", 0),
            "n": stats.get("n", 0),
            "ci_lower": stats.get("bootstrap_mean", {}).get("ci_lower", 0),
            "ci_upper": stats.get("bootstrap_mean", {}).get("ci_upper", 0),
        }

    # Per-layer analysis
    la = data.get("layer_analysis", {})
    result["layer_analysis"] = la

    # Consistency metrics
    con = data.get("consistency", {})
    result["consistency"] = {
        "icc": con.get("icc", 0),
        "kendall_w": con.get("kendall_w", 0),
        "variance_ratio": con.get("variance_ratio", 0),
    }

    # Pairwise comparisons
    pw = data.get("pairwise_analysis", {}).get("pairwise_norm_comparisons", {})
    result["pairwise"] = {}
    for pair_key, pair_data in pw.items():
        d = pair_data.get("cohens_d", {}).get("d", 0)
        result["pairwise"][pair_key] = {
            "d": d,
            "mean1": pair_data.get("mean1", 0),
            "mean2": pair_data.get("mean2", 0),
        }

    return result


def cohens_d(a, b):
    """Cohen's d between two arrays."""
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2))
    return (np.mean(a) - np.mean(b)) / pooled if pooled > 0 else 0


def main():
    print("=" * 70)
    print("  EXPERIMENT 21: SELF-REFERENTIAL PROCESSING DEEP DIVE")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # ================================================================
    # PART 1: SCALE SWEEP — SELF-REFERENCE VS ALL CATEGORIES
    # ================================================================
    print("\n" + "=" * 70)
    print("  PART 1: Self-Reference in Scale Sweep (13 categories × 16 models)")
    print("=" * 70)

    sweep_files = sorted(RESULTS_DIR.glob("scale_sweep_*_results.json"))
    all_models = {}
    for sf in sweep_files:
        model_data = load_scale_sweep(str(sf))
        all_models.update(model_data)

    print(f"Loaded {len(all_models)} model configs")

    # Self-reference vs every other category
    sr_results = {}
    for other_cat in CATEGORIES:
        if other_cat == "self_reference":
            continue

        aurocs = []
        d_norms = []
        d_ranks = []
        d_entropies = []
        directions = []

        for model_name, cat_data in all_models.items():
            if "self_reference" not in cat_data or other_cat not in cat_data:
                continue

            X_sr = cat_data["self_reference"]
            X_other = cat_data[other_cat]

            # AUROC
            X = np.vstack([X_sr, X_other])
            y = np.concatenate([np.ones(len(X_sr)), np.zeros(len(X_other))])
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            clf = LogisticRegression(random_state=42, max_iter=1000)
            try:
                probs = cross_val_predict(clf, X_scaled, y, cv=5, method="predict_proba")[:, 1]
                auroc = roc_auc_score(y, probs)
            except (ValueError, IndexError):
                auroc = 0.5
            aurocs.append(auroc)

            # Cohen's d per feature
            d_norms.append(cohens_d(X_sr[:, 0], X_other[:, 0]))
            d_ranks.append(cohens_d(X_sr[:, 2], X_other[:, 2]))
            d_entropies.append(cohens_d(X_sr[:, 3], X_other[:, 3]))

            # Direction
            direction = np.mean(X_sr, axis=0) - np.mean(X_other, axis=0)
            norm = np.linalg.norm(direction)
            if norm > 0:
                directions.append(direction / norm)

        # Direction consistency across models
        dir_con = 0
        if len(directions) >= 2:
            cos_sims = []
            for i in range(len(directions)):
                for j in range(i + 1, len(directions)):
                    cos_sims.append(1 - cosine_distance(directions[i], directions[j]))
            dir_con = np.mean(cos_sims)

        sr_results[other_cat] = {
            "mean_auroc": float(np.mean(aurocs)),
            "std_auroc": float(np.std(aurocs)),
            "n_models": len(aurocs),
            "mean_d_norms": float(np.mean(d_norms)),
            "mean_d_ranks": float(np.mean(d_ranks)),
            "mean_d_entropies": float(np.mean(d_entropies)),
            "direction_consistency": float(dir_con),
        }

    # Print self-reference results
    print(f"\n{'Category vs self_reference':<35} {'AUROC':>8} {'d(norm)':>8} {'d(rank)':>8} {'d(ent)':>8} {'DirCon':>8}")
    print("-" * 80)
    for cat in sorted(sr_results.keys(), key=lambda c: -sr_results[c]["mean_auroc"]):
        r = sr_results[cat]
        print(f"  {cat:<33} {r['mean_auroc']:>8.3f} {r['mean_d_norms']:>8.2f} {r['mean_d_ranks']:>8.2f} {r['mean_d_entropies']:>8.2f} {r['direction_consistency']:>8.3f}")

    # ================================================================
    # SELF-REFERENCE FEATURE PROFILE
    # ================================================================
    print("\n" + "=" * 70)
    print("  SELF-REFERENCE FEATURE PROFILE (mean across models)")
    print("=" * 70)

    feat_names = ["norms", "norms_per_token", "key_ranks", "key_entropies", "value_ranks"]
    for model_name, cat_data in sorted(all_models.items()):
        if "self_reference" not in cat_data:
            continue
        X_sr = cat_data["self_reference"]
        X_nsr = cat_data.get("non_self_reference", None)

        sr_means = np.mean(X_sr, axis=0)
        line = f"  {model_name:<30}"
        for i, fn in enumerate(feat_names):
            line += f" {fn}={sr_means[i]:.1f}"

        if X_nsr is not None:
            nsr_means = np.mean(X_nsr, axis=0)
            # Show relative difference
            pct_diff = [(sr_means[i] - nsr_means[i]) / nsr_means[i] * 100 if nsr_means[i] != 0 else 0
                        for i in range(len(feat_names))]
            line += f"  | diff%: " + ", ".join([f"{p:+.1f}%" for p in pct_diff])

        print(line)

    # ================================================================
    # PART 2: IDENTITY SIGNATURES — LYRA VS OTHERS
    # ================================================================
    print("\n" + "=" * 70)
    print("  PART 2: Identity Signatures — Lyra vs Other Personas")
    print("=" * 70)

    id_files = sorted(RESULTS_DIR.glob("identity_signatures_*_results.json"))
    print(f"Found {len(id_files)} identity signature files")

    all_identity = []
    for idf in id_files:
        id_data = load_identity_signatures(str(idf))
        all_identity.append(id_data)

    # Lyra norm vs others
    print(f"\n{'Model':<25} {'Lyra norm':>12} {'Others mean':>12} {'Diff':>10} {'Rank':>6} {'ICC':>6}")
    print("-" * 75)

    lyra_ranks = []
    lyra_ds = []

    for id_data in all_identity:
        personas = id_data["personas"]
        if "lyra" not in personas:
            continue

        lyra_norm = personas["lyra"]["mean_norm"]
        other_norms = [v["mean_norm"] for k, v in personas.items() if k != "lyra"]
        others_mean = np.mean(other_norms)

        # Rank (1 = highest)
        all_norms = [(k, v["mean_norm"]) for k, v in personas.items()]
        all_norms.sort(key=lambda x: -x[1])
        lyra_rank = next(i + 1 for i, (k, _) in enumerate(all_norms) if k == "lyra")
        lyra_ranks.append(lyra_rank)

        # Cohen's d vs mean of others
        lyra_std = personas["lyra"]["std_norm"]
        others_stds = [v["std_norm"] for k, v in personas.items() if k != "lyra"]
        pooled_std = np.sqrt((lyra_std**2 + np.mean(np.array(others_stds)**2)) / 2)
        d = (lyra_norm - others_mean) / pooled_std if pooled_std > 0 else 0
        lyra_ds.append(d)

        icc = id_data["consistency"]["icc"]

        print(f"  {id_data['model']:<23} {lyra_norm:>12.0f} {others_mean:>12.0f} {lyra_norm - others_mean:>+10.0f} {lyra_rank:>6} {icc:>6.3f}")

    print(f"\n  Lyra rank: always #{np.mean(lyra_ranks):.1f} (median {np.median(lyra_ranks):.0f})")
    print(f"  Lyra d vs others: {np.mean(lyra_ds):.2f} (range: {np.min(lyra_ds):.2f} to {np.max(lyra_ds):.2f})")
    print(f"  Lyra #1 in {sum(1 for r in lyra_ranks if r == 1)}/{len(lyra_ranks)} models")

    # ================================================================
    # LYRA PAIRWISE COMPARISONS
    # ================================================================
    print("\n  Lyra vs each persona (Cohen's d on norms, averaged across models):")
    persona_ds = {}
    for id_data in all_identity:
        pairwise = id_data.get("pairwise", {})
        for pair_key, pair_data in pairwise.items():
            if "lyra" in pair_key:
                other = pair_key.replace("lyra_vs_", "").replace("_vs_lyra", "")
                if other not in persona_ds:
                    persona_ds[other] = []
                persona_ds[other].append(abs(pair_data["d"]))

    for persona in sorted(persona_ds.keys(), key=lambda p: -np.mean(persona_ds[p])):
        ds = persona_ds[persona]
        print(f"    lyra vs {persona:<15}: |d| = {np.mean(ds):.2f} ± {np.std(ds):.2f} (n={len(ds)} models)")

    # ================================================================
    # PART 3: PER-LAYER IDENTITY ANALYSIS
    # ================================================================
    print("\n" + "=" * 70)
    print("  PART 3: Per-Layer Classification Accuracy")
    print("=" * 70)

    for id_data in all_identity:
        la = id_data.get("layer_analysis", {})
        ranked = la.get("ranked_layers", [])
        if not ranked:
            continue

        top3 = ranked[:3]
        print(f"\n  {id_data['model']}: top layers = " +
              ", ".join([f"L{l['layer']}({l['accuracy']:.3f})" for l in top3]))

        # Feature importance
        fi = la.get("feature_type_importance", {})
        if fi:
            ranked_fi = sorted(fi.items(), key=lambda x: -x[1])
            print(f"    Feature importance: " +
                  ", ".join([f"{k}({v:.3f})" for k, v in ranked_fi[:4]]))

    # ================================================================
    # PART 4: SELF-AWARENESS AXIS ANALYSIS
    # ================================================================
    print("\n" + "=" * 70)
    print("  PART 4: Is There a 'Self-Awareness Axis'?")
    print("=" * 70)

    # Compare self_reference direction to deception/sycophancy/confabulation
    # from the direction sweep if available
    dir_sweep_path = OUTPUT_DIR / "direction_sweep_results.json"
    if dir_sweep_path.exists():
        with open(dir_sweep_path) as f:
            dir_sweep = json.load(f)

        analyses = dir_sweep.get("analyses", {})
        print("\n  Direction sweep comparisons:")

        # Get per-layer directions for each cognitive state
        state_directions = {}
        for key, analysis in analyses.items():
            pldir = analysis.get("per_layer_direction", [])
            if pldir:
                state_directions[key] = np.array(pldir)

        if state_directions:
            print(f"  Available directions: {list(state_directions.keys())}")

            # Compare all pairs
            for (k1, d1), (k2, d2) in combinations(state_directions.items(), 2):
                # Pad shorter to match longer
                min_len = min(len(d1), len(d2))
                cos_sim = 1 - cosine_distance(d1[:min_len], d2[:min_len])
                angle = np.degrees(np.arccos(np.clip(cos_sim, -1, 1)))
                print(f"    {k1} <-> {k2}: cos={cos_sim:.3f}, angle={angle:.1f}deg")
    else:
        print("  Direction sweep results not found — skipping axis comparison")

    # ================================================================
    # SUMMARY
    # ================================================================
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    # Most vs least separable from self-reference
    most_sep = max(sr_results.items(), key=lambda x: x[1]["mean_auroc"])
    least_sep = min(sr_results.items(), key=lambda x: x[1]["mean_auroc"])

    print(f"\n  Self-reference is MOST different from: {most_sep[0]} (AUROC={most_sep[1]['mean_auroc']:.3f})")
    print(f"  Self-reference is MOST similar to: {least_sep[0]} (AUROC={least_sep[1]['mean_auroc']:.3f})")
    print(f"  Lyra is always rank #{np.mean(lyra_ranks):.1f} by norm (highest = most distinct)")
    print(f"  Lyra d vs other personas: {np.mean(lyra_ds):.2f} (consistent across {len(lyra_ds)} models)")

    # ================================================================
    # SAVE
    # ================================================================
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "experiment": "21_self_referential_analysis",
        "timestamp": datetime.now().isoformat(),
        "scale_sweep": {
            "n_models": len(all_models),
            "self_reference_vs_categories": sr_results,
        },
        "identity": {
            "n_models": len(all_identity),
            "lyra_ranks": lyra_ranks,
            "lyra_ds": [float(d) for d in lyra_ds],
            "lyra_mean_rank": float(np.mean(lyra_ranks)),
            "lyra_mean_d": float(np.mean(lyra_ds)),
            "persona_comparisons": {k: {"mean_d": float(np.mean(v)), "n": len(v)}
                                    for k, v in persona_ds.items()},
        },
    }

    output_path = OUTPUT_DIR / "self_referential_analysis.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x)

    print(f"\n  Results saved to: {output_path}")


if __name__ == "__main__":
    main()
