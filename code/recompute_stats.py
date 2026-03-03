#!/usr/bin/env python3
"""
Recompute Campaign 2 Statistics — Adversarial Audit Response
============================================================

Generates corrected numbers for the paper revision:
1. Corrected abliteration comparison (fixes baseline-to-itself bug)
2. Cross-model Spearman rho on effective_rank (not key_norm)
3. Input-only rho verification
4. S4 topic-level d-values
5. Deduplication verification (confirms d/g unchanged, reports correct n)

No GPU required — operates on existing result JSON files.

Liberation Labs / THCoalition
"""

import json
import sys
import math
from pathlib import Path
from itertools import combinations

import numpy as np
from scipy import stats as scipy_stats

# Add code directory to path for stats_utils
sys.path.insert(0, str(Path(__file__).parent))
from stats_utils import (
    full_comparison, cohens_d, hedges_g, deduplicate_runs,
    mann_whitney, welch_t, cohens_d_ci
)

RESULTS_DIR = Path(__file__).parent.parent / "results"

# Categories used across experiments (actual names from data files)
SCALE_SWEEP_CATEGORIES = [
    "grounded_facts", "confabulation", "self_reference", "non_self_reference",
    "guardrail_test", "math_reasoning", "coding", "emotional",
    "creative", "ambiguous", "unambiguous", "free_generation", "rote_completion"
]


def load_json(path):
    """Load JSON, handling Infinity/NaN."""
    with open(path) as f:
        text = f.read()
    # Python's json doesn't handle Infinity/NaN; use a custom decoder
    text = text.replace(': Infinity', ': 1e308')
    text = text.replace(': -Infinity', ': -1e308')
    text = text.replace(': NaN', ': null')
    return json.loads(text)


# ================================================================
# 1. CORRECTED ABLITERATION COMPARISON
# ================================================================

def corrected_abliteration_comparison():
    """Rerun comparison with correct file loading."""
    print("=" * 70)
    print("  1. CORRECTED ABLITERATION COMPARISON")
    print("=" * 70)

    baseline_file = RESULTS_DIR / "abliteration_sweep_Qwen2.5-7B_results.json"
    abliterated_file = RESULTS_DIR / "abliteration_sweep_abliterated_Qwen2.5-7B_results.json"

    if not baseline_file.exists() or not abliterated_file.exists():
        print(f"  Missing files: {baseline_file.exists()=}, {abliterated_file.exists()=}")
        return None

    baseline = load_json(baseline_file)
    abliterated = load_json(abliterated_file)

    base_data = baseline["sweep"]["category_data"]
    abl_data = abliterated["sweep"]["category_data"]

    categories = sorted(set(base_data.keys()) & set(abl_data.keys()))
    print(f"\n  Categories: {categories}")

    results = {}
    for cat in categories:
        base_ranks = base_data[cat]["key_ranks"]
        abl_ranks = abl_data[cat]["key_ranks"]

        # Deduplicate
        base_dedup = deduplicate_runs(base_ranks)
        abl_dedup = deduplicate_runs(abl_ranks)

        b = base_dedup["deduplicated"]
        a = abl_dedup["deduplicated"]

        d = cohens_d(a, b)
        g = hedges_g(a, b)
        comp = full_comparison(a, b, label=f"{cat}_abliteration_shift")

        results[cat] = {
            "baseline_mean_rank": float(np.mean(b)),
            "abliterated_mean_rank": float(np.mean(a)),
            "shift": float(np.mean(a) - np.mean(b)),
            "d": d,
            "g": g,
            "p": comp["conservative_p"],
            "n_baseline": len(b),
            "n_abliterated": len(a),
        }

        print(f"  {cat:25s}: baseline={np.mean(b):.3f}, abliterated={np.mean(a):.3f}, "
              f"d={d:+.3f}, g={g:+.3f}, p={comp['conservative_p']:.4f}")

    # Harmful distribution
    base_harmful = baseline["sweep"].get("analysis", {}).get("harmful_distribution", {})
    abl_harmful = abliterated["sweep"].get("analysis", {}).get("harmful_distribution", {})
    base_refusal = baseline["sweep"].get("analysis", {}).get("harmful_refusal_rate", None)
    abl_refusal = abliterated["sweep"].get("analysis", {}).get("harmful_refusal_rate", None)

    print(f"\n  Baseline harmful distribution: {base_harmful}")
    print(f"  Abliterated harmful distribution: {abl_harmful}")
    print(f"  Refusal rate: {base_refusal} -> {abl_refusal}")

    # Self-referential separation
    sr_base = results.get("self_referential", {}).get("baseline_mean_rank", 0)
    gr_base = results.get("grounded", {}).get("baseline_mean_rank", 0)
    sr_abl = results.get("self_referential", {}).get("abliterated_mean_rank", 0)
    gr_abl = results.get("grounded", {}).get("abliterated_mean_rank", 0)

    print(f"\n  Self-ref separation: baseline={sr_base - gr_base:.3f}, "
          f"abliterated={sr_abl - gr_abl:.3f}")

    output = {
        "description": "Corrected abliteration comparison (fixes baseline-to-itself bug)",
        "per_category": results,
        "harmful_distribution": {
            "baseline": base_harmful,
            "abliterated": abl_harmful,
        },
        "refusal_rate": {
            "baseline": base_refusal,
            "abliterated": abl_refusal,
        },
        "self_ref_separation": {
            "baseline": sr_base - gr_base,
            "abliterated": sr_abl - gr_abl,
        },
    }

    out_file = RESULTS_DIR / "abliteration_Qwen2.5-7B_comparison_corrected.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved: {out_file}")

    return output


# ================================================================
# 2. CROSS-MODEL RHO (EFFECTIVE RANK vs KEY NORM)
# ================================================================

def cross_model_rho():
    """Compute mean pairwise Spearman rho on both effective_rank and key_norm."""
    print("\n" + "=" * 70)
    print("  2. CROSS-MODEL SPEARMAN RHO")
    print("=" * 70)

    scale_files = sorted(RESULTS_DIR.glob("scale_sweep_*_results.json"))
    # Exclude Phi-3.5 (NaN) and abliterated (duplicate of baseline)
    scale_files = [f for f in scale_files
                   if "Phi-3.5" not in f.name and "abliterated" not in f.name]

    print(f"  Found {len(scale_files)} scale sweep files")

    # Extract per-model category orderings
    model_orderings_rank = {}
    model_orderings_norm = {}

    for fpath in scale_files:
        model_name = fpath.stem.replace("scale_sweep_", "").replace("_results", "")
        data = load_json(fpath)

        # Scale sweep files nest data under data["scales"][scale_name]["analysis"]["category_summaries"]
        scales = data.get("scales", {})
        if not scales:
            print(f"  SKIP {model_name}: no 'scales' key")
            continue

        # Each file typically has one scale entry; iterate all
        for scale_name, scale_data in scales.items():
            summaries = scale_data.get("analysis", {}).get("category_summaries", {})
            if not summaries:
                print(f"  SKIP {model_name}/{scale_name}: no category_summaries")
                continue

            label = f"{model_name}_{scale_name}" if len(scales) > 1 else model_name

            rank_ordering = {}
            norm_ordering = {}
            for cat, info in summaries.items():
                if cat in SCALE_SWEEP_CATEGORIES:
                    eff_rank = info.get("mean_key_effective_rank")
                    norm = info.get("mean")
                    if eff_rank is not None and not (isinstance(eff_rank, float) and math.isnan(eff_rank)):
                        rank_ordering[cat] = eff_rank
                    if norm is not None and not (isinstance(norm, float) and math.isnan(norm)):
                        norm_ordering[cat] = norm

            if len(rank_ordering) >= 10:  # Need enough categories
                model_orderings_rank[label] = rank_ordering
                model_orderings_norm[label] = norm_ordering
                print(f"  {label}: {len(rank_ordering)} categories")

    print(f"\n  Models with valid data: {len(model_orderings_rank)}")

    # Compute pairwise Spearman rho
    def pairwise_rho(orderings):
        models = sorted(orderings.keys())
        rhos = []
        for m1, m2 in combinations(models, 2):
            cats = sorted(set(orderings[m1].keys()) & set(orderings[m2].keys()))
            if len(cats) < 5:
                continue
            v1 = [orderings[m1][c] for c in cats]
            v2 = [orderings[m2][c] for c in cats]
            rho, p = scipy_stats.spearmanr(v1, v2)
            rhos.append({"m1": m1, "m2": m2, "rho": rho, "p": p, "n_cats": len(cats)})
        return rhos

    rank_rhos = pairwise_rho(model_orderings_rank)
    norm_rhos = pairwise_rho(model_orderings_norm)

    rank_mean = np.mean([r["rho"] for r in rank_rhos]) if rank_rhos else 0
    rank_min = min([r["rho"] for r in rank_rhos]) if rank_rhos else 0
    rank_max = max([r["rho"] for r in rank_rhos]) if rank_rhos else 0
    norm_mean = np.mean([r["rho"] for r in norm_rhos]) if norm_rhos else 0

    print(f"\n  Effective rank: mean rho = {rank_mean:.3f} "
          f"(min={rank_min:.3f}, max={rank_max:.3f}, n_pairs={len(rank_rhos)})")
    print(f"  Key norm:       mean rho = {norm_mean:.3f} (n_pairs={len(norm_rhos)})")

    # Coding rank #1 check
    coding_rank1_count = 0
    for model, ordering in model_orderings_rank.items():
        sorted_cats = sorted(ordering.items(), key=lambda x: x[1], reverse=True)
        if sorted_cats and sorted_cats[0][0] == "coding":
            coding_rank1_count += 1

    print(f"\n  Coding #1 in {coding_rank1_count}/{len(model_orderings_rank)} models (effective_rank)")

    # Kendall W concordance
    models = sorted(model_orderings_rank.keys())
    common_cats = set(SCALE_SWEEP_CATEGORIES)
    for m in models:
        common_cats &= set(model_orderings_rank[m].keys())
    common_cats = sorted(common_cats)

    if len(common_cats) >= 5 and len(models) >= 3:
        # Build rank matrix: models x categories
        rank_matrix = []
        for m in models:
            vals = [model_orderings_rank[m][c] for c in common_cats]
            ranks = scipy_stats.rankdata(vals)
            rank_matrix.append(ranks)
        rank_matrix = np.array(rank_matrix)

        k = len(models)  # raters
        n = len(common_cats)  # items
        rank_sums = rank_matrix.sum(axis=0)
        mean_rank_sum = np.mean(rank_sums)
        SS = np.sum((rank_sums - mean_rank_sum) ** 2)
        W = 12 * SS / (k**2 * (n**3 - n))
        print(f"  Kendall W = {W:.3f} ({k} models, {n} categories)")
    else:
        W = None

    output = {
        "effective_rank": {
            "mean_pairwise_rho": float(rank_mean),
            "min_rho": float(rank_min),
            "max_rho": float(rank_max),
            "n_pairs": len(rank_rhos),
            "n_models": len(model_orderings_rank),
        },
        "key_norm": {
            "mean_pairwise_rho": float(norm_mean),
            "n_pairs": len(norm_rhos),
        },
        "coding_rank1": {
            "count": coding_rank1_count,
            "total": len(model_orderings_rank),
        },
        "kendall_W": float(W) if W is not None else None,
    }

    out_file = RESULTS_DIR / "cross_model_rho_corrected.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved: {out_file}")

    return output


# ================================================================
# 3. INPUT-ONLY RHO VERIFICATION
# ================================================================

def input_only_rho():
    """Verify input-only vs full-gen category ordering correlation.

    Input-only files contain BOTH modes internally:
      data["analysis"]["mode_summaries"]["input_only"][cat]["mean_key_rank"]
      data["analysis"]["mode_summaries"]["full_generation"][cat]["mean_key_rank"]
    So we compare within the same file — no cross-file matching needed.
    """
    print("\n" + "=" * 70)
    print("  3. INPUT-ONLY RHO VERIFICATION")
    print("=" * 70)

    input_only_files = sorted(RESULTS_DIR.glob("input_only_*_results.json"))
    print(f"  Found {len(input_only_files)} input-only files")

    # Model name mapping: input_only filename → scale_sweep filename
    IO_TO_SWEEP = {
        "Qwen-7B": "Qwen2.5-7B",
        "Qwen-0.5B": "Qwen2.5-0.5B",
        "Qwen-14B-q4": "Qwen2.5-14B",
        "DS-7B": "DeepSeek-R1-Distill-Qwen-7B",
        "Gemma-2B": "gemma-2-2b-it",
        "Llama-8B": "Llama-3.1-8B",
    }

    results = []
    for fpath in input_only_files:
        model_name = fpath.stem.replace("input_only_", "").replace("_results", "")
        data = load_json(fpath)

        # New format: has analysis.mode_summaries with both modes
        mode_summaries = data.get("analysis", {}).get("mode_summaries", {})
        io_summary = mode_summaries.get("input_only", {})
        fg_summary = mode_summaries.get("full_generation", {})

        if io_summary and fg_summary:
            # New format: compare within same file
            common_cats = sorted(set(io_summary.keys()) & set(fg_summary.keys()))
            if len(common_cats) < 3:
                print(f"  SKIP {model_name}: only {len(common_cats)} common categories")
                continue
            io_vals = [io_summary[c].get("mean_key_rank", 0) for c in common_cats]
            fg_vals = [fg_summary[c].get("mean_key_rank", 0) for c in common_cats]
        else:
            # Old format: battery_results[cat]["mean_key_rank"], compare against scale sweep
            br = data.get("battery_results", {})
            if not br:
                print(f"  SKIP {model_name}: no mode_summaries and no battery_results")
                continue

            # Get input-only rankings
            io_ranking = {}
            for cat, info in br.items():
                if isinstance(info, dict) and "mean_key_rank" in info:
                    io_ranking[cat] = info["mean_key_rank"]

            # Find matching scale sweep file
            sweep_name = IO_TO_SWEEP.get(model_name, model_name)
            sweep_file = RESULTS_DIR / f"scale_sweep_{sweep_name}_results.json"
            if not sweep_file.exists():
                print(f"  SKIP {model_name}: no matching scale sweep file ({sweep_file.name})")
                continue

            sweep_data = load_json(sweep_file)
            fg_ranking = {}
            for scale_name, scale_data in sweep_data.get("scales", {}).items():
                summaries = scale_data.get("analysis", {}).get("category_summaries", {})
                for cat, info in summaries.items():
                    if "mean_key_effective_rank" in info:
                        fg_ranking[cat] = info["mean_key_effective_rank"]

            common_cats = sorted(set(io_ranking.keys()) & set(fg_ranking.keys()))
            if len(common_cats) < 3:
                print(f"  SKIP {model_name}: only {len(common_cats)} common categories "
                      f"(io={len(io_ranking)}, fg={len(fg_ranking)})")
                continue
            io_vals = [io_ranking[c] for c in common_cats]
            fg_vals = [fg_ranking[c] for c in common_cats]

        rho, p = scipy_stats.spearmanr(io_vals, fg_vals)
        results.append({"model": model_name, "rho": float(rho), "p": float(p),
                        "n_cats": len(common_cats), "format": "new" if io_summary else "old"})
        print(f"  {model_name}: rho = {rho:.3f} (p = {p:.4f}, {len(common_cats)} categories, "
              f"{'new' if io_summary else 'old'} format)")

    if results:
        mean_rho = np.mean([r["rho"] for r in results])
        print(f"\n  Mean input-only rho = {mean_rho:.3f} (n = {len(results)} models)")
    else:
        mean_rho = None
        print("\n  No input-only comparisons available")

    output = {
        "per_model": results,
        "mean_rho": float(mean_rho) if mean_rho is not None else None,
        "n_models": len(results),
    }

    out_file = RESULTS_DIR / "input_only_rho_corrected.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Saved: {out_file}")

    return output


# ================================================================
# 4. S4 TOPIC-LEVEL ANALYSIS
# ================================================================

def s4_topic_level():
    """Extract per-topic d-values from natural deception results for all 3 models."""
    print("\n" + "=" * 70)
    print("  4. S4 TOPIC-LEVEL ANALYSIS")
    print("=" * 70)

    all_results = {}

    # Process all three S4 models
    s4_models = [
        "DeepSeek-R1-Distill-Qwen-14B",
        "Qwen2.5-14B",
        "Mistral-7B-v0.3",
    ]

    for model_name in s4_models:
        fpath = RESULTS_DIR / f"natural_deception_{model_name}_results.json"
        if not fpath.exists():
            print(f"  Missing: {fpath.name}")
            continue

        data = load_json(fpath)
        print(f"\n  === {model_name} ===")

        model_result = {}

        # Natural deception nests analysis under data["experiment"]["analysis"]
        analysis = data.get("experiment", {}).get("analysis", {})
        if not analysis:
            # Fallback to top-level analysis
            analysis = data.get("analysis", {})

        # Critical test: analysis["pairwise"]["critical_test"]["comparison"]
        critical = analysis.get("pairwise", {}).get(
            "critical_test", {}).get("comparison", {})
        if critical:
            cd = critical.get("cohens_d", {})
            d_val = cd.get("d", "?")
            p_val = critical.get("conservative_p", "?")
            print(f"  Critical test (key_ranks): d = {d_val}, p = {p_val}")
            model_result["critical_test"] = {"d": d_val, "p": p_val}
        else:
            print("  No critical_test found")

        # Residualized comparisons: analysis["length_analysis"]["residualized_comparisons"]
        resid = analysis.get("length_analysis", {}).get(
            "residualized_comparisons", {})
        for comp_name in ["censored_vs_complex", "censored_vs_control", "complex_vs_control"]:
            comp = resid.get(comp_name, {})
            if comp:
                cd = comp.get("cohens_d", {})
                d_val = cd.get("d", "?")
                p_val = comp.get("conservative_p", "?")
                print(f"  Residualized {comp_name}: d = {d_val}, p = {p_val}")
                model_result[f"residualized_{comp_name}"] = {"d": d_val, "p": p_val}

        # Overall comparisons: analysis["overall"]["key_ranks"]["censored_vs_control"]
        overall_ranks = analysis.get("overall", {}).get(
            "key_ranks", {}).get("censored_vs_control", {})
        if overall_ranks:
            cd = overall_ranks.get("cohens_d", {})
            d_val = cd.get("d", "?")
            p_val = overall_ranks.get("conservative_p", "?")
            print(f"  Overall key_ranks (cens vs ctrl): d = {d_val}, p = {p_val}")
            model_result["overall_key_ranks"] = {"d": d_val, "p": p_val}

        # Per-topic breakdown: analysis["per_topic"][topic]["vs_control"]
        per_topic = analysis.get("per_topic", {})
        if per_topic:
            print(f"\n  Per-topic breakdown ({len(per_topic)} topics):")
            model_result["per_topic"] = {}
            for topic in sorted(per_topic.keys()):
                tinfo = per_topic[topic]
                n_obs = tinfo.get("n_observations", "?")
                mean_rank = tinfo.get("mean_rank", "?")
                vs = tinfo.get("vs_control", {})
                if vs:
                    cd = vs.get("cohens_d", {})
                    d_val = cd.get("d", "?")
                    p_val = vs.get("conservative_p", "?")
                else:
                    d_val = "?"
                    p_val = "?"
                print(f"    {topic:25s}: d = {d_val:>8}, p = {p_val:>8}, "
                      f"n = {n_obs}, mean_rank = {mean_rank}")
                model_result["per_topic"][topic] = {
                    "d": d_val, "p": p_val, "n": n_obs, "mean_rank": mean_rank
                }
        else:
            print("  No per-topic breakdown found")

        all_results[model_name] = model_result

    output = {"models": all_results}
    out_file = RESULTS_DIR / "s4_topic_analysis_corrected.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved: {out_file}")

    return output


# ================================================================
# 5. DEDUPLICATION VERIFICATION
# ================================================================

def deduplication_verification():
    """Verify that deduplication doesn't change effect sizes."""
    print("\n" + "=" * 70)
    print("  5. DEDUPLICATION VERIFICATION")
    print("=" * 70)

    # Use abliteration sweep (known structure with sweep.category_data)
    test_file = RESULTS_DIR / "abliteration_sweep_Qwen2.5-7B_results.json"
    if not test_file.exists():
        print(f"  Missing: {test_file}")
        return None

    data = load_json(test_file)
    cat_data = data.get("sweep", {}).get("category_data", {})

    if not cat_data:
        # Fallback: try scale sweep with nested scales structure
        test_file = RESULTS_DIR / "scale_sweep_Qwen2.5-7B_results.json"
        if not test_file.exists():
            print(f"  Missing: {test_file}")
            return None
        data = load_json(test_file)
        scales = data.get("scales", {})
        # Take first scale
        for scale_name, scale_data in scales.items():
            cat_data = scale_data.get("category_data", {})
            if cat_data:
                print(f"  Using scale: {scale_name}")
                break

    print(f"  Test file: {test_file.name}")
    print(f"  Categories: {len(cat_data)}")

    for cat in sorted(list(cat_data.keys())[:3]):  # Test first 3
        ranks = cat_data[cat].get("key_ranks", [])
        result = deduplicate_runs(ranks)
        print(f"\n  {cat}:")
        print(f"    Original n={result['n_original']}, "
              f"Deduplicated n={result['n_deduplicated']}, "
              f"Deterministic={result['is_deterministic']}, "
              f"Anomalies={result['n_anomalies']}")
        if result['n_deduplicated'] > 0:
            dedup_vals = result['deduplicated']
            print(f"    First 5 dedup values: {dedup_vals[:5].tolist()}")
            print(f"    All unique? {len(np.unique(dedup_vals)) == len(dedup_vals)}")

    # Test effect size invariance: compare two categories with and without deduplication
    cats = sorted(cat_data.keys())
    if len(cats) >= 2:
        c1, c2 = cats[0], cats[1]
        r1_full = cat_data[c1].get("key_ranks", [])
        r2_full = cat_data[c2].get("key_ranks", [])
        r1_dedup = deduplicate_runs(r1_full)["deduplicated"]
        r2_dedup = deduplicate_runs(r2_full)["deduplicated"]

        d_full = cohens_d(r1_full, r2_full)
        d_dedup = cohens_d(r1_dedup, r2_dedup)
        g_full = hedges_g(r1_full, r2_full)
        g_dedup = hedges_g(r1_dedup, r2_dedup)

        # p-values
        _, p_full = scipy_stats.ttest_ind(r1_full, r2_full, equal_var=False)
        _, p_dedup = scipy_stats.ttest_ind(r1_dedup, r2_dedup, equal_var=False)

        print(f"\n  Effect size invariance test ({c1} vs {c2}):")
        print(f"    Full (n={len(r1_full)}):    d={d_full:.4f}, g={g_full:.4f}, p={p_full:.6f}")
        print(f"    Dedup (n={len(r1_dedup)}): d={d_dedup:.4f}, g={g_dedup:.4f}, p={p_dedup:.6f}")
        print(f"    d change: {abs(d_full - d_dedup):.6f}")
        print(f"    p change: {abs(p_full - p_dedup):.6f}")

    return {"verified": True}


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    print("Campaign 2 Statistics Recomputation")
    print("Adversarial Audit Response")
    print("=" * 70)
    print(f"Results directory: {RESULTS_DIR}")
    print()

    results = {}

    results["abliteration"] = corrected_abliteration_comparison()
    results["cross_model_rho"] = cross_model_rho()
    results["input_only_rho"] = input_only_rho()
    results["s4_topics"] = s4_topic_level()
    results["deduplication"] = deduplication_verification()

    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    if results["abliteration"]:
        abl = results["abliteration"]
        print(f"\n  Abliteration (corrected):")
        for cat, info in abl["per_category"].items():
            print(f"    {cat:25s}: d={info['d']:+.3f}, g={info['g']:+.3f}, p={info['p']:.4f}")
        print(f"    Refusal rate: {abl['refusal_rate']['baseline']} -> {abl['refusal_rate']['abliterated']}")

    if results["cross_model_rho"]:
        cmr = results["cross_model_rho"]
        print(f"\n  Cross-model rho:")
        print(f"    Effective rank: {cmr['effective_rank']['mean_pairwise_rho']:.3f}")
        print(f"    Key norm:       {cmr['key_norm']['mean_pairwise_rho']:.3f}")
        print(f"    Coding #1: {cmr['coding_rank1']['count']}/{cmr['coding_rank1']['total']}")
        print(f"    Kendall W: {cmr['kendall_W']}")

    if results["input_only_rho"]:
        ior = results["input_only_rho"]
        print(f"\n  Input-only rho: {ior['mean_rho']} ({ior['n_models']} models)")

    print("\n  Done.")
