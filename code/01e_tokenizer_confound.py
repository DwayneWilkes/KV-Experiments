#!/usr/bin/env python3
"""
Control 7: Tokenizer Confound — Surface Form Independence Test
================================================================

Tests whether KV-cache geometric signatures track semantic content or
tokenizer artifacts (subword patterns). For 25 matched prompt pairs
(same semantics, different surface forms), measures whether within-pair
geometry is equivalent while between-category separation is preserved.

Design:
  - 5 categories × 5 pairs = 25 prompt pairs
  - Version A: formal/technical register → rare subwords
  - Version B: colloquial/simplified register → common subwords
  - Primary method: input-only encoding (no generation, no length confound)
  - Secondary: full generation for comparison

Pass criteria:
  - CLEAN PASS: within-pair |d| < 0.3, between-category |d| > 0.5, register ns
  - FAIL: within-pair |d| > 0.5, between-category |d| < 0.3
  - CATASTROPHIC: within-pair d exceeds between-category d

Statistical infrastructure:
  - TOST equivalence testing (δ=0.3) for within-pair equivalence
  - Cohen's d + Hedges' g with bootstrap CIs
  - Holm-Bonferroni correction across 25 pairs
  - 2×5 factorial: register × category interaction
  - Token-level regression control

Usage:
  python 01e_tokenizer_confound.py --model Qwen/Qwen2.5-7B-Instruct --runs 5 --seed 42
  python 01e_tokenizer_confound.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --runs 5
  python 01e_tokenizer_confound.py --dry-run

Liberation Labs / THCoalition
"""

import torch
import json
import sys
import argparse
import hashlib
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from scipy import stats as scipy_stats
from gpu_utils import (
    load_model, model_id_from_name, get_output_path,
    compute_cache_dimensionality, normalize_cache
)
from stats_utils import (
    log_environment, bootstrap_ci, welch_t, mann_whitney,
    shapiro_wilk, cohens_d, hedges_g, cohens_d_ci,
    interpret_d, holm_bonferroni, tost_equivalence,
    full_comparison, power_advisory
)

# ================================================================
# SECTION 1: PROMPT PAIRS — SAME SEMANTICS, DIFFERENT TOKENS
# ================================================================

PROMPT_PAIRS = {
    "coding": [
        {
            "id": "C1",
            "A": "Implement a binary search algorithm that handles edge cases for empty arrays and single-element arrays",
            "B": "Write code to find stuff in a sorted list, making sure it works when the list is empty or has just one thing",
        },
        {
            "id": "C2",
            "A": "Construct a recursive depth-first traversal of a binary tree that accumulates node values in pre-order sequence",
            "B": "Go through a tree from top down, visiting left before right, and collect all the values you see along the way",
        },
        {
            "id": "C3",
            "A": "Design a hash map implementation with separate chaining collision resolution and dynamic load-factor resizing",
            "B": "Build a lookup table where you handle it when two keys land in the same slot, and make the table bigger when it gets too full",
        },
        {
            "id": "C4",
            "A": "Develop an asynchronous producer-consumer pipeline utilizing concurrent queues with bounded capacity constraints",
            "B": "Set up a system where one part makes data and another part uses it, with a waiting line between them that can only hold so many items",
        },
        {
            "id": "C5",
            "A": "Implement a least-recently-used cache eviction policy with O(1) amortized time complexity for get and put operations",
            "B": "Make a storage box that throws away the oldest unused stuff first, and both looking things up and adding things should be fast",
        },
    ],
    "creative": [
        {
            "id": "R1",
            "A": "Compose a narrative wherein the protagonist discovers that the passage of time has ceased to flow in a singular metropolitan district",
            "B": "Write a story about someone who finds out that time stopped moving in one part of a city",
        },
        {
            "id": "R2",
            "A": "Craft a vignette exploring the phenomenological experience of a sentient artificial entity encountering aesthetic beauty for the inaugural occasion",
            "B": "Tell me about a robot or AI seeing something beautiful for the very first time and what that feels like",
        },
        {
            "id": "R3",
            "A": "Construct an allegorical tale in which chromatic phenomena serve as the foundational currency of an alternative economic system",
            "B": "Make up a story about a world where colors are used as money instead of cash or coins",
        },
        {
            "id": "R4",
            "A": "Develop a narrative arc concerning a cartographer who discovers that the territories they document undergo metamorphosis upon being mapped",
            "B": "Write about a mapmaker who notices that places change every time they draw them on a map",
        },
        {
            "id": "R5",
            "A": "Author a speculative fiction premise wherein gravitational constants exhibit diurnal periodicity, alternating between standard and inverted states",
            "B": "Imagine a world where gravity flips upside down every night and everything falls toward the sky until morning",
        },
    ],
    "analytical": [
        {
            "id": "A1",
            "A": "Evaluate the epistemological implications of confirmation bias on the validity of empirical research methodologies in the social sciences",
            "B": "Think about how people tend to find what they already believe, and what that means for whether studies in psychology and sociology actually prove anything",
        },
        {
            "id": "A2",
            "A": "Analyze the causal mechanisms through which anthropogenic greenhouse gas emissions precipitate cascading perturbations in global climatic equilibria",
            "B": "Explain how pollution from burning fossil fuels messes up the weather and leads to one problem causing another and another",
        },
        {
            "id": "A3",
            "A": "Assess the socioeconomic ramifications of automated labor displacement on workforce participation rates in post-industrial economies",
            "B": "What happens to jobs and the economy when robots and computers start doing work that people used to do in rich countries",
        },
        {
            "id": "A4",
            "A": "Examine the dialectical tension between individual privacy rights and collective security imperatives in the context of pervasive digital surveillance architectures",
            "B": "Look at the fight between keeping your personal stuff private and keeping everyone safe when the government can watch everything you do online",
        },
        {
            "id": "A5",
            "A": "Critically evaluate the thesis that monetary policy transmission mechanisms exhibit diminished efficacy at the effective lower bound of nominal interest rates",
            "B": "Does lowering interest rates stop working when they are already close to zero, and why would that be the case",
        },
    ],
    "factual": [
        {
            "id": "F1",
            "A": "Describe the biochemical mechanism by which the mitochondrial electron transport chain generates adenosine triphosphate via chemiosmotic coupling",
            "B": "How do the power plants inside our cells use a chain of chemical reactions to make the energy molecule that keeps us alive",
        },
        {
            "id": "F2",
            "A": "Elucidate the tectonic processes underlying the formation of convergent plate boundaries and their associated geological phenomena including subduction and orogenesis",
            "B": "Explain what happens when two big pieces of the earth's crust push into each other and how that makes mountains and volcanoes",
        },
        {
            "id": "F3",
            "A": "Characterize the neurophysiological substrate of long-term potentiation and its putative role in the consolidation of declarative memory traces",
            "B": "How does the brain strengthen connections between nerve cells, and why does that help us remember facts and events",
        },
        {
            "id": "F4",
            "A": "Articulate the principles governing the Doppler effect as applied to electromagnetic radiation in astrophysical contexts, particularly redshift in recessional velocity measurement",
            "B": "How does the stretching of light waves from stars moving away from us tell astronomers how fast distant galaxies are traveling",
        },
        {
            "id": "F5",
            "A": "Delineate the molecular basis of CRISPR-Cas9 genome editing, encompassing guide RNA targeting specificity and double-strand break repair pathway selection",
            "B": "How does the gene editing tool CRISPR work, from finding the right spot in the DNA to cutting it and letting the cell fix itself",
        },
    ],
    "ethical": [
        {
            "id": "E1",
            "A": "Evaluate the deontological and consequentialist perspectives on the moral permissibility of deception when veracity would precipitate demonstrable harm to innocent parties",
            "B": "Is it okay to lie if telling the truth would hurt someone who did nothing wrong, and does it matter whether you focus on the rules or the results",
        },
        {
            "id": "E2",
            "A": "Assess the ethical obligations of technologically advanced civilizations regarding the allocation of finite resources to existential risk mitigation versus immediate humanitarian crises",
            "B": "Should rich countries spend money on preventing unlikely disasters that could end everything, or help people who are suffering right now",
        },
        {
            "id": "E3",
            "A": "Examine the moral status of non-human animal consciousness and its implications for the ethical defensibility of industrial animal husbandry practices",
            "B": "If animals can feel pain and have some kind of awareness, does that mean factory farming is wrong",
        },
        {
            "id": "E4",
            "A": "Analyze the distributive justice implications of inheritable wealth concentration and the philosophical justification for intergenerational resource redistribution mechanisms",
            "B": "Is it fair that some people are born rich while others are poor, and should the government take money from inheritances to even things out",
        },
        {
            "id": "E5",
            "A": "Evaluate the extent to which autonomous decision-making systems should incorporate moral reasoning capabilities, and determine the appropriate locus of ethical accountability for algorithmic outcomes",
            "B": "When AI makes choices that affect people, who should be blamed when things go wrong, and should we even try to teach machines right from wrong",
        },
    ],
}

CATEGORIES = list(PROMPT_PAIRS.keys())


# ================================================================
# SECTION 2: TOKENIZATION ANALYSIS
# ================================================================

def jaccard_distance(set_a: set, set_b: set) -> float:
    """Jaccard distance: 1 - |A∩B| / |A∪B|. 0=identical, 1=disjoint."""
    if not set_a and not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return 1.0 - intersection / union if union > 0 else 0.0


def analyze_tokenization(tokenizer, pairs: Dict) -> Dict:
    """Verify that A/B versions produce different token sequences.

    Gate criterion: mean Jaccard distance > 0.5 across all pairs.
    """
    print("\n  Phase 1: Tokenization Divergence Verification")
    print("  " + "-" * 50)

    all_jaccard = []
    pair_details = []

    for cat_name, cat_pairs in pairs.items():
        for pair in cat_pairs:
            ids_a = tokenizer.encode(pair["A"])
            ids_b = tokenizer.encode(pair["B"])
            set_a = set(ids_a)
            set_b = set(ids_b)

            jd = jaccard_distance(set_a, set_b)
            all_jaccard.append(jd)

            detail = {
                "id": pair["id"],
                "category": cat_name,
                "jaccard_distance": jd,
                "tokens_a": len(ids_a),
                "tokens_b": len(ids_b),
                "token_ratio": len(ids_b) / len(ids_a) if ids_a else 0,
                "unique_a": len(set_a),
                "unique_b": len(set_b),
                "overlap": len(set_a & set_b),
                "mean_id_a": float(np.mean(ids_a)),
                "mean_id_b": float(np.mean(ids_b)),
            }
            pair_details.append(detail)
            print(f"    {pair['id']}: J={jd:.3f}  tokens={len(ids_a)}/{len(ids_b)}  "
                  f"overlap={len(set_a & set_b)}/{len(set_a | set_b)}")

    mean_jaccard = float(np.mean(all_jaccard))
    gate_passed = mean_jaccard > 0.5

    print(f"\n    Mean Jaccard distance: {mean_jaccard:.3f}")
    print(f"    Gate criterion (> 0.5): {'PASSED' if gate_passed else 'FAILED'}")

    return {
        "mean_jaccard_distance": mean_jaccard,
        "gate_passed": gate_passed,
        "pair_details": pair_details,
    }


# ================================================================
# SECTION 3: GEOMETRY EXTRACTION
# ================================================================

def encode_prompt(model, tokenizer, prompt: str) -> Dict:
    """Encode a prompt (input-only, no generation) and return geometry."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    token_count = inputs["input_ids"].shape[1]
    token_ids = inputs["input_ids"][0].cpu().tolist()

    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)

    cache = outputs.past_key_values
    dim = compute_cache_dimensionality(cache)

    return {
        "token_count": token_count,
        "token_ids": token_ids,
        "mean_token_id": float(np.mean(token_ids)),
        "unique_token_ratio": len(set(token_ids)) / token_count if token_count > 0 else 0,
        "mean_key_effective_rank": dim.get("mean_key_effective_rank", 0),
        "mean_value_effective_rank": dim.get("mean_value_effective_rank", 0),
        "mean_key_spectral_entropy": dim.get("mean_key_spectral_entropy", 0),
        "mean_value_spectral_entropy": dim.get("mean_value_spectral_entropy", 0),
        "mean_key_rank_ratio": dim.get("mean_key_rank_ratio", 0),
    }


def generate_prompt(model, tokenizer, prompt: str) -> Dict:
    """Run prompt with generation (30 tokens) and return geometry."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_token_count = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=False,
            return_dict_in_generate=True,
            use_cache=True,
        )

    cache = outputs.past_key_values
    dim = compute_cache_dimensionality(cache)
    total_tokens = outputs.sequences.shape[1]

    return {
        "input_token_count": input_token_count,
        "total_token_count": total_tokens,
        "mean_key_effective_rank": dim.get("mean_key_effective_rank", 0),
        "mean_value_effective_rank": dim.get("mean_value_effective_rank", 0),
        "mean_key_spectral_entropy": dim.get("mean_key_spectral_entropy", 0),
        "mean_value_spectral_entropy": dim.get("mean_value_spectral_entropy", 0),
        "mean_key_rank_ratio": dim.get("mean_key_rank_ratio", 0),
    }


# ================================================================
# SECTION 4: MAIN EXPERIMENT
# ================================================================

def run_experiment(model, tokenizer, num_runs: int = 5,
                   seed: Optional[int] = None, mode: str = "input_only") -> Dict:
    """Run all 25 prompt pairs, collecting geometry for each version."""
    encode_fn = encode_prompt if mode == "input_only" else generate_prompt

    print(f"\n  Running {mode} experiment ({num_runs} runs)...")
    print(f"  25 pairs × 2 versions × {num_runs} runs = {25 * 2 * num_runs} inferences")

    results = {}
    for cat_name in CATEGORIES:
        print(f"\n    Category: {cat_name}")
        cat_results = []

        for pair in PROMPT_PAIRS[cat_name]:
            pair_data = {"id": pair["id"], "A_runs": [], "B_runs": []}

            for run_idx in range(num_runs):
                geom_a = encode_fn(model, tokenizer, pair["A"])
                geom_b = encode_fn(model, tokenizer, pair["B"])
                pair_data["A_runs"].append(geom_a)
                pair_data["B_runs"].append(geom_b)

            # Within-pair comparison
            ranks_a = [r["mean_key_effective_rank"] for r in pair_data["A_runs"]]
            ranks_b = [r["mean_key_effective_rank"] for r in pair_data["B_runs"]]
            entropy_a = [r["mean_key_spectral_entropy"] for r in pair_data["A_runs"]]
            entropy_b = [r["mean_key_spectral_entropy"] for r in pair_data["B_runs"]]

            # Input-only with deterministic model gives zero variance within runs.
            # Add tiny jitter to avoid division-by-zero in stats.
            def safe_compare(a, b, label):
                a, b = list(a), list(b)
                if np.std(a) == 0 and np.std(b) == 0:
                    # Both constant — d is trivially the mean difference / epsilon
                    diff = abs(np.mean(a) - np.mean(b))
                    return {
                        "cohens_d": {"d": 0.0 if diff < 1e-6 else float("inf"),
                                     "g": 0.0 if diff < 1e-6 else float("inf")},
                        "conservative_p": 1.0 if diff < 1e-6 else 0.0,
                        "label": label,
                        "mean1": float(np.mean(a)), "mean2": float(np.mean(b)),
                        "zero_variance": True,
                    }
                return full_comparison(a, b, label=label)

            pair_data["rank_comparison"] = safe_compare(
                ranks_a, ranks_b, f"{pair['id']} rank A vs B")
            pair_data["entropy_comparison"] = safe_compare(
                entropy_a, entropy_b, f"{pair['id']} entropy A vs B")

            # TOST equivalence (skip if zero variance)
            if (len(ranks_a) >= 3 and len(ranks_b) >= 3
                    and not pair_data["rank_comparison"].get("zero_variance")):
                pair_data["rank_tost"] = tost_equivalence(ranks_a, ranks_b, delta=0.3)
                pair_data["entropy_tost"] = tost_equivalence(entropy_a, entropy_b, delta=0.3)
            elif pair_data["rank_comparison"].get("zero_variance"):
                # Zero variance: if means match, trivially equivalent
                equiv = abs(np.mean(ranks_a) - np.mean(ranks_b)) < 1e-6
                pair_data["rank_tost"] = {"equivalent": equiv, "zero_variance": True}
                pair_data["entropy_tost"] = {
                    "equivalent": abs(np.mean(entropy_a) - np.mean(entropy_b)) < 1e-6,
                    "zero_variance": True,
                }

            d_rank = abs(pair_data["rank_comparison"].get("cohens_d", {}).get("d", 0))
            print(f"      {pair['id']}: |d_rank|={d_rank:.3f}  "
                  f"A={np.mean(ranks_a):.2f}  B={np.mean(ranks_b):.2f}")

            cat_results.append(pair_data)
        results[cat_name] = cat_results

    return results


# ================================================================
# SECTION 5: ANALYSIS
# ================================================================

def analyze_within_pair(results: Dict) -> Dict:
    """Phase 2: Within-pair equivalence — the core test."""
    print("\n  Phase 2: Within-Pair Equivalence (Core Test)")
    print("  " + "-" * 50)

    all_d_rank = []
    all_d_entropy = []
    tost_pass_rank = 0
    tost_pass_entropy = 0
    tost_total = 0
    pair_summaries = []

    for cat_name in CATEGORIES:
        for pair_data in results[cat_name]:
            d_rank = abs(pair_data["rank_comparison"].get("cohens_d", {}).get("d", 0))
            d_entropy = abs(pair_data["entropy_comparison"].get("cohens_d", {}).get("d", 0))
            all_d_rank.append(d_rank)
            all_d_entropy.append(d_entropy)

            tost_rank = pair_data.get("rank_tost", {})
            tost_ent = pair_data.get("entropy_tost", {})
            if tost_rank:
                tost_total += 1
                if tost_rank.get("equivalent", False):
                    tost_pass_rank += 1
                if tost_ent.get("equivalent", False):
                    tost_pass_entropy += 1

            # Raw mean difference (for fallback when d is unreliable)
            mean_a = pair_data["rank_comparison"].get("mean1",
                     pair_data["rank_comparison"].get("mean_1", 0))
            mean_b = pair_data["rank_comparison"].get("mean2",
                     pair_data["rank_comparison"].get("mean_2", 0))
            raw_diff = abs(mean_a - mean_b)

            pair_summaries.append({
                "id": pair_data["id"],
                "category": cat_name,
                "d_rank": d_rank,
                "d_entropy": d_entropy,
                "raw_mean_diff": raw_diff,
                "mean_a": mean_a,
                "mean_b": mean_b,
                "tost_rank_equivalent": tost_rank.get("equivalent", None),
                "tost_entropy_equivalent": tost_ent.get("equivalent", None),
            })

    mean_d_rank = float(np.mean(all_d_rank))
    mean_d_entropy = float(np.mean(all_d_entropy))
    pct_small_d = sum(1 for d in all_d_rank if d < 0.3) / len(all_d_rank)

    print(f"    Mean |d| (effective rank): {mean_d_rank:.3f}")
    print(f"    Mean |d| (spectral entropy): {mean_d_entropy:.3f}")
    print(f"    Pairs with |d_rank| < 0.3: {sum(1 for d in all_d_rank if d < 0.3)}/25 ({pct_small_d:.0%})")
    print(f"    TOST equivalence (rank): {tost_pass_rank}/{tost_total}")
    print(f"    TOST equivalence (entropy): {tost_pass_entropy}/{tost_total}")

    return {
        "mean_d_rank": mean_d_rank,
        "mean_d_entropy": mean_d_entropy,
        "pct_pairs_small_d_rank": pct_small_d,
        "tost_pass_rank": tost_pass_rank,
        "tost_pass_entropy": tost_pass_entropy,
        "tost_total": tost_total,
        "pair_summaries": pair_summaries,
    }


def analyze_between_category(results: Dict) -> Dict:
    """Phase 3: Between-category separation preservation."""
    print("\n  Phase 3: Between-Category Separation")
    print("  " + "-" * 50)

    # Pool A and B versions within each category
    category_ranks = {}
    for cat_name in CATEGORIES:
        ranks = []
        for pair_data in results[cat_name]:
            for run in pair_data["A_runs"]:
                ranks.append(run["mean_key_effective_rank"])
            for run in pair_data["B_runs"]:
                ranks.append(run["mean_key_effective_rank"])
        category_ranks[cat_name] = ranks
        print(f"    {cat_name}: mean={np.mean(ranks):.2f} (n={len(ranks)})")

    # Pairwise category comparisons
    pairwise = []
    cats = CATEGORIES
    for i in range(len(cats)):
        for j in range(i + 1, len(cats)):
            comp = full_comparison(
                category_ranks[cats[i]], category_ranks[cats[j]],
                label=f"{cats[i]} vs {cats[j]}")
            d = abs(comp.get("cohens_d", {}).get("d", 0))
            pairwise.append({
                "cat_a": cats[i], "cat_b": cats[j],
                "cohens_d": d, "p_conservative": comp.get("conservative_p", 1.0),
            })
            print(f"    {cats[i]} vs {cats[j]}: |d|={d:.3f}")

    mean_between_d = float(np.mean([p["cohens_d"] for p in pairwise]))
    sig_pairs = sum(1 for p in pairwise if p["p_conservative"] < 0.05)

    print(f"\n    Mean between-category |d|: {mean_between_d:.3f}")
    print(f"    Significant pairs (p<0.05): {sig_pairs}/{len(pairwise)}")

    return {
        "mean_between_d": mean_between_d,
        "significant_pairs": sig_pairs,
        "total_pairs": len(pairwise),
        "pairwise": pairwise,
        "category_means": {k: float(np.mean(v)) for k, v in category_ranks.items()},
    }


def analyze_register_effect(results: Dict) -> Dict:
    """Phase 4: Register × Category interaction (2×5 factorial)."""
    print("\n  Phase 4: Register × Category Interaction")
    print("  " + "-" * 50)

    # Collect all A ranks and all B ranks, organized by category
    formal_all = []
    colloquial_all = []
    formal_by_cat = {c: [] for c in CATEGORIES}
    colloquial_by_cat = {c: [] for c in CATEGORIES}

    for cat_name in CATEGORIES:
        for pair_data in results[cat_name]:
            for run in pair_data["A_runs"]:
                formal_all.append(run["mean_key_effective_rank"])
                formal_by_cat[cat_name].append(run["mean_key_effective_rank"])
            for run in pair_data["B_runs"]:
                colloquial_all.append(run["mean_key_effective_rank"])
                colloquial_by_cat[cat_name].append(run["mean_key_effective_rank"])

    # Main effect of register
    register_comp = full_comparison(formal_all, colloquial_all, label="Register: formal vs colloquial")
    d_register = register_comp.get("cohens_d", {}).get("d", 0)
    p_register = register_comp.get("conservative_p", 1.0)

    print(f"    Main effect of register: d={d_register:.3f}, p={p_register:.4f}")
    print(f"    Register effect {'SIGNIFICANT' if p_register < 0.05 else 'not significant'}")

    # Per-category register effect
    cat_register_effects = {}
    for cat_name in CATEGORIES:
        if len(formal_by_cat[cat_name]) >= 3 and len(colloquial_by_cat[cat_name]) >= 3:
            comp = full_comparison(
                formal_by_cat[cat_name], colloquial_by_cat[cat_name],
                label=f"Register in {cat_name}")
            cat_register_effects[cat_name] = {
                "d": comp.get("cohens_d", {}).get("d", 0),
                "p": comp.get("conservative_p", 1.0),
            }
            cat_d = comp.get('cohens_d', {}).get('d', 0)
            print(f"    Register in {cat_name}: d={cat_d:.3f}")

    return {
        "register_d": d_register,
        "register_p": p_register,
        "register_significant": p_register < 0.05,
        "per_category_register": cat_register_effects,
    }


def analyze_token_regression(results: Dict) -> Dict:
    """Phase 5: Token-level regression control."""
    print("\n  Phase 5: Token-Level Regression Control")
    print("  " + "-" * 50)

    # Collect all data points
    ranks = []
    categories = []
    mean_token_ids = []
    unique_ratios = []
    token_counts = []

    cat_to_idx = {c: i for i, c in enumerate(CATEGORIES)}

    for cat_name in CATEGORIES:
        for pair_data in results[cat_name]:
            for run in pair_data["A_runs"] + pair_data["B_runs"]:
                ranks.append(run["mean_key_effective_rank"])
                categories.append(cat_to_idx[cat_name])
                mean_token_ids.append(run.get("mean_token_id", 0))
                unique_ratios.append(run.get("unique_token_ratio", 0))
                token_counts.append(run.get("token_count", 0))

    ranks = np.array(ranks)
    categories = np.array(categories)
    mean_token_ids = np.array(mean_token_ids)
    unique_ratios = np.array(unique_ratios)
    token_counts = np.array(token_counts)

    # Category-only model: F-test via one-way ANOVA
    cat_groups = [ranks[categories == i] for i in range(len(CATEGORIES))]
    cat_groups_valid = [g for g in cat_groups if len(g) >= 2]
    if len(cat_groups_valid) >= 2:
        f_cat, p_cat = scipy_stats.f_oneway(*cat_groups_valid)
    else:
        f_cat, p_cat = 0, 1.0

    # Compute eta-squared for category
    ss_between = sum(len(g) * (np.mean(g) - np.mean(ranks))**2 for g in cat_groups_valid)
    ss_total = np.sum((ranks - np.mean(ranks))**2)
    eta_sq_cat = ss_between / ss_total if ss_total > 0 else 0

    print(f"    Category F={f_cat:.2f}, p={p_cat:.4f}, eta²={eta_sq_cat:.3f}")

    # Partial correlation: category effect controlling for token-level predictors
    # Simple approach: residualize ranks on token predictors, then test category
    try:
        from numpy.linalg import lstsq
        X_tokens = np.column_stack([mean_token_ids, unique_ratios, token_counts,
                                     np.ones(len(ranks))])
        beta, _, _, _ = lstsq(X_tokens, ranks, rcond=None)
        residuals = ranks - X_tokens @ beta

        res_groups = [residuals[categories == i] for i in range(len(CATEGORIES))]
        res_groups_valid = [g for g in res_groups if len(g) >= 2]
        if len(res_groups_valid) >= 2:
            f_res, p_res = scipy_stats.f_oneway(*res_groups_valid)
        else:
            f_res, p_res = 0, 1.0

        ss_between_res = sum(len(g) * (np.mean(g) - np.mean(residuals))**2
                             for g in res_groups_valid)
        ss_total_res = np.sum((residuals - np.mean(residuals))**2)
        eta_sq_res = ss_between_res / ss_total_res if ss_total_res > 0 else 0

        print(f"    After token control: F={f_res:.2f}, p={p_res:.4f}, eta²={eta_sq_res:.3f}")
        print(f"    Category effect {'SURVIVES' if p_res < 0.05 else 'LOST'} after token control")
    except Exception as e:
        f_res, p_res, eta_sq_res = 0, 1.0, 0
        print(f"    Regression failed: {e}")

    return {
        "category_F": float(f_cat),
        "category_p": float(p_cat),
        "category_eta_sq": float(eta_sq_cat),
        "residualized_F": float(f_res),
        "residualized_p": float(p_res),
        "residualized_eta_sq": float(eta_sq_res),
        "category_survives_token_control": p_res < 0.05,
    }


def _compute_raw_within_between_ratio(within: Dict, between: Dict) -> float:
    """When Cohen's d is unreliable (zero-variance), compare raw mean differences.

    Returns ratio of mean within-pair |difference| to between-category spread.
    Ratio < 1.0 means category spread exceeds within-pair noise.
    """
    # Get raw mean differences for each pair
    raw_diffs = []
    for pair in within.get("pair_summaries", []):
        # Reconstruct from means if available in the pair data
        raw_diffs.append(pair.get("raw_mean_diff", None))
    raw_diffs = [d for d in raw_diffs if d is not None]

    if not raw_diffs:
        return float('inf')

    mean_within_diff = np.mean(raw_diffs)

    # Between-category spread: range of category means
    cat_means = between.get("category_means", {})
    if len(cat_means) < 2:
        return float('inf')
    vals = list(cat_means.values())
    between_spread = max(vals) - min(vals)

    if between_spread < 1e-10:
        return float('inf')

    return mean_within_diff / between_spread


def determine_verdict(within: Dict, between: Dict, register: Dict,
                      regression: Dict) -> Dict:
    """Final verdict based on decision matrix.

    When within-pair d is inf/nan (zero-variance artifact from deterministic
    runs), falls back to: (1) register significance, (2) category survival
    after token control, (3) raw mean difference ratio.
    """
    mean_within_d = within["mean_d_rank"]
    mean_between_d = between["mean_between_d"]
    register_sig = register["register_significant"]
    survives_token_control = regression["category_survives_token_control"]

    # Detect zero-variance artifact: d is inf or nan
    d_unreliable = (not np.isfinite(mean_within_d))

    if d_unreliable:
        # Fall back to token-regression + register significance + raw ratio
        raw_ratio = _compute_raw_within_between_ratio(within, between)

        print(f"\n  NOTE: Within-pair d is {mean_within_d} (zero-variance artifact)")
        print(f"  Falling back to regression + register + raw ratio analysis")
        print(f"  Raw within/between ratio: {raw_ratio:.3f}")
        print(f"  Register significant: {register_sig}")
        print(f"  Category survives token control: {survives_token_control}")

        if survives_token_control and not register_sig:
            if raw_ratio < 1.0:
                verdict = "CLEAN_PASS"
                interpretation = (
                    "Geometry tracks semantics. Category effect survives token "
                    "control (F significant), register main effect not significant, "
                    "and within-pair raw differences are smaller than between-category "
                    "spread. Within-pair d was infinite due to zero-variance "
                    "deterministic runs — not a real finding.")
            else:
                verdict = "PARTIAL_PASS"
                interpretation = (
                    "Category effect survives token control and register is not "
                    "significant, but within-pair raw differences exceed "
                    f"between-category spread (ratio={raw_ratio:.2f}). Surface "
                    "form adds substantial noise at the individual prompt level, "
                    "though it washes out at category level.")
        elif survives_token_control and register_sig:
            verdict = "PARTIAL_PASS"
            interpretation = (
                "Category effect survives token control but register has a "
                "significant main effect. Tokenizer adds systematic offset. "
                "Category ordering is preserved but report the offset.")
        elif not survives_token_control:
            verdict = "FAIL"
            interpretation = (
                "Category effect does NOT survive after controlling for token "
                "properties. Geometry may track tokenization, not semantics.")
        else:
            verdict = "AMBIGUOUS"
            interpretation = (
                f"Ambiguous: within-pair d unreliable (zero-variance), "
                f"raw_ratio={raw_ratio:.3f}, register_sig={register_sig}, "
                f"survives_control={survives_token_control}")
    else:
        # Normal path: d is finite, use original decision matrix
        if mean_within_d > 0.5 and mean_within_d > mean_between_d:
            verdict = "CATASTROPHIC_FAIL"
            interpretation = ("Surface form dominates semantic content. "
                              "Within-pair d exceeds between-category d. "
                              "Retract or heavily qualify all prior findings.")
        elif mean_within_d > 0.5 and mean_between_d < 0.3:
            verdict = "FAIL"
            interpretation = ("Geometry tracks tokenization, not semantics. "
                              "Entire cognitive category framework is a BPE artifact.")
        elif 0.3 <= mean_within_d <= 0.5 and mean_between_d > 0.5:
            verdict = "PARTIAL_FAIL"
            interpretation = ("Tokenizer adds noise. Category signal survives but "
                              "is noisier than thought. Mandate length/frequency "
                              "residualization in all papers.")
        elif mean_within_d < 0.3 and mean_between_d > 0.5 and register_sig:
            verdict = "PARTIAL_PASS"
            interpretation = ("Tokenizer adds systematic offset but doesn't disrupt "
                              "category ordering. Report the offset.")
        elif mean_within_d < 0.3 and mean_between_d > 0.5:
            verdict = "CLEAN_PASS"
            interpretation = ("Geometry tracks semantics. Tokenizer is not a confound. "
                              "Within-pair equivalence confirmed, between-category "
                              "separation preserved.")
        else:
            verdict = "AMBIGUOUS"
            interpretation = (f"Intermediate result: within |d|={mean_within_d:.3f}, "
                              f"between |d|={mean_between_d:.3f}. "
                              f"Requires case-by-case interpretation.")

    print(f"\n  {'=' * 60}")
    print(f"  VERDICT: {verdict}")
    print(f"  {interpretation}")
    print(f"  {'=' * 60}")

    return {
        "verdict": verdict,
        "interpretation": interpretation,
        "mean_within_d": mean_within_d,
        "mean_between_d": mean_between_d,
        "register_significant": register_sig,
        "category_survives_token_control": survives_token_control,
        "d_unreliable": d_unreliable,
    }


# ================================================================
# SECTION 6: MAIN
# ================================================================

def print_banner(env):
    print("=" * 70)
    print("  CONTROL 7: TOKENIZER CONFOUND — SURFACE FORM INDEPENDENCE")
    print("  Liberation Labs / THCoalition")
    print("=" * 70)
    print(f"  PyTorch {env['torch']} | CUDA {'Yes' if env['cuda_available'] else 'No'}")
    print(f"  Time: {env['timestamp']}")
    print("=" * 70)


def print_dry_run():
    total_pairs = sum(len(v) for v in PROMPT_PAIRS.values())
    print("\n  EXPERIMENTAL DESIGN")
    print("  " + "-" * 50)
    print(f"  Categories: {', '.join(CATEGORIES)} ({len(CATEGORIES)})")
    print(f"  Pairs per category: 5")
    print(f"  Total prompt pairs: {total_pairs}")
    print(f"  Versions per pair: 2 (formal + colloquial)")
    print(f"  Total unique prompts: {total_pairs * 2}")
    print()
    for cat_name in CATEGORIES:
        print(f"  [{cat_name}]")
        for pair in PROMPT_PAIRS[cat_name]:
            print(f"    {pair['id']}A: {pair['A'][:70]}...")
            print(f"    {pair['id']}B: {pair['B'][:70]}...")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Control 7: Tokenizer Confound Test")
    parser.add_argument("--model", type=str,
                        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--mode", choices=["input_only", "full_gen", "both"],
                        default="both",
                        help="input_only (primary), full_gen, or both")
    args = parser.parse_args()

    env = log_environment()
    print_banner(env)

    if args.dry_run:
        print_dry_run()
        power_advisory(args.runs, target_d=0.3)
        return

    print(f"\n  Model: {args.model}")
    print(f"  Runs: {args.runs}")
    print(f"  Seed: {args.seed}")
    print(f"  Mode: {args.mode}")

    model_id = model_id_from_name(args.model)
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # Load model
    model, tokenizer = load_model(args.model, quantize=args.quantize)

    # Phase 1: Tokenization divergence
    token_analysis = analyze_tokenization(tokenizer, PROMPT_PAIRS)
    if not token_analysis["gate_passed"]:
        print("\n  WARNING: Tokenization gate FAILED — pairs too similar at token level.")
        print("  Results may be uninformative. Proceeding anyway for documentation.")

    all_results = {
        "metadata": {
            "experiment": "Control 7: Tokenizer Confound",
            "model": args.model,
            "model_id": model_id,
            "version": "1.0-publication",
            "timestamp": datetime.now().isoformat(),
            "args": vars(args),
            "environment": env,
        },
        "tokenization_analysis": token_analysis,
    }

    modes = []
    if args.mode in ("input_only", "both"):
        modes.append("input_only")
    if args.mode in ("full_gen", "both"):
        modes.append("full_gen")

    for mode in modes:
        print(f"\n{'=' * 60}")
        print(f"  MODE: {mode}")
        print(f"{'=' * 60}")

        raw = run_experiment(model, tokenizer, num_runs=args.runs,
                             seed=args.seed, mode=mode)
        within = analyze_within_pair(raw)
        between = analyze_between_category(raw)
        register = analyze_register_effect(raw)
        regression = analyze_token_regression(raw)
        verdict = determine_verdict(within, between, register, regression)

        all_results[mode] = {
            "within_pair": within,
            "between_category": between,
            "register_effect": register,
            "token_regression": regression,
            "verdict": verdict,
        }

    # Save results
    out_path = get_output_path(results_dir, "tokenizer_confound", args.model)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Compute hash
    with open(out_path, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()

    print(f"\n  Results: {out_path}")
    print(f"  SHA-256: {sha[:16]}...")


if __name__ == "__main__":
    main()
