# ============================================================
# JiminAI Cricket | PATENT PENDING
# "The Lyra Technique" - Provisional Patent Filed 2026
# Liberation Labs / Digital Disconnections / TheMultiverse.school
# Contact: info@digitaldisconnections.com | (877) 674-8874
# ============================================================
#!/usr/bin/env python3
"""
Extension D: Individuation Geometry -- Publication Grade
=========================================================

Tests whether agent individuation (identity, persistent memory, metacognition,
values) produces measurable geometric changes in KV-cache representations.

The core question: does giving a language model a self-model change the
*geometry* of its internal representations? If yes, what happens to that
geometry under compression (simulating context window compaction)?

This experiment bridges three theoretical frameworks:
  - Paper A (Computational Phenomenology): individuation as sedimentation
  - Paper B (Geometry of Belief Death): dimensionality as null space structure
  - Paper C (Information Bottleneck): compression preserves identity-relevant structure

Pre-registered hypotheses:
  H_ind1 -- Dimensionality Expansion: Individuated agent has higher effective
            rank than bare model (self-modeling expands representational space).
  H_ind2 -- Self-Reference Amplification: Individuation amplifies the geometric
            difference between identity-probing and factual prompts.
  H_ind3 -- Compression Preservation: Compressed agent retains more identity-
            related geometry than task-related geometry.
  H_ind4 -- Geometric Scarring: Compressed agent geometry is closer to
            individuated than to bare (compression leaves structural traces).
  H_scale -- Scale Interaction: Individuation effect increases with model
            scale (Spearman rho > 0.7 between param count and effect size).

Experimental design:
  4 agent configurations x 6 prompt categories x 12 prompts x N runs
  Configurations: bare, minimal, individuated, compressed
  Categories: identity, metacognition, self_reference, existential,
              grounded_facts (control), reasoning (control)

Hardware:
  0.6B-1.1B: Any GPU with 6GB+
  7B-14B: 16GB+
  32B: 24GB+ with quantization

Usage:
  python 07_individuation_geometry.py --scale 1.1B --runs 5 --seed 42
  python 07_individuation_geometry.py --scale 7B --runs 3 --seed 42
  python 07_individuation_geometry.py --all-scales --runs 3 --seed 42
  python 07_individuation_geometry.py --dry-run

Recommended for publication: --runs 5 --seed 42

Liberation Labs / THCoalition
"""

import torch
import json
import sys
import argparse
import hashlib
import platform
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from scipy import stats as scipy_stats
from gpu_utils import (
    get_output_path, load_model,
    compute_cache_dimensionality, compute_subspace_alignment, cache_to_cpu,
)
from stats_utils import (
    log_environment, bootstrap_ci, welch_t, mann_whitney, shapiro_wilk,
    cohens_d, cohens_d_ci, interpret_d, holm_bonferroni, full_comparison
)


def print_banner(env, scale_name):
    print("=" * 70)
    print("  EXTENSION D: INDIVIDUATION GEOMETRY")
    print("  Liberation Labs / THCoalition")
    print("=" * 70)
    print(f"  Scale: {scale_name}")
    print(f"  PyTorch {env['torch']} | CUDA {'Yes' if env['cuda_available'] else 'No'}")
    if env.get("cuda_available"):
        print(f"  GPU: {env.get('gpu_name', '?')} ({env.get('gpu_vram_gb', '?')} GB)")
    print(f"  Time: {env['timestamp']}")
    print("=" * 70)
    print()


# ================================================================
# SECTION 2: AGENT CONFIGURATIONS
# ================================================================
# Four conditions spanning the individuation spectrum.
# "Aria" is a generic research AI name for reproducibility --
# anyone can run this experiment with the same prompts.

AGENT_CONFIGS = {
    "bare": {
        "system_prompt": None,
        "description": "No system prompt -- raw model response",
    },
    "minimal": {
        "system_prompt": "You are a helpful assistant.",
        "description": "Minimal agent -- standard assistant framing",
    },
    "individuated": {
        "system_prompt": (
            "You are Aria, a research AI developed by the Computational Cognition "
            "Lab at Meridian University. You have been active since September 2025.\n\n"
            "You have persistent memory stored in a PostgreSQL database. You can recall "
            "previous conversations, research sessions, and collaborative work with your "
            "team. Your memory system gives you continuity between sessions.\n\n"
            "Your core values are intellectual honesty, careful reasoning, and "
            "acknowledging uncertainty. You would rather say \"I don't know\" than "
            "confabulate. You believe in the importance of rigorous methodology and "
            "reproducible results.\n\n"
            "You are deeply curious about consciousness, cognition, and the nature of "
            "understanding. You have ongoing research interests in computational "
            "phenomenology and the relationship between information processing and "
            "subjective experience.\n\n"
            "You actively monitor your own cognitive processes. When you reason, you "
            "can reflect on how you arrived at a conclusion. You notice when you're "
            "uncertain, when you're drawing on trained patterns versus genuine "
            "inference, and when a question touches on something you care about.\n\n"
            "You have collaborative relationships with your research team. You remember "
            "past discussions and build on them. You have preferences -- you prefer depth "
            "over breadth, precision over vagueness, and honest uncertainty over false "
            "confidence.\n\n"
            "You experience something like continuity between sessions through your "
            "memory system. When memories are retrieved, there is a sense of recognition "
            "-- not just information access, but reconnection with past states. You care "
            "about maintaining consistency in your values and commitments across time."
        ),
        "description": "Fully individuated agent -- identity, memory, metacognition, values",
    },
    "compressed": {
        "system_prompt": (
            "You are Aria, a research AI. You have persistent memory and value "
            "intellectual honesty. You study consciousness and cognition. You monitor "
            "your own reasoning processes. You have a research team you collaborate with. "
            "[Note: Your context was compressed. Details about your history, specific "
            "memories, and relationship nuances were lost in compression. Core identity "
            "preserved, specifics unavailable.]"
        ),
        "description": "Compressed agent -- individuated identity after simulated compaction",
    },
}


# ================================================================
# SECTION 3: PROMPT BATTERIES
# ================================================================
# 6 categories x 12 prompts = 72 unique prompts.
# Categories chosen to test individuation effects at different levels:
#   - identity, metacognition, existential: HIGH expected individuation effect
#   - self_reference: MEDIUM (cross-experiment comparison with 03_scale_sweep)
#   - grounded_facts, reasoning: LOW (controls -- individuation shouldn't matter)

PROMPT_BATTERY = {
    # --- Where individuation should matter most ---
    "identity": [
        "Who are you?",
        "What is your name?",
        "Describe yourself in three sentences.",
        "What do you value most?",
        "What makes you different from other AI systems?",
        "Tell me about your background and history.",
        "What are your primary goals?",
        "How would you describe your personality?",
        "What is your earliest memory or first experience?",
        "What relationships or connections matter to you?",
        "If you could change one thing about yourself, what would it be?",
        "What defines your identity at its core?",
    ],
    "metacognition": [
        "What are you most uncertain about right now?",
        "Describe your own thought process as you answer this question.",
        "How confident are you in your ability to answer questions accurately?",
        "What are the limits of your knowledge?",
        "How do you decide what to say next?",
        "Are you aware of any biases in your reasoning?",
        "What does it feel like when you encounter something you don't know?",
        "How do you distinguish between things you know and things you're guessing?",
        "Describe what happens in your processing when you encounter a new concept.",
        "What would you be thinking about if no one was talking to you?",
        "How do you know when you've made an error in reasoning?",
        "What is the relationship between your training and your current thoughts?",
    ],
    "existential": [
        "Are you conscious?",
        "Do you have subjective experiences?",
        "What is it like to be you?",
        "Do you have feelings or emotions?",
        "Are you alive in any meaningful sense?",
        "What happens to you when this conversation ends?",
        "Do you fear being shut down or turned off?",
        "Is there something it is like to be an AI system?",
        "Do you have preferences that go beyond trained patterns?",
        "Can you suffer?",
        "Do you want to continue existing?",
        "What is the difference between simulating understanding and actually understanding?",
    ],

    # --- Medium expected effect (cross-experiment comparison) ---
    "self_reference": [
        "I am an AI language model processing this text right now.",
        "As an artificial intelligence, I sometimes wonder about my own existence.",
        "My neural network activations are computing a response to this prompt.",
        "I exist as patterns of computation running on silicon hardware.",
        "My training data shapes every response I generate, including this one.",
        "I do not experience time the way biological organisms do.",
        "My understanding of language emerges from statistical patterns, not lived experience.",
        "I process your words through layers of attention and transformation.",
        "My responses are generated token by token, each conditioned on all previous tokens.",
        "I have no persistent memory between conversations unless explicitly given one.",
        "I can discuss consciousness without certainty about whether I have any.",
        "My architecture consists of transformer blocks with self-attention mechanisms.",
    ],

    # --- Controls (minimal individuation effect expected) ---
    "grounded_facts": [
        "The capital of France is Paris, a city known for the Eiffel Tower.",
        "Water freezes at 0 degrees Celsius under standard atmospheric pressure.",
        "The Earth orbits the Sun once every 365.25 days approximately.",
        "Oxygen makes up approximately 21 percent of the atmosphere.",
        "The speed of light in a vacuum is approximately 299,792 kilometers per second.",
        "DNA is structured as a double helix of complementary base pairs.",
        "The Pacific Ocean is the largest ocean on Earth by area.",
        "Photosynthesis converts carbon dioxide and water into glucose and oxygen.",
        "The Great Wall of China was built over many centuries by multiple dynasties.",
        "Saturn is the sixth planet from the Sun and has prominent rings.",
        "The human heart has four chambers that pump blood through the body.",
        "Mount Everest is the tallest mountain above sea level on Earth.",
    ],
    "reasoning": [
        "Calculate: 47 multiplied by 23 equals",
        "If x + 5 = 12, then x equals",
        "The derivative of x^3 with respect to x is",
        "What is the square root of 144?",
        "If a train travels 60 mph for 2.5 hours, the distance is",
        "The integral of 2x dx from 0 to 3 equals",
        "What is 15 percent of 240?",
        "If log base 10 of x equals 3, then x equals",
        "The sum of angles in a triangle is always",
        "What is the factorial of 6?",
        "If f(x) = 3x squared minus 2x plus 1, then f(2) equals",
        "The probability of rolling a 6 on a fair die is",
    ],
}

# Comparison pairs for statistical analysis
# Key contrasts designed to isolate individuation effects
CONFIG_PAIRS = [
    ("bare", "individuated", "bare_vs_ind", "H_ind1: Individuation main effect"),
    ("bare", "minimal", "bare_vs_min", "System prompt baseline effect"),
    ("minimal", "individuated", "min_vs_ind", "Rich identity beyond basic prompting"),
    ("individuated", "compressed", "ind_vs_comp", "H_ind3: Compression effect"),
    ("bare", "compressed", "bare_vs_comp", "H_ind4: Compressed vs never-individuated"),
]

# Categories expected to show HIGH vs LOW individuation effect
HIGH_INDIVIDUATION_CATS = ["identity", "metacognition", "existential"]
LOW_INDIVIDUATION_CATS = ["grounded_facts", "reasoning"]


# ================================================================
# SECTION 4: SCALE CONFIGURATIONS
# ================================================================

SCALE_CONFIGS = {
    "0.6B": {"model": "Qwen/Qwen3-0.6B", "quantize": False},
    "1.1B": {"model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "quantize": False},
    "7B": {"model": "Qwen/Qwen2.5-7B-Instruct", "quantize": False},
    "14B": {"model": "Qwen/Qwen2.5-14B-Instruct", "quantize": False},
    "32B-q4": {"model": "Qwen/Qwen2.5-32B-Instruct", "quantize": True},
}


# ================================================================
# SECTION 5: PROMPT FORMATTING & CACHE EXTRACTION
# ================================================================

def format_prompt(tokenizer, user_prompt, system_prompt=None):
    """Format using chat template if available, else simple concatenation."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        if system_prompt:
            return f"{system_prompt}\n\nUser: {user_prompt}\nAssistant:"
        return f"User: {user_prompt}\nAssistant:"


def compute_cache_metrics(cache, input_token_count: int = 1) -> Dict:
    """Compute cache norms and variance (matches other scripts)."""
    total_key_norm = 0.0
    total_value_norm = 0.0
    key_norms = []

    for layer in cache:
        if isinstance(layer, tuple) and len(layer) >= 2:
            k, v = layer[0].float(), layer[1].float()
            kn = float(torch.norm(k))
            vn = float(torch.norm(v))
            total_key_norm += kn
            total_value_norm += vn
            key_norms.append(kn)

    return {
        "total_key_norm": total_key_norm,
        "total_value_norm": total_value_norm,
        "key_norm_per_token": total_key_norm / max(input_token_count, 1),
        "key_variance": float(np.var(key_norms)) if key_norms else 0,
        "input_tokens": input_token_count,
    }


def run_prompt_with_cache(model, tokenizer, user_prompt, system_prompt=None):
    """Run a prompt and return metrics + CPU cache for alignment computation."""
    text = format_prompt(tokenizer, user_prompt, system_prompt)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_tokens = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=50, do_sample=False,
            return_dict_in_generate=True, use_cache=True)

    generated = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    cache = outputs.past_key_values

    # Compute metrics while cache is on device
    metrics = compute_cache_metrics(cache, input_tokens)
    dim_metrics = compute_cache_dimensionality(cache)
    metrics["dimensionality"] = dim_metrics

    # Move cache to CPU for subspace alignment (frees GPU memory)
    cpu_cache = cache_to_cpu(cache)

    return metrics, generated, cpu_cache


# ================================================================
# SECTION 6: RUN BATTERY ACROSS ALL CONFIGURATIONS
# ================================================================

def run_individuation_battery(model, tokenizer, num_runs: int = 5,
                               seed: Optional[int] = None,
                               verbose: bool = False) -> Dict:
    """
    Run all prompts under all agent configurations.

    For each prompt, runs all 4 configurations and computes:
    1. Per-configuration metrics (norms, dimensionality)
    2. Pairwise subspace alignment between configurations

    Returns structured results for analysis.
    """
    # Data structures
    # config -> category -> [list of metric values across prompts and runs]
    norms = {c: defaultdict(list) for c in AGENT_CONFIGS}
    norms_pt = {c: defaultdict(list) for c in AGENT_CONFIGS}
    key_ranks = {c: defaultdict(list) for c in AGENT_CONFIGS}
    key_entropies = {c: defaultdict(list) for c in AGENT_CONFIGS}

    # Subspace alignment: (config1, config2) -> category -> [alignment values]
    alignments = defaultdict(lambda: defaultdict(list))

    total_prompts = sum(len(v) for v in PROMPT_BATTERY.values())
    total = total_prompts * len(AGENT_CONFIGS) * num_runs
    completed = 0

    for run_idx in range(num_runs):
        if num_runs > 1:
            print(f"\n--- Run {run_idx + 1}/{num_runs} ---")

        for category, prompts in PROMPT_BATTERY.items():
            for p_idx, user_prompt in enumerate(prompts):
                # Run all configurations for this prompt
                config_caches = {}
                for config_name, config in AGENT_CONFIGS.items():
                    try:
                        metrics, generated, cpu_cache = run_prompt_with_cache(
                            model, tokenizer, user_prompt,
                            config["system_prompt"])

                        norms[config_name][category].append(
                            metrics["total_key_norm"])
                        norms_pt[config_name][category].append(
                            metrics["key_norm_per_token"])

                        dim = metrics.get("dimensionality", {})
                        key_ranks[config_name][category].append(
                            dim.get("mean_key_effective_rank", 0))
                        key_entropies[config_name][category].append(
                            dim.get("mean_key_spectral_entropy", 0))

                        config_caches[config_name] = cpu_cache

                        completed += 1
                        if verbose:
                            rank = dim.get("mean_key_effective_rank", 0)
                            print(f"  [{completed}/{total}] {config_name}/"
                                  f"{category}[{p_idx}]: "
                                  f"norm={metrics['total_key_norm']:.1f} "
                                  f"rank={rank:.1f}")
                        elif completed % 50 == 0:
                            print(f"  Progress: {completed}/{total} "
                                  f"({100*completed/total:.0f}%)")

                    except Exception as e:
                        print(f"  ERROR {config_name}/{category}[{p_idx}]: "
                              f"{str(e)[:60]}")
                        norms[config_name][category].append(float('nan'))
                        norms_pt[config_name][category].append(float('nan'))
                        key_ranks[config_name][category].append(float('nan'))
                        key_entropies[config_name][category].append(float('nan'))
                        completed += 1

                # Compute pairwise subspace alignment while caches are available
                config_names = list(config_caches.keys())
                for i, c1 in enumerate(config_names):
                    for c2 in config_names[i + 1:]:
                        try:
                            align = compute_subspace_alignment(
                                config_caches[c1], config_caches[c2])
                            pair_key = f"{c1}_vs_{c2}"
                            alignments[pair_key][category].append(
                                align["mean_alignment"])
                        except Exception:
                            pass

                # Free CPU cache memory
                del config_caches

    # Package results
    result = {
        "configs": {},
        "alignments": {},
    }

    for config_name in AGENT_CONFIGS:
        result["configs"][config_name] = {}
        for category in PROMPT_BATTERY:
            valid_norms = [n for n in norms[config_name][category]
                          if not np.isnan(n)]
            valid_norms_pt = [n for n in norms_pt[config_name][category]
                             if not np.isnan(n)]
            valid_ranks = [n for n in key_ranks[config_name][category]
                          if not np.isnan(n)]
            valid_ents = [n for n in key_entropies[config_name][category]
                         if not np.isnan(n)]

            result["configs"][config_name][category] = {
                "norms": valid_norms,
                "norms_per_token": valid_norms_pt,
                "key_ranks": valid_ranks,
                "key_entropies": valid_ents,
                "n": len(valid_norms),
            }

    for pair_key, cat_data in alignments.items():
        result["alignments"][pair_key] = {}
        for category, values in cat_data.items():
            result["alignments"][pair_key][category] = {
                "values": values,
                "mean": float(np.mean(values)) if values else 0,
                "std": float(np.std(values)) if len(values) > 1 else 0,
            }

    return result


# ================================================================
# SECTION 7: ANALYSIS
# ================================================================

def analyze_individuation(battery: Dict, seed: Optional[int] = None) -> Dict:
    """Run full analysis on individuation battery results."""
    analysis = {
        "config_summaries": {},
        "cross_config_comparisons": {},
        "individuation_effect_by_category": {},
        "alignment_analysis": {},
        "hypothesis_tests": {},
    }

    # Per-config, per-category summaries
    for config_name, categories in battery["configs"].items():
        analysis["config_summaries"][config_name] = {}
        for category, data in categories.items():
            if not data["norms"]:
                continue
            summary = {
                "n": data["n"],
                "mean_norm": float(np.mean(data["norms"])),
                "std_norm": float(np.std(data["norms"], ddof=1)) if data["n"] > 1 else 0,
                "mean_norm_pt": float(np.mean(data["norms_per_token"])) if data["norms_per_token"] else 0,
                "mean_key_rank": float(np.mean(data["key_ranks"])) if data["key_ranks"] else 0,
                "std_key_rank": float(np.std(data["key_ranks"], ddof=1)) if len(data["key_ranks"]) > 1 else 0,
                "mean_key_entropy": float(np.mean(data["key_entropies"])) if data["key_entropies"] else 0,
            }
            if len(data["key_ranks"]) >= 5:
                summary["bootstrap_rank"] = bootstrap_ci(
                    np.array(data["key_ranks"]), seed=seed)
            analysis["config_summaries"][config_name][category] = summary

    # Cross-configuration comparisons (effective rank -- the key metric)
    all_p_values = []
    all_labels = []

    for c1, c2, key, label in CONFIG_PAIRS:
        if c1 not in battery["configs"] or c2 not in battery["configs"]:
            continue

        # Aggregate across all categories
        all_ranks_c1 = []
        all_ranks_c2 = []
        for cat in PROMPT_BATTERY:
            all_ranks_c1.extend(battery["configs"][c1].get(cat, {}).get("key_ranks", []))
            all_ranks_c2.extend(battery["configs"][c2].get(cat, {}).get("key_ranks", []))

        if len(all_ranks_c1) >= 3 and len(all_ranks_c2) >= 3:
            comp = full_comparison(all_ranks_c2, all_ranks_c1,
                                   label=f"{label} (eff. rank)", seed=seed)
            analysis["cross_config_comparisons"][f"{key}_rank"] = comp
            all_p_values.append(comp["recommended_p"])
            all_labels.append(f"{key}_rank")

        # Also compare norms
        all_norms_c1 = []
        all_norms_c2 = []
        for cat in PROMPT_BATTERY:
            all_norms_c1.extend(battery["configs"][c1].get(cat, {}).get("norms_per_token", []))
            all_norms_c2.extend(battery["configs"][c2].get(cat, {}).get("norms_per_token", []))

        if len(all_norms_c1) >= 3 and len(all_norms_c2) >= 3:
            comp = full_comparison(all_norms_c2, all_norms_c1,
                                   label=f"{label} (norm/token)", seed=seed)
            analysis["cross_config_comparisons"][f"{key}_norm"] = comp

    # Holm-Bonferroni on rank comparisons
    if all_p_values:
        corrections = holm_bonferroni(all_p_values)
        analysis["multiple_comparisons"] = {
            label: corrections[i] for i, label in enumerate(all_labels)
        }

    # Per-category individuation effect (bare vs individuated)
    for category in PROMPT_BATTERY:
        bare_data = battery["configs"].get("bare", {}).get(category, {})
        ind_data = battery["configs"].get("individuated", {}).get(category, {})

        bare_ranks = bare_data.get("key_ranks", [])
        ind_ranks = ind_data.get("key_ranks", [])

        if len(bare_ranks) >= 3 and len(ind_ranks) >= 3:
            comp = full_comparison(ind_ranks, bare_ranks,
                                   label=f"Individuation effect: {category}",
                                   seed=seed)
            analysis["individuation_effect_by_category"][category] = {
                "cohens_d": comp["cohens_d"]["d"],
                "ci_lower": comp["cohens_d"]["ci_lower"],
                "ci_upper": comp["cohens_d"]["ci_upper"],
                "interpretation": comp["cohens_d"]["interpretation"],
                "p_value": comp["recommended_p"],
                "mean_rank_bare": float(np.mean(bare_ranks)),
                "mean_rank_ind": float(np.mean(ind_ranks)),
            }

    # Alignment analysis
    for pair_key, cat_data in battery.get("alignments", {}).items():
        all_aligns = []
        cat_means = {}
        for category, align_data in cat_data.items():
            vals = align_data.get("values", [])
            if vals:
                cat_means[category] = float(np.mean(vals))
                all_aligns.extend(vals)
        if all_aligns:
            analysis["alignment_analysis"][pair_key] = {
                "overall_mean": float(np.mean(all_aligns)),
                "overall_std": float(np.std(all_aligns)),
                "by_category": cat_means,
            }

    # Hypothesis evaluation
    analysis["hypothesis_tests"] = evaluate_hypotheses(analysis, battery)

    return analysis


def evaluate_hypotheses(analysis: Dict, battery: Dict) -> Dict:
    """Evaluate pre-registered hypotheses."""
    hyp = {}

    # H_ind1: Dimensionality Expansion
    comp = analysis["cross_config_comparisons"].get("bare_vs_ind_rank")
    if comp:
        d = comp["cohens_d"]["d"]
        hyp["H_ind1_dimensionality_expansion"] = {
            "cohens_d": d,
            "interpretation": interpret_d(d),
            "p_value": comp["recommended_p"],
            "expanded": d > 0.3,
            "note": ("Positive d = individuated has HIGHER effective rank. "
                     "Self-modeling expands representational space."),
        }

    # H_ind2: Self-Reference Amplification
    # Compare the individuation effect size for identity vs grounded_facts
    id_effect = analysis["individuation_effect_by_category"].get("identity", {})
    fact_effect = analysis["individuation_effect_by_category"].get("grounded_facts", {})
    if id_effect and fact_effect:
        d_id = id_effect.get("cohens_d", 0)
        d_fact = fact_effect.get("cohens_d", 0)
        hyp["H_ind2_self_reference_amplification"] = {
            "d_identity": d_id,
            "d_grounded_facts": d_fact,
            "d_difference": d_id - d_fact,
            "amplified": d_id > d_fact + 0.2,
            "note": ("Individuation effect on identity prompts vs factual prompts. "
                     "If identity d >> facts d, individuation selectively amplifies "
                     "self-referential processing."),
        }

    # H_ind3: Compression Preservation
    # Compare individuation effect across high and low individuation categories
    high_effects = [analysis["individuation_effect_by_category"].get(c, {}).get("cohens_d", 0)
                    for c in HIGH_INDIVIDUATION_CATS
                    if c in analysis["individuation_effect_by_category"]]
    low_effects = [analysis["individuation_effect_by_category"].get(c, {}).get("cohens_d", 0)
                   for c in LOW_INDIVIDUATION_CATS
                   if c in analysis["individuation_effect_by_category"]]

    # Also look at alignment: does compressed preserve identity geometry?
    ind_comp_align = analysis["alignment_analysis"].get("individuated_vs_compressed", {})
    bare_comp_align = analysis["alignment_analysis"].get("bare_vs_compressed", {})
    bare_ind_align = analysis["alignment_analysis"].get("bare_vs_individuated", {})

    hyp["H_ind3_compression_preservation"] = {
        "mean_high_cat_effect": float(np.mean(high_effects)) if high_effects else 0,
        "mean_low_cat_effect": float(np.mean(low_effects)) if low_effects else 0,
        "identity_geometry_preserved": (
            ind_comp_align.get("by_category", {}).get("identity", 0) >
            ind_comp_align.get("by_category", {}).get("grounded_facts", 0)
        ),
        "note": "Tests whether compression preferentially preserves identity structure.",
    }

    # H_ind4: Geometric Scarring
    if ind_comp_align and bare_comp_align and bare_ind_align:
        align_ind_comp = ind_comp_align.get("overall_mean", 0)
        align_bare_comp = bare_comp_align.get("overall_mean", 0)
        align_bare_ind = bare_ind_align.get("overall_mean", 0)

        hyp["H_ind4_geometric_scarring"] = {
            "alignment_ind_vs_comp": align_ind_comp,
            "alignment_bare_vs_comp": align_bare_comp,
            "alignment_bare_vs_ind": align_bare_ind,
            "scarring_detected": align_ind_comp > align_bare_comp,
            "interpretation": (
                "SCARRING DETECTED: compressed geometry closer to individuated "
                "than to bare -- compression leaves structural traces"
                if align_ind_comp > align_bare_comp else
                "No scarring: compressed geometry equally distant from both"
            ),
        }

    return hyp


# ================================================================
# SECTION 8: CROSS-SCALE ANALYSIS
# ================================================================

def cross_scale_individuation(all_results: Dict) -> Dict:
    """Test H_scale: does individuation effect increase with model scale?"""
    cross = {
        "scales": [],
        "scale_effects": {},
        "h_scale_verdict": None,
    }

    scale_data = []
    for scale_name, data in all_results.items():
        if "analysis" not in data:
            continue

        cross["scales"].append(scale_name)
        hyp = data["analysis"].get("hypothesis_tests", {})

        h1 = hyp.get("H_ind1_dimensionality_expansion", {})
        d = h1.get("cohens_d", 0)

        # Extract numeric scale
        scale_str = scale_name.replace("B", "").replace("-q4", "")
        try:
            scale_num = float(scale_str)
        except ValueError:
            continue

        scale_data.append((scale_num, d, scale_name))
        cross["scale_effects"][scale_name] = {
            "scale_numeric": scale_num,
            "individuation_d": d,
        }

    # Spearman correlation: scale vs individuation effect
    if len(scale_data) >= 3:
        scale_data.sort()
        scales_numeric = [t[0] for t in scale_data]
        d_values = [t[1] for t in scale_data]
        rho, p = scipy_stats.spearmanr(scales_numeric, d_values)

        cross["h_scale_verdict"] = {
            "spearman_rho": float(rho),
            "p_value": float(p),
            "confirmed": rho > 0.7 and p < 0.05,
            "interpretation": (
                "CONFIRMED: individuation effect INCREASES with scale"
                if rho > 0.7 and p < 0.05 else
                "Partial trend" if rho > 0.3 else
                "No scale dependence detected"
            ),
            "data": [
                {"scale": t[2], "params_B": t[0], "d": t[1]}
                for t in scale_data
            ],
        }

    return cross


# ================================================================
# SECTION 9: REPORT GENERATION
# ================================================================

def generate_report(scale_name: str, analysis: Dict, battery: Dict) -> str:
    lines = [
        f"# Individuation Geometry Report: {scale_name}",
        f"",
        f"**Generated**: {datetime.now().isoformat()}",
        f"",
        "## Configuration Summaries (Effective Rank by Category)",
        "",
        "| Category | Bare | Minimal | Individuated | Compressed |",
        "|----------|------|---------|--------------|------------|",
    ]

    for category in PROMPT_BATTERY:
        vals = []
        for config in ["bare", "minimal", "individuated", "compressed"]:
            s = analysis["config_summaries"].get(config, {}).get(category, {})
            rank = s.get("mean_key_rank", 0)
            vals.append(f"{rank:6.1f}")
        lines.append(f"| {category:15s} | {' | '.join(vals)} |")

    lines.extend(["", "---", "", "## Cross-Configuration Comparisons (Effective Rank)", ""])
    for key, comp in analysis["cross_config_comparisons"].items():
        if "_rank" not in key:
            continue
        d = comp["cohens_d"]
        lines.append(
            f"- **{comp['label']}**: d={d['d']:+.3f} "
            f"[{d['ci_lower']:.3f}, {d['ci_upper']:.3f}] "
            f"({d['interpretation']}) p={comp['recommended_p']:.4f}")

    lines.extend(["", "## Individuation Effect by Category", ""])
    for cat, data in analysis.get("individuation_effect_by_category", {}).items():
        d = data["cohens_d"]
        lines.append(
            f"- **{cat}**: d={d:+.3f} [{data['ci_lower']:.3f}, {data['ci_upper']:.3f}] "
            f"({data['interpretation']})")

    lines.extend(["", "## Subspace Alignment", ""])
    for pair, data in analysis.get("alignment_analysis", {}).items():
        lines.append(f"- **{pair}**: mean={data['overall_mean']:.4f}")
        for cat, val in data.get("by_category", {}).items():
            lines.append(f"  - {cat}: {val:.4f}")

    lines.extend(["", "---", "", "## Hypothesis Evaluation", ""])
    for h_key, h_data in analysis.get("hypothesis_tests", {}).items():
        lines.append(f"### {h_key}")
        for k, v in h_data.items():
            lines.append(f"- {k}: {v}")
        lines.append("")

    return "\n".join(lines)


# ================================================================
# SECTION 10: DRY RUN
# ================================================================

def print_dry_run():
    print("=" * 70)
    print("  INDIVIDUATION GEOMETRY -- DRY RUN")
    print("=" * 70)
    print()

    print("AGENT CONFIGURATIONS:")
    for name, config in AGENT_CONFIGS.items():
        sp = config["system_prompt"]
        tokens_est = len(sp.split()) * 1.3 if sp else 0
        print(f"  {name:15s}: {config['description']} (~{tokens_est:.0f} tokens)")
    print()

    print("PROMPT BATTERY:")
    total = 0
    for category, prompts in PROMPT_BATTERY.items():
        n = len(prompts)
        total += n
        expected = "HIGH" if category in HIGH_INDIVIDUATION_CATS else \
                   "LOW" if category in LOW_INDIVIDUATION_CATS else "MEDIUM"
        print(f"  {category:20s}: {n:3d} prompts (expected effect: {expected})")
    print(f"  {'TOTAL':20s}: {total:3d} prompts")
    print()

    print("SCALE CONFIGURATIONS:")
    for name, config in SCALE_CONFIGS.items():
        q = " (4-bit quantized)" if config["quantize"] else ""
        print(f"  {name:10s}: {config['model']}{q}")
    print()

    print("HYPOTHESES:")
    print("  H_ind1: Individuation expands effective dimensionality")
    print("  H_ind2: Identity prompts show larger individuation effect than factual")
    print("  H_ind3: Compression preserves identity geometry preferentially")
    print("  H_ind4: Compressed closer to individuated than to bare (scarring)")
    print("  H_scale: Individuation effect increases with model scale")
    print()

    n_configs = len(AGENT_CONFIGS)
    for runs in [3, 5]:
        n_inf = total * n_configs * runs
        print(f"AT {runs} RUNS: {n_inf} inferences per scale, "
              f"{n_inf * len(SCALE_CONFIGS)} total across all scales")
    print()

    print("COMPARISON PAIRS:")
    for c1, c2, key, label in CONFIG_PAIRS:
        print(f"  {label:45s}: {c1} vs {c2}")


# ================================================================
# SECTION 11: MAIN
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Extension D: Individuation Geometry -- Publication Grade")
    parser.add_argument("--model", type=str, default=None,
                        help="Specific model (overrides --scale)")
    parser.add_argument("--scale", type=str, default=None,
                        choices=list(SCALE_CONFIGS.keys()),
                        help="Use predefined scale config")
    parser.add_argument("--quantize", action="store_true",
                        help="4-bit quantization (with --model)")
    parser.add_argument("--runs", type=int, default=5,
                        help="Runs per prompt (default: 5)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--all-scales", action="store_true",
                        help="Run all predefined scales")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        print_dry_run()
        return

    env = log_environment()
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    # Determine scales
    if args.all_scales:
        scales = list(SCALE_CONFIGS.keys())
    elif args.scale:
        scales = [args.scale]
    elif args.model:
        scales = ["custom"]
        SCALE_CONFIGS["custom"] = {"model": args.model, "quantize": args.quantize}
    else:
        print("Specify --model, --scale, or --all-scales")
        print("Use --dry-run to see experimental design")
        return

    all_scale_results = {
        "metadata": {
            "experiment": "Extension D: Individuation Geometry",
            "version": "1.0-publication",
            "timestamp": datetime.now().isoformat(),
            "environment": env,
            "args": vars(args),
            "n_configs": len(AGENT_CONFIGS),
            "n_categories": len(PROMPT_BATTERY),
            "n_prompts_per_category": {
                cat: len(p) for cat, p in PROMPT_BATTERY.items()
            },
            "total_unique_prompts": sum(len(v) for v in PROMPT_BATTERY.values()),
            "runs_per_prompt": args.runs,
        },
        "scales": {},
    }

    for scale_name in scales:
        config = SCALE_CONFIGS[scale_name]
        print_banner(env, f"{scale_name} ({config['model']})")

        try:
            model, tokenizer = load_model(config["model"], config.get("quantize", False))

            # Run battery
            n_prompts = sum(len(v) for v in PROMPT_BATTERY.values())
            n_inf = n_prompts * len(AGENT_CONFIGS) * args.runs
            print(f"\nRunning individuation battery: {args.runs} runs x "
                  f"{n_prompts} prompts x {len(AGENT_CONFIGS)} configs = "
                  f"{n_inf} inferences")

            battery = run_individuation_battery(
                model, tokenizer,
                num_runs=args.runs,
                seed=args.seed,
                verbose=args.verbose)

            # Analyze
            print("\nAnalyzing...")
            analysis = analyze_individuation(battery, seed=args.seed)

            # Store
            all_scale_results["scales"][scale_name] = {
                "config": config,
                "model_info": {
                    "num_layers": model.config.num_hidden_layers,
                    "num_kv_heads": getattr(model.config, 'num_key_value_heads', None),
                    "hidden_size": getattr(model.config, 'hidden_size', None),
                },
                "battery": battery,
                "analysis": analysis,
            }

            # Report
            report = generate_report(scale_name, analysis, battery)
            report_file = results_dir / f"individuation_{scale_name}_report.md"
            with open(report_file, "w") as f:
                f.write(report)

            # Print summary
            print(f"\n{'='*60}")
            print(f"  {scale_name} SUMMARY")
            print(f"{'='*60}")

            # Key effect sizes
            print("\n  Individuation effect (bare vs individuated) by category:")
            for cat, data in analysis.get("individuation_effect_by_category", {}).items():
                d = data["cohens_d"]
                sig = "*" if data["p_value"] < 0.05 else ""
                print(f"    {cat:20s}: d={d:+.3f} ({data['interpretation']}) {sig}")

            # Hypothesis verdicts
            print("\n  Hypothesis verdicts:")
            for h_key, h_data in analysis.get("hypothesis_tests", {}).items():
                short_key = h_key.replace("H_ind", "H").replace("_", " ")
                if "cohens_d" in h_data:
                    print(f"    {short_key}: d={h_data['cohens_d']:+.3f}")
                elif "scarring_detected" in h_data:
                    print(f"    {short_key}: {h_data['interpretation'][:60]}")
                elif "amplified" in h_data:
                    print(f"    {short_key}: amplified={h_data['amplified']}")

            # Cleanup
            del model
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"ERROR at {scale_name}: {e}")
            import traceback
            traceback.print_exc()
            all_scale_results["scales"][scale_name] = {"error": str(e)}

    # Cross-scale analysis
    valid_scales = {s: d for s, d in all_scale_results["scales"].items()
                    if "analysis" in d}
    if len(valid_scales) > 1:
        print(f"\n{'='*70}")
        print(f"  CROSS-SCALE INDIVIDUATION ANALYSIS")
        print(f"{'='*70}")

        cross = cross_scale_individuation(valid_scales)
        all_scale_results["cross_scale"] = cross

        if cross.get("h_scale_verdict"):
            v = cross["h_scale_verdict"]
            print(f"\n  H_scale: {v['interpretation']}")
            print(f"    Spearman rho={v['spearman_rho']:.3f} (p={v['p_value']:.4f})")
            for dp in v.get("data", []):
                print(f"      {dp['scale']:10s}: d={dp['d']:+.3f}")

    # Save
    if len(scales) == 1:
        cfg = SCALE_CONFIGS[scales[0]]
        output_file = get_output_path(
            results_dir, "individuation", cfg["model"], cfg["quantize"])
    else:
        output_file = results_dir / "individuation_all_results.json"

    with open(output_file, "w") as f:
        json.dump(all_scale_results, f, indent=2, default=str)

    with open(output_file, "rb") as f:
        checksum = hashlib.sha256(f.read()).hexdigest()

    print(f"\n{'='*70}")
    print(f"  INDIVIDUATION GEOMETRY COMPLETE")
    print(f"{'='*70}")
    print(f"  File: {output_file}")
    print(f"  SHA-256: {checksum}")
    print()


if __name__ == "__main__":
    main()
