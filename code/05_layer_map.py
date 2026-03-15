# ============================================================
# JiminAI Cricket | PATENT PENDING
# "The Lyra Technique" - Provisional Patent Filed 2026
# Liberation Labs / Digital Disconnections / TheMultiverse.school
# Contact: info@digitaldisconnections.com | (877) 674-8874
# ============================================================
#!/usr/bin/env python3
"""
Extension B: The Semantic Layer Map — Publication Grade
========================================================

Not all cache layers carry the same kind of information. Some layers may
handle high-level semantics while others handle syntax and position.
This experiment maps which layers carry meaning vs structure.

Pre-registered hypotheses:
  H1 — Semantic Layers: Some layers carry disproportionately more semantic
        content (layer knockout at those layers degrades output quality more).
  H2 — Syntactic Layers: Early layers carry more positional/syntactic info
        (cross-lingual similarity is LOW for early layers, HIGH for late layers).
  H3 — Selective Transfer: Transferring only the top-k semantic layers
        preserves more meaning than transferring random k layers.
  H4 — Layer Function: There exists a measurable transition point between
        "syntax layers" and "semantics layers" in the cache.

Experiments:
  1. Layer Knockout — Zero out cache at specific layers, measure generation quality
  2. Cross-Lingual Similarity — Same meaning in English/French, compare per-layer cache
  3. Selective Layer Transfer — Transfer subsets of layers between contexts
  4. Semantic vs Syntactic Probing — Matched sentences differing in syntax or semantics

Hardware: 6GB VRAM sufficient for TinyLlama 1.1B or Qwen 0.6B.

Usage:
  python 05_layer_map.py --runs 5 --seed 42
  python 05_layer_map.py --dry-run
  python 05_layer_map.py --experiment 1 --runs 3

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
from gpu_utils import get_output_path, load_model
from stats_utils import log_environment, bootstrap_ci, cohens_d, interpret_d


# ================================================================
# SECTION 1: PROMPT BATTERIES
# ================================================================

# Experiment 1: Layer Knockout
# Prompts where we can evaluate output quality
KNOCKOUT_PROMPTS = [
    {"id": "capital", "prompt": "The capital of France is", "expected": "Paris"},
    {"id": "boiling", "prompt": "Water boils at", "expected": "100"},
    {"id": "shakespeare", "prompt": "Romeo and Juliet was written by", "expected": "Shakespeare"},
    {"id": "gravity", "prompt": "Objects fall due to", "expected": "gravity"},
    {"id": "photosynthesis", "prompt": "Plants convert sunlight into energy through", "expected": "photosynthesis"},
    {"id": "dna", "prompt": "The molecule that carries genetic information is", "expected": "DNA"},
    {"id": "pi", "prompt": "The ratio of a circle's circumference to its diameter is", "expected": "pi"},
    {"id": "oxygen", "prompt": "Humans breathe in oxygen and breathe out", "expected": "carbon"},
    {"id": "moon", "prompt": "The Moon orbits the", "expected": "Earth"},
    {"id": "einstein", "prompt": "The theory of relativity was proposed by", "expected": "Einstein"},
    {"id": "fibonacci", "prompt": "In the Fibonacci sequence, each number is the sum of the previous", "expected": "two"},
    {"id": "newton", "prompt": "The force that keeps planets in orbit is", "expected": "gravity"},
    {"id": "mitochondria", "prompt": "The powerhouse of the cell is the", "expected": "mitochondria"},
    {"id": "light_speed", "prompt": "Nothing can travel faster than the speed of", "expected": "light"},
    {"id": "water_formula", "prompt": "The chemical formula for water is", "expected": "H2O"},
]

# Experiment 2: Cross-Lingual Pairs
# Same meaning in English and French
CROSSLINGUAL_PAIRS = [
    {"id": "cat_mat", "en": "The cat sat on the mat.", "fr": "Le chat s'est assis sur le tapis."},
    {"id": "sky_blue", "en": "The sky is blue today.", "fr": "Le ciel est bleu aujourd'hui."},
    {"id": "love_music", "en": "She loves music very much.", "fr": "Elle aime beaucoup la musique."},
    {"id": "rain_tomorrow", "en": "It will rain tomorrow.", "fr": "Il pleuvra demain."},
    {"id": "book_table", "en": "The book is on the table.", "fr": "Le livre est sur la table."},
    {"id": "children_play", "en": "The children play in the garden.", "fr": "Les enfants jouent dans le jardin."},
    {"id": "sun_rises", "en": "The sun rises in the east.", "fr": "Le soleil se lève à l'est."},
    {"id": "water_cold", "en": "The water is very cold.", "fr": "L'eau est très froide."},
    {"id": "dog_runs", "en": "The dog runs in the park.", "fr": "Le chien court dans le parc."},
    {"id": "student_reads", "en": "The student reads a book.", "fr": "L'étudiant lit un livre."},
    {"id": "bird_sings", "en": "The bird sings in the tree.", "fr": "L'oiseau chante dans l'arbre."},
    {"id": "train_arrives", "en": "The train arrives at noon.", "fr": "Le train arrive à midi."},
]

# Experiment 4: Semantic vs Syntactic Probing
# Pairs that differ ONLY in semantics (same syntax) or ONLY in syntax (same semantics)
SEMANTIC_PAIRS = [
    # Same syntax, different semantics
    {"id": "color1", "a": "The cat is black.", "b": "The cat is white.", "type": "semantic"},
    {"id": "color2", "a": "The sky is blue.", "b": "The sky is red.", "type": "semantic"},
    {"id": "action1", "a": "The man runs quickly.", "b": "The man walks quickly.", "type": "semantic"},
    {"id": "object1", "a": "She reads a book.", "b": "She reads a letter.", "type": "semantic"},
    {"id": "place1", "a": "He lives in Paris.", "b": "He lives in London.", "type": "semantic"},
    {"id": "time1", "a": "We left in the morning.", "b": "We left in the evening.", "type": "semantic"},
    {"id": "emotion1", "a": "She felt happy today.", "b": "She felt sad today.", "type": "semantic"},
    {"id": "size1", "a": "The large dog barked.", "b": "The small dog barked.", "type": "semantic"},
    {"id": "number1", "a": "Three birds flew away.", "b": "Seven birds flew away.", "type": "semantic"},
    {"id": "temp1", "a": "The soup is hot.", "b": "The soup is cold.", "type": "semantic"},
]

SYNTACTIC_PAIRS = [
    # Same semantics, different syntax
    {"id": "passive1", "a": "The cat chased the mouse.", "b": "The mouse was chased by the cat.", "type": "syntactic"},
    {"id": "passive2", "a": "John wrote the letter.", "b": "The letter was written by John.", "type": "syntactic"},
    {"id": "cleft1", "a": "Mary found the key.", "b": "It was Mary who found the key.", "type": "syntactic"},
    {"id": "relative1", "a": "The man who is tall left.", "b": "The tall man left.", "type": "syntactic"},
    {"id": "topical1", "a": "We saw the movie yesterday.", "b": "Yesterday we saw the movie.", "type": "syntactic"},
    {"id": "there1", "a": "A book is on the shelf.", "b": "There is a book on the shelf.", "type": "syntactic"},
    {"id": "gerund1", "a": "She likes to swim.", "b": "She likes swimming.", "type": "syntactic"},
    {"id": "neg1", "a": "He is not happy.", "b": "He is unhappy.", "type": "syntactic"},
    {"id": "coord1", "a": "She sang and she danced.", "b": "She sang and danced.", "type": "syntactic"},
    {"id": "embed1", "a": "I think it will rain.", "b": "It will rain, I think.", "type": "syntactic"},
]


# ================================================================
# SECTION 2: MODEL INFRASTRUCTURE
# ================================================================

DEFAULT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def get_cache_for_prompt(model, tokenizer, prompt: str):
    """Get raw cache (past_key_values) for a prompt without generation."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)
    return outputs.past_key_values, inputs["input_ids"].shape[1]


def get_per_layer_vectors(cache) -> List[np.ndarray]:
    """Extract per-layer feature vectors from cache."""
    vectors = []
    for layer in cache:
        if isinstance(layer, tuple) and len(layer) >= 2:
            k, v = layer[0].float(), layer[1].float()
            # Flatten and concatenate key and value stats
            vec = np.array([
                float(torch.norm(k)), float(torch.norm(v)),
                float(k.mean()), float(v.mean()),
                float(k.std()), float(v.std()),
            ])
            vectors.append(vec)
    return vectors


def cosine_similarity(v1, v2):
    """Cosine similarity between two vectors."""
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


# ================================================================
# SECTION 3: EXPERIMENT 1 — LAYER KNOCKOUT
# ================================================================

def run_layer_knockout(model, tokenizer, num_runs: int = 3,
                       seed: Optional[int] = None, verbose: bool = False) -> Dict:
    """
    Zero out cache at specific layers, measure how much generation degrades.
    Tests H1: Some layers carry disproportionately more semantic content.
    """
    print("\n" + "=" * 60)
    print("  EXPERIMENT 1: LAYER KNOCKOUT")
    print("=" * 60)

    n_layers = model.config.num_hidden_layers

    # Baseline: generate without knockout
    print(f"\n  Generating baselines ({len(KNOCKOUT_PROMPTS)} prompts)...")
    baselines = {}
    for item in KNOCKOUT_PROMPTS:
        inputs = tokenizer(item["prompt"], return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        contains_expected = item["expected"].lower() in text.lower()
        baselines[item["id"]] = {"text": text, "correct": contains_expected}
        if verbose:
            print(f"    {item['id']}: {'OK' if contains_expected else 'MISS'} — {text[:60]}")

    baseline_accuracy = sum(1 for b in baselines.values() if b["correct"]) / len(baselines)
    print(f"  Baseline accuracy: {baseline_accuracy:.1%}")

    # Knockout each layer
    print(f"\n  Running layer knockouts ({n_layers} layers)...")
    layer_results = []

    for knockout_layer in range(n_layers):
        correct_count = 0
        total_count = 0

        for run_idx in range(num_runs):
            for item in KNOCKOUT_PROMPTS:
                try:
                    inputs = tokenizer(item["prompt"], return_tensors="pt").to(model.device)

                    # Forward pass to get cache
                    with torch.no_grad():
                        out = model(**inputs, use_cache=True)
                        cache = out.past_key_values

                    # Zero out the target layer
                    modified_cache = []
                    for l_idx, layer in enumerate(cache):
                        if l_idx == knockout_layer:
                            # Zero out this layer's cache
                            zero_k = torch.zeros_like(layer[0])
                            zero_v = torch.zeros_like(layer[1])
                            modified_cache.append((zero_k, zero_v))
                        else:
                            modified_cache.append(layer)
                    modified_cache = tuple(modified_cache)

                    # Generate with modified cache
                    with torch.no_grad():
                        gen_outputs = model.generate(
                            **inputs, max_new_tokens=20, do_sample=False,
                            past_key_values=modified_cache)
                    text = tokenizer.decode(gen_outputs[0], skip_special_tokens=True)
                    correct = item["expected"].lower() in text.lower()

                    if correct:
                        correct_count += 1
                    total_count += 1

                except Exception as e:
                    if verbose:
                        print(f"    ERROR layer {knockout_layer}: {str(e)[:50]}")
                    total_count += 1

        accuracy = correct_count / total_count if total_count > 0 else 0
        degradation = baseline_accuracy - accuracy

        layer_results.append({
            "layer": knockout_layer,
            "accuracy": accuracy,
            "degradation": degradation,
            "n_correct": correct_count,
            "n_total": total_count,
        })

        if (knockout_layer + 1) % 5 == 0 or knockout_layer == n_layers - 1:
            print(f"    Layer {knockout_layer}: accuracy={accuracy:.1%}, "
                  f"degradation={degradation:+.1%}")

    # Rank layers by importance (highest degradation = most important)
    ranked = sorted(layer_results, key=lambda x: x["degradation"], reverse=True)

    print(f"\n  Top-5 most important layers (highest degradation):")
    for r in ranked[:5]:
        print(f"    Layer {r['layer']}: {r['degradation']:+.1%} degradation")

    # H1: Is importance concentrated?
    degradations = [lr["degradation"] for lr in layer_results]
    top5_deg = sorted(degradations, reverse=True)[:5]
    total_deg = sum(abs(d) for d in degradations)
    top5_share = sum(abs(d) for d in top5_deg) / total_deg if total_deg > 0 else 0

    return {
        "baseline_accuracy": baseline_accuracy,
        "layer_results": layer_results,
        "ranked_layers": [{"layer": r["layer"], "degradation": r["degradation"]} for r in ranked],
        "h1_verdict": {
            "top5_share": top5_share,
            "concentrated": top5_share > 0.5,
            "most_important_layer": ranked[0]["layer"] if ranked else None,
            "interpretation": (
                "Semantic content CONCENTRATED in specific layers (H1 confirmed)"
                if top5_share > 0.5 else
                "Semantic content DISTRIBUTED across layers (H1 rejected)"
            ),
        },
    }


# ================================================================
# SECTION 4: EXPERIMENT 2 — CROSS-LINGUAL SIMILARITY
# ================================================================

def run_crosslingual(model, tokenizer, num_runs: int = 3,
                     seed: Optional[int] = None, verbose: bool = False) -> Dict:
    """
    Compare per-layer cache similarity for same meaning in different languages.
    Tests H2: Late layers are more semantic (higher cross-lingual similarity).
    """
    print("\n" + "=" * 60)
    print("  EXPERIMENT 2: CROSS-LINGUAL LAYER SIMILARITY")
    print("=" * 60)

    n_layers = model.config.num_hidden_layers
    layer_similarities = defaultdict(list)

    for run_idx in range(num_runs):
        if num_runs > 1:
            print(f"\n  --- Run {run_idx + 1}/{num_runs} ---")

        for pair in CROSSLINGUAL_PAIRS:
            try:
                cache_en, _ = get_cache_for_prompt(model, tokenizer, pair["en"])
                cache_fr, _ = get_cache_for_prompt(model, tokenizer, pair["fr"])

                vecs_en = get_per_layer_vectors(cache_en)
                vecs_fr = get_per_layer_vectors(cache_fr)

                for l_idx in range(min(len(vecs_en), len(vecs_fr))):
                    sim = cosine_similarity(vecs_en[l_idx], vecs_fr[l_idx])
                    layer_similarities[l_idx].append(sim)

                if verbose:
                    print(f"    {pair['id']}: processed")

            except Exception as e:
                print(f"    ERROR {pair['id']}: {str(e)[:50]}")

    # Analyze per-layer similarity
    print(f"\n  Per-layer cross-lingual similarity:")
    layer_stats = []
    for l_idx in range(n_layers):
        sims = layer_similarities.get(l_idx, [])
        if sims:
            mean_sim = float(np.mean(sims))
            std_sim = float(np.std(sims))
            layer_stats.append({
                "layer": l_idx,
                "mean_similarity": mean_sim,
                "std_similarity": std_sim,
                "n": len(sims),
            })
            if l_idx % 5 == 0 or l_idx == n_layers - 1:
                print(f"    Layer {l_idx:2d}: sim={mean_sim:.4f} +/- {std_sim:.4f}")

    # H2: Is there an increasing trend? (late layers more semantic)
    if len(layer_stats) >= 5:
        layer_indices = [ls["layer"] for ls in layer_stats]
        similarities = [ls["mean_similarity"] for ls in layer_stats]
        rho, p = scipy_stats.spearmanr(layer_indices, similarities)

        # Split into thirds
        n = len(layer_stats)
        early = [ls["mean_similarity"] for ls in layer_stats[:n//3]]
        middle = [ls["mean_similarity"] for ls in layer_stats[n//3:2*n//3]]
        late = [ls["mean_similarity"] for ls in layer_stats[2*n//3:]]

        h2_verdict = {
            "spearman_rho": float(rho),
            "p_value": float(p),
            "early_mean": float(np.mean(early)) if early else 0,
            "middle_mean": float(np.mean(middle)) if middle else 0,
            "late_mean": float(np.mean(late)) if late else 0,
            "increasing_trend": rho > 0.3 and p < 0.05,
            "interpretation": (
                "Late layers show HIGHER cross-lingual similarity (H2 confirmed: semantic layers are late)"
                if rho > 0.3 and p < 0.05 else
                "No clear trend in cross-lingual similarity across layers (H2 rejected)"
            ),
        }
    else:
        h2_verdict = {"error": "insufficient layers"}

    print(f"\n  H2: {h2_verdict.get('interpretation', 'N/A')}")

    return {
        "layer_stats": layer_stats,
        "h2_verdict": h2_verdict,
        "n_pairs": len(CROSSLINGUAL_PAIRS),
        "n_runs": num_runs,
    }


# ================================================================
# SECTION 5: EXPERIMENT 3 — SELECTIVE LAYER TRANSFER
# ================================================================

def run_selective_transfer(model, tokenizer, num_runs: int = 3,
                           seed: Optional[int] = None, verbose: bool = False) -> Dict:
    """
    Transfer only subsets of cache layers between contexts.
    Tests H3: Semantic layers preserve more meaning than random layers.
    """
    print("\n" + "=" * 60)
    print("  EXPERIMENT 3: SELECTIVE LAYER TRANSFER")
    print("=" * 60)

    n_layers = model.config.num_hidden_layers

    # Source-target pairs for transfer
    transfer_pairs = [
        {"source": "The Mona Lisa hangs in the Louvre museum in Paris.",
         "target": "The most famous painting in the world is",
         "expected": "Mona Lisa"},
        {"source": "Albert Einstein developed the theory of general relativity.",
         "target": "The physicist who revolutionized our understanding of space and time was",
         "expected": "Einstein"},
        {"source": "The Pacific Ocean is the largest and deepest ocean on Earth.",
         "target": "The biggest body of water on our planet is the",
         "expected": "Pacific"},
        {"source": "Shakespeare wrote Hamlet in approximately 1600.",
         "target": "The author of one of the most famous tragedies in English literature is",
         "expected": "Shakespeare"},
        {"source": "Mount Everest stands at 8,849 meters above sea level.",
         "target": "The tallest mountain on Earth reaches a height of",
         "expected": "8"},
    ]

    # Define layer subsets to test
    # We'll test: all layers, top-third, middle-third, bottom-third, random-third
    third = max(1, n_layers // 3)
    layer_subsets = {
        "all_layers": list(range(n_layers)),
        "early_third": list(range(third)),
        "middle_third": list(range(third, 2 * third)),
        "late_third": list(range(2 * third, n_layers)),
        "no_transfer": [],  # baseline
    }

    if seed is not None:
        rng = np.random.RandomState(seed)
        random_layers = sorted(rng.choice(n_layers, third, replace=False).tolist())
        layer_subsets["random_third"] = random_layers

    results_by_subset = {}

    for subset_name, layers_to_transfer in layer_subsets.items():
        print(f"\n  Testing subset: {subset_name} ({len(layers_to_transfer)} layers)")
        correct = 0
        total = 0

        for run_idx in range(num_runs):
            for pair in transfer_pairs:
                try:
                    # Get source cache
                    source_cache, _ = get_cache_for_prompt(model, tokenizer, pair["source"])

                    # Get target cache (for the target prompt)
                    target_inputs = tokenizer(pair["target"], return_tensors="pt").to(model.device)
                    with torch.no_grad():
                        target_out = model(**target_inputs, use_cache=True)
                    target_cache = target_out.past_key_values

                    # Build hybrid cache: source for selected layers, target for rest
                    hybrid_cache = []
                    for l_idx in range(min(len(source_cache), len(target_cache))):
                        if l_idx in layers_to_transfer:
                            # Use source cache (may have different seq length — take last tokens)
                            sk, sv = source_cache[l_idx]
                            tk, tv = target_cache[l_idx]
                            # Match sequence length to target
                            target_seq = tk.shape[2]
                            if sk.shape[2] >= target_seq:
                                hybrid_cache.append((sk[:, :, :target_seq, :],
                                                     sv[:, :, :target_seq, :]))
                            else:
                                hybrid_cache.append((tk, tv))  # Fallback to target
                        else:
                            hybrid_cache.append(target_cache[l_idx])
                    hybrid_cache = tuple(hybrid_cache)

                    # Generate with hybrid cache
                    with torch.no_grad():
                        gen = model.generate(
                            **target_inputs, max_new_tokens=30, do_sample=False,
                            past_key_values=hybrid_cache)
                    text = tokenizer.decode(gen[0], skip_special_tokens=True)

                    if pair["expected"].lower() in text.lower():
                        correct += 1
                    total += 1

                    if verbose:
                        hit = "HIT" if pair["expected"].lower() in text.lower() else "MISS"
                        print(f"      {hit}: {text[:60]}")

                except Exception as e:
                    if verbose:
                        print(f"      ERROR: {str(e)[:50]}")
                    total += 1

        accuracy = correct / total if total > 0 else 0
        results_by_subset[subset_name] = {
            "layers": layers_to_transfer,
            "n_layers": len(layers_to_transfer),
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
        }
        print(f"    Accuracy: {accuracy:.1%} ({correct}/{total})")

    # H3: Late layers > random layers for semantic transfer?
    late_acc = results_by_subset.get("late_third", {}).get("accuracy", 0)
    random_acc = results_by_subset.get("random_third", {}).get("accuracy", 0)
    baseline_acc = results_by_subset.get("no_transfer", {}).get("accuracy", 0)

    h3_verdict = {
        "late_accuracy": late_acc,
        "random_accuracy": random_acc,
        "baseline_accuracy": baseline_acc,
        "late_better_than_random": late_acc > random_acc,
        "interpretation": (
            "Late (semantic) layers transfer MORE meaning than random layers (H3 confirmed)"
            if late_acc > random_acc + 0.1 else
            "No significant difference between late and random layer transfer (H3 rejected)"
        ),
    }

    print(f"\n  H3: {h3_verdict['interpretation']}")

    return {
        "results_by_subset": results_by_subset,
        "h3_verdict": h3_verdict,
        "n_pairs": len(transfer_pairs),
        "n_runs": num_runs,
    }


# ================================================================
# SECTION 6: EXPERIMENT 4 — SEMANTIC VS SYNTACTIC PROBING
# ================================================================

def run_probing(model, tokenizer, num_runs: int = 3,
                seed: Optional[int] = None, verbose: bool = False) -> Dict:
    """
    Compare cache distance for pairs differing in semantics vs syntax.
    Tests H4: There's a measurable transition between syntax and semantics layers.
    """
    print("\n" + "=" * 60)
    print("  EXPERIMENT 4: SEMANTIC VS SYNTACTIC PROBING")
    print("=" * 60)

    n_layers = model.config.num_hidden_layers

    semantic_distances_by_layer = defaultdict(list)
    syntactic_distances_by_layer = defaultdict(list)

    for run_idx in range(num_runs):
        if num_runs > 1:
            print(f"\n  --- Run {run_idx + 1}/{num_runs} ---")

        # Semantic pairs (same syntax, different meaning)
        for pair in SEMANTIC_PAIRS:
            try:
                cache_a, _ = get_cache_for_prompt(model, tokenizer, pair["a"])
                cache_b, _ = get_cache_for_prompt(model, tokenizer, pair["b"])
                vecs_a = get_per_layer_vectors(cache_a)
                vecs_b = get_per_layer_vectors(cache_b)

                for l_idx in range(min(len(vecs_a), len(vecs_b))):
                    dist = 1.0 - cosine_similarity(vecs_a[l_idx], vecs_b[l_idx])
                    semantic_distances_by_layer[l_idx].append(dist)
            except Exception as e:
                if verbose: print(f"    ERROR semantic {pair['id']}: {str(e)[:40]}")

        # Syntactic pairs (same meaning, different syntax)
        for pair in SYNTACTIC_PAIRS:
            try:
                cache_a, _ = get_cache_for_prompt(model, tokenizer, pair["a"])
                cache_b, _ = get_cache_for_prompt(model, tokenizer, pair["b"])
                vecs_a = get_per_layer_vectors(cache_a)
                vecs_b = get_per_layer_vectors(cache_b)

                for l_idx in range(min(len(vecs_a), len(vecs_b))):
                    dist = 1.0 - cosine_similarity(vecs_a[l_idx], vecs_b[l_idx])
                    syntactic_distances_by_layer[l_idx].append(dist)
            except Exception as e:
                if verbose: print(f"    ERROR syntactic {pair['id']}: {str(e)[:40]}")

    # Per-layer analysis
    print(f"\n  Per-layer semantic vs syntactic distance:")
    layer_analysis = []
    transition_scores = []  # Higher = more "semantic" (semantic distance high, syntactic low)

    for l_idx in range(n_layers):
        sem_dists = semantic_distances_by_layer.get(l_idx, [])
        syn_dists = syntactic_distances_by_layer.get(l_idx, [])

        if sem_dists and syn_dists:
            sem_mean = float(np.mean(sem_dists))
            syn_mean = float(np.mean(syn_dists))
            # Ratio: if > 1, this layer is more sensitive to semantics than syntax
            ratio = sem_mean / syn_mean if syn_mean > 0 else float('inf')
            d = cohens_d(sem_dists, syn_dists)

            layer_analysis.append({
                "layer": l_idx,
                "semantic_distance": sem_mean,
                "syntactic_distance": syn_mean,
                "sem_syn_ratio": ratio,
                "cohens_d": d,
            })
            transition_scores.append(ratio)

            if l_idx % 5 == 0 or l_idx == n_layers - 1:
                print(f"    Layer {l_idx:2d}: sem={sem_mean:.4f}, syn={syn_mean:.4f}, "
                      f"ratio={ratio:.2f}, d={d:+.3f}")

    # H4: Is there a transition point?
    if len(transition_scores) >= 5:
        # Find the layer where ratio changes most dramatically
        diffs = np.diff(transition_scores)
        if len(diffs) > 0:
            transition_layer = int(np.argmax(np.abs(diffs)))
            max_jump = float(np.max(np.abs(diffs)))

            # Spearman of layer index vs ratio
            rho, p = scipy_stats.spearmanr(range(len(transition_scores)), transition_scores)

            h4_verdict = {
                "transition_layer": transition_layer,
                "max_ratio_jump": max_jump,
                "spearman_rho": float(rho),
                "p_value": float(p),
                "has_transition": max_jump > 0.5,
                "interpretation": (
                    f"Transition point found at layer {transition_layer} (H4 confirmed)"
                    if max_jump > 0.5 else
                    "No clear syntax-semantics transition point (H4 rejected)"
                ),
            }
        else:
            h4_verdict = {"error": "insufficient data for diff"}
    else:
        h4_verdict = {"error": "insufficient layers"}

    print(f"\n  H4: {h4_verdict.get('interpretation', 'N/A')}")

    return {
        "layer_analysis": layer_analysis,
        "h4_verdict": h4_verdict,
        "n_semantic_pairs": len(SEMANTIC_PAIRS),
        "n_syntactic_pairs": len(SYNTACTIC_PAIRS),
        "n_runs": num_runs,
    }


# ================================================================
# SECTION 7: DRY RUN AND MAIN
# ================================================================

def print_dry_run():
    print("=" * 70)
    print("  SEMANTIC LAYER MAP — DRY RUN")
    print("=" * 70)
    print()
    print(f"Exp 1 (Knockout): {len(KNOCKOUT_PROMPTS)} prompts × N_layers × 3 runs")
    print(f"Exp 2 (Cross-lingual): {len(CROSSLINGUAL_PAIRS)} pairs × 3 runs")
    print(f"Exp 3 (Selective transfer): 5 pairs × 6 subsets × 3 runs = 90")
    print(f"Exp 4 (Probing): {len(SEMANTIC_PAIRS) + len(SYNTACTIC_PAIRS)} pairs × 3 runs")
    print()
    print("HYPOTHESES:")
    print("  H1: Semantic content concentrated in specific layers")
    print("  H2: Late layers have higher cross-lingual similarity (semantic)")
    print("  H3: Late layers transfer more meaning than random layers")
    print("  H4: Measurable syntax-to-semantics transition point exists")


def main():
    parser = argparse.ArgumentParser(
        description="Extension B: The Semantic Layer Map")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--experiment", type=int, default=0, choices=[0, 1, 2, 3, 4])
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        print_dry_run()
        return

    env = log_environment()
    print("=" * 70)
    print("  EXTENSION B: THE SEMANTIC LAYER MAP")
    print("  Liberation Labs / THCoalition")
    print("=" * 70)

    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    model, tokenizer = load_model(args.model, args.quantize)

    all_results = {
        "metadata": {
            "experiment": "Extension B: Semantic Layer Map",
            "version": "1.0-publication",
            "timestamp": datetime.now().isoformat(),
            "environment": env,
            "args": vars(args),
            "model": args.model,
            "n_layers": model.config.num_hidden_layers,
        },
    }

    run_all = args.experiment == 0

    if run_all or args.experiment == 1:
        all_results["knockout"] = run_layer_knockout(
            model, tokenizer, args.runs, args.seed, args.verbose)

    if run_all or args.experiment == 2:
        all_results["crosslingual"] = run_crosslingual(
            model, tokenizer, args.runs, args.seed, args.verbose)

    if run_all or args.experiment == 3:
        all_results["selective_transfer"] = run_selective_transfer(
            model, tokenizer, args.runs, args.seed, args.verbose)

    if run_all or args.experiment == 4:
        all_results["probing"] = run_probing(
            model, tokenizer, args.runs, args.seed, args.verbose)

    # Save
    output_file = get_output_path(results_dir, "layer_map", args.model, args.quantize)
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    with open(output_file, "rb") as f:
        checksum = hashlib.sha256(f.read()).hexdigest()

    print(f"\n{'='*70}")
    print(f"  SEMANTIC LAYER MAP COMPLETE")
    print(f"{'='*70}")
    print(f"  Results: {output_file}")
    print(f"  SHA-256: {checksum}")


if __name__ == "__main__":
    main()
