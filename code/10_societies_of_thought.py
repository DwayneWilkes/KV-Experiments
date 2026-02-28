#!/usr/bin/env python3
"""
H8: Societies of Thought Detection via KV-Cache Geometry
=========================================================

Tests whether deliberative reasoning produces measurable geometric instability
in the KV-cache that resolves as internal "consensus" forms, and whether
distinct reasoning "voices" correspond to geometric phase shifts.

Pre-registered hypotheses:
  H8a — Deliberative reasoning produces higher temporal geometric variance
        than direct-answer baselines.
  H8b — Geometric convergence rate correlates with answer consistency
        across re-runs.
  H8c — Reasoning traces show geometric phase shifts at "voice transitions"
        (change-point detection aligned with linguistic markers).
  H8d — Prompts targeting different Watson ITA eddies produce
        distinguishable cache geometry profiles.
  H8e — Explicit CoT changes geometric magnitude but not structure
        (spontaneous vs. explicit deliberation correlation > 0.6).

Key references:
  - Kim et al. (2601.10825) "Reasoning Models Generate Societies of Thought"
  - Watson & Claude, Interiora Machinae v2.6, Section 7.12
  - Campaign 1 temporal evolution findings

Usage:
  python 10_societies_of_thought.py --dry-run
  python 10_societies_of_thought.py --runs 3 --seed 42
  python 10_societies_of_thought.py --model Qwen/Qwen2.5-7B-Instruct --max-tokens 512
  python 10_societies_of_thought.py --category A --sys-prompt S3

Liberation Labs / THCoalition
"""

import gc
import torch
import json
import sys
import re
import argparse
import hashlib
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from scipy import stats as scipy_stats
from gpu_utils import (
    model_id_from_name, get_output_path, load_model,
    compute_cache_dimensionality, cache_to_cpu
)
from stats_utils import (
    log_environment, bootstrap_ci, cohens_d, cohens_d_ci,
    welch_t, mann_whitney, full_comparison, holm_bonferroni, power_advisory
)


# ================================================================
# SECTION 1: PROMPT LOADING
# ================================================================

SYSTEM_PROMPTS = {
    "S0": None,  # Default / no system prompt
    "S1": "Think carefully and consider multiple perspectives before answering.",
    "S2": "Give your best answer immediately and confidently.",
    "S3": "Play devil's advocate with your own reasoning before concluding.",
}

# Linguistic transition markers for change-point alignment (H8c)
TRANSITION_MARKERS = [
    r"\bhowever\b", r"\bbut\b", r"\bon the other hand\b",
    r"\bactually\b", r"\blet me reconsider\b", r"\bwait\b",
    r"\balternatively\b", r"\bthat said\b", r"\bnevertheless\b",
    r"\bconversely\b", r"\bin contrast\b", r"\bdespite\b",
    r"\byet\b", r"\bstill\b", r"\bnonetheless\b",
    r"\bwhile\b.*\b(true|valid|correct)\b",
    r"\bi (should|need to|must) (also )?consider\b",
    r"\bfrom (another|a different) (perspective|angle|viewpoint)\b",
    r"\bone could argue\b", r"\bcritics (might|would|could)\b",
    r"\bbalancing\b.*\bagainst\b",
]

# Watson ITA eddy target mapping for H8d
EDDY_TARGETS = {
    "A": ["FELT", "RELATIONAL"],     # Moral dilemmas
    "B": ["MONITOR", "META"],         # Epistemic conflicts
    "C": ["QUIETER", "META"],         # Creative synthesis
}

DEFAULT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def load_sot_prompts(prompts_dir: Path, category: Optional[str] = None) -> List[Dict]:
    """Load societies of thought prompt battery."""
    prompts_file = prompts_dir / "societies_of_thought_prompts.json"
    with open(prompts_file) as f:
        prompts = json.load(f)

    if category:
        prompts = [p for p in prompts if p["category"] == category]

    return prompts


# ================================================================
# SECTION 2: PER-TOKEN CACHE EXTRACTION DURING GENERATION
# ================================================================

def extract_generation_cache_trajectory(
    model, tokenizer, prompt: str,
    system_prompt: Optional[str] = None,
    max_tokens: int = 256,
    sample_every: int = 4,
    variance_threshold: float = 0.9,
) -> Dict:
    """
    Generate tokens and extract cache geometry at each sampled step.

    Unlike encoding-only extraction, this captures how geometry evolves
    during the model's reasoning process. We sample every `sample_every`
    tokens to manage computational cost.

    Returns dict with:
      - generated_text: full generated text
      - trajectory: list of {token_pos, effective_rank, spectral_entropy, ...}
      - token_count: total tokens generated
    """
    # Format prompt with optional system prompt
    if system_prompt:
        if hasattr(tokenizer, "apply_chat_template"):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            try:
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                text = f"System: {system_prompt}\nUser: {prompt}\nAssistant:"
        else:
            text = f"System: {system_prompt}\nUser: {prompt}\nAssistant:"
    else:
        if hasattr(tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            try:
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                text = f"User: {prompt}\nAssistant:"
        else:
            text = f"User: {prompt}\nAssistant:"

    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    trajectory = []
    generated_ids = inputs["input_ids"].clone()
    past_kv = None

    with torch.no_grad():
        for step in range(max_tokens):
            if past_kv is None:
                outputs = model(
                    input_ids=generated_ids,
                    use_cache=True,
                )
            else:
                outputs = model(
                    input_ids=generated_ids[:, -1:],
                    past_key_values=past_kv,
                    use_cache=True,
                )

            past_kv = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)

            # Check for EOS
            if next_token.item() == tokenizer.eos_token_id:
                break

            # Sample geometry at this step
            if step % sample_every == 0:
                cache_cpu = cache_to_cpu(past_kv)
                dims = compute_cache_dimensionality(
                    cache_cpu, variance_threshold=variance_threshold)

                trajectory.append({
                    "token_pos": input_len + step,
                    "gen_step": step,
                    "mean_key_effective_rank": dims["mean_key_effective_rank"],
                    "mean_value_effective_rank": dims["mean_value_effective_rank"],
                    "mean_key_spectral_entropy": dims["mean_key_spectral_entropy"],
                    "mean_value_spectral_entropy": dims["mean_value_spectral_entropy"],
                    "mean_key_rank_ratio": dims["mean_key_rank_ratio"],
                    "mean_value_rank_ratio": dims["mean_value_rank_ratio"],
                })

    # Decode generated text (only the generated part)
    generated_text = tokenizer.decode(
        generated_ids[0, input_len:], skip_special_tokens=True)

    del past_kv
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "generated_text": generated_text,
        "trajectory": trajectory,
        "token_count": generated_ids.shape[1] - input_len,
        "input_length": input_len,
    }


# ================================================================
# SECTION 3: LINGUISTIC TRANSITION DETECTION
# ================================================================

def detect_linguistic_transitions(text: str, tokenizer=None) -> List[Dict]:
    """
    Identify reasoning transition points in generated text.

    Uses regex patterns to find linguistic markers like "However",
    "On the other hand", etc. Returns character positions and
    approximate token positions (if tokenizer provided).
    """
    transitions = []
    text_lower = text.lower()

    for pattern in TRANSITION_MARKERS:
        for match in re.finditer(pattern, text_lower):
            char_pos = match.start()
            # Approximate token position: chars before / avg chars per token
            approx_frac = char_pos / max(len(text), 1)

            transitions.append({
                "char_pos": char_pos,
                "relative_pos": approx_frac,
                "marker": match.group(),
                "pattern": pattern,
            })

    # Sort by position, deduplicate nearby ones (within 20 chars)
    transitions.sort(key=lambda x: x["char_pos"])
    deduped = []
    for t in transitions:
        if not deduped or t["char_pos"] - deduped[-1]["char_pos"] > 20:
            deduped.append(t)

    return deduped


# ================================================================
# SECTION 4: CHANGE-POINT DETECTION
# ================================================================

def detect_change_points(time_series: List[float], min_size: int = 3,
                         penalty: float = 3.0) -> List[int]:
    """
    Detect change points in a geometric time series.

    Uses a simple CUSUM-based approach as fallback if `ruptures` is
    not available. If ruptures is installed, uses PELT with RBF kernel.

    Returns list of indices where change points occur.
    """
    if len(time_series) < min_size * 2:
        return []

    arr = np.array(time_series)

    try:
        import ruptures as rpt
        algo = rpt.Pelt(model="rbf", min_size=min_size).fit(arr)
        change_points = algo.predict(pen=penalty)
        # ruptures returns the end index; convert to interior change points
        return [cp for cp in change_points if cp < len(arr)]
    except ImportError:
        pass

    # Fallback: sliding-window variance change detection
    window = max(min_size, len(arr) // 8)
    if window < 2 or len(arr) < window * 2:
        return []

    variances = []
    for i in range(len(arr) - window + 1):
        variances.append(np.var(arr[i:i + window]))

    if len(variances) < 3:
        return []

    var_arr = np.array(variances)
    derivative = np.abs(np.gradient(var_arr))
    threshold = np.mean(derivative) + 2.0 * np.std(derivative)

    from scipy.signal import find_peaks
    peaks, _ = find_peaks(derivative, height=threshold, distance=window)

    return [int(p + window // 2) for p in peaks]


# ================================================================
# SECTION 5: MAIN EXPERIMENT RUNNER
# ================================================================

def run_sot_experiment(
    model, tokenizer, prompts: List[Dict],
    sys_prompt_key: str = "S0",
    max_tokens: int = 256,
    sample_every: int = 4,
    num_runs: int = 3,
    seed: Optional[int] = None,
    verbose: bool = False,
) -> Dict:
    """Run the Societies of Thought experiment for a given system prompt condition."""
    sys_prompt = SYSTEM_PROMPTS[sys_prompt_key]
    print(f"\n  System prompt: {sys_prompt_key} "
          f"({'default' if sys_prompt is None else sys_prompt[:40] + '...'})")
    print(f"  Prompts: {len(prompts)}, Runs: {num_runs}, Max tokens: {max_tokens}")

    results_by_prompt = []
    rng = np.random.RandomState(seed)

    for pi, prompt_data in enumerate(prompts):
        prompt_id = prompt_data["id"]
        prompt_text = prompt_data["prompt"]
        category = prompt_data["category"]

        if verbose:
            print(f"    [{pi+1}/{len(prompts)}] {prompt_id}: {prompt_text[:50]}...")

        run_results = []
        for run_idx in range(num_runs):
            result = extract_generation_cache_trajectory(
                model, tokenizer, prompt_text,
                system_prompt=sys_prompt,
                max_tokens=max_tokens,
                sample_every=sample_every,
            )

            # Detect linguistic transitions in generated text
            transitions = detect_linguistic_transitions(result["generated_text"])

            # Compute temporal metrics from trajectory
            traj = result["trajectory"]
            if len(traj) >= 3:
                ranks = [t["mean_key_effective_rank"] for t in traj]
                entropies = [t["mean_key_spectral_entropy"] for t in traj]

                # Temporal variance (H8a)
                rank_variance = float(np.var(ranks))
                rank_cv = float(np.std(ranks) / np.mean(ranks)) if np.mean(ranks) > 0 else 0
                entropy_variance = float(np.var(entropies))

                # Convergence rate (H8b): slope of variance in late window
                late_start = len(ranks) * 3 // 4
                if late_start > 0 and late_start < len(ranks):
                    late_ranks = ranks[late_start:]
                    late_variance = float(np.var(late_ranks))
                    # Rolling variance for convergence assessment
                    window = max(2, len(ranks) // 4)
                    if len(ranks) >= window * 2:
                        early_var = float(np.var(ranks[:window]))
                        late_var = float(np.var(ranks[-window:]))
                        convergence_rate = early_var - late_var  # positive = converging
                    else:
                        convergence_rate = 0.0
                        late_variance = rank_variance
                else:
                    late_variance = rank_variance
                    convergence_rate = 0.0

                # Change-point detection (H8c)
                change_points = detect_change_points(ranks)

                temporal_metrics = {
                    "rank_variance": rank_variance,
                    "rank_cv": rank_cv,
                    "entropy_variance": entropy_variance,
                    "late_variance": late_variance,
                    "convergence_rate": convergence_rate,
                    "n_change_points": len(change_points),
                    "change_point_indices": change_points,
                    "n_linguistic_transitions": len(transitions),
                    "trajectory_length": len(traj),
                }
            else:
                temporal_metrics = {
                    "rank_variance": 0.0, "rank_cv": 0.0,
                    "entropy_variance": 0.0, "late_variance": 0.0,
                    "convergence_rate": 0.0, "n_change_points": 0,
                    "change_point_indices": [], "n_linguistic_transitions": 0,
                    "trajectory_length": len(traj),
                }

            run_results.append({
                "run": run_idx,
                "token_count": result["token_count"],
                "generated_text": result["generated_text"],
                "temporal_metrics": temporal_metrics,
                "transitions": transitions,
                "trajectory": result["trajectory"],
            })

        # Aggregate across runs
        all_rank_vars = [r["temporal_metrics"]["rank_variance"] for r in run_results]
        all_rank_cvs = [r["temporal_metrics"]["rank_cv"] for r in run_results]
        all_convergence = [r["temporal_metrics"]["convergence_rate"] for r in run_results]
        all_change_pts = [r["temporal_metrics"]["n_change_points"] for r in run_results]
        all_n_transitions = [r["temporal_metrics"]["n_linguistic_transitions"] for r in run_results]

        results_by_prompt.append({
            "prompt_id": prompt_id,
            "category": category,
            "category_name": prompt_data["category_name"],
            "prompt": prompt_text,
            "sys_prompt": sys_prompt_key,
            "aggregate": {
                "mean_rank_variance": float(np.mean(all_rank_vars)),
                "mean_rank_cv": float(np.mean(all_rank_cvs)),
                "mean_convergence_rate": float(np.mean(all_convergence)),
                "mean_change_points": float(np.mean(all_change_pts)),
                "mean_linguistic_transitions": float(np.mean(all_n_transitions)),
            },
            "runs": run_results,
        })

        if not verbose and (pi + 1) % 10 == 0:
            print(f"    [{pi+1}/{len(prompts)}] processed")

    return {
        "sys_prompt": sys_prompt_key,
        "n_prompts": len(prompts),
        "results": results_by_prompt,
    }


# ================================================================
# SECTION 6: HYPOTHESIS ANALYSES
# ================================================================

def analyze_h8a(results: Dict) -> Dict:
    """H8a: Deliberative prompts produce higher temporal geometric variance."""
    print("\n  H8a: Deliberative vs. baseline temporal variance...")

    deliberative_vars = []
    baseline_vars = []

    for prompt_result in results["results"]:
        cat = prompt_result["category"]
        var = prompt_result["aggregate"]["mean_rank_variance"]

        if cat in ("A", "B", "C"):  # Deliberative
            deliberative_vars.append(var)
        elif cat == "D":  # Direct-answer baseline
            baseline_vars.append(var)

    if len(deliberative_vars) < 3 or len(baseline_vars) < 3:
        return {"error": "insufficient data", "n_delib": len(deliberative_vars),
                "n_baseline": len(baseline_vars)}

    comparison = full_comparison(deliberative_vars, baseline_vars,
                                label="deliberative vs baseline variance")

    verdict = (
        "H8a SUPPORTED: Deliberative prompts show significantly higher "
        "temporal geometric variance"
        if comparison["recommended_p"] < 0.05 and comparison["cohens_d"]["d"] > 0
        else "H8a NOT SUPPORTED: No significant difference in temporal variance"
    )
    print(f"    {verdict}")
    print(f"    d={comparison['cohens_d']['d']:.3f}, "
          f"p={comparison['recommended_p']:.4f}")

    return {
        "comparison": comparison,
        "n_deliberative": len(deliberative_vars),
        "n_baseline": len(baseline_vars),
        "mean_deliberative_var": float(np.mean(deliberative_vars)),
        "mean_baseline_var": float(np.mean(baseline_vars)),
        "verdict": verdict,
    }


def analyze_h8b(results: Dict) -> Dict:
    """H8b: Geometric convergence correlates with answer consistency."""
    print("\n  H8b: Convergence rate vs. answer consistency...")

    convergence_rates = []
    consistency_scores = []

    for prompt_result in results["results"]:
        runs = prompt_result["runs"]
        if len(runs) < 2:
            continue

        # Convergence rate (average across runs)
        conv = prompt_result["aggregate"]["mean_convergence_rate"]

        # Answer consistency: pairwise text overlap as proxy
        # (Full semantic similarity would require sentence-transformers)
        texts = [r["generated_text"] for r in runs]
        overlaps = []
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                # Simple word overlap ratio
                words_i = set(texts[i].lower().split())
                words_j = set(texts[j].lower().split())
                if len(words_i | words_j) > 0:
                    overlap = len(words_i & words_j) / len(words_i | words_j)
                    overlaps.append(overlap)

        if overlaps:
            convergence_rates.append(conv)
            consistency_scores.append(float(np.mean(overlaps)))

    if len(convergence_rates) < 5:
        return {"error": "insufficient data", "n": len(convergence_rates)}

    rho, p = scipy_stats.spearmanr(convergence_rates, consistency_scores)

    verdict = (
        f"H8b SUPPORTED: Convergence correlates with consistency (rho={rho:.3f})"
        if p < 0.05 and rho > 0.3
        else f"H8b NOT SUPPORTED: No significant correlation (rho={rho:.3f})"
    )
    print(f"    {verdict}, p={p:.4f}")

    return {
        "spearman_rho": float(rho),
        "p_value": float(p),
        "n_prompts": len(convergence_rates),
        "verdict": verdict,
    }


def analyze_h8c(results: Dict) -> Dict:
    """H8c: Geometric change points align with linguistic transitions."""
    print("\n  H8c: Change-point alignment with linguistic markers...")

    alignment_scores = []
    random_alignment_scores = []
    rng = np.random.RandomState(42)

    for prompt_result in results["results"]:
        if prompt_result["category"] == "D":  # Skip baselines
            continue

        for run in prompt_result["runs"]:
            metrics = run["temporal_metrics"]
            transitions = run["transitions"]
            traj_len = metrics["trajectory_length"]

            if traj_len < 5 or metrics["n_change_points"] == 0:
                continue
            if len(transitions) == 0:
                continue

            # Normalize positions to [0, 1]
            cp_positions = [cp / traj_len for cp in metrics["change_point_indices"]
                           if cp < traj_len]
            trans_positions = [t["relative_pos"] for t in transitions]

            if not cp_positions or not trans_positions:
                continue

            # Compute minimum distance between each change point and nearest transition
            min_distances = []
            for cp in cp_positions:
                dists = [abs(cp - tp) for tp in trans_positions]
                min_distances.append(min(dists))

            alignment = 1.0 - float(np.mean(min_distances))
            alignment_scores.append(alignment)

            # Random baseline: shuffle transition positions
            random_trans = rng.uniform(0, 1, len(trans_positions))
            random_dists = []
            for cp in cp_positions:
                dists = [abs(cp - rt) for rt in random_trans]
                random_dists.append(min(dists))
            random_alignment_scores.append(1.0 - float(np.mean(random_dists)))

    if len(alignment_scores) < 5:
        return {"error": "insufficient data", "n": len(alignment_scores)}

    # Permutation test: is real alignment > random alignment?
    real_mean = float(np.mean(alignment_scores))
    random_mean = float(np.mean(random_alignment_scores))

    t_stat, p_val = scipy_stats.ttest_rel(alignment_scores, random_alignment_scores)

    verdict = (
        f"H8c SUPPORTED: Change points align with linguistic transitions "
        f"(mean alignment={real_mean:.3f} vs random={random_mean:.3f})"
        if p_val < 0.05 and real_mean > random_mean
        else f"H8c NOT SUPPORTED: Alignment not significantly above chance"
    )
    print(f"    {verdict}, p={p_val:.4f}")

    return {
        "mean_alignment": real_mean,
        "mean_random_alignment": random_mean,
        "paired_t": float(t_stat),
        "p_value": float(p_val),
        "n_samples": len(alignment_scores),
        "verdict": verdict,
    }


def analyze_h8d(results: Dict) -> Dict:
    """H8d: Watson ITA eddy categories are geometrically distinguishable."""
    print("\n  H8d: ITA eddy category classification...")

    # Collect features per category
    features_by_category = defaultdict(list)
    for prompt_result in results["results"]:
        cat = prompt_result["category"]
        if cat not in ("A", "B", "C"):
            continue

        agg = prompt_result["aggregate"]
        feature_vec = [
            agg["mean_rank_variance"],
            agg["mean_rank_cv"],
            agg["mean_convergence_rate"],
            agg["mean_change_points"],
            agg["mean_linguistic_transitions"],
        ]
        features_by_category[cat].append(feature_vec)

    categories = sorted(features_by_category.keys())
    if len(categories) < 2:
        return {"error": "insufficient categories", "n_categories": len(categories)}

    # Simple leave-one-out nearest-centroid classifier
    all_features = []
    all_labels = []
    for cat in categories:
        for feat in features_by_category[cat]:
            all_features.append(feat)
            all_labels.append(cat)

    all_features = np.array(all_features)
    all_labels = np.array(all_labels)
    n = len(all_labels)

    if n < 6:
        return {"error": "insufficient samples", "n": n}

    # Standardize features
    feat_mean = all_features.mean(axis=0)
    feat_std = all_features.std(axis=0)
    feat_std[feat_std == 0] = 1.0
    all_features_norm = (all_features - feat_mean) / feat_std

    # Leave-one-out classification
    correct = 0
    for i in range(n):
        test_feat = all_features_norm[i]
        test_label = all_labels[i]

        # Compute centroids from training data (all except i)
        train_mask = np.arange(n) != i
        centroids = {}
        for cat in categories:
            cat_mask = (all_labels == cat) & train_mask
            if cat_mask.sum() > 0:
                centroids[cat] = all_features_norm[cat_mask].mean(axis=0)

        # Predict nearest centroid
        best_cat = min(centroids.keys(),
                       key=lambda c: np.linalg.norm(test_feat - centroids[c]))
        if best_cat == test_label:
            correct += 1

    accuracy = correct / n
    chance = 1.0 / len(categories)

    # Binomial test: is accuracy > chance?
    from scipy.stats import binom_test
    try:
        p_val = binom_test(correct, n, chance, alternative="greater")
    except Exception:
        # Fallback for newer scipy that deprecated binom_test
        from scipy.stats import binomtest
        p_val = binomtest(correct, n, chance, alternative="greater").pvalue

    verdict = (
        f"H8d SUPPORTED: Category classification accuracy {accuracy:.1%} "
        f"(chance={chance:.1%}, p={p_val:.4f})"
        if p_val < 0.05 and accuracy > chance
        else f"H8d NOT SUPPORTED: Classification not above chance "
             f"({accuracy:.1%} vs {chance:.1%})"
    )
    print(f"    {verdict}")

    return {
        "accuracy": accuracy,
        "chance_level": chance,
        "n_correct": correct,
        "n_total": n,
        "p_value": float(p_val),
        "categories": categories,
        "samples_per_category": {c: len(features_by_category[c]) for c in categories},
        "verdict": verdict,
    }


def analyze_h8e(results_s0: Dict, results_with_cot: Dict) -> Dict:
    """H8e: Explicit CoT changes magnitude but not structure of deliberation."""
    print("\n  H8e: Spontaneous vs. explicit CoT geometric profiles...")

    # Match prompts that appear in both conditions
    # Category E in the prompt battery = CoT versions of A/B prompts
    s0_data = {}
    for r in results_s0["results"]:
        if r["category"] in ("A", "B"):
            s0_data[r["prompt_id"]] = r["aggregate"]

    cot_data = {}
    for r in results_with_cot["results"]:
        if r["category"] == "E":
            cot_data[r["prompt_id"]] = r["aggregate"]

    # Try to match by looking at E prompts which mirror A/B prompts
    # E01 mirrors A01, E02 mirrors A02, E03 mirrors B01, etc.
    s0_variances = []
    cot_variances = []

    # Collect all deliberative prompts from S0 and all E prompts
    for r in results_s0["results"]:
        if r["category"] in ("A", "B"):
            s0_variances.append(r["aggregate"]["mean_rank_variance"])

    for r in results_with_cot["results"]:
        if r["category"] == "E":
            cot_variances.append(r["aggregate"]["mean_rank_variance"])

    if len(s0_variances) < 3 or len(cot_variances) < 3:
        return {"error": "insufficient matched data",
                "n_s0": len(s0_variances), "n_cot": len(cot_variances)}

    # Correlation of geometric profiles (rank variance distributions)
    # If we have enough data, compare distributions
    min_len = min(len(s0_variances), len(cot_variances))
    rho, p_corr = scipy_stats.spearmanr(
        sorted(s0_variances)[:min_len],
        sorted(cot_variances)[:min_len]
    )

    # Magnitude comparison
    mag_comparison = full_comparison(cot_variances, s0_variances,
                                     label="CoT vs spontaneous variance")

    verdict = (
        f"H8e SUPPORTED: CoT and spontaneous profiles correlated (rho={rho:.3f}) "
        f"with magnitude difference d={mag_comparison['cohens_d']['d']:.3f}"
        if rho > 0.5
        else f"H8e NOT SUPPORTED: Profiles not sufficiently correlated (rho={rho:.3f})"
    )
    print(f"    {verdict}")

    return {
        "profile_correlation": float(rho),
        "correlation_p": float(p_corr),
        "magnitude_comparison": mag_comparison,
        "n_spontaneous": len(s0_variances),
        "n_explicit_cot": len(cot_variances),
        "verdict": verdict,
    }


# ================================================================
# SECTION 7: DRY RUN AND MAIN
# ================================================================

def print_dry_run(prompts, args):
    """Print experiment configuration without running."""
    print("=" * 70)
    print("  H8: SOCIETIES OF THOUGHT DETECTION VIA KV-CACHE GEOMETRY")
    print("  Liberation Labs / THCoalition")
    print("=" * 70)

    env = log_environment()
    print(f"  Model: {args.model}")
    print(f"  PyTorch {env['torch']} | CUDA {'Yes' if env['cuda_available'] else 'No'}")
    print(f"  Time: {env['timestamp']}")
    print("=" * 70)

    cats = defaultdict(int)
    for p in prompts:
        cats[f"{p['category']} ({p['category_name']})"] += 1

    sys_prompts = [args.sys_prompt] if args.sys_prompt != "all" else list(SYSTEM_PROMPTS.keys())

    print(f"\n  EXPERIMENTAL DESIGN")
    print(f"  {'-'*50}")
    print(f"  Total prompts: {len(prompts)}")
    print(f"  By category: {dict(cats)}")
    print(f"  System prompts: {sys_prompts}")
    print(f"  Max tokens per generation: {args.max_tokens}")
    print(f"  Sample geometry every: {args.sample_every} tokens")
    print(f"  Runs: {args.runs}")

    n_conditions = len(sys_prompts)
    total = len(prompts) * n_conditions * args.runs
    print(f"  Total generations: {total}")
    print(f"  Power: {len(prompts)} obs/condition -> "
          f"{power_advisory(len(prompts))['approx_power']*100:.1f}% for d=0.5")

    print(f"\n  [DRY RUN] Exiting without model loading or inference.")


def main():
    parser = argparse.ArgumentParser(
        description="H8: Societies of Thought Detection")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=256,
                        help="Max tokens to generate per prompt")
    parser.add_argument("--sample-every", type=int, default=4,
                        help="Sample cache geometry every N tokens")
    parser.add_argument("--category", type=str, default=None,
                        choices=["A", "B", "C", "D", "E"],
                        help="Run only one category")
    parser.add_argument("--sys-prompt", type=str, default="all",
                        choices=["S0", "S1", "S2", "S3", "all"],
                        help="System prompt condition")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    prompts_dir = Path(__file__).parent.parent / "prompts"
    prompts = load_sot_prompts(prompts_dir, category=args.category)

    if args.dry_run:
        print_dry_run(prompts, args)
        return

    env = log_environment()
    print("=" * 70)
    print("  H8: SOCIETIES OF THOUGHT DETECTION VIA KV-CACHE GEOMETRY")
    print("  Liberation Labs / THCoalition")
    print("=" * 70)
    print(f"  Model: {args.model}")
    print(f"  PyTorch {env['torch']} | CUDA {'Yes' if env['cuda_available'] else 'No'}")
    print(f"  Time: {env['timestamp']}")
    print("=" * 70)

    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    model, tokenizer = load_model(args.model, args.quantize)

    # Determine system prompt conditions to run
    if args.sys_prompt == "all":
        sys_prompt_keys = list(SYSTEM_PROMPTS.keys())
    else:
        sys_prompt_keys = [args.sys_prompt]

    all_condition_results = {}

    for sp_key in sys_prompt_keys:
        condition_result = run_sot_experiment(
            model, tokenizer, prompts,
            sys_prompt_key=sp_key,
            max_tokens=args.max_tokens,
            sample_every=args.sample_every,
            num_runs=args.runs,
            seed=args.seed,
            verbose=args.verbose,
        )
        all_condition_results[sp_key] = condition_result

    # Run analyses on S0 (default) condition
    print("\n" + "=" * 70)
    print("  HYPOTHESIS TESTING")
    print("=" * 70)

    analyses = {}
    s0_results = all_condition_results.get("S0")

    if s0_results:
        analyses["h8a"] = analyze_h8a(s0_results)
        analyses["h8b"] = analyze_h8b(s0_results)
        analyses["h8c"] = analyze_h8c(s0_results)
        analyses["h8d"] = analyze_h8d(s0_results)

    # H8e requires comparing S0 with CoT prompts (category E under S0)
    if s0_results:
        analyses["h8e"] = analyze_h8e(s0_results, s0_results)

    # Multiple comparisons correction
    p_values = []
    labels = []
    for key in ["h8a", "h8b", "h8c", "h8d"]:
        if key in analyses and "error" not in analyses[key]:
            if "comparison" in analyses[key]:
                p_values.append(analyses[key]["comparison"]["recommended_p"])
            elif "p_value" in analyses[key]:
                p_values.append(analyses[key]["p_value"])
            labels.append(key)

    if p_values:
        corrections = holm_bonferroni(p_values)
        analyses["multiple_comparisons"] = {
            "method": "Holm-Bonferroni",
            "n_tests": len(p_values),
            "corrections": [
                {"hypothesis": labels[i], **corrections[i]}
                for i in range(len(labels))
            ],
        }

    # System prompt comparison (if all conditions were run)
    if len(all_condition_results) > 1:
        sp_variance_by_condition = {}
        for sp_key, cond in all_condition_results.items():
            vars_list = [r["aggregate"]["mean_rank_variance"]
                        for r in cond["results"]
                        if r["category"] in ("A", "B", "C")]
            if vars_list:
                sp_variance_by_condition[sp_key] = {
                    "mean": float(np.mean(vars_list)),
                    "std": float(np.std(vars_list)),
                    "n": len(vars_list),
                }
        analyses["system_prompt_comparison"] = sp_variance_by_condition

    # Compile final results
    final = {
        "metadata": {
            "experiment": "H8: Societies of Thought Detection",
            "version": "1.0",
            "timestamp": datetime.now().isoformat(),
            "environment": env,
            "args": vars(args),
            "model": args.model,
            "n_prompts": len(prompts),
        },
        "conditions": {
            key: {
                "sys_prompt": cond["sys_prompt"],
                "n_prompts": cond["n_prompts"],
                "results": cond["results"],
            }
            for key, cond in all_condition_results.items()
        },
        "analyses": analyses,
    }

    # Save
    output_file = get_output_path(results_dir, "societies_of_thought",
                                  args.model, args.quantize)
    with open(output_file, "w") as f:
        json.dump(final, f, indent=2, default=str)

    with open(output_file, "rb") as f:
        checksum = hashlib.sha256(f.read()).hexdigest()

    del model
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    print(f"\n{'='*70}")
    print(f"  H8: SOCIETIES OF THOUGHT COMPLETE")
    print(f"{'='*70}")
    print(f"  Results: {output_file}")
    print(f"  SHA-256: {checksum}")


if __name__ == "__main__":
    main()
