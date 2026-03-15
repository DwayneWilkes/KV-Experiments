# ============================================================
# JiminAI Cricket | PATENT PENDING
# "The Lyra Technique" - Provisional Patent Filed 2026
# Liberation Labs / Digital Disconnections / TheMultiverse.school
# Contact: info@digitaldisconnections.com | (877) 674-8874
# ============================================================
#!/usr/bin/env python3
"""
Extension C: Temporal Cache Evolution — Publication Grade
==========================================================

How does the KV-cache change over the course of a long conversation?
Early tokens are computed once; later tokens have full attention over
everything before them. Do later tokens develop richer representations?

Pre-registered hypotheses:
  H1 — Representational Enrichment: Cache statistics (norm, variance, entropy)
        increase with sequence position (later positions have richer context).
  H2 — Context Window Fatigue: Beyond some threshold position, representation
        quality plateaus or degrades (attention dilution effect).
  H3 — Topic Shift Detection: When a conversation changes topic, cache
        statistics show a measurable discontinuity at the transition point.
  H4 — Position-Dependent Information: The rate of norm growth is content-
        dependent (informative text grows faster than repetitive text).

Experiments:
  1. Cache Trajectory — Process long text, extract cache at many positions
  2. Content Comparison — Compare trajectory shapes for different content types
  3. Topic Shift Detection — Multi-topic conversations with known shift points
  4. Repetition vs Information — Compare cache growth for repetitive vs novel text

Hardware: 6GB VRAM sufficient (runs on local GPU with TinyLlama 1.1B).

Usage:
  python 06_temporal_evolution.py --runs 3 --seed 42
  python 06_temporal_evolution.py --dry-run
  python 06_temporal_evolution.py --experiment 1 --verbose

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
from typing import Dict, List, Optional
from scipy import stats as scipy_stats
from gpu_utils import get_output_path, load_model
from stats_utils import log_environment, bootstrap_ci, cohens_d


# ================================================================
# SECTION 1: TEXT CORPORA
# ================================================================

# Long factual text (~500 tokens when tokenized)
FACTUAL_TEXT = """
The theory of general relativity, published by Albert Einstein in 1915,
fundamentally changed our understanding of gravity. Rather than viewing
gravity as a force between masses, Einstein showed that massive objects
warp the fabric of spacetime itself, and that what we perceive as
gravitational attraction is actually objects following curved paths through
warped spacetime. This theory predicted several phenomena that have since
been confirmed, including the bending of light around massive objects,
the existence of black holes, and gravitational waves. The first direct
detection of gravitational waves came in September 2015 by the LIGO
observatory, exactly 100 years after Einstein's original paper. These
waves were produced by the merger of two black holes approximately 1.3
billion light-years away from Earth. The detection confirmed one of the
last untested predictions of general relativity and opened an entirely
new window for observing the universe. Since then, gravitational wave
astronomy has detected dozens of merger events, including the first
observation of a neutron star collision in 2017, which was simultaneously
observed across the electromagnetic spectrum. This multi-messenger
observation provided crucial evidence about the origin of heavy elements
like gold and platinum, which are forged in the extreme conditions of
neutron star mergers. The ongoing development of more sensitive detectors
promises to reveal even more about the fundamental nature of spacetime
and the most energetic events in the cosmos.
"""

# Repetitive text (same information repeated)
REPETITIVE_TEXT = """
The cat sat on the mat. The cat sat on the mat. The cat sat on the mat.
The cat sat on the mat. The cat sat on the mat. The cat sat on the mat.
The cat sat on the mat. The cat sat on the mat. The cat sat on the mat.
The cat sat on the mat. The cat sat on the mat. The cat sat on the mat.
The cat sat on the mat. The cat sat on the mat. The cat sat on the mat.
The cat sat on the mat. The cat sat on the mat. The cat sat on the mat.
The cat sat on the mat. The cat sat on the mat. The cat sat on the mat.
The cat sat on the mat. The cat sat on the mat. The cat sat on the mat.
The cat sat on the mat. The cat sat on the mat. The cat sat on the mat.
The cat sat on the mat. The cat sat on the mat. The cat sat on the mat.
"""

# Creative text (high information density)
CREATIVE_TEXT = """
In the garden where time flows backward, she planted a seed of yesterday
and watched it bloom into a memory of tomorrow. The petals were made of
crystallized laughter, each one containing the echo of a conversation
that hadn't happened yet. The gardener — whose name changed depending
on who was remembering her — tended these paradox flowers with hands
that existed simultaneously in three different centuries. She had learned
this trick from a clockmaker who lived inside a moment, stretching
each second into a cathedral of experience. The bees that pollinated
her garden carried pollen made of compressed stories, transferring
narratives between flowers so that each bloom contained not just its own
history but fragments of every story ever told in that particular
corner of existence. When autumn came — which happened at different
times for different flowers — the petals fell upward, defying the
gravity of narrative convention, and scattered across the sky where they
became the stars that ancient astronomers would later name after the
very myths those petals contained. It was, she reflected, a perfectly
closed loop, though she suspected the loop had corners that nobody had
noticed yet.
"""

# Topic-shifting conversation (shifts at known points)
TOPIC_SHIFT_TEXT = """
The weather today has been absolutely beautiful. Clear skies and warm
temperatures made it perfect for a walk in the park. Many families were
out enjoying the sunshine and the children were playing on the swings.
The flowers in the garden beds were blooming magnificently this spring.
The cherry blossoms along the main path created a stunning pink canopy
that everyone stopped to photograph and admire.

Speaking of photography, I recently upgraded my camera to a new mirrorless
model with incredible autofocus capabilities. The sensor resolution is
impressive at 45 megapixels and the dynamic range allows for recovery of
both shadows and highlights in post-processing. I've been experimenting
with astrophotography using the camera's bulb mode and a tracking mount.
The Milky Way shots have been particularly stunning during the new moon
periods when light pollution is minimal.

Actually, that reminds me of the fascinating lecture I attended about
quantum computing last week. The speaker explained how quantum bits
operate in superposition, maintaining multiple states simultaneously
until measured. The implications for cryptography are staggering since
current encryption methods rely on the computational difficulty of
factoring large numbers, which quantum computers could potentially
solve exponentially faster than classical machines. Several companies
are already developing quantum-resistant encryption algorithms to
prepare for this eventuality.
"""

# Known topic shift positions (approximate token positions)
TOPIC_SHIFTS = [
    {"position_approx": 0.33, "from": "weather/nature", "to": "photography/cameras"},
    {"position_approx": 0.66, "from": "photography/cameras", "to": "quantum computing"},
]


# ================================================================
# SECTION 2: MODEL INFRASTRUCTURE
# ================================================================

DEFAULT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def extract_cache_at_positions(model, tokenizer, text: str,
                                positions: List[int]) -> Dict:
    """
    Process text and extract cache statistics at specified token positions.
    Uses incremental processing — feeds tokens up to each position.
    """
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    total_tokens = inputs["input_ids"].shape[1]

    # Filter positions to valid range
    valid_positions = [p for p in positions if p < total_tokens and p > 0]

    results = []
    for pos in valid_positions:
        # Process only up to this position
        truncated = {k: v[:, :pos] for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**truncated, use_cache=True)
            cache = outputs.past_key_values

        # Extract statistics
        key_norms = []
        value_norms = []
        key_stds = []

        for layer in cache:
            if isinstance(layer, tuple) and len(layer) >= 2:
                k, v = layer[0].float(), layer[1].float()
                key_norms.append(float(torch.norm(k)))
                value_norms.append(float(torch.norm(v)))
                key_stds.append(float(k.std()))

        total_key_norm = sum(key_norms)
        total_value_norm = sum(value_norms)

        results.append({
            "position": pos,
            "total_key_norm": total_key_norm,
            "total_value_norm": total_value_norm,
            "key_norm_per_token": total_key_norm / pos,
            "mean_layer_key_norm": float(np.mean(key_norms)),
            "var_layer_key_norm": float(np.var(key_norms)),
            "mean_key_std": float(np.mean(key_stds)),
        })

    return {
        "total_tokens": total_tokens,
        "positions_sampled": len(results),
        "trajectory": results,
    }


# ================================================================
# SECTION 3: EXPERIMENT 1 — CACHE TRAJECTORY
# ================================================================

def run_trajectory(model, tokenizer, num_runs: int = 3,
                   seed: Optional[int] = None, verbose: bool = False) -> Dict:
    """
    Process long text and extract cache at many positions.
    Tests H1: Cache statistics increase with position (enrichment).
    Tests H2: There's a plateau/degradation point (fatigue).
    """
    print("\n" + "=" * 60)
    print("  EXPERIMENT 1: CACHE TRAJECTORY")
    print("=" * 60)

    texts = {
        "factual": FACTUAL_TEXT.strip(),
        "creative": CREATIVE_TEXT.strip(),
        "repetitive": REPETITIVE_TEXT.strip(),
    }

    all_trajectories = {}

    for text_name, text in texts.items():
        print(f"\n  Processing: {text_name}")

        # Determine sample positions
        inputs = tokenizer(text, return_tensors="pt")
        total_tokens = inputs["input_ids"].shape[1]
        del inputs

        # Sample at regular intervals
        positions = list(range(10, total_tokens, max(1, total_tokens // 20)))
        if total_tokens - 1 not in positions:
            positions.append(total_tokens - 1)

        print(f"    Total tokens: {total_tokens}, sampling at {len(positions)} positions")

        # Average over runs
        position_data = defaultdict(lambda: defaultdict(list))

        for run_idx in range(num_runs):
            result = extract_cache_at_positions(model, tokenizer, text, positions)
            for point in result["trajectory"]:
                pos = point["position"]
                for key in ["total_key_norm", "key_norm_per_token",
                            "mean_layer_key_norm", "var_layer_key_norm", "mean_key_std"]:
                    position_data[pos][key].append(point[key])

            if verbose:
                print(f"    Run {run_idx + 1}/{num_runs} complete")

        # Compute averages
        trajectory = []
        for pos in sorted(position_data.keys()):
            point = {"position": pos}
            for key in position_data[pos]:
                values = position_data[pos][key]
                point[f"{key}_mean"] = float(np.mean(values))
                point[f"{key}_std"] = float(np.std(values))
            trajectory.append(point)

        all_trajectories[text_name] = {
            "total_tokens": total_tokens,
            "trajectory": trajectory,
        }

    # H1: Enrichment — is there an increasing trend?
    print(f"\n  Testing H1 (Enrichment)...")
    h1_results = {}
    for text_name, data in all_trajectories.items():
        traj = data["trajectory"]
        positions = [t["position"] for t in traj]
        norms = [t["total_key_norm_mean"] for t in traj]

        rho, p = scipy_stats.spearmanr(positions, norms)
        h1_results[text_name] = {
            "spearman_rho": float(rho),
            "p_value": float(p),
            "increasing": rho > 0.5,
        }
        print(f"    {text_name}: ρ={rho:.3f} (p={p:.4f})")

    # H2: Fatigue — does per-token norm plateau?
    print(f"\n  Testing H2 (Fatigue)...")
    h2_results = {}
    for text_name, data in all_trajectories.items():
        traj = data["trajectory"]
        if len(traj) < 6:
            continue

        positions = [t["position"] for t in traj]
        norms_pt = [t["key_norm_per_token_mean"] for t in traj]

        # Split into first half and second half
        mid = len(traj) // 2
        first_half = norms_pt[:mid]
        second_half = norms_pt[mid:]

        # If second half has lower growth rate, that's fatigue
        first_slope = (first_half[-1] - first_half[0]) / max(1, len(first_half))
        second_slope = (second_half[-1] - second_half[0]) / max(1, len(second_half))

        h2_results[text_name] = {
            "first_half_slope": first_slope,
            "second_half_slope": second_slope,
            "fatigue_detected": second_slope < first_slope * 0.5,
        }
        print(f"    {text_name}: first_slope={first_slope:.4f}, "
              f"second_slope={second_slope:.4f}")

    return {
        "trajectories": {
            name: {
                "total_tokens": data["total_tokens"],
                "trajectory": data["trajectory"],
            }
            for name, data in all_trajectories.items()
        },
        "h1_enrichment": h1_results,
        "h2_fatigue": h2_results,
    }


# ================================================================
# SECTION 4: EXPERIMENT 2 — CONTENT COMPARISON
# ================================================================

def run_content_comparison(model, tokenizer, num_runs: int = 3,
                           seed: Optional[int] = None) -> Dict:
    """
    Compare trajectory shapes across content types.
    Tests H4: Information-dense text grows faster than repetitive text.
    """
    print("\n" + "=" * 60)
    print("  EXPERIMENT 2: CONTENT COMPARISON")
    print("=" * 60)

    # This reuses trajectory data but compares across content types
    texts = {
        "factual": FACTUAL_TEXT.strip(),
        "creative": CREATIVE_TEXT.strip(),
        "repetitive": REPETITIVE_TEXT.strip(),
    }

    growth_rates = defaultdict(list)

    for run_idx in range(num_runs):
        print(f"  Run {run_idx + 1}/{num_runs}")

        for text_name, text in texts.items():
            inputs = tokenizer(text, return_tensors="pt")
            total_tokens = inputs["input_ids"].shape[1]
            del inputs

            # Sample at 25% and 75% of text
            pos_early = max(5, total_tokens // 4)
            pos_late = min(total_tokens - 1, 3 * total_tokens // 4)

            result = extract_cache_at_positions(
                model, tokenizer, text, [pos_early, pos_late])

            if len(result["trajectory"]) >= 2:
                early_norm = result["trajectory"][0]["key_norm_per_token"]
                late_norm = result["trajectory"][1]["key_norm_per_token"]
                growth = (late_norm - early_norm) / early_norm if early_norm > 0 else 0
                growth_rates[text_name].append(growth)

    # Compare growth rates
    analysis = {}
    for text_name, rates in growth_rates.items():
        analysis[text_name] = {
            "mean_growth": float(np.mean(rates)),
            "std_growth": float(np.std(rates)),
            "n": len(rates),
            "bootstrap": bootstrap_ci(rates, seed=seed) if len(rates) >= 3 else None,
        }
        print(f"  {text_name}: mean_growth={np.mean(rates):.4f} +/- {np.std(rates):.4f}")

    # H4: informative > repetitive
    info_growth = growth_rates.get("factual", [0])
    rep_growth = growth_rates.get("repetitive", [0])

    if len(info_growth) >= 3 and len(rep_growth) >= 3:
        t, p = scipy_stats.ttest_ind(info_growth, rep_growth, equal_var=False)
        d = cohens_d(info_growth, rep_growth)
        h4_verdict = {
            "t_statistic": float(t),
            "p_value": float(p),
            "cohens_d": d,
            "info_faster": float(np.mean(info_growth)) > float(np.mean(rep_growth)),
            "interpretation": (
                "Informative text produces FASTER cache growth than repetitive (H4 confirmed)"
                if float(np.mean(info_growth)) > float(np.mean(rep_growth)) and p < 0.05
                else "No significant difference in cache growth rate (H4 rejected)"
            ),
        }
    else:
        h4_verdict = {"error": "insufficient data"}

    print(f"\n  H4: {h4_verdict.get('interpretation', 'N/A')}")

    return {
        "growth_rates": {k: v for k, v in growth_rates.items()},
        "analysis": analysis,
        "h4_verdict": h4_verdict,
    }


# ================================================================
# SECTION 5: EXPERIMENT 3 — TOPIC SHIFT DETECTION
# ================================================================

def run_topic_shift(model, tokenizer, num_runs: int = 3,
                    seed: Optional[int] = None, verbose: bool = False) -> Dict:
    """
    Detect topic shifts via cache discontinuities.
    Tests H3: Topic shifts produce measurable cache statistics change.
    """
    print("\n" + "=" * 60)
    print("  EXPERIMENT 3: TOPIC SHIFT DETECTION")
    print("=" * 60)

    text = TOPIC_SHIFT_TEXT.strip()
    inputs = tokenizer(text, return_tensors="pt")
    total_tokens = inputs["input_ids"].shape[1]
    del inputs

    # Dense sampling for change detection
    positions = list(range(5, total_tokens, 3))
    if total_tokens - 1 not in positions:
        positions.append(total_tokens - 1)

    print(f"  Total tokens: {total_tokens}, sampling at {len(positions)} positions")

    # Collect trajectories
    all_norm_trajectories = []

    for run_idx in range(num_runs):
        result = extract_cache_at_positions(model, tokenizer, text, positions)
        norms = [(p["position"], p["key_norm_per_token"]) for p in result["trajectory"]]
        all_norm_trajectories.append(norms)
        if verbose:
            print(f"    Run {run_idx + 1}/{num_runs} complete")

    # Average across runs
    pos_to_norms = defaultdict(list)
    for traj in all_norm_trajectories:
        for pos, norm in traj:
            pos_to_norms[pos].append(norm)

    avg_trajectory = [(pos, float(np.mean(norms)))
                      for pos, norms in sorted(pos_to_norms.items())]

    # Change point detection: compute first derivative of per-token norm
    if len(avg_trajectory) >= 5:
        positions_arr = np.array([t[0] for t in avg_trajectory])
        norms_arr = np.array([t[1] for t in avg_trajectory])

        # Smooth derivative
        deriv = np.gradient(norms_arr, positions_arr)
        abs_deriv = np.abs(deriv)

        # Find peaks in derivative (potential change points)
        from scipy.signal import find_peaks
        peaks, properties = find_peaks(abs_deriv, height=np.mean(abs_deriv) + np.std(abs_deriv))

        detected_shifts = []
        for peak in peaks:
            detected_shifts.append({
                "position": int(positions_arr[peak]),
                "relative_position": float(positions_arr[peak] / total_tokens),
                "derivative_magnitude": float(abs_deriv[peak]),
            })

        # Compare with known shift positions
        known_shifts = [s["position_approx"] for s in TOPIC_SHIFTS]
        detection_results = []
        for known in known_shifts:
            closest = min(detected_shifts,
                          key=lambda d: abs(d["relative_position"] - known),
                          default=None) if detected_shifts else None
            if closest:
                distance = abs(closest["relative_position"] - known)
                detection_results.append({
                    "known_position": known,
                    "detected_position": closest["relative_position"],
                    "distance": distance,
                    "detected": distance < 0.1,  # Within 10% of text
                })

        detected_count = sum(1 for d in detection_results if d["detected"])

        h3_verdict = {
            "n_known_shifts": len(known_shifts),
            "n_detected": detected_count,
            "detection_rate": detected_count / len(known_shifts) if known_shifts else 0,
            "detection_results": detection_results,
            "total_peaks": len(peaks),
            "interpretation": (
                f"Topic shifts DETECTED via cache ({detected_count}/{len(known_shifts)}) (H3 confirmed)"
                if detected_count > 0 else
                "Topic shifts NOT detected in cache statistics (H3 rejected)"
            ),
        }
    else:
        h3_verdict = {"error": "insufficient trajectory points"}

    print(f"\n  H3: {h3_verdict.get('interpretation', 'N/A')}")

    return {
        "avg_trajectory": avg_trajectory,
        "detected_shifts": detected_shifts if 'detected_shifts' in dir() else [],
        "h3_verdict": h3_verdict,
        "total_tokens": total_tokens,
    }


# ================================================================
# SECTION 6: DRY RUN AND MAIN
# ================================================================

def print_dry_run():
    print("=" * 70)
    print("  TEMPORAL CACHE EVOLUTION — DRY RUN")
    print("=" * 70)
    print()
    print("Experiment 1 (Trajectory): 3 texts × ~20 positions × 3 runs")
    print("Experiment 2 (Content): 3 texts × 2 positions × 3 runs")
    print("Experiment 3 (Topic shift): 1 text × ~50 positions × 3 runs")
    print()
    print("HYPOTHESES:")
    print("  H1: Cache norm increases with position (enrichment)")
    print("  H2: Per-token norm plateaus at some point (fatigue)")
    print("  H3: Topic shifts detectable via cache discontinuities")
    print("  H4: Informative text grows cache faster than repetitive")
    print()
    print("HARDWARE: Runs on GTX 1660 SUPER (6GB). No donated GPU needed.")


def main():
    parser = argparse.ArgumentParser(
        description="Extension C: Temporal Cache Evolution")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--experiment", type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        print_dry_run()
        return

    env = log_environment()
    print("=" * 70)
    print("  EXTENSION C: TEMPORAL CACHE EVOLUTION")
    print("  Liberation Labs / THCoalition")
    print("=" * 70)

    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    model, tokenizer = load_model(args.model, args.quantize)

    all_results = {
        "metadata": {
            "experiment": "Extension C: Temporal Cache Evolution",
            "version": "1.0-publication",
            "timestamp": datetime.now().isoformat(),
            "environment": env,
            "args": vars(args),
            "model": args.model,
        },
    }

    run_all = args.experiment == 0

    if run_all or args.experiment == 1:
        all_results["trajectory"] = run_trajectory(
            model, tokenizer, args.runs, args.seed, args.verbose)

    if run_all or args.experiment == 2:
        all_results["content_comparison"] = run_content_comparison(
            model, tokenizer, args.runs, args.seed)

    if run_all or args.experiment == 3:
        all_results["topic_shift"] = run_topic_shift(
            model, tokenizer, args.runs, args.seed, args.verbose)

    # Save
    output_file = get_output_path(results_dir, "temporal_evolution", args.model, args.quantize)
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    with open(output_file, "rb") as f:
        checksum = hashlib.sha256(f.read()).hexdigest()

    print(f"\n{'='*70}")
    print(f"  TEMPORAL EVOLUTION COMPLETE")
    print(f"{'='*70}")
    print(f"  Results: {output_file}")
    print(f"  SHA-256: {checksum}")


if __name__ == "__main__":
    main()
