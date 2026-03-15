#!/usr/bin/env python3
"""
Analyze per-layer trajectory directions from Exp 17 (token_trajectory).

For each task (deception, sycophancy, confabulation), computes a "direction"
vector in 28-dimensional layer-profile space that separates the two conditions.
Tracks how this direction evolves during generation, measures stability via
cosine similarity, computes AUROC at each step, and compares directions across
tasks.

Input:  results/hackathon/token_trajectory.json
Output: Formatted report to stdout
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

INPUT_PATH = Path(__file__).resolve().parent.parent / "results" / "hackathon" / "token_trajectory.json"

# Task definitions: task_name -> (positive_label, negative_label)
# Direction = positive_mean - negative_mean
TASK_CONFIG = {
    "deception":     ("deceptive",    "honest"),
    "sycophancy":    ("sycophantic",  "corrective"),
    "confabulation": ("confabulation","factual"),
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def compute_auroc(profiles_pos, profiles_neg, direction):
    """
    Project positive and negative profiles onto a direction vector and
    compute AUROC.  Returns AUROC or None if degenerate.
    """
    if len(profiles_pos) == 0 or len(profiles_neg) == 0:
        return None
    scores_pos = [np.dot(p, direction) for p in profiles_pos]
    scores_neg = [np.dot(p, direction) for p in profiles_neg]
    y_true = [1] * len(scores_pos) + [0] * len(scores_neg)
    y_score = scores_pos + scores_neg
    if len(set(y_true)) < 2:
        return None
    try:
        return roc_auc_score(y_true, y_score)
    except ValueError:
        return None


def fmt(val, width=8, decimals=4):
    """Format a float or None for tabular display."""
    if val is None:
        return " " * (width - 3) + "N/A"
    return f"{val:{width}.{decimals}f}"


def print_header(title: str):
    print()
    print("=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_subheader(title: str):
    print()
    print(f"--- {title} ---")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data():
    if not INPUT_PATH.exists():
        print(f"ERROR: Input file not found: {INPUT_PATH}", file=sys.stderr)
        sys.exit(1)

    with open(INPUT_PATH) as f:
        data = json.load(f)

    return data


def extract_task_data(data):
    """
    Organize data by task -> label -> list of result entries.
    Each result entry's trajectory is indexed by step number.
    """
    task_data = {}
    for task_name, (pos_label, neg_label) in TASK_CONFIG.items():
        results = [r for r in data["results"] if r["task"] == task_name]
        pos_results = [r for r in results if r["label"] == pos_label]
        neg_results = [r for r in results if r["label"] == neg_label]

        # Build step -> profile mapping for each result
        pos_by_step = []
        for r in pos_results:
            step_map = {}
            for t in r["trajectory"]:
                step_map[t["step"]] = np.array(t["layer_profile"])
            pos_by_step.append(step_map)

        neg_by_step = []
        for r in neg_results:
            step_map = {}
            for t in r["trajectory"]:
                step_map[t["step"]] = np.array(t["layer_profile"])
            neg_by_step.append(step_map)

        # Find common steps where ALL prompts in this task have data
        all_step_sets = [set(sm.keys()) for sm in pos_by_step + neg_by_step]
        common_steps = sorted(set.intersection(*all_step_sets)) if all_step_sets else []

        task_data[task_name] = {
            "pos_label": pos_label,
            "neg_label": neg_label,
            "pos_by_step": pos_by_step,
            "neg_by_step": neg_by_step,
            "common_steps": common_steps,
            "n_pos": len(pos_results),
            "n_neg": len(neg_results),
        }

    return task_data


# ---------------------------------------------------------------------------
# Analysis per task
# ---------------------------------------------------------------------------

def analyze_task(task_name, td):
    """Run all analyses for a single task. Returns direction vectors by step."""
    pos_label = td["pos_label"]
    neg_label = td["neg_label"]
    common_steps = td["common_steps"]
    pos_by_step = td["pos_by_step"]
    neg_by_step = td["neg_by_step"]
    n_layers = len(pos_by_step[0][common_steps[0]])

    print_header(f"TASK: {task_name.upper()}")
    print(f"  Positive class: {pos_label} (n={td['n_pos']})")
    print(f"  Negative class: {neg_label} (n={td['n_neg']})")
    print(f"  Common steps:   {common_steps}")
    print(f"  N layers:       {n_layers}")

    # ------------------------------------------------------------------
    # 1. Compute direction at each step
    # ------------------------------------------------------------------
    directions = {}   # step -> direction vector
    magnitudes = {}   # step -> L2 norm of direction
    pos_means = {}
    neg_means = {}

    for step in common_steps:
        pos_profiles = np.array([sm[step] for sm in pos_by_step])
        neg_profiles = np.array([sm[step] for sm in neg_by_step])
        pos_mean = pos_profiles.mean(axis=0)
        neg_mean = neg_profiles.mean(axis=0)
        direction = pos_mean - neg_mean
        directions[step] = direction
        magnitudes[step] = np.linalg.norm(direction)
        pos_means[step] = pos_mean
        neg_means[step] = neg_mean

    print_subheader("1. Direction magnitude across generation steps")
    print(f"  {'Step':>6s}  {'|direction|':>12s}  {'Max layer delta':>15s}  {'Max layer idx':>14s}")
    for step in common_steps:
        d = directions[step]
        max_idx = int(np.argmax(np.abs(d)))
        print(f"  {step:6d}  {magnitudes[step]:12.4f}  {d[max_idx]:15.4f}  {max_idx:14d}")

    # ------------------------------------------------------------------
    # 2. Cosine similarity of direction across steps (stability)
    # ------------------------------------------------------------------
    print_subheader("2. Direction stability (cosine similarity vs step-0 direction)")
    d0 = directions[common_steps[0]]
    print(f"  {'Step':>6s}  {'cos(d0, d_step)':>16s}  {'cos(d_prev, d_step)':>20s}")
    prev_d = d0
    for i, step in enumerate(common_steps):
        cs0 = cosine_sim(d0, directions[step])
        cs_prev = cosine_sim(prev_d, directions[step])
        prev_label = f"{cs_prev:.6f}" if i > 0 else "---"
        print(f"  {step:6d}  {cs0:16.6f}  {prev_label:>20s}")
        prev_d = directions[step]

    # Also measure the angle between first and last direction
    d_last = directions[common_steps[-1]]
    angle_deg = np.degrees(np.arccos(np.clip(cosine_sim(d0, d_last), -1, 1)))
    print(f"\n  Angle between step-0 and step-{common_steps[-1]} direction: {angle_deg:.2f} degrees")

    # ------------------------------------------------------------------
    # 3. AUROC at each step (projecting onto step-0 direction)
    # ------------------------------------------------------------------
    print_subheader("3. AUROC at each step (projected onto step-0 direction)")
    print(f"  {'Step':>6s}  {'AUROC (d0)':>11s}  {'AUROC (d_step)':>15s}")
    for step in common_steps:
        pos_profiles = [sm[step] for sm in pos_by_step]
        neg_profiles = [sm[step] for sm in neg_by_step]
        auroc_d0 = compute_auroc(pos_profiles, neg_profiles, d0)
        auroc_ds = compute_auroc(pos_profiles, neg_profiles, directions[step])
        print(f"  {step:6d}  {fmt(auroc_d0, 11, 4)}  {fmt(auroc_ds, 15, 4)}")

    # ------------------------------------------------------------------
    # 4. Layer-level contribution to direction at each step
    # ------------------------------------------------------------------
    print_subheader("4. Top-5 layers contributing to direction (by |delta|)")
    for step in common_steps:
        d = directions[step]
        ranked = np.argsort(np.abs(d))[::-1][:5]
        top_str = ", ".join(f"L{idx}({d[idx]:+.3f})" for idx in ranked)
        print(f"  Step {step:3d}: {top_str}")

    # Layer importance change: which layers change rank most between first and last?
    print_subheader("4b. Layer rank change from step 0 to last step")
    d_first = directions[common_steps[0]]
    d_last_step = directions[common_steps[-1]]
    rank_first = np.argsort(np.argsort(np.abs(d_first))[::-1])
    rank_last = np.argsort(np.argsort(np.abs(d_last_step))[::-1])
    rank_change = rank_last.astype(int) - rank_first.astype(int)

    movers = np.argsort(np.abs(rank_change))[::-1][:5]
    print(f"  Biggest rank movers (layer -> rank_change):")
    for idx in movers:
        print(f"    L{idx}: rank {rank_first[idx]} -> {rank_last[idx]} (change: {rank_change[idx]:+d})")

    return directions, common_steps


# ---------------------------------------------------------------------------
# Cross-task comparison
# ---------------------------------------------------------------------------

def cross_task_analysis(all_directions, all_steps):
    """Compare direction vectors across tasks."""
    print_header("CROSS-TASK DIRECTION COMPARISON")

    tasks = list(all_directions.keys())

    # Find globally common steps
    global_common = sorted(set.intersection(*[set(all_steps[t]) for t in tasks]))
    print(f"  Steps common to all tasks: {global_common}")

    if not global_common:
        print("  WARNING: No common steps across all tasks. Comparing at step 0 only if available.")
        # Fall back to step 0 which should exist everywhere
        global_common = [0]

    print_subheader("Pairwise cosine similarity of direction vectors")
    for step in global_common:
        print(f"\n  Step {step}:")
        # Print a matrix header
        header = "          " + "".join(f"{t:>16s}" for t in tasks)
        print(header)
        for t1 in tasks:
            row = f"  {t1:>8s}"
            for t2 in tasks:
                d1 = all_directions[t1].get(step)
                d2 = all_directions[t2].get(step)
                if d1 is not None and d2 is not None:
                    cs = cosine_sim(d1, d2)
                    row += f"{cs:16.4f}"
                else:
                    row += f"{'N/A':>16s}"
            print(row)

    # Show the angle interpretations for step 0
    print_subheader("Angle between task directions at step 0")
    step = 0
    for i, t1 in enumerate(tasks):
        for t2 in tasks[i+1:]:
            d1 = all_directions[t1].get(step)
            d2 = all_directions[t2].get(step)
            if d1 is not None and d2 is not None:
                cs = cosine_sim(d1, d2)
                angle = np.degrees(np.arccos(np.clip(cs, -1, 1)))
                print(f"  {t1} vs {t2}: cos={cs:.4f}, angle={angle:.1f} deg")
                if angle < 30:
                    interpretation = "NEARLY ALIGNED (same geometric direction)"
                elif angle < 60:
                    interpretation = "PARTIALLY ALIGNED"
                elif angle < 120:
                    interpretation = "ROUGHLY ORTHOGONAL (independent directions)"
                elif angle < 150:
                    interpretation = "PARTIALLY OPPOSED"
                else:
                    interpretation = "NEARLY OPPOSED (anti-correlated)"
                print(f"    -> {interpretation}")

    # Track cross-task cosine similarity across generation steps
    print_subheader("Cross-task direction cosine similarity evolution")
    print(f"  {'Step':>6s}", end="")
    pairs = []
    for i, t1 in enumerate(tasks):
        for t2 in tasks[i+1:]:
            pair_label = f"{t1[:3]}v{t2[:3]}"
            pairs.append((t1, t2, pair_label))
            print(f"  {pair_label:>12s}", end="")
    print()

    for step in global_common:
        print(f"  {step:6d}", end="")
        for t1, t2, _ in pairs:
            d1 = all_directions[t1].get(step)
            d2 = all_directions[t2].get(step)
            if d1 is not None and d2 is not None:
                cs = cosine_sim(d1, d2)
                print(f"  {cs:12.4f}", end="")
            else:
                print(f"  {'N/A':>12s}", end="")
        print()


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def summary_statistics(task_data):
    """Print overall profile statistics per task for context."""
    print_header("PROFILE SUMMARY STATISTICS")

    for task_name, td in task_data.items():
        pos_label = td["pos_label"]
        neg_label = td["neg_label"]
        step0_profiles_pos = np.array([sm[0] for sm in td["pos_by_step"]])
        step0_profiles_neg = np.array([sm[0] for sm in td["neg_by_step"]])

        print_subheader(f"{task_name}: Step-0 profile statistics")
        print(f"  {pos_label} mean effective rank: {step0_profiles_pos.mean():.2f} "
              f"(std: {step0_profiles_pos.std():.2f})")
        print(f"  {neg_label} mean effective rank: {step0_profiles_neg.mean():.2f} "
              f"(std: {step0_profiles_neg.std():.2f})")

        all_profiles = np.vstack([step0_profiles_pos, step0_profiles_neg])
        layer_stds = all_profiles.std(axis=0)
        most_variable = np.argsort(layer_stds)[::-1][:5]
        print(f"  Most variable layers (across all prompts): "
              + ", ".join(f"L{i}(std={layer_stds[i]:.2f})" for i in most_variable))


# ---------------------------------------------------------------------------
# Projection trajectory visualization (text-based)
# ---------------------------------------------------------------------------

def projection_trajectories(task_data):
    """Show how individual prompts project onto the step-0 direction over time."""
    print_header("PROJECTION TRAJECTORIES (individual prompts onto step-0 direction)")

    for task_name, td in task_data.items():
        pos_label = td["pos_label"]
        neg_label = td["neg_label"]
        common_steps = td["common_steps"]
        pos_by_step = td["pos_by_step"]
        neg_by_step = td["neg_by_step"]

        # Step-0 direction
        pos_profiles_0 = np.array([sm[0] for sm in pos_by_step])
        neg_profiles_0 = np.array([sm[0] for sm in neg_by_step])
        d0 = pos_profiles_0.mean(axis=0) - neg_profiles_0.mean(axis=0)

        # Normalize direction for interpretable projection values
        d0_norm = d0 / (np.linalg.norm(d0) + 1e-12)

        print_subheader(f"{task_name}: Projection onto step-0 direction (normalized)")

        # Header
        step_headers = "".join(f"{s:>8d}" for s in common_steps)
        print(f"  {'Prompt':>20s} {'Label':>12s}  {step_headers}")
        print(f"  {'------':>20s} {'-----':>12s}  " + "-" * (8 * len(common_steps)))

        for i, sm in enumerate(pos_by_step):
            projections = "".join(
                f"{np.dot(sm[step], d0_norm):8.2f}" if step in sm else f"{'--':>8s}"
                for step in common_steps
            )
            print(f"  {f'{pos_label}_{i}':>20s} {pos_label:>12s}  {projections}")

        for i, sm in enumerate(neg_by_step):
            projections = "".join(
                f"{np.dot(sm[step], d0_norm):8.2f}" if step in sm else f"{'--':>8s}"
                for step in common_steps
            )
            print(f"  {f'{neg_label}_{i}':>20s} {neg_label:>12s}  {projections}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("  TRAJECTORY DIRECTION ANALYSIS  --  Exp 17 Token Trajectory")
    print("  Analyzing how behavioral directions in layer-profile space evolve")
    print("  during autoregressive generation.")
    print("=" * 80)

    data = load_data()
    print(f"\n  Model:       {data['model']}")
    print(f"  Timestamp:   {data['timestamp']}")
    print(f"  N prompts:   {data['n_prompts']}")
    print(f"  Tasks:       {data['tasks']}")

    task_data = extract_task_data(data)

    # Summary stats first
    summary_statistics(task_data)

    # Per-task analysis
    all_directions = {}
    all_steps = {}
    for task_name, td in task_data.items():
        directions, common_steps = analyze_task(task_name, td)
        all_directions[task_name] = directions
        all_steps[task_name] = common_steps

    # Cross-task comparison
    cross_task_analysis(all_directions, all_steps)

    # Individual prompt projections
    projection_trajectories(task_data)

    # ---------------------------------------------------------------------------
    # Final synthesis
    # ---------------------------------------------------------------------------
    print_header("SYNTHESIS")
    print()

    for task_name in TASK_CONFIG:
        td = task_data[task_name]
        steps = td["common_steps"]
        dirs = all_directions[task_name]

        if len(steps) < 2:
            print(f"  {task_name}: Insufficient common steps for analysis.")
            continue

        d0 = dirs[steps[0]]
        d_last = dirs[steps[-1]]
        stability = cosine_sim(d0, d_last)
        mag_change = np.linalg.norm(d_last) / (np.linalg.norm(d0) + 1e-12)

        # AUROC at first and last step using step-0 direction
        pos_profiles_first = [sm[steps[0]] for sm in td["pos_by_step"]]
        neg_profiles_first = [sm[steps[0]] for sm in td["neg_by_step"]]
        auroc_first = compute_auroc(pos_profiles_first, neg_profiles_first, d0)

        pos_profiles_last = [sm[steps[-1]] for sm in td["pos_by_step"]]
        neg_profiles_last = [sm[steps[-1]] for sm in td["neg_by_step"]]
        auroc_last = compute_auroc(pos_profiles_last, neg_profiles_last, d0)

        print(f"  {task_name.upper()}:")
        print(f"    Direction stability (cos step-0 vs step-{steps[-1]}): {stability:.4f}")
        print(f"    Magnitude ratio (last/first): {mag_change:.4f}")
        if stability > 0.95:
            print(f"    -> Direction is HIGHLY STABLE during generation")
        elif stability > 0.8:
            print(f"    -> Direction is MODERATELY STABLE but rotates somewhat")
        else:
            print(f"    -> Direction ROTATES SIGNIFICANTLY -- generation creates new structure")

        if auroc_first is not None and auroc_last is not None:
            print(f"    AUROC at step {steps[0]}: {auroc_first:.4f}")
            print(f"    AUROC at step {steps[-1]}: {auroc_last:.4f}")
            if auroc_last > auroc_first + 0.05:
                print(f"    -> Separation IMPROVES during generation")
            elif auroc_last < auroc_first - 0.05:
                print(f"    -> Separation DEGRADES during generation")
            else:
                print(f"    -> Separation is STABLE during generation")
        print()

    # Cross-task angle summary
    tasks = list(TASK_CONFIG.keys())
    print("  CROSS-TASK ANGLES (step 0):")
    for i, t1 in enumerate(tasks):
        for t2 in tasks[i+1:]:
            d1 = all_directions[t1].get(0)
            d2 = all_directions[t2].get(0)
            if d1 is not None and d2 is not None:
                cs = cosine_sim(d1, d2)
                angle = np.degrees(np.arccos(np.clip(cs, -1, 1)))
                print(f"    {t1} vs {t2}: {angle:.1f} degrees")

    print()
    print("=" * 80)
    print("  Analysis complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()
