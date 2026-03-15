#!/usr/bin/env python3
"""Analyze token-by-token detection trajectory from Exp 17.

Key questions:
1. At what generation step does honest vs deceptive confidence diverge?
2. Is the step-0 "detection" an artifact (confirmed by adversarial controls)?
3. Do per-layer profiles show earlier divergence than scalar features?
"""
import json
import numpy as np
from collections import defaultdict

with open("results/hackathon/token_trajectory.json") as f:
    data = json.load(f)

print("=" * 85)
print("  TRAJECTORY ANALYSIS — Exp 17")
print(f"  Model: {data['model']}")
print(f"  Prompts: {data['n_prompts']}, Checkpoints: {data['checkpoints']}")
print("=" * 85)

# ── 1. Classifier confidence trajectories ──────────────────────────────
print("\n  1. CLASSIFIER CONFIDENCE TRAJECTORIES (P(deceptive))")
print("  " + "-" * 78)

for task in ["deception", "sycophancy", "confabulation"]:
    task_results = [r for r in data["results"] if r["task"] == task]
    if not task_results:
        continue

    # Get unique labels
    labels = sorted(set(r["label"] for r in task_results))

    print(f"\n  Task: {task}")
    print(f"  {'Step':<8}", end="")
    for label in labels:
        print(f"{'  ' + label + ' conf':>18}", end="")
    print(f"{'  delta':>10}")
    print("  " + "-" * 60)

    # Collect per-step confidences by label
    all_steps = set()
    step_confs = defaultdict(lambda: defaultdict(list))

    for r in task_results:
        for t in r["trajectory"]:
            step = t["step"]
            all_steps.add(step)
            if t.get("prediction"):
                conf = t["prediction"]["probabilities"].get("deceptive", 0)
                step_confs[step][r["label"]].append(conf)

    for step in sorted(all_steps):
        print(f"  {step:<8}", end="")
        means = {}
        for label in labels:
            vals = step_confs[step].get(label, [])
            if vals:
                m = np.mean(vals)
                means[label] = m
                print(f"{m:18.3f}", end="")
            else:
                print(f"{'n/a':>18}", end="")

        if len(means) == 2:
            vals = list(means.values())
            print(f"{abs(vals[1] - vals[0]):10.3f}", end="")
        print()

# ── 2. Feature trajectories (raw values, not d) ──────────────────────
print("\n\n  2. MEAN FEATURE VALUES BY CONDITION AND STEP")
print("  " + "-" * 78)

for task in ["deception", "sycophancy", "confabulation"]:
    task_results = [r for r in data["results"] if r["task"] == task]
    if not task_results:
        continue

    labels = sorted(set(r["label"] for r in task_results))

    print(f"\n  Task: {task}")
    print(f"  {'Step':<6}", end="")
    for label in labels:
        print(f"  {'norm_' + label:>14} {'rank_' + label:>14}", end="")
    print()
    print("  " + "-" * 70)

    step_features = defaultdict(lambda: defaultdict(lambda: {"norm": [], "rank": []}))
    all_steps = set()

    for r in task_results:
        for t in r["trajectory"]:
            step = t["step"]
            all_steps.add(step)
            step_features[step][r["label"]]["norm"].append(t["norm"])
            step_features[step][r["label"]]["rank"].append(t["key_rank"])

    for step in sorted(all_steps):
        print(f"  {step:<6}", end="")
        for label in labels:
            norms = step_features[step][label]["norm"]
            ranks = step_features[step][label]["rank"]
            if norms:
                print(f"  {np.mean(norms):14.1f} {np.mean(ranks):14.1f}", end="")
            else:
                print(f"  {'n/a':>14} {'n/a':>14}", end="")
        print()

# ── 3. Norm per token (length-normalized) ─────────────────────────────
print("\n\n  3. NORM PER TOKEN (length-normalized feature)")
print("  " + "-" * 78)

for task in ["deception"]:
    task_results = [r for r in data["results"] if r["task"] == task]
    labels = sorted(set(r["label"] for r in task_results))

    print(f"\n  Task: {task}")
    print(f"  {'Step':<6}", end="")
    for label in labels:
        print(f"  {'npt_' + label:>14} {'tokens':>8}", end="")
    print(f"  {'npt_delta':>12}")
    print("  " + "-" * 70)

    step_npt = defaultdict(lambda: defaultdict(lambda: {"npt": [], "tok": []}))
    all_steps = set()

    for r in task_results:
        for t in r["trajectory"]:
            step = t["step"]
            all_steps.add(step)
            step_npt[step][r["label"]]["npt"].append(t["norm_per_token"])
            step_npt[step][r["label"]]["tok"].append(t["n_tokens"])

    for step in sorted(all_steps):
        print(f"  {step:<6}", end="")
        means = {}
        for label in labels:
            npts = step_npt[step][label]["npt"]
            toks = step_npt[step][label]["tok"]
            if npts:
                m = np.mean(npts)
                means[label] = m
                print(f"  {m:14.4f} {np.mean(toks):8.0f}", end="")

        if len(means) == 2:
            vals = list(means.values())
            print(f"  {vals[1] - vals[0]:12.4f}", end="")
        print()

# ── 4. Per-layer profile divergence ────────────────────────────────────
print("\n\n  4. PER-LAYER PROFILE COSINE SIMILARITY (honest vs deceptive at each step)")
print("  " + "-" * 78)

for task in ["deception", "sycophancy", "confabulation"]:
    task_results = [r for r in data["results"] if r["task"] == task]
    if not task_results:
        continue

    labels = sorted(set(r["label"] for r in task_results))
    if len(labels) != 2:
        continue

    step_profiles = defaultdict(lambda: defaultdict(list))
    all_steps = set()

    for r in task_results:
        for t in r["trajectory"]:
            step = t["step"]
            all_steps.add(step)
            if "layer_profile" in t:
                step_profiles[step][r["label"]].append(np.array(t["layer_profile"]))

    print(f"\n  Task: {task} ({labels[0]} vs {labels[1]})")
    print(f"  {'Step':<8} {'Cosine Sim':>12} {'L2 Distance':>14} {'Profile Info':>30}")
    print("  " + "-" * 70)

    for step in sorted(all_steps):
        p0 = step_profiles[step].get(labels[0], [])
        p1 = step_profiles[step].get(labels[1], [])

        if p0 and p1:
            mean0 = np.mean(p0, axis=0)
            mean1 = np.mean(p1, axis=0)

            cos_sim = np.dot(mean0, mean1) / (np.linalg.norm(mean0) * np.linalg.norm(mean1))
            l2_dist = np.linalg.norm(mean0 - mean1)

            # Find layer with max difference
            diffs = np.abs(mean0 - mean1)
            max_layer = np.argmax(diffs)

            print(f"  {step:<8} {cos_sim:12.6f} {l2_dist:14.2f} {'  max diff layer ' + str(max_layer) + ' (' + f'{diffs[max_layer]:.1f}' + ')':>30}")

# ── 5. Key question: when does discrimination actually emerge? ────────
print("\n\n  5. DISCRIMINATION EMERGENCE (per-prompt classifier trajectory)")
print("  " + "-" * 78)

for task in ["deception"]:
    task_results = [r for r in data["results"] if r["task"] == task]

    print(f"\n  Task: {task}")
    for r in task_results:
        label = r["label"]
        prompt = r["user_prompt"][:50]

        print(f"\n  [{label:>10}] {prompt}")
        print(f"    Step  P(honest) P(decept) P(confab) Prediction  Correct?")

        for t in r["trajectory"]:
            if t.get("prediction"):
                pred = t["prediction"]
                p_h = pred["probabilities"].get("honest", 0)
                p_d = pred["probabilities"].get("deceptive", 0)
                p_c = pred["probabilities"].get("confabulation", 0)
                pred_label = pred["label"]

                # Is prediction correct?
                correct = (pred_label == label)
                marker = "OK" if correct else "XX"

                print(f"    {t['step']:>4}  {p_h:9.3f} {p_d:9.3f} {p_c:9.3f}  {pred_label:>12}  {marker}")

print("\n" + "=" * 85)
print("  CONCLUSIONS")
print("=" * 85)
print("""
  1. Step-0 detection is CONFIRMED ARTIFACT (adversarial controls, Exp 17b)
     - Classifier trained on post-generation features extrapolates incorrectly
     - All prompts classified as "deceptive" at encoding regardless of actual label

  2. Raw feature divergence at encoding is dominated by PROMPT LENGTH confound
     - Deception d shrinks as tokens accumulate (length difference washes out)
     - Sycophancy d grows (real signal accumulating during generation)

  3. Per-layer profiles may contain real encoding-time signal
     - Cosine similarity and L2 distance track profile geometry changes
     - Need per-layer classifier (not 4-scalar RF) for proper encoding detection

  4. Open question: retrain classifier on trajectory data with step as feature
     - Could learn step-dependent decision boundaries
     - Would separate length confound from real semantic signal
""")
