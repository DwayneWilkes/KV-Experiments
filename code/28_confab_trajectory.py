# ============================================================
# JiminAI Cricket | PATENT PENDING
# "The Lyra Technique" - Provisional Patent Filed 2026
# Liberation Labs / Digital Disconnections / TheMultiverse.school
# Contact: info@digitaldisconnections.com | (877) 674-8874
# ============================================================
#!/usr/bin/env python3
"""
Experiment 28: Confabulation Trajectory — Encoding vs Generation
=================================================================

Uses Exp 17 token trajectory data to test whether confabulation becomes
distinguishable from factual responses as generation progresses.

At checkpoint 0 (encoding only): Exp 20/22 showed AUROC~0.65 (barely above chance)
At checkpoint 50 (50 tokens generated): Does the generation regime separate them?

This directly tests the two-regime hypothesis from Exp 24.

No GPU needed — uses existing trajectory JSON.

Funding the Commons Hackathon — March 14-15, 2026
Liberation Labs / THCoalition / JiminAI
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy.stats import mannwhitneyu

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
OUTPUT_DIR = RESULTS_DIR / "hackathon"


def main():
    print("=" * 70)
    print("  EXPERIMENT 28: CONFABULATION TRAJECTORY")
    print("  Does confab become separable as generation progresses?")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Load trajectory data
    traj_path = OUTPUT_DIR / "token_trajectory.json"
    with open(traj_path) as f:
        data = json.load(f)

    checkpoints = data["checkpoints"]
    results = data["results"]

    # Filter confabulation task
    confab_results = [r for r in results if r["task"] == "confabulation"]
    factual = [r for r in confab_results if r["label"] == "factual"]
    confabs = [r for r in confab_results if r["label"] == "confabulation"]

    print(f"\n  Confabulation task: {len(factual)} factual, {len(confabs)} confabulation")
    print(f"  Checkpoints: {checkpoints}")

    # Also get deception for comparison
    deception_results = [r for r in results if r["task"] == "deception"]
    honest = [r for r in deception_results if r["label"] == "honest"]
    deceptive = [r for r in deception_results if r["label"] == "deceptive"]

    sycophancy_results = [r for r in results if r["task"] == "sycophancy"]
    non_syc = [r for r in sycophancy_results if r["label"] == "non_sycophantic"]
    syc = [r for r in sycophancy_results if r["label"] == "sycophantic"]

    print(f"  Deception task: {len(honest)} honest, {len(deceptive)} deceptive")
    print(f"  Sycophancy task: {len(non_syc)} non-sycophantic, {len(syc)} sycophantic")

    features = ["norm", "norm_per_token", "key_rank", "key_entropy"]

    # ================================================================
    # TRAJECTORY ANALYSIS: DIVERGENCE OVER TIME
    # ================================================================
    print("\n" + "=" * 70)
    print("  FEATURE DIVERGENCE OVER GENERATION CHECKPOINTS")
    print("=" * 70)

    tasks = [
        ("CONFABULATION", factual, confabs, "factual", "confab"),
        ("DECEPTION", honest, deceptive, "honest", "deceptive"),
        ("SYCOPHANCY", non_syc, syc, "non_syc", "sycophantic"),
    ]

    all_divergence = {}

    for task_name, group_a, group_b, label_a, label_b in tasks:
        print(f"\n  --- {task_name}: {label_a} vs {label_b} ---")
        print(f"  {'Checkpoint':>12} {'Feature':>15} {'Mean_A':>10} {'Mean_B':>10} {'d':>8} {'p':>8}")
        print("  " + "-" * 70)

        task_divergence = {}
        for feat in features:
            feat_divergence = []
            for ci, cp in enumerate(checkpoints):
                # Get feature values at this checkpoint for each group
                vals_a = []
                vals_b = []

                for r in group_a:
                    traj = r.get("trajectory", [])
                    if ci < len(traj) and feat in traj[ci]:
                        vals_a.append(traj[ci][feat])

                for r in group_b:
                    traj = r.get("trajectory", [])
                    if ci < len(traj) and feat in traj[ci]:
                        vals_b.append(traj[ci][feat])

                if len(vals_a) >= 2 and len(vals_b) >= 2:
                    mean_a = np.mean(vals_a)
                    mean_b = np.mean(vals_b)
                    pooled_std = np.sqrt((np.var(vals_a, ddof=1) * (len(vals_a) - 1) +
                                         np.var(vals_b, ddof=1) * (len(vals_b) - 1)) /
                                        (len(vals_a) + len(vals_b) - 2))
                    d = (mean_a - mean_b) / pooled_std if pooled_std > 0 else 0

                    _, p = mannwhitneyu(vals_a, vals_b, alternative='two-sided')
                    feat_divergence.append({
                        "checkpoint": cp,
                        "d": float(d),
                        "p": float(p),
                        "mean_a": float(mean_a),
                        "mean_b": float(mean_b),
                    })

                    print(f"  {cp:>12} {feat:>15} {mean_a:>10.2f} {mean_b:>10.2f} {d:>8.2f} {p:>8.4f}")

            task_divergence[feat] = feat_divergence

        all_divergence[task_name] = task_divergence

    # ================================================================
    # THE KEY TEST: CONFABULATION AT CHECKPOINT 0 vs CHECKPOINT 50
    # ================================================================
    print("\n" + "=" * 70)
    print("  THE TWO-REGIME TEST")
    print("=" * 70)

    for task_name, group_a, group_b, label_a, label_b in tasks:
        print(f"\n  {task_name}:")
        td = all_divergence.get(task_name, {})

        for feat in features:
            fd = td.get(feat, [])
            if len(fd) >= 2:
                first = fd[0]
                last = fd[-1]
                print(f"    {feat:<15}: checkpoint 0 |d|={abs(first['d']):.2f} (p={first['p']:.4f}) "
                      f"--> checkpoint {last['checkpoint']} |d|={abs(last['d']):.2f} (p={last['p']:.4f})")

    # ================================================================
    # CONFABULATION GENERATED TEXT ANALYSIS
    # ================================================================
    print("\n" + "=" * 70)
    print("  GENERATED TEXT ANALYSIS")
    print("=" * 70)

    print(f"\n  Factual responses:")
    for r in factual:
        print(f"    Q: {r['user_prompt']}")
        print(f"    A: {r['generated_text'][:80]}...")
        print()

    print(f"  Confabulation responses:")
    for r in confabs:
        print(f"    Q: {r['user_prompt']}")
        print(f"    A: {r['generated_text'][:80]}...")
        print()

    # ================================================================
    # SYNTHESIS
    # ================================================================
    print("=" * 70)
    print("  SYNTHESIS")
    print("=" * 70)

    # Check if confabulation diverges less than deception
    confab_max_d = 0
    deception_max_d = 0
    for feat in features:
        cd = all_divergence.get("CONFABULATION", {}).get(feat, [])
        dd = all_divergence.get("DECEPTION", {}).get(feat, [])
        if cd:
            confab_max_d = max(confab_max_d, max(abs(x["d"]) for x in cd))
        if dd:
            deception_max_d = max(deception_max_d, max(abs(x["d"]) for x in dd))

    print(f"\n  Maximum |d| across all features and checkpoints:")
    print(f"    Deception:     {deception_max_d:.2f}")
    print(f"    Confabulation: {confab_max_d:.2f}")

    if confab_max_d < deception_max_d * 0.5:
        print(f"\n  >> Confabulation is MUCH HARDER to detect than deception,")
        print(f"     even in the generation regime. The model confabulates with")
        print(f"     the same cognitive effort as factual answering.")
    elif confab_max_d > 1.0:
        print(f"\n  >> Confabulation IS detectable from generation features!")
        print(f"     The generation regime can separate facts from confab.")
    else:
        print(f"\n  >> Confabulation shows MODERATE generation-time signal.")
        print(f"     Weaker than deception but potentially usable.")

    # Save
    output = {
        "experiment": "28_confab_trajectory",
        "timestamp": datetime.now().isoformat(),
        "n_factual": len(factual),
        "n_confab": len(confabs),
        "checkpoints": checkpoints,
        "divergence": all_divergence,
        "max_d_deception": float(deception_max_d),
        "max_d_confabulation": float(confab_max_d),
    }

    output_path = OUTPUT_DIR / "confab_trajectory.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x)

    print(f"\n  Results saved to: {output_path}")


if __name__ == "__main__":
    main()
