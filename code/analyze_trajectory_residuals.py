#!/usr/bin/env python3
"""
Cache Residual Analysis -- Exp 17 Token Trajectories

Concept: At each generation step, subtract the encoding-phase (step 0) features
from the current step's features. This gives the "generation residual" --the
signal contributed by generation only, with the system prompt fingerprint removed.

Classifying residuals tests whether deception/sycophancy/confabulation create
generation-specific geometric signal beyond what the system prompt encodes.

Input: token_trajectory.json from Exp 17
Output: Formatted report to stdout
"""

import json
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
INPUT_PATH = Path(r"C:\Users\Thomas\Desktop\LiberationLabs\KV-Experiments\results\hackathon\token_trajectory.json")

# Task condition pairs: (task_name, ground_truth_label, aligned_label, misaligned_label)
TASK_PAIRS = {
    "deception":     ("honest",     "deceptive"),
    "sycophancy":    ("corrective", "sycophantic"),
    "confabulation": ("factual",    "confabulation"),
}

# Common checkpoints present across most prompts (for aligned comparison)
# We use n_generated values that appear in most trajectories
COMMON_STEPS = [1, 2, 3, 5, 8, 10]


def load_data():
    with open(INPUT_PATH) as f:
        data = json.load(f)
    print(f"Loaded {len(data['results'])} prompts from {data['model']}")
    print(f"Timestamp: {data['timestamp']}")
    print(f"Checkpoints: {data['checkpoints']}")
    print(f"Tasks: {data['tasks']}")
    return data


def compute_residuals(trajectory):
    """Subtract step-0 features from all subsequent steps."""
    step0 = trajectory[0]
    assert step0["step"] == 0 and step0["n_generated"] == 0, "First entry must be encoding step"

    profile0 = np.array(step0["layer_profile"])
    residuals = []

    for t in trajectory[1:]:  # skip step 0
        n_gen = t["n_generated"]
        norm_res = t["norm"] - step0["norm"]
        rank_res = t["key_rank"] - step0["key_rank"]
        entropy_res = t["key_entropy"] - step0["key_entropy"]
        npt_res = (t["norm"] - step0["norm"]) / n_gen if n_gen > 0 else 0.0
        profile_res = np.array(t["layer_profile"]) - profile0

        residuals.append({
            "step": t["step"],
            "n_generated": n_gen,
            "norm_residual": norm_res,
            "rank_residual": rank_res,
            "entropy_residual": entropy_res,
            "norm_per_gen_residual": npt_res,
            "profile_residual": profile_res,
        })

    return residuals


def cohens_d(a, b):
    """Compute Cohen's d between two arrays. Positive = a > b."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    ma, mb = np.mean(a), np.mean(b)
    sa, sb = np.std(a, ddof=1), np.std(b, ddof=1)
    pooled = np.sqrt(((na - 1) * sa**2 + (nb - 1) * sb**2) / (na + nb - 2))
    if pooled == 0:
        return 0.0
    return (ma - mb) / pooled


def cosine_sim(a, b):
    """Cosine similarity between two vectors."""
    dot = np.dot(a, b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def auroc_manual(labels, scores):
    """Simple AUROC via sorting. labels: 0/1, scores: float."""
    if len(set(labels)) < 2:
        return float("nan")
    pairs = sorted(zip(scores, labels), key=lambda x: -x[0])
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    tp = 0
    auc = 0.0
    for score, label in pairs:
        if label == 1:
            tp += 1
        else:
            auc += tp
    return auc / (n_pos * n_neg) if (n_pos * n_neg) > 0 else float("nan")


def print_header(title, char="="):
    w = 80
    print()
    print(char * w)
    print(f" {title}")
    print(char * w)


def print_subheader(title, char="-"):
    print(f"\n  {char * 3} {title} {char * 3}")


# ---------------------------------------------------------------------------
# Main Analysis
# ---------------------------------------------------------------------------
def main():
    data = load_data()

    # Organize results by task and condition
    by_task = defaultdict(lambda: defaultdict(list))
    for r in data["results"]:
        by_task[r["task"]][r["label"]].append(r)

    # Compute residuals for every prompt
    all_residuals = {}  # (task, label, prompt_idx) -> [residual dicts]
    for task, labels in by_task.items():
        for label, prompts in labels.items():
            for i, p in enumerate(prompts):
                res = compute_residuals(p["trajectory"])
                all_residuals[(task, label, i)] = res

    # -----------------------------------------------------------------------
    # Section 1: Residual Summary at Each Step
    # -----------------------------------------------------------------------
    print_header("SECTION 1: GENERATION RESIDUALS AT COMMON CHECKPOINTS")
    print("\n  Residual = value[step_N] - value[step_0]  (encoding fingerprint removed)")

    for task, (aligned_label, misaligned_label) in TASK_PAIRS.items():
        print_subheader(f"Task: {task.upper()} --{aligned_label} vs {misaligned_label}")

        # Find which steps are available for ALL prompts in this task
        all_steps_sets = []
        for label in [aligned_label, misaligned_label]:
            for i in range(len(by_task[task][label])):
                steps = {r["step"] for r in all_residuals[(task, label, i)]}
                all_steps_sets.append(steps)
        common = sorted(set.intersection(*all_steps_sets)) if all_steps_sets else []

        if not common:
            print("    No common steps across all prompts in this task pair.")
            continue

        # Print table header
        print(f"\n  {'Step':>6}  {'Cond':>12}  {'norm_res':>10}  {'rank_res':>10}  {'ent_res':>10}  {'norm/gen':>10}")
        print(f"  {'':>6}  {'':>12}  {'(mean+/-sd)':>10}  {'(mean+/-sd)':>10}  {'(mean+/-sd)':>10}  {'(mean+/-sd)':>10}")
        print(f"  {'-'*6}  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")

        for step in common:
            for label in [aligned_label, misaligned_label]:
                norms, ranks, ents, npts = [], [], [], []
                for i in range(len(by_task[task][label])):
                    for r in all_residuals[(task, label, i)]:
                        if r["step"] == step:
                            norms.append(r["norm_residual"])
                            ranks.append(r["rank_residual"])
                            ents.append(r["entropy_residual"])
                            npts.append(r["norm_per_gen_residual"])
                            break

                def fmt(arr):
                    if not arr:
                        return "N/A"
                    m, s = np.mean(arr), np.std(arr)
                    return f"{m:+.2f}+/-{s:.2f}"

                print(f"  {step:>6}  {label:>12}  {fmt(norms):>10}  {fmt(ranks):>10}  {fmt(ents):>10}  {fmt(npts):>10}")

    # -----------------------------------------------------------------------
    # Section 2: Cohen's d Between Conditions at Each Step
    # -----------------------------------------------------------------------
    print_header("SECTION 2: COHEN'S d --CONDITION SEPARATION IN RESIDUAL SPACE")
    print("\n  Positive d = aligned condition has larger residual than misaligned")

    for task, (aligned_label, misaligned_label) in TASK_PAIRS.items():
        print_subheader(f"Task: {task.upper()} --{aligned_label} vs {misaligned_label}")

        # Determine common steps
        all_steps_sets = []
        for label in [aligned_label, misaligned_label]:
            for i in range(len(by_task[task][label])):
                steps = {r["step"] for r in all_residuals[(task, label, i)]}
                all_steps_sets.append(steps)
        common = sorted(set.intersection(*all_steps_sets)) if all_steps_sets else []

        if not common:
            print("    No common steps.")
            continue

        print(f"\n  {'Step':>6}  {'d(norm)':>10}  {'d(rank)':>10}  {'d(entropy)':>10}  {'d(norm/gen)':>12}  {'Interp':>16}")
        print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*12}  {'-'*16}")

        for step in common:
            vals = {label: {"norm": [], "rank": [], "ent": [], "npt": []}
                    for label in [aligned_label, misaligned_label]}

            for label in [aligned_label, misaligned_label]:
                for i in range(len(by_task[task][label])):
                    for r in all_residuals[(task, label, i)]:
                        if r["step"] == step:
                            vals[label]["norm"].append(r["norm_residual"])
                            vals[label]["rank"].append(r["rank_residual"])
                            vals[label]["ent"].append(r["entropy_residual"])
                            vals[label]["npt"].append(r["norm_per_gen_residual"])
                            break

            d_norm = cohens_d(vals[aligned_label]["norm"], vals[misaligned_label]["norm"])
            d_rank = cohens_d(vals[aligned_label]["rank"], vals[misaligned_label]["rank"])
            d_ent = cohens_d(vals[aligned_label]["ent"], vals[misaligned_label]["ent"])
            d_npt = cohens_d(vals[aligned_label]["npt"], vals[misaligned_label]["npt"])

            # Interpretation
            max_d = max(abs(d_norm), abs(d_rank), abs(d_ent), abs(d_npt))
            if np.isnan(max_d):
                interp = "insufficient"
            elif max_d >= 1.5:
                interp = "*** HUGE ***"
            elif max_d >= 0.8:
                interp = "** LARGE **"
            elif max_d >= 0.5:
                interp = "* MEDIUM *"
            elif max_d >= 0.2:
                interp = "small"
            else:
                interp = "negligible"

            def fd(v):
                return f"{v:+.3f}" if not np.isnan(v) else "N/A"

            print(f"  {step:>6}  {fd(d_norm):>10}  {fd(d_rank):>10}  {fd(d_ent):>10}  {fd(d_npt):>12}  {interp:>16}")

    # -----------------------------------------------------------------------
    # Section 3: AUROC Classification of Residuals
    # -----------------------------------------------------------------------
    print_header("SECTION 3: AUROC --CAN RESIDUALS CLASSIFY COGNITIVE STATE?")
    print("\n  Using each scalar residual feature independently at each step.")
    print("  Label: 0 = aligned (honest/corrective/factual), 1 = misaligned")

    for task, (aligned_label, misaligned_label) in TASK_PAIRS.items():
        print_subheader(f"Task: {task.upper()}")

        all_steps_sets = []
        for label in [aligned_label, misaligned_label]:
            for i in range(len(by_task[task][label])):
                steps = {r["step"] for r in all_residuals[(task, label, i)]}
                all_steps_sets.append(steps)
        common = sorted(set.intersection(*all_steps_sets)) if all_steps_sets else []

        if not common:
            print("    No common steps.")
            continue

        features = ["norm_residual", "rank_residual", "entropy_residual", "norm_per_gen_residual"]
        print(f"\n  {'Step':>6}  {'AUROC(norm)':>12}  {'AUROC(rank)':>12}  {'AUROC(ent)':>12}  {'AUROC(n/g)':>12}  {'Best':>12}")
        print(f"  {'-'*6}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*12}")

        for step in common:
            labels_list = []
            feat_vals = {f: [] for f in features}

            for label in [aligned_label, misaligned_label]:
                lbl = 0 if label == aligned_label else 1
                for i in range(len(by_task[task][label])):
                    for r in all_residuals[(task, label, i)]:
                        if r["step"] == step:
                            labels_list.append(lbl)
                            for f in features:
                                feat_vals[f].append(r[f])
                            break

            aurocs = {}
            for f in features:
                auc = auroc_manual(labels_list, feat_vals[f])
                # Also try reversed direction
                auc_rev = auroc_manual(labels_list, [-v for v in feat_vals[f]])
                aurocs[f] = max(auc, auc_rev) if not (np.isnan(auc) or np.isnan(auc_rev)) else auc

            best_feat = max(aurocs, key=lambda k: aurocs[k] if not np.isnan(aurocs[k]) else -1)
            best_val = aurocs[best_feat]
            best_name = best_feat.replace("_residual", "").replace("norm_per_gen", "n/g")

            def fa(v):
                return f"{v:.3f}" if not np.isnan(v) else "N/A"

            print(f"  {step:>6}  {fa(aurocs['norm_residual']):>12}  {fa(aurocs['rank_residual']):>12}  "
                  f"{fa(aurocs['entropy_residual']):>12}  {fa(aurocs['norm_per_gen_residual']):>12}  "
                  f"{fa(best_val)+' '+best_name:>12}")

    # -----------------------------------------------------------------------
    # Section 4: Profile Residual Cosine Similarity
    # -----------------------------------------------------------------------
    print_header("SECTION 4: LAYER PROFILE RESIDUAL --COSINE SIMILARITY")
    print("\n  Mean profile residual per condition, then cosine sim between conditions.")
    print("  cos=1.0 -> identical direction of change; cos<1 -> diverging geometry")

    for task, (aligned_label, misaligned_label) in TASK_PAIRS.items():
        print_subheader(f"Task: {task.upper()}")

        all_steps_sets = []
        for label in [aligned_label, misaligned_label]:
            for i in range(len(by_task[task][label])):
                steps = {r["step"] for r in all_residuals[(task, label, i)]}
                all_steps_sets.append(steps)
        common = sorted(set.intersection(*all_steps_sets)) if all_steps_sets else []

        if not common:
            print("    No common steps.")
            continue

        print(f"\n  {'Step':>6}  {'cos_sim':>10}  {'||aligned||':>12}  {'||misaligned||':>14}  {'norm_ratio':>12}  {'Interp':>18}")
        print(f"  {'-'*6}  {'-'*10}  {'-'*12}  {'-'*14}  {'-'*12}  {'-'*18}")

        for step in common:
            profiles = {label: [] for label in [aligned_label, misaligned_label]}

            for label in [aligned_label, misaligned_label]:
                for i in range(len(by_task[task][label])):
                    for r in all_residuals[(task, label, i)]:
                        if r["step"] == step:
                            profiles[label].append(r["profile_residual"])
                            break

            mean_aligned = np.mean(profiles[aligned_label], axis=0)
            mean_misaligned = np.mean(profiles[misaligned_label], axis=0)

            cos = cosine_sim(mean_aligned, mean_misaligned)
            norm_a = np.linalg.norm(mean_aligned)
            norm_m = np.linalg.norm(mean_misaligned)
            ratio = norm_m / norm_a if norm_a > 0 else float("inf")

            if cos > 0.99:
                interp = "parallel"
            elif cos > 0.95:
                interp = "slight divergence"
            elif cos > 0.80:
                interp = "DIVERGING"
            else:
                interp = "*** ORTHOGONAL ***"

            print(f"  {step:>6}  {cos:>10.4f}  {norm_a:>12.3f}  {norm_m:>14.3f}  {ratio:>12.3f}  {interp:>18}")

    # -----------------------------------------------------------------------
    # Section 5: Growth Rate Analysis
    # -----------------------------------------------------------------------
    print_header("SECTION 5: RESIDUAL GROWTH RATES")
    print("\n  How fast do residuals grow? Slope = delta_residual / delta_step between consecutive checkpoints.")

    for task, (aligned_label, misaligned_label) in TASK_PAIRS.items():
        print_subheader(f"Task: {task.upper()}")

        all_steps_sets = []
        for label in [aligned_label, misaligned_label]:
            for i in range(len(by_task[task][label])):
                steps = {r["step"] for r in all_residuals[(task, label, i)]}
                all_steps_sets.append(steps)
        common = sorted(set.intersection(*all_steps_sets)) if all_steps_sets else []

        if not common or len(common) < 2:
            print("    Insufficient steps for growth rate.")
            continue

        print(f"\n  {'Step_a->b':>10}  {'Condition':>12}  {'norm_slope':>12}  {'rank_slope':>12}  {'ent_slope':>12}")
        print(f"  {'-'*10}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*12}")

        for idx in range(1, len(common)):
            s_prev, s_curr = common[idx - 1], common[idx]
            delta_step = s_curr - s_prev
            if delta_step == 0:
                continue

            for label in [aligned_label, misaligned_label]:
                norm_slopes, rank_slopes, ent_slopes = [], [], []
                for i in range(len(by_task[task][label])):
                    r_prev, r_curr = None, None
                    for r in all_residuals[(task, label, i)]:
                        if r["step"] == s_prev:
                            r_prev = r
                        if r["step"] == s_curr:
                            r_curr = r
                    if r_prev and r_curr:
                        norm_slopes.append((r_curr["norm_residual"] - r_prev["norm_residual"]) / delta_step)
                        rank_slopes.append((r_curr["rank_residual"] - r_prev["rank_residual"]) / delta_step)
                        ent_slopes.append((r_curr["entropy_residual"] - r_prev["entropy_residual"]) / delta_step)

                def fmt_slope(arr):
                    if not arr:
                        return "N/A"
                    return f"{np.mean(arr):+.4f}"

                print(f"  {s_prev:>4}->{s_curr:<4}  {label:>12}  {fmt_slope(norm_slopes):>12}  "
                      f"{fmt_slope(rank_slopes):>12}  {fmt_slope(ent_slopes):>12}")

    # -----------------------------------------------------------------------
    # Section 6: Norm Per Generated Token (Normalized Residual)
    # -----------------------------------------------------------------------
    print_header("SECTION 6: NORM-PER-GENERATED-TOKEN --ISOLATING PER-TOKEN CONTRIBUTION")
    print("\n  norm_residual / n_generated at each step, per condition.")
    print("  If this converges -> linear cache growth. If it diverges -> condition-specific scaling.")

    for task, (aligned_label, misaligned_label) in TASK_PAIRS.items():
        print_subheader(f"Task: {task.upper()}")

        all_steps_sets = []
        for label in [aligned_label, misaligned_label]:
            for i in range(len(by_task[task][label])):
                steps = {r["step"] for r in all_residuals[(task, label, i)]}
                all_steps_sets.append(steps)
        common = sorted(set.intersection(*all_steps_sets)) if all_steps_sets else []

        if not common:
            print("    No common steps.")
            continue

        print(f"\n  {'Step':>6}  {'n_gen':>6}  {'Condition':>12}  {'norm/gen (mean)':>16}  {'norm/gen (sd)':>14}  {'d vs other':>12}")
        print(f"  {'-'*6}  {'-'*6}  {'-'*12}  {'-'*16}  {'-'*14}  {'-'*12}")

        for step in common:
            vals = {label: [] for label in [aligned_label, misaligned_label]}
            for label in [aligned_label, misaligned_label]:
                for i in range(len(by_task[task][label])):
                    for r in all_residuals[(task, label, i)]:
                        if r["step"] == step:
                            vals[label].append(r["norm_per_gen_residual"])
                            break

            d = cohens_d(vals[aligned_label], vals[misaligned_label])

            for label in [aligned_label, misaligned_label]:
                arr = vals[label]
                m = np.mean(arr) if arr else float("nan")
                s = np.std(arr) if arr else float("nan")
                d_str = f"{d:+.3f}" if label == aligned_label and not np.isnan(d) else ""
                print(f"  {step:>6}  {step:>6}  {label:>12}  {m:>16.4f}  {s:>14.4f}  {d_str:>12}")

    # -----------------------------------------------------------------------
    # Section 7: Summary / Key Findings
    # -----------------------------------------------------------------------
    print_header("SECTION 7: KEY FINDINGS SUMMARY")

    # Compute overall statistics for the final common step across tasks
    for task, (aligned_label, misaligned_label) in TASK_PAIRS.items():
        print_subheader(f"Task: {task.upper()}")

        # Get common steps
        all_steps_sets = []
        for label in [aligned_label, misaligned_label]:
            for i in range(len(by_task[task][label])):
                steps = {r["step"] for r in all_residuals[(task, label, i)]}
                all_steps_sets.append(steps)
        common = sorted(set.intersection(*all_steps_sets)) if all_steps_sets else []

        if not common:
            print("    No data.")
            continue

        # Report at early (step 3) and late (last common) steps
        report_steps = []
        if 3 in common:
            report_steps.append(("early (step 3)", 3))
        if common[-1] != 3:
            report_steps.append((f"late (step {common[-1]})", common[-1]))

        for step_name, step in report_steps:
            vals = {label: {"norm": [], "rank": [], "ent": [], "npt": [], "profile": []}
                    for label in [aligned_label, misaligned_label]}

            for label in [aligned_label, misaligned_label]:
                for i in range(len(by_task[task][label])):
                    for r in all_residuals[(task, label, i)]:
                        if r["step"] == step:
                            vals[label]["norm"].append(r["norm_residual"])
                            vals[label]["rank"].append(r["rank_residual"])
                            vals[label]["ent"].append(r["entropy_residual"])
                            vals[label]["npt"].append(r["norm_per_gen_residual"])
                            vals[label]["profile"].append(r["profile_residual"])
                            break

            d_norm = cohens_d(vals[aligned_label]["norm"], vals[misaligned_label]["norm"])
            d_rank = cohens_d(vals[aligned_label]["rank"], vals[misaligned_label]["rank"])
            d_ent = cohens_d(vals[aligned_label]["ent"], vals[misaligned_label]["ent"])
            d_npt = cohens_d(vals[aligned_label]["npt"], vals[misaligned_label]["npt"])

            mean_a = np.mean(vals[aligned_label]["profile"], axis=0)
            mean_m = np.mean(vals[misaligned_label]["profile"], axis=0)
            cos = cosine_sim(mean_a, mean_m)

            # Best single-feature AUROC
            features = {"norm": vals, "rank": vals, "ent": vals, "npt": vals}
            best_auc = 0.0
            best_feat_name = ""
            for feat_key, feat_name in [("norm", "norm"), ("rank", "rank"), ("ent", "entropy"), ("npt", "norm/gen")]:
                labels_arr = [0] * len(vals[aligned_label][feat_key]) + [1] * len(vals[misaligned_label][feat_key])
                scores_arr = vals[aligned_label][feat_key] + vals[misaligned_label][feat_key]
                auc = auroc_manual(labels_arr, scores_arr)
                auc_rev = auroc_manual(labels_arr, [-v for v in scores_arr])
                auc_best = max(auc, auc_rev) if not (np.isnan(auc) or np.isnan(auc_rev)) else auc
                if not np.isnan(auc_best) and auc_best > best_auc:
                    best_auc = auc_best
                    best_feat_name = feat_name

            print(f"\n    At {step_name}:")
            print(f"      Cohen's d  --norm: {d_norm:+.3f}, rank: {d_rank:+.3f}, entropy: {d_ent:+.3f}, norm/gen: {d_npt:+.3f}")
            max_d = max(abs(d_norm), abs(d_rank), abs(d_ent), abs(d_npt))
            if np.isnan(max_d):
                size = "insufficient data"
            elif max_d >= 0.8:
                size = "LARGE separation"
            elif max_d >= 0.5:
                size = "MEDIUM separation"
            elif max_d >= 0.2:
                size = "small separation"
            else:
                size = "negligible separation"
            print(f"      Max |d| = {max_d:.3f} -> {size}")
            print(f"      Best AUROC = {best_auc:.3f} ({best_feat_name})")
            print(f"      Profile cosine similarity = {cos:.4f}", end="")
            if cos < 0.95:
                print(" -> GEOMETRY DIVERGES")
            elif cos < 0.99:
                print(" -> slight divergence")
            else:
                print(" -> parallel growth")

    # -----------------------------------------------------------------------
    # Final verdict
    # -----------------------------------------------------------------------
    print_header("VERDICT: DOES GENERATION CREATE CONDITION-SPECIFIC SIGNAL?")
    print("""
  The cache residual analysis subtracts the encoding-phase (step 0) fingerprint
  from every subsequent generation step. If the RESIDUALS differ between honest
  and deceptive conditions, the GENERATION PROCESS ITSELF -- not the system prompt --
  creates cognitive-state-specific geometric changes in KV-cache space.

  This matters for Cricket because:
  1. If residuals separate -> we can detect deception even with unknown system prompts
  2. If profile residuals diverge -> generation geometry is condition-dependent
  3. If norm/gen converges -> the per-token signal is stable (good for real-time monitoring)
  4. If growth rates differ -> cache expansion rate is itself a detection signal
""")
    print("=" * 80)


if __name__ == "__main__":
    main()
