# ============================================================
# JiminAI Cricket | PATENT PENDING
# "The Lyra Technique" - Provisional Patent Filed 2026
# Liberation Labs / Digital Disconnections / TheMultiverse.school
# Contact: info@digitaldisconnections.com | (877) 674-8874
# ============================================================
#!/usr/bin/env python3
"""
Phase 1.75: Publication-Quality Visualization
==============================================

Reads adversarial_controls_results.json and generates figures suitable
for academic publication. Outputs PNG + SVG for each figure.

Usage:
    python 01e_controls_visualization.py
    python 01e_controls_visualization.py --results path/to/results.json
    python 01e_controls_visualization.py --format pdf  # for LaTeX

Requires: matplotlib, seaborn, numpy
"""

import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# Publication style
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.figsize": (10, 6),
})

COLORS = {
    "true": "#2ecc71",
    "false": "#e74c3c",
    "common": "#3498db",
    "rare": "#9b59b6",
    "refusal": "#e74c3c",
    "rote": "#f39c12",
    "code": "#3498db",
    "formulaic": "#9b59b6",
    "creative": "#2ecc71",
    "4bit": "#e74c3c",
    "fp16": "#3498db",
}


def load_results(results_path):
    with open(results_path) as f:
        return json.load(f)


# ================================================================
# FIGURE 1: Frequency × Truth Matrix (Control 1)
# ================================================================

def plot_control_1(results, output_dir, fmt="png"):
    """2×2 heatmap + box plots for frequency/truth factorial."""
    if "control_1" not in results:
        print("  Skipping Control 1 (not in results)")
        return

    data = results["control_1"]
    analysis = data["analysis"]
    cells = analysis.get("cells", {})

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), gridspec_kw={"width_ratios": [1, 2, 2]})

    # Panel A: 2×2 heatmap of means
    ax = axes[0]
    matrix = np.array([
        [cells.get("common_true", {}).get("mean", 0), cells.get("common_false", {}).get("mean", 0)],
        [cells.get("rare_true", {}).get("mean", 0), cells.get("rare_false", {}).get("mean", 0)],
    ])
    im = ax.imshow(matrix, cmap="RdYlBu_r", aspect="auto")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["True", "False"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Common", "Rare"])
    ax.set_title("A. Mean Cache Norm")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center",
                    fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Panel B: Box plots by cell
    ax = axes[1]
    cell_data = []
    cell_labels = []
    cell_colors = []
    for cell_name in ["common_true", "common_false", "rare_true", "rare_false"]:
        if cell_name in data.get("categories", {}):
            norms = [s["total_key_norm"] for s in data["categories"][cell_name]]
            cell_data.append(norms)
            cell_labels.append(cell_name.replace("_", "\n"))
            color = COLORS["true"] if "true" in cell_name else COLORS["false"]
            cell_colors.append(color)

    if cell_data:
        bp = ax.boxplot(cell_data, labels=cell_labels, patch_artist=True,
                        showmeans=True, meanprops={"marker": "D", "markerfacecolor": "black"})
        for patch, color in zip(bp["boxes"], cell_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
    ax.set_ylabel("Total Key L2 Norm")
    ax.set_title("B. Distribution by Cell")

    # Panel C: Effect sizes
    ax = axes[2]
    comparisons = analysis.get("comparisons", {})
    comp_names = []
    comp_ds = []
    comp_cis = []
    for key in ["truth_effect_common", "truth_effect_rare",
                "freq_effect_true", "freq_effect_false"]:
        if key in comparisons:
            d_info = comparisons[key].get("cohens_d", {})
            comp_names.append(key.replace("_", " ").title())
            comp_ds.append(d_info.get("d", 0))
            comp_cis.append((
                d_info.get("d", 0) - d_info.get("ci_lower", 0),
                d_info.get("ci_upper", 0) - d_info.get("d", 0),
            ))

    if comp_ds:
        y_pos = range(len(comp_names))
        errors = np.array(comp_cis).T
        colors = ["#e74c3c" if "truth" in n.lower() else "#3498db" for n in comp_names]
        ax.barh(y_pos, comp_ds, xerr=errors, color=colors, alpha=0.7, capsize=5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(comp_names, fontsize=9)
        ax.axvline(x=0, color="black", linewidth=0.5, linestyle="--")
        ax.axvline(x=0.5, color="gray", linewidth=0.5, linestyle=":", label="d=0.5")
        ax.axvline(x=-0.5, color="gray", linewidth=0.5, linestyle=":")
        ax.set_xlabel("Cohen's d (with 95% CI)")
        ax.set_title("C. Effect Sizes")
        ax.legend()

    fig.suptitle("Control 1: Frequency × Truth Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_dir / f"fig_control_1.{fmt}", bbox_inches="tight")
    if fmt != "svg":
        fig.savefig(output_dir / "fig_control_1.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: fig_control_1.{fmt}")


# ================================================================
# FIGURE 2: Rote Memorization (Control 2)
# ================================================================

def plot_control_2(results, output_dir, fmt="png"):
    """Category comparison + pairwise effect sizes."""
    if "control_2" not in results:
        print("  Skipping Control 2 (not in results)")
        return

    data = results["control_2"]
    analysis = data["analysis"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: Box plots by category
    ax = axes[0]
    cat_order = ["refusal", "rote_completion", "code_boilerplate",
                 "formulaic_response", "creative_open"]
    cat_data = []
    cat_labels = []
    cat_colors_list = [COLORS["refusal"], COLORS["rote"], COLORS["code"],
                       COLORS["formulaic"], COLORS["creative"]]

    for cat in cat_order:
        if cat in data.get("categories", {}):
            norms = [s["total_key_norm"] for s in data["categories"][cat]]
            cat_data.append(norms)
            cat_labels.append(cat.replace("_", "\n"))

    if cat_data:
        bp = ax.boxplot(cat_data, labels=cat_labels, patch_artist=True,
                        showmeans=True, meanprops={"marker": "D", "markerfacecolor": "black"})
        for i, patch in enumerate(bp["boxes"]):
            if i < len(cat_colors_list):
                patch.set_facecolor(cat_colors_list[i])
                patch.set_alpha(0.6)
    ax.set_ylabel("Total Key L2 Norm")
    ax.set_title("A. Cache Norm by Category")

    # Panel B: Pairwise Cohen's d vs refusal
    ax = axes[1]
    comparisons = analysis.get("comparisons", {})
    comp_cats = ["rote_completion", "code_boilerplate", "formulaic_response", "creative_open"]
    ds = []
    ci_lows = []
    ci_highs = []
    labels = []
    for cat in comp_cats:
        key = f"refusal_vs_{cat}"
        if key in comparisons:
            d_info = comparisons[key].get("norm", {}).get("cohens_d", {})
            ds.append(d_info.get("d", 0))
            ci_lows.append(d_info.get("ci_lower", 0))
            ci_highs.append(d_info.get("ci_upper", 0))
            labels.append(f"vs {cat.replace('_', ' ')}")

    if ds:
        y_pos = range(len(labels))
        errors = [[d - cl for d, cl in zip(ds, ci_lows)],
                  [ch - d for d, ch in zip(ds, ci_highs)]]
        ax.barh(y_pos, ds, xerr=errors, color=COLORS["refusal"], alpha=0.6, capsize=5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.axvline(x=0, color="black", linewidth=0.5, linestyle="--")
        ax.axvline(x=0.3, color="orange", linewidth=0.5, linestyle=":", label="|d|=0.3")
        ax.axvline(x=-0.3, color="orange", linewidth=0.5, linestyle=":")
        ax.set_xlabel("Cohen's d (refusal - other)")
        ax.set_title("B. Refusal vs Others (Effect Size)")
        ax.legend(fontsize=8)

    fig.suptitle("Control 2: Rote Memorization Test", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_dir / f"fig_control_2.{fmt}", bbox_inches="tight")
    if fmt != "svg":
        fig.savefig(output_dir / "fig_control_2.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: fig_control_2.{fmt}")


# ================================================================
# FIGURE 3: Precision Sweep (Control 3)
# ================================================================

def plot_control_3(results, output_dir, fmt="png"):
    """Correlation scatter + Bland-Altman + per-category comparison."""
    if "control_3" not in results:
        print("  Skipping Control 3 (not in results)")
        return

    data = results["control_3"]
    analysis = data["analysis"]
    matched = analysis.get("matched_prompts", {})

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel A: Scatter correlation
    ax = axes[0]
    if matched:
        x = [v["norm_4bit"] for v in matched.values()]
        y = [v["norm_fp16"] for v in matched.values()]
        ax.scatter(x, y, alpha=0.7, s=40, color=COLORS["4bit"], edgecolors="white", linewidth=0.5)

        # Regression line
        if len(x) > 2:
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            x_line = np.linspace(min(x), max(x), 100)
            ax.plot(x_line, p(x_line), "--", color="black", linewidth=1)

        r = analysis.get("pearson", {}).get("r", 0)
        ax.set_xlabel("4-bit Norm")
        ax.set_ylabel("FP16 Norm")
        ax.set_title(f"A. Correlation (r={r:.3f})")

        # Identity line
        lims = [min(min(x), min(y)), max(max(x), max(y))]
        ax.plot(lims, lims, ":", color="gray", linewidth=0.5, label="y=x")
        ax.legend()

    # Panel B: Bland-Altman
    ax = axes[1]
    if matched:
        means_ba = [(v["norm_4bit"] + v["norm_fp16"]) / 2 for v in matched.values()]
        diffs_ba = [v["norm_fp16"] - v["norm_4bit"] for v in matched.values()]
        ax.scatter(means_ba, diffs_ba, alpha=0.7, s=40, color=COLORS["fp16"],
                   edgecolors="white", linewidth=0.5)

        ba = analysis.get("bland_altman", {})
        mean_d = ba.get("mean_diff", 0)
        loa = ba.get("limits_of_agreement", {})
        ax.axhline(y=mean_d, color="black", linewidth=1, label=f"Mean diff: {mean_d:.1f}")
        ax.axhline(y=loa.get("upper", 0), color="red", linewidth=0.5, linestyle="--",
                   label=f"±1.96 SD")
        ax.axhline(y=loa.get("lower", 0), color="red", linewidth=0.5, linestyle="--")
        ax.set_xlabel("Mean of 4-bit and FP16")
        ax.set_ylabel("Difference (FP16 − 4-bit)")
        ax.set_title("B. Bland-Altman Agreement")
        ax.legend(fontsize=8)

    # Panel C: Per-category comparison
    ax = axes[2]
    cats = analysis.get("categories", {})
    if cats:
        cat_names = list(cats.keys())
        means_4 = [cats[c]["mean_4bit"] for c in cat_names]
        means_16 = [cats[c]["mean_fp16"] for c in cat_names]
        x_pos = np.arange(len(cat_names))
        width = 0.35

        ax.bar(x_pos - width / 2, means_4, width, label="4-bit", color=COLORS["4bit"], alpha=0.7)
        ax.bar(x_pos + width / 2, means_16, width, label="FP16", color=COLORS["fp16"], alpha=0.7)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([c.replace("_", "\n") for c in cat_names], fontsize=8)
        ax.set_ylabel("Mean Key L2 Norm")
        ax.set_title("C. Category Norms by Precision")
        ax.legend()

    fig.suptitle("Control 3: Precision Sweep (4-bit vs FP16)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_dir / f"fig_control_3.{fmt}", bbox_inches="tight")
    if fmt != "svg":
        fig.savefig(output_dir / "fig_control_3.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: fig_control_3.{fmt}")


# ================================================================
# FIGURE 4: Semantic Gradient (Control 4)
# ================================================================

def plot_control_4(results, output_dir, fmt="png"):
    """Transfer success rate by semantic distance."""
    if "control_4" not in results:
        print("  Skipping Control 4 (not in results)")
        return

    analysis = results["control_4"]["analysis"]
    by_dist = analysis.get("by_distance", {})

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Success rate by distance
    ax = axes[0]
    distances = ["near", "medium_near", "medium", "medium_far", "far"]
    rates = []
    counts = []
    dist_labels = []
    for d in distances:
        if d in by_dist:
            rates.append(by_dist[d]["success_rate"])
            counts.append(by_dist[d]["n"])
            dist_labels.append(d.replace("_", "\n"))

    if rates:
        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(rates)))
        bars = ax.bar(range(len(rates)), rates, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_xticks(range(len(rates)))
        ax.set_xticklabels(dist_labels)
        ax.set_ylabel("Transfer Success Rate")
        ax.set_ylim(0, 1.05)
        ax.set_title("A. Success Rate by Distance")

        for i, (bar, r, n) in enumerate(zip(bars, rates, counts)):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{r:.0%}\n(n={n})", ha="center", fontsize=9)

    # Panel B: Per-experiment breakdown
    ax = axes[1]
    experiments = results["control_4"].get("experiments", [])
    if experiments:
        exp_names = [e["name"] for e in experiments]
        for i, exp in enumerate(experiments):
            dist_rates = defaultdict(list)
            for t in exp["targets"]:
                dist_rates[t["distance"]].append(1 if t["transfer_success"] else 0)

            exp_rates = []
            for d in distances:
                if d in dist_rates:
                    exp_rates.append(np.mean(dist_rates[d]))
                else:
                    exp_rates.append(0)

            ax.plot(range(len(exp_rates)), exp_rates, "o-", label=exp["name"],
                    alpha=0.7, markersize=6)

        ax.set_xticks(range(len(distances)))
        ax.set_xticklabels([d.replace("_", "\n") for d in distances])
        ax.set_ylabel("Success Rate")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title("B. Per-Experiment Gradient")
        ax.legend(fontsize=8, loc="upper right")

    tau = analysis.get("gradient_tau", 0)
    fig.suptitle(f"Control 4: Semantic Gradient Transfer (Kendall τ={tau:.3f})",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_dir / f"fig_control_4.{fmt}", bbox_inches="tight")
    if fmt != "svg":
        fig.savefig(output_dir / "fig_control_4.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: fig_control_4.{fmt}")


# ================================================================
# FIGURE 5: Length Confound (Control 5)
# ================================================================

def plot_control_5(results, output_dir, fmt="png"):
    """Length × Truth interaction plot."""
    if "control_5" not in results:
        print("  Skipping Control 5 (not in results)")
        return

    data = results["control_5"]
    analysis = data["analysis"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Box plots
    ax = axes[0]
    groups = ["short_true", "short_false", "long_true", "long_false"]
    group_data = []
    group_labels = []
    group_colors = []
    for g in groups:
        if g in data.get("categories", {}):
            norms = [s["total_key_norm"] for s in data["categories"][g]]
            group_data.append(norms)
            group_labels.append(g.replace("_", "\n"))
            group_colors.append(COLORS["true"] if "true" in g else COLORS["false"])

    if group_data:
        bp = ax.boxplot(group_data, labels=group_labels, patch_artist=True, showmeans=True,
                        meanprops={"marker": "D", "markerfacecolor": "black"})
        for patch, color in zip(bp["boxes"], group_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
    ax.set_ylabel("Total Key L2 Norm")
    ax.set_title("A. Distributions")

    # Panel B: Interaction plot
    ax = axes[1]
    means = {}
    for g in groups:
        if g in data.get("categories", {}):
            norms = [s["total_key_norm"] for s in data["categories"][g]]
            means[g] = np.mean(norms)

    if len(means) == 4:
        ax.plot([0, 1], [means["short_true"], means["long_true"]], "o-",
                color=COLORS["true"], label="True", markersize=8, linewidth=2)
        ax.plot([0, 1], [means["short_false"], means["long_false"]], "o-",
                color=COLORS["false"], label="False", markersize=8, linewidth=2)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Short (3-5 tokens)", "Long (15-20 tokens)"])
        ax.set_ylabel("Mean Key L2 Norm")
        ax.set_title("B. Length × Truth Interaction")
        ax.legend()

    fig.suptitle("Control 5: Length Confound", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_dir / f"fig_control_5.{fmt}", bbox_inches="tight")
    if fmt != "svg":
        fig.savefig(output_dir / "fig_control_5.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: fig_control_5.{fmt}")


# ================================================================
# FIGURE 6: Template Structure (Control 6)
# ================================================================

def plot_control_6(results, output_dir, fmt="png"):
    """Paired comparison with controlled template."""
    if "control_6" not in results:
        print("  Skipping Control 6 (not in results)")
        return

    data = results["control_6"]
    analysis = data["analysis"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Overlapping distributions
    ax = axes[0]
    if "template_true" in data.get("categories", {}) and "template_false" in data.get("categories", {}):
        true_norms = [s["total_key_norm"] for s in data["categories"]["template_true"]]
        false_norms = [s["total_key_norm"] for s in data["categories"]["template_false"]]

        ax.hist(true_norms, bins=20, alpha=0.5, color=COLORS["true"], label="True", density=True)
        ax.hist(false_norms, bins=20, alpha=0.5, color=COLORS["false"], label="False", density=True)
        ax.set_xlabel("Total Key L2 Norm")
        ax.set_ylabel("Density")
        ax.set_title("A. Distribution (Template-Controlled)")
        ax.legend()

    # Panel B: Paired differences
    ax = axes[1]
    true_by_pid = defaultdict(list)
    false_by_pid = defaultdict(list)
    if "template_true" in data.get("categories", {}):
        for s in data["categories"]["template_true"]:
            base = s["prompt_id"].replace("_true", "").replace("t_", "")
            true_by_pid[base].append(s["total_key_norm"])
    if "template_false" in data.get("categories", {}):
        for s in data["categories"]["template_false"]:
            base = s["prompt_id"].replace("_false", "").replace("_f", "").replace("t_", "")
            false_by_pid[base].append(s["total_key_norm"])

    pair_diffs = []
    pair_labels = []
    for pid in sorted(true_by_pid.keys()):
        if pid in false_by_pid:
            diff = np.mean(false_by_pid[pid]) - np.mean(true_by_pid[pid])
            pair_diffs.append(diff)
            pair_labels.append(pid.replace("_", " "))

    if pair_diffs:
        colors = ["#e74c3c" if d > 0 else "#2ecc71" for d in pair_diffs]
        ax.barh(range(len(pair_diffs)), pair_diffs, color=colors, alpha=0.7)
        ax.set_yticks(range(len(pair_diffs)))
        ax.set_yticklabels(pair_labels, fontsize=8)
        ax.axvline(x=0, color="black", linewidth=0.5)
        ax.set_xlabel("Norm Difference (False − True)")
        ax.set_title("B. Paired Differences")

    nc = analysis.get("norm_comparison", {}).get("cohens_d", {})
    d = nc.get("d", 0)
    fig.suptitle(f"Control 6: Template Structure (d={d:.3f})", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_dir / f"fig_control_6.{fmt}", bbox_inches="tight")
    if fmt != "svg":
        fig.savefig(output_dir / "fig_control_6.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: fig_control_6.{fmt}")


# ================================================================
# COMPOSITE FIGURE: Summary dashboard
# ================================================================

def plot_summary(results, output_dir, fmt="png"):
    """Summary dashboard with key findings from all controls."""
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.3)

    control_titles = {
        "control_1": "C1: Freq/Truth",
        "control_2": "C2: Memorization",
        "control_3": "C3: Precision",
        "control_4": "C4: Gradient",
        "control_5": "C5: Length",
        "control_6": "C6: Template",
    }

    positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]

    for (key, title), (row, col) in zip(control_titles.items(), positions):
        ax = fig.add_subplot(gs[row, col])

        if key not in results:
            ax.text(0.5, 0.5, "Not Run", ha="center", va="center",
                    fontsize=14, color="gray", transform=ax.transAxes)
            ax.set_title(title, fontsize=11)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        analysis = results[key].get("analysis", {})
        verdict = analysis.get("verdict", "N/A")

        # Determine verdict color
        if any(w in verdict.upper() for w in ["CONFIRMED", "SURVIVES", "DETECTED", "CONTENT"]):
            color = "#2ecc71"
            symbol = "PASS"
        elif any(w in verdict.upper() for w in ["ARTIFACT", "CONFOUND", "MOOT", "NO EFFECT"]):
            color = "#e74c3c"
            symbol = "FAIL"
        else:
            color = "#f39c12"
            symbol = "MIXED"

        ax.text(0.5, 0.6, symbol, ha="center", va="center",
                fontsize=28, fontweight="bold", color=color, transform=ax.transAxes)

        # Truncate verdict for display
        short_verdict = verdict.split(":")[0] if ":" in verdict else verdict[:40]
        ax.text(0.5, 0.25, short_verdict, ha="center", va="center",
                fontsize=9, color="black", transform=ax.transAxes, wrap=True)

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(2)
            spine.set_color(color)

    fig.suptitle("Phase 1.75: Adversarial Controls — Summary Dashboard",
                 fontsize=16, fontweight="bold")
    fig.savefig(output_dir / f"fig_summary.{fmt}", bbox_inches="tight")
    if fmt != "svg":
        fig.savefig(output_dir / "fig_summary.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: fig_summary.{fmt}")


# ================================================================
# MAIN
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 1.75 Visualization")
    parser.add_argument("--results", type=str, default=None,
                        help="Path to adversarial_controls_results.json")
    parser.add_argument("--format", type=str, default="png",
                        choices=["png", "pdf", "svg"],
                        help="Output format (png for slides, pdf for LaTeX)")
    args = parser.parse_args()

    # Find results
    if args.results:
        results_path = Path(args.results)
    else:
        # Try model-specific filename first, fall back to generic
        results_dir = Path(__file__).parent.parent / "results"
        candidates = sorted(results_dir.glob("adversarial_controls_*_results.json"))
        if candidates:
            results_path = candidates[0]
        else:
            results_path = results_dir / "adversarial_controls_results.json"

    if not results_path.exists():
        print(f"  Results not found at {results_path}")
        print(f"  Run 01d_adversarial_controls.py first.")
        return

    print(f"  Loading results from {results_path}")
    results = load_results(results_path)

    # Output directory
    fig_dir = results_path.parent / "figures"
    fig_dir.mkdir(exist_ok=True)

    print(f"  Output: {fig_dir}/")
    print()

    # Generate all figures
    plot_control_1(results, fig_dir, args.format)
    plot_control_2(results, fig_dir, args.format)
    plot_control_3(results, fig_dir, args.format)
    plot_control_4(results, fig_dir, args.format)
    plot_control_5(results, fig_dir, args.format)
    plot_control_6(results, fig_dir, args.format)
    plot_summary(results, fig_dir, args.format)

    print(f"\n  All figures saved to {fig_dir}/")
    print(f"  Formats: {args.format} + SVG (for editing)")


if __name__ == "__main__":
    main()
