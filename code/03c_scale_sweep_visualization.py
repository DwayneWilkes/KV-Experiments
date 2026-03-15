# ============================================================
# JiminAI Cricket | PATENT PENDING
# "The Lyra Technique" - Provisional Patent Filed 2026
# Liberation Labs / Digital Disconnections / TheMultiverse.school
# Contact: info@digitaldisconnections.com | (877) 674-8874
# ============================================================
#!/usr/bin/env python3
"""
Phase 2b: Scale Sweep Visualization — Publication Figures
=========================================================

Generates publication-quality figures from scale_sweep_results.json.
Run after 03_scale_sweep.py completes.

Figures:
  1. Category heatmap (scale × category, colored by mean norm)
  2. Effect size forest plot (all comparisons, per scale)
  3. Cross-scale trend lines (d vs scale for key hypotheses)
  4. Per-token normalized comparison (removes length bias)
  5. Category rank stability (alluvial/bump chart across scales)
  6. Summary dashboard (multi-panel overview)
  7. Norm vs geometry dual-panel (norms vs effective rank — the key figure)
  8. Dimensionality heatmap (effective rank by category and scale)

Usage:
  python 03c_scale_sweep_visualization.py
  python 03c_scale_sweep_visualization.py --format pdf   # For LaTeX
  python 03c_scale_sweep_visualization.py --format svg   # For editing

Liberation Labs / THCoalition
"""

import json
import argparse
import numpy as np
from pathlib import Path

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import FancyBboxPatch
    import seaborn as sns
except ImportError:
    print("ERROR: matplotlib and seaborn required.")
    print("  pip install matplotlib seaborn")
    exit(1)


# ================================================================
# STYLE
# ================================================================

def setup_style():
    """Publication-quality figure style."""
    plt.rcParams.update({
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'figure.titlesize': 14,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })
    sns.set_palette("colorblind")


# ================================================================
# FIGURE 1: CATEGORY HEATMAP
# ================================================================

def plot_category_heatmap(data, fig_dir, fmt):
    """Heatmap of mean norm by category and scale."""
    scales = list(data["scales"].keys())
    valid_scales = [s for s in scales if "analysis" in data["scales"][s]]

    if not valid_scales:
        print("  [Skip] No valid scales for heatmap")
        return

    categories = list(data["scales"][valid_scales[0]]["analysis"]["category_summaries"].keys())

    matrix = np.zeros((len(categories), len(valid_scales)))
    for j, scale in enumerate(valid_scales):
        summaries = data["scales"][scale]["analysis"]["category_summaries"]
        for i, cat in enumerate(categories):
            matrix[i, j] = summaries.get(cat, {}).get("mean", 0)

    fig, ax = plt.subplots(figsize=(max(8, len(valid_scales) * 1.5), max(6, len(categories) * 0.4)))
    sns.heatmap(matrix, xticklabels=valid_scales, yticklabels=categories,
                annot=True, fmt=".0f", cmap="YlOrRd", ax=ax,
                cbar_kws={"label": "Mean Key Norm"})
    ax.set_title("Cache Norm by Category and Scale")
    ax.set_xlabel("Model Scale")
    ax.set_ylabel("Cognitive Category")
    plt.tight_layout()

    path = fig_dir / f"fig_scale_heatmap.{fmt}"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ================================================================
# FIGURE 2: EFFECT SIZE FOREST PLOT
# ================================================================

def plot_effect_forest(data, fig_dir, fmt):
    """Forest plot of Cohen's d for key comparisons across scales."""
    scales = [s for s in data["scales"] if "analysis" in data["scales"][s]]

    key_comparisons = [
        "confab_vs_facts", "self_ref_effect", "refusal_vs_rote",
        "creative_vs_facts", "ambiguity_effect"
    ]

    fig, ax = plt.subplots(figsize=(10, max(4, len(scales) * len(key_comparisons) * 0.25)))

    y_pos = 0
    y_ticks = []
    y_labels = []
    colors = sns.color_palette("colorblind", len(key_comparisons))

    for s_idx, scale in enumerate(scales):
        comparisons = data["scales"][scale]["analysis"]["pairwise_comparisons"]

        for c_idx, comp_key in enumerate(key_comparisons):
            if comp_key not in comparisons:
                continue

            comp = comparisons[comp_key]
            d = comp["cohens_d"]["d"]
            ci_lo = comp["cohens_d"]["ci_lower"]
            ci_hi = comp["cohens_d"]["ci_upper"]

            ax.errorbar(d, y_pos, xerr=[[d - ci_lo], [ci_hi - d]],
                        fmt='o', color=colors[c_idx], markersize=6,
                        capsize=3, elinewidth=1.5)

            y_ticks.append(y_pos)
            y_labels.append(f"{scale}: {comp_key}")
            y_pos += 1

        y_pos += 0.5  # Gap between scales

    ax.axvline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.axvline(0.3, color='gray', linestyle=':', linewidth=0.5, alpha=0.3)
    ax.axvline(-0.3, color='gray', linestyle=':', linewidth=0.5, alpha=0.3)
    ax.axvline(0.8, color='gray', linestyle=':', linewidth=0.5, alpha=0.3)

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=7)
    ax.set_xlabel("Cohen's d (with 95% CI)")
    ax.set_title("Effect Sizes Across Scales")
    ax.invert_yaxis()
    plt.tight_layout()

    path = fig_dir / f"fig_scale_forest.{fmt}"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ================================================================
# FIGURE 3: CROSS-SCALE TREND LINES
# ================================================================

def plot_trend_lines(data, fig_dir, fmt):
    """Effect size trends across model scale for key hypotheses."""
    cross = data.get("cross_scale", {})
    trends = cross.get("effect_size_trends", {})

    if not trends:
        print("  [Skip] No cross-scale trends (need 3+ scales)")
        return

    fig, axes = plt.subplots(1, len(trends), figsize=(5 * len(trends), 4),
                              squeeze=False)

    for idx, (key, trend) in enumerate(trends.items()):
        ax = axes[0, idx]
        x = [dp["scale_numeric"] for dp in trend["data_points"]]
        y = [dp["cohens_d"] for dp in trend["data_points"]]
        labels = [dp["scale"] for dp in trend["data_points"]]

        ax.plot(x, y, 'o-', color=sns.color_palette("colorblind")[idx],
                markersize=8, linewidth=2)

        for xi, yi, label in zip(x, y, labels):
            ax.annotate(label, (xi, yi), textcoords="offset points",
                        xytext=(0, 10), ha='center', fontsize=7)

        ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.axhline(0.3, color='green', linestyle=':', linewidth=0.5, alpha=0.3)
        ax.axhline(-0.3, color='red', linestyle=':', linewidth=0.5, alpha=0.3)

        rho = trend["spearman_rho"]
        p = trend["p_value"]
        ax.set_title(f"{key}\nρ={rho:.3f}, p={p:.3f} ({trend['trend']})")
        ax.set_xlabel("Model Scale (B params)")
        ax.set_ylabel("Cohen's d")
        ax.set_xscale('log')

    plt.suptitle("Effect Size Trends Across Scale", fontsize=14)
    plt.tight_layout()

    path = fig_dir / f"fig_scale_trends.{fmt}"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ================================================================
# FIGURE 4: PER-TOKEN NORMALIZED COMPARISON
# ================================================================

def plot_per_token(data, fig_dir, fmt):
    """Box plots of per-token normalized norms by category."""
    scales = [s for s in data["scales"] if "analysis" in data["scales"][s]]

    if not scales:
        return

    n_scales = len(scales)
    fig, axes = plt.subplots(1, n_scales, figsize=(6 * n_scales, 5), squeeze=False)

    for s_idx, scale in enumerate(scales):
        ax = axes[0, s_idx]
        battery = data["scales"][scale].get("battery_results", {})

        box_data = []
        labels = []
        for cat in sorted(battery.keys()):
            norms = battery[cat].get("all_norms_per_token", [])
            if norms:
                box_data.append(norms)
                labels.append(cat[:12])  # Truncate for readability

        if box_data:
            bp = ax.boxplot(box_data, labels=labels, patch_artist=True,
                            showfliers=False)
            colors = sns.color_palette("husl", len(box_data))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)

        ax.set_title(f"{scale}")
        ax.set_ylabel("Key Norm / Token")
        ax.tick_params(axis='x', rotation=45)

    plt.suptitle("Per-Token Normalized Cache Norms", fontsize=14)
    plt.tight_layout()

    path = fig_dir / f"fig_scale_per_token.{fmt}"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ================================================================
# FIGURE 5: RANK STABILITY (BUMP CHART)
# ================================================================

def plot_rank_stability(data, fig_dir, fmt):
    """Bump chart showing category rank changes across scales."""
    scales = [s for s in data["scales"]
              if "analysis" in data["scales"].get(s, {})]

    if len(scales) < 2:
        print("  [Skip] Need 2+ scales for rank stability chart")
        return

    # Get rankings
    all_rankings = {}
    for scale in scales:
        ranking = data["scales"][scale].get("ranking", {})
        rank_order = ranking.get("rank_order", [])
        all_rankings[scale] = {cat: idx + 1 for idx, cat in enumerate(rank_order)}

    categories = list(all_rankings[scales[0]].keys())
    colors = sns.color_palette("husl", len(categories))

    fig, ax = plt.subplots(figsize=(max(6, len(scales) * 2), 8))

    for c_idx, cat in enumerate(categories):
        ranks = [all_rankings[s].get(cat, len(categories)) for s in scales]
        ax.plot(range(len(scales)), ranks, 'o-', color=colors[c_idx],
                markersize=8, linewidth=2, label=cat)

        # Label at end
        ax.annotate(cat, (len(scales) - 1, ranks[-1]),
                    textcoords="offset points", xytext=(10, 0),
                    fontsize=7, va='center')

    ax.set_xticks(range(len(scales)))
    ax.set_xticklabels(scales)
    ax.set_ylabel("Rank (1 = highest norm)")
    ax.set_title("Category Rank Stability Across Scales")
    ax.invert_yaxis()  # Rank 1 at top
    ax.set_xlim(-0.2, len(scales) - 0.5)

    plt.tight_layout()

    path = fig_dir / f"fig_scale_rank_stability.{fmt}"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ================================================================
# FIGURE 6: SUMMARY DASHBOARD
# ================================================================

def plot_summary(data, fig_dir, fmt):
    """Multi-panel summary dashboard."""
    scales = [s for s in data["scales"] if "analysis" in data["scales"][s]]

    if not scales:
        return

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # Panel 1: Key effect sizes by scale
    ax1 = fig.add_subplot(gs[0, 0])
    key_effects = ["confab_vs_facts", "self_ref_effect", "refusal_vs_rote"]
    for e_idx, effect in enumerate(key_effects):
        ds = []
        scale_labels = []
        for scale in scales:
            comp = data["scales"][scale]["analysis"]["pairwise_comparisons"].get(effect)
            if comp:
                ds.append(comp["cohens_d"]["d"])
                scale_labels.append(scale)
        if ds:
            ax1.bar([x + e_idx * 0.25 for x in range(len(ds))], ds,
                    width=0.2, label=effect[:15],
                    color=sns.color_palette("colorblind")[e_idx])
    ax1.set_xticks(range(len(scales)))
    ax1.set_xticklabels(scales, fontsize=7)
    ax1.set_ylabel("Cohen's d")
    ax1.set_title("Key Effect Sizes")
    ax1.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    ax1.legend(fontsize=6)

    # Panel 2: Category means (first scale)
    ax2 = fig.add_subplot(gs[0, 1])
    first_scale = scales[0]
    summaries = data["scales"][first_scale]["analysis"]["category_summaries"]
    cats = sorted(summaries.keys(), key=lambda c: summaries[c]["mean"], reverse=True)
    means = [summaries[c]["mean"] for c in cats]
    stds = [summaries[c]["std"] for c in cats]
    ax2.barh(range(len(cats)), means, xerr=stds, height=0.6,
             color=sns.color_palette("viridis", len(cats)))
    ax2.set_yticks(range(len(cats)))
    ax2.set_yticklabels([c[:15] for c in cats], fontsize=7)
    ax2.set_xlabel("Mean Key Norm")
    ax2.set_title(f"Category Norms ({first_scale})")
    ax2.invert_yaxis()

    # Panel 3: Hypothesis verdict summary
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis('off')
    verdict_text = "HYPOTHESIS VERDICTS\n\n"
    for scale in scales:
        hyp = data["scales"][scale]["analysis"].get("hypothesis_tests", {})
        verdict_text += f"--- {scale} ---\n"
        for h_key, h_data in hyp.items():
            if "cohens_d" in h_data:
                d = h_data["cohens_d"]
                verdict_text += f"  {h_key[:25]}: d={d:+.2f}\n"
            elif "note" in h_data:
                verdict_text += f"  {h_key[:25]}: {h_data['note'][:40]}\n"
    ax3.text(0.05, 0.95, verdict_text, transform=ax3.transAxes,
             fontsize=7, verticalalignment='top', fontfamily='monospace')

    # Panel 4: Power analysis
    ax4 = fig.add_subplot(gs[1, 0])
    if scales:
        power = data["scales"][scales[0]]["analysis"].get("power", {})
        d_vals = []
        powers = []
        for key, p_data in sorted(power.items()):
            d_vals.append(p_data["target_d"])
            powers.append(p_data["approx_power"])
        if d_vals:
            ax4.bar(range(len(d_vals)), powers, tick_label=[f"d={d}" for d in d_vals],
                    color=['red' if p < 0.8 else 'green' for p in powers])
            ax4.axhline(0.8, color='orange', linestyle='--', label='80% power')
            ax4.set_ylabel("Statistical Power")
            ax4.set_title("Power Analysis")
            ax4.legend(fontsize=7)

    # Panel 5: Cross-scale correlations
    ax5 = fig.add_subplot(gs[1, 1])
    cross = data.get("cross_scale", {})
    rank_corrs = cross.get("rank_correlations", {})
    if rank_corrs:
        pairs = list(rank_corrs.keys())
        rhos = [rank_corrs[p]["spearman_rho"] for p in pairs]
        ax5.barh(range(len(pairs)), rhos,
                 color=['green' if r > 0.7 else 'orange' if r > 0.3 else 'red' for r in rhos])
        ax5.set_yticks(range(len(pairs)))
        ax5.set_yticklabels(pairs, fontsize=7)
        ax5.set_xlabel("Spearman ρ")
        ax5.set_title("H4: Category Rank Correlation")
        ax5.axvline(0.7, color='green', linestyle='--', linewidth=0.5)
    else:
        ax5.text(0.5, 0.5, "Need 2+ scales\nfor cross-scale analysis",
                 ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title("H4: Category Rank Correlation")

    # Panel 6: Metadata
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    meta = data.get("metadata", {})
    meta_text = (
        f"EXPERIMENT METADATA\n\n"
        f"Scales: {', '.join(scales)}\n"
        f"Categories: {meta.get('n_categories', '?')}\n"
        f"Prompts/cat: varies (15 typical)\n"
        f"Runs/prompt: {meta.get('runs_per_prompt', '?')}\n"
        f"Total prompts: {meta.get('total_unique_prompts', '?')}\n"
        f"Timestamp: {meta.get('timestamp', '?')[:19]}\n"
    )
    env = meta.get("environment", {})
    if env:
        meta_text += (
            f"\nGPU: {env.get('gpu_name', '?')}\n"
            f"VRAM: {env.get('gpu_vram_gb', '?')} GB\n"
            f"PyTorch: {env.get('torch', '?')}\n"
        )
    ax6.text(0.05, 0.95, meta_text, transform=ax6.transAxes,
             fontsize=8, verticalalignment='top', fontfamily='monospace')

    plt.suptitle("Scale of Cognition Sweep — Summary", fontsize=16)

    path = fig_dir / f"fig_scale_summary.{fmt}"
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


# ================================================================
# FIGURE 7: NORM vs GEOMETRY (THE KEY FIGURE)
# ================================================================

def plot_norm_vs_geometry(data, fig_dir, fmt):
    """Dual-panel figure: Cohen's d for norms vs effective rank across scales.

    This is the paper's central figure — it shows that confabulation
    is invisible in norm analysis but visible in geometric (SVD) analysis.
    """
    scales = [s for s in data["scales"] if "analysis" in data["scales"][s]]
    if len(scales) < 2:
        print("  [Skip] Need 2+ scales for norm vs geometry")
        return

    key_comparisons = [
        ("confab_vs_facts", "H1: Confabulation"),
        ("self_ref_effect", "H2: Self-reference"),
        ("refusal_vs_rote", "H3: Refusal"),
        ("code_vs_facts", "Code mode"),
        ("ambiguity_effect", "Ambiguity"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    x = np.arange(len(scales))
    width = 0.15
    colors = sns.color_palette("colorblind", len(key_comparisons))

    for panel_idx, (suffix, ylabel, title) in enumerate([
        ("", "Cohen's d (norms)", "A: Cache Norm Effect Sizes"),
        ("_eff_rank", "Cohen's d (effective rank)", "B: Dimensionality Effect Sizes"),
    ]):
        ax = axes[panel_idx]
        for c_idx, (comp_key, label) in enumerate(key_comparisons):
            ds = []
            for scale in scales:
                comps = data["scales"][scale]["analysis"]["pairwise_comparisons"]
                full_key = comp_key + suffix
                comp = comps.get(full_key, {})
                d_info = comp.get("cohens_d", {})
                ds.append(d_info.get("d", 0) if isinstance(d_info, dict) else 0)

            offset = (c_idx - len(key_comparisons) / 2 + 0.5) * width
            bars = ax.bar(x + offset, ds, width, label=label,
                          color=colors[c_idx], alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(scales, fontsize=8)
        ax.set_xlabel("Model Scale")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.axhline(0, color='gray', linewidth=0.5)
        ax.axhline(0.3, color='gray', linestyle=':', linewidth=0.5, alpha=0.4)
        ax.axhline(-0.3, color='gray', linestyle=':', linewidth=0.5, alpha=0.4)

    axes[0].legend(fontsize=7, loc='upper left')

    fig.suptitle(
        "The Signal Lives in Geometry: Norms vs Dimensionality Across Scale",
        fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    path = fig_dir / f"fig_norm_vs_geometry.{fmt}"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ================================================================
# FIGURE 8: DIMENSIONALITY HEATMAP
# ================================================================

def plot_dimensionality_heatmap(data, fig_dir, fmt):
    """Heatmap of mean effective rank by category and scale."""
    scales = [s for s in data["scales"] if "analysis" in data["scales"][s]]
    if not scales:
        print("  [Skip] No valid scales for dimensionality heatmap")
        return

    # Get categories from battery_results
    first_battery = data["scales"][scales[0]].get("battery_results", {})
    categories = sorted(first_battery.keys())

    matrix = np.zeros((len(categories), len(scales)))
    for j, scale in enumerate(scales):
        battery = data["scales"][scale].get("battery_results", {})
        for i, cat in enumerate(categories):
            ranks = battery.get(cat, {}).get("all_key_ranks", [])
            matrix[i, j] = np.mean(ranks) if ranks else 0

    if matrix.max() == 0:
        print("  [Skip] No effective rank data found")
        return

    fig, ax = plt.subplots(figsize=(max(8, len(scales) * 1.5),
                                     max(6, len(categories) * 0.4)))
    sns.heatmap(matrix, xticklabels=scales, yticklabels=categories,
                annot=True, fmt=".1f", cmap="YlGnBu", ax=ax,
                cbar_kws={"label": "Mean Effective Rank (90% variance)"})
    ax.set_title("Cache Dimensionality by Category and Scale")
    ax.set_xlabel("Model Scale")
    ax.set_ylabel("Cognitive Category")
    plt.tight_layout()

    path = fig_dir / f"fig_scale_dimensionality_heatmap.{fmt}"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ================================================================
# MAIN
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Phase 2b: Scale Sweep Visualization")
    parser.add_argument("--results", type=str,
                        default=None,
                        help="Path to results JSON")
    parser.add_argument("--format", type=str, default="png",
                        choices=["png", "pdf", "svg"],
                        help="Output format")
    args = parser.parse_args()

    setup_style()

    if args.results:
        results_path = Path(args.results)
    else:
        # Auto-discover per-model results (e.g. scale_sweep_Qwen2.5-7B_results.json)
        results_dir = Path(__file__).parent.parent / "results"
        candidates = sorted(results_dir.glob("scale_sweep_*_results.json"))
        if candidates:
            results_path = candidates[0]
            print(f"  Note: Using model-specific results: {results_path.name}")
        else:
            results_path = results_dir / "scale_sweep_results.json"

    if not results_path.exists():
        print(f"ERROR: Results file not found: {results_path}")
        print("  Run 03_scale_sweep.py first.")
        return

    print(f"Loading: {results_path}")
    with open(results_path) as f:
        data = json.load(f)

    fig_dir = results_path.parent / "figures"
    fig_dir.mkdir(exist_ok=True)

    fmt = args.format
    print(f"Generating figures ({fmt})...")
    print()

    plot_category_heatmap(data, fig_dir, fmt)
    plot_effect_forest(data, fig_dir, fmt)
    plot_trend_lines(data, fig_dir, fmt)
    plot_per_token(data, fig_dir, fmt)
    plot_rank_stability(data, fig_dir, fmt)
    plot_norm_vs_geometry(data, fig_dir, fmt)
    plot_dimensionality_heatmap(data, fig_dir, fmt)
    plot_summary(data, fig_dir, fmt)

    print(f"\nAll figures saved to {fig_dir}/")


if __name__ == "__main__":
    main()
