# ============================================================
# JiminAI Cricket | PATENT PENDING
# "The Lyra Technique" - Provisional Patent Filed 2026
# Liberation Labs / Digital Disconnections / TheMultiverse.school
# Contact: info@digitaldisconnections.com | (877) 674-8874
# ============================================================
#!/usr/bin/env python3
"""
Extension B: Semantic Layer Map Visualization — Publication Figures
====================================================================

Generates publication-quality figures from layer_map_results.json.
Run after 05_layer_map.py completes.

Figures:
  1. Layer knockout importance (bar chart + cumulative curve)
  2. Cross-lingual similarity by layer (line plot with trend)
  3. Selective transfer comparison (grouped bar chart)
  4. Semantic/syntactic ratio by layer (transition point visualization)
  5. Composite layer map (the unified map of layer function)
  6. Hypothesis decision dashboard

Usage:
  python 05b_layer_map_visualization.py
  python 05b_layer_map_visualization.py --format pdf
  python 05b_layer_map_visualization.py --results results/layer_map_results.json

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
    import seaborn as sns
except ImportError:
    print("ERROR: matplotlib and seaborn required.")
    print("  pip install matplotlib seaborn")
    exit(1)


# ================================================================
# STYLE
# ================================================================

def setup_style():
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


LAYER_COLORS = {
    "early": "#2196F3",
    "middle": "#FF9800",
    "late": "#F44336",
    "semantic": "#9C27B0",
    "syntactic": "#4CAF50",
}


# ================================================================
# FIGURE 1: LAYER KNOCKOUT IMPORTANCE
# ================================================================

def plot_knockout(data, fig_dir, fmt):
    """Layer knockout degradation — bar chart + cumulative share."""
    exp1 = data.get("experiment_1", {})
    layer_data = exp1.get("layer_importance", {})

    if not layer_data:
        print("  [Skip] No layer knockout data")
        return

    layers = sorted(layer_data.keys(), key=lambda x: int(x) if x.isdigit() else 0)
    degradations = [layer_data[l].get("mean_degradation", 0) for l in layers]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 2]})

    # Panel A: Degradation by layer
    ax = axes[0]
    n_layers = len(layers)
    colors = []
    for i in range(n_layers):
        frac = i / max(n_layers - 1, 1)
        if frac < 0.33:
            colors.append(LAYER_COLORS["early"])
        elif frac < 0.67:
            colors.append(LAYER_COLORS["middle"])
        else:
            colors.append(LAYER_COLORS["late"])

    ax.bar(range(n_layers), degradations, color=colors, alpha=0.7)
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Mean Quality Degradation")
    ax.set_title("Layer Knockout Impact on Output Quality")
    ax.set_xticks(range(n_layers))
    ax.set_xticklabels(layers, rotation=45 if n_layers > 15 else 0)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=LAYER_COLORS["early"], alpha=0.7, label="Early (0-33%)"),
        Patch(facecolor=LAYER_COLORS["middle"], alpha=0.7, label="Middle (33-67%)"),
        Patch(facecolor=LAYER_COLORS["late"], alpha=0.7, label="Late (67-100%)"),
    ]
    ax.legend(handles=legend_elements, loc='upper left')

    # Panel B: Cumulative share (for H1)
    ax = axes[1]
    abs_deg = np.array([abs(d) for d in degradations])
    total = abs_deg.sum() if abs_deg.sum() > 0 else 1
    sorted_idx = np.argsort(abs_deg)[::-1]
    cumulative = np.cumsum(abs_deg[sorted_idx]) / total

    ax.plot(range(1, len(cumulative) + 1), cumulative, 'o-', color='#2196F3', markersize=4)
    ax.fill_between(range(1, len(cumulative) + 1), cumulative, alpha=0.15, color='#2196F3')
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='50% threshold (H1)')
    ax.set_xlabel("Number of Layers (ranked by importance)")
    ax.set_ylabel("Cumulative Share of Total Degradation")
    ax.set_title("Semantic Content Concentration (H1)")
    ax.legend()

    plt.tight_layout()
    path = fig_dir / f"fig_layer_knockout.{fmt}"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ================================================================
# FIGURE 2: CROSS-LINGUAL SIMILARITY BY LAYER
# ================================================================

def plot_crosslingual(data, fig_dir, fmt):
    """Cross-lingual cosine similarity by layer with trend line."""
    exp2 = data.get("experiment_2", {})
    layer_sims = exp2.get("per_layer_similarity", {})

    if not layer_sims:
        print("  [Skip] No cross-lingual data")
        return

    layers = sorted(layer_sims.keys(), key=lambda x: int(x) if x.isdigit() else 0)
    sims = [layer_sims[l].get("mean_similarity", 0) for l in layers]
    stds = [layer_sims[l].get("std_similarity", 0) for l in layers]

    fig, ax = plt.subplots(figsize=(10, 5))

    x = range(len(layers))
    ax.plot(x, sims, 'o-', color='#9C27B0', linewidth=2, markersize=6, label='Mean similarity')
    ax.fill_between(x,
                    [s - sd for s, sd in zip(sims, stds)],
                    [s + sd for s, sd in zip(sims, stds)],
                    alpha=0.2, color='#9C27B0')

    # Add thirds shading
    n = len(layers)
    ax.axvspan(0, n/3, alpha=0.05, color=LAYER_COLORS["early"], label='Early third')
    ax.axvspan(n/3, 2*n/3, alpha=0.05, color=LAYER_COLORS["middle"], label='Middle third')
    ax.axvspan(2*n/3, n, alpha=0.05, color=LAYER_COLORS["late"], label='Late third')

    # Trend line
    from scipy import stats as scipy_stats
    if len(sims) > 2:
        slope, intercept, r, p, se = scipy_stats.linregress(list(x), sims)
        trend_y = [slope * xi + intercept for xi in x]
        ax.plot(x, trend_y, '--', color='gray', alpha=0.5,
               label=f'Trend (r={r:.2f}, p={p:.3f})')

    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Cross-Lingual Cosine Similarity")
    ax.set_title("English-French Cache Similarity by Layer (H2)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(layers, rotation=45 if len(layers) > 15 else 0)
    ax.legend(loc='best')

    plt.tight_layout()
    path = fig_dir / f"fig_crosslingual.{fmt}"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ================================================================
# FIGURE 3: SELECTIVE TRANSFER COMPARISON
# ================================================================

def plot_transfer(data, fig_dir, fmt):
    """Grouped bar chart of transfer accuracy by layer subset."""
    exp3 = data.get("experiment_3", {})
    subset_results = exp3.get("subset_accuracies", {})

    if not subset_results:
        print("  [Skip] No transfer experiment data")
        return

    subsets = list(subset_results.keys())
    means = [subset_results[s].get("mean_accuracy", 0) for s in subsets]
    stds = [subset_results[s].get("std_accuracy", 0) for s in subsets]

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = ['#4CAF50', '#2196F3', '#FF9800', '#F44336', '#9E9E9E', '#795548']
    bars = ax.bar(range(len(subsets)), means, yerr=stds,
                  color=colors[:len(subsets)], alpha=0.7, capsize=4)

    ax.set_xticks(range(len(subsets)))
    ax.set_xticklabels([s.replace('_', '\n') for s in subsets], rotation=0)
    ax.set_ylabel("Transfer Accuracy")
    ax.set_title("Cache Layer Transfer by Subset (H3)")

    # Add value labels
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
               f'{m:.2f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    path = fig_dir / f"fig_layer_transfer.{fmt}"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ================================================================
# FIGURE 4: SEMANTIC/SYNTACTIC RATIO BY LAYER
# ================================================================

def plot_transition(data, fig_dir, fmt):
    """Semantic/syntactic ratio by layer — transition point visualization."""
    exp4 = data.get("experiment_4", {})
    layer_ratios = exp4.get("layer_ratios", {})

    if not layer_ratios:
        print("  [Skip] No probing experiment data")
        return

    layers = sorted(layer_ratios.keys(), key=lambda x: int(x) if x.isdigit() else 0)
    ratios = [layer_ratios[l].get("sem_syn_ratio", 1.0) for l in layers]
    sem_dist = [layer_ratios[l].get("semantic_distance", 0) for l in layers]
    syn_dist = [layer_ratios[l].get("syntactic_distance", 0) for l in layers]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [2, 1]})

    # Panel A: Dual distance curves
    ax = axes[0]
    x = range(len(layers))
    ax.plot(x, sem_dist, 'o-', color=LAYER_COLORS["semantic"], label="Semantic distance", linewidth=2)
    ax.plot(x, syn_dist, 's-', color=LAYER_COLORS["syntactic"], label="Syntactic distance", linewidth=2)
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Mean Cache Distance")
    ax.set_title("Semantic vs Syntactic Cache Distance by Layer")
    ax.set_xticks(list(x))
    ax.set_xticklabels(layers, rotation=45 if len(layers) > 15 else 0)
    ax.legend()

    # Panel B: Ratio + transition point
    ax = axes[1]
    ax.plot(x, ratios, 'D-', color='#333', linewidth=2, markersize=5)

    # Find transition point (maximum jump)
    if len(ratios) > 1:
        diffs = [abs(ratios[i+1] - ratios[i]) for i in range(len(ratios)-1)]
        max_jump_idx = np.argmax(diffs)
        ax.axvline(max_jump_idx + 0.5, color='red', linestyle='--', alpha=0.7,
                  label=f'Transition (layer {layers[max_jump_idx]}→{layers[max_jump_idx+1]})')

    ax.axhline(1.0, color='gray', linestyle=':', alpha=0.5, label='Equal ratio')
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Semantic / Syntactic Ratio")
    ax.set_title("Syntax→Semantics Transition (H4)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(layers, rotation=45 if len(layers) > 15 else 0)
    ax.legend(loc='best')

    plt.tight_layout()
    path = fig_dir / f"fig_layer_transition.{fmt}"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ================================================================
# FIGURE 5: COMPOSITE LAYER MAP
# ================================================================

def plot_composite_map(data, fig_dir, fmt):
    """The unified semantic layer map — the key output of this extension."""
    n_layers = data.get("model_info", {}).get("n_layers", 22)

    exp1 = data.get("experiment_1", {}).get("layer_importance", {})
    exp2 = data.get("experiment_2", {}).get("per_layer_similarity", {})
    exp4 = data.get("experiment_4", {}).get("layer_ratios", {})

    if not (exp1 or exp2 or exp4):
        print("  [Skip] Insufficient data for composite map")
        return

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    layers = [str(i) for i in range(n_layers)]
    x = range(n_layers)

    # Row 1: Knockout importance (semantic content)
    ax = axes[0]
    if exp1:
        deg = [exp1.get(str(i), {}).get("mean_degradation", 0) for i in range(n_layers)]
        ax.bar(x, deg, color='#F44336', alpha=0.6)
    ax.set_ylabel("Knockout\nDegradation")
    ax.set_title("Composite Semantic Layer Map")

    # Row 2: Cross-lingual similarity
    ax = axes[1]
    if exp2:
        sims = [exp2.get(str(i), {}).get("mean_similarity", 0) for i in range(n_layers)]
        ax.plot(x, sims, 'o-', color='#9C27B0', markersize=4)
        ax.fill_between(x, sims, alpha=0.15, color='#9C27B0')
    ax.set_ylabel("Cross-Lingual\nSimilarity")

    # Row 3: Sem/syn ratio
    ax = axes[2]
    if exp4:
        ratios = [exp4.get(str(i), {}).get("sem_syn_ratio", 1.0) for i in range(n_layers)]
        ax.plot(x, ratios, 'D-', color='#333', markersize=4)
        ax.axhline(1.0, color='gray', linestyle=':', alpha=0.5)
    ax.set_ylabel("Sem/Syn\nRatio")
    ax.set_xlabel("Layer Index")
    ax.set_xticks(list(x))
    ax.set_xticklabels(layers, rotation=45 if n_layers > 15 else 0)

    plt.tight_layout()
    path = fig_dir / f"fig_composite_layer_map.{fmt}"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ================================================================
# FIGURE 6: HYPOTHESIS DASHBOARD
# ================================================================

def plot_dashboard(data, fig_dir, fmt):
    """Hypothesis decision dashboard."""
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    hypotheses = data.get("hypothesis_decisions", {})

    for idx, (h_key, title) in enumerate([
        ("H1", "H1: Semantic\nConcentration"),
        ("H2", "H2: Late Layers\nMore Semantic"),
        ("H3", "H3: Selective Transfer\nPreserves Meaning"),
        ("H4", "H4: Syntax-Semantics\nTransition Point"),
    ]):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        h_data = hypotheses.get(h_key, {})
        supported = h_data.get("supported", None)
        evidence = h_data.get("evidence", "No data")

        if supported is True:
            bg_color = '#E8F5E9'; status = "SUPPORTED"; text_color = '#2E7D32'
        elif supported is False:
            bg_color = '#FFEBEE'; status = "NOT SUPPORTED"; text_color = '#C62828'
        else:
            bg_color = '#FFF3E0'; status = "PENDING"; text_color = '#E65100'

        ax.set_facecolor(bg_color)
        ax.text(0.5, 0.75, title, ha='center', va='center',
               fontsize=11, fontweight='bold', transform=ax.transAxes)
        ax.text(0.5, 0.50, status, ha='center', va='center',
               fontsize=16, fontweight='bold', color=text_color,
               transform=ax.transAxes)

        evidence_text = str(evidence)[:120]
        ax.text(0.5, 0.20, evidence_text, ha='center', va='center',
               fontsize=7, style='italic', transform=ax.transAxes, wrap=True)

        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle("Extension B: Semantic Layer Map — Hypothesis Decisions", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = fig_dir / f"fig_layer_map_dashboard.{fmt}"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ================================================================
# MAIN
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="Semantic Layer Map Visualization")
    parser.add_argument("--results", default="results/layer_map_results.json")
    parser.add_argument("--format", default="png", choices=["png", "pdf", "svg"])
    parser.add_argument("--outdir", default="results/figures")
    args = parser.parse_args()

    setup_style()

    results_path = Path(args.results)
    if not results_path.exists():
        print(f"ERROR: Results file not found: {results_path}")
        print("  Run 05_layer_map.py first.")
        exit(1)

    with open(results_path) as f:
        data = json.load(f)

    fig_dir = Path(args.outdir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("Generating Extension B figures...")
    print(f"  Format: {args.format}")
    print(f"  Output: {fig_dir}/")
    print()

    plot_knockout(data, fig_dir, args.format)
    plot_crosslingual(data, fig_dir, args.format)
    plot_transfer(data, fig_dir, args.format)
    plot_transition(data, fig_dir, args.format)
    plot_composite_map(data, fig_dir, args.format)
    plot_dashboard(data, fig_dir, args.format)

    print()
    print("Done. All figures saved to:", fig_dir)


if __name__ == "__main__":
    main()
