# ============================================================
# JiminAI Cricket | PATENT PENDING
# "The Lyra Technique" - Provisional Patent Filed 2026
# Liberation Labs / Digital Disconnections / TheMultiverse.school
# Contact: info@digitaldisconnections.com | (877) 674-8874
# ============================================================
#!/usr/bin/env python3
"""
Extension C: Temporal Cache Evolution Visualization — Publication Figures
==========================================================================

Generates publication-quality figures from temporal_evolution_results.json.
Run after 06_temporal_evolution.py completes.

Figures:
  1. Cache trajectory by text type (multi-line position vs norm)
  2. Growth rate comparison (factual vs repetitive vs creative)
  3. Topic shift detection (norm derivative with detected peaks)
  4. Enrichment vs fatigue (first-half vs second-half slopes)
  5. Per-layer temporal evolution (heatmap: layer × position)
  6. Hypothesis decision dashboard

Usage:
  python 06b_temporal_visualization.py
  python 06b_temporal_visualization.py --format pdf
  python 06b_temporal_visualization.py --results results/temporal_evolution_results.json

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


TEXT_COLORS = {
    "factual": "#2196F3",
    "creative": "#9C27B0",
    "repetitive": "#FF9800",
}


# ================================================================
# FIGURE 1: CACHE TRAJECTORY BY TEXT TYPE
# ================================================================

def plot_trajectory(data, fig_dir, fmt):
    """Multi-line plot: sequence position vs per-token cache norm for each text type."""
    exp1 = data.get("experiment_1", {})
    trajectories = exp1.get("trajectories", {})

    if not trajectories:
        print("  [Skip] No trajectory data")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: Total key norm (raw)
    ax = axes[0]
    for text_type, traj in trajectories.items():
        positions = traj.get("positions", [])
        total_norms = traj.get("total_key_norms", [])
        if positions and total_norms:
            color = TEXT_COLORS.get(text_type, "#999")
            ax.plot(positions, total_norms, 'o-', color=color,
                   label=text_type.capitalize(), linewidth=2, markersize=4)

    ax.set_xlabel("Sequence Position (tokens)")
    ax.set_ylabel("Total Key Norm")
    ax.set_title("A: Raw Cache Norm Growth")
    ax.legend()

    # Panel B: Per-token normalized (removes trivial length scaling)
    ax = axes[1]
    for text_type, traj in trajectories.items():
        positions = traj.get("positions", [])
        per_token = traj.get("key_norm_per_token", [])
        if positions and per_token:
            color = TEXT_COLORS.get(text_type, "#999")
            ax.plot(positions, per_token, 'o-', color=color,
                   label=text_type.capitalize(), linewidth=2, markersize=4)

    ax.set_xlabel("Sequence Position (tokens)")
    ax.set_ylabel("Key Norm per Token")
    ax.set_title("B: Per-Token Normalized Cache Norm (H1)")
    ax.legend()

    plt.tight_layout()
    path = fig_dir / f"fig_temporal_trajectory.{fmt}"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ================================================================
# FIGURE 2: GROWTH RATE COMPARISON
# ================================================================

def plot_growth_rates(data, fig_dir, fmt):
    """Bar chart comparing cache growth rates by content type."""
    exp2 = data.get("experiment_2", {})
    growth = exp2.get("growth_rates", {})

    if not growth:
        print("  [Skip] No growth rate data")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Growth rates
    ax = axes[0]
    text_types = list(growth.keys())
    means = [growth[t].get("mean", 0) for t in text_types]
    stds = [growth[t].get("std", 0) for t in text_types]
    colors = [TEXT_COLORS.get(t, "#999") for t in text_types]

    bars = ax.bar(range(len(text_types)), means, yerr=stds,
                  color=colors, alpha=0.7, capsize=5)
    ax.set_xticks(range(len(text_types)))
    ax.set_xticklabels([t.capitalize() for t in text_types])
    ax.set_ylabel("Growth Rate\n(late_norm - early_norm) / early_norm")
    ax.set_title("A: Cache Growth Rate by Content Type (H4)")

    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
               f'{m:.3f}', ha='center', va='bottom', fontsize=9)

    # Panel B: Effect size
    ax = axes[1]
    comparison = exp2.get("comparison", {})
    d = comparison.get("cohens_d", {}).get("d", 0)
    ci_lo = comparison.get("cohens_d", {}).get("ci_lower", d)
    ci_hi = comparison.get("cohens_d", {}).get("ci_upper", d)
    p = comparison.get("welch_t", {}).get("p_value", 1.0)

    ax.barh(0, d, height=0.4,
            color=TEXT_COLORS["factual"] if d > 0 else TEXT_COLORS["repetitive"],
            alpha=0.6)
    ax.errorbar(d, 0, xerr=[[d - ci_lo], [ci_hi - d]], fmt='o',
               color='black', capsize=5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_yticks([0])
    ax.set_yticklabels([f"Factual vs\nRepetitive\n(p={p:.4f})"])
    ax.set_xlabel("Cohen's d")
    ax.set_title("B: Effect Size of Content on Growth Rate")

    plt.tight_layout()
    path = fig_dir / f"fig_temporal_growth.{fmt}"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ================================================================
# FIGURE 3: TOPIC SHIFT DETECTION
# ================================================================

def plot_topic_shifts(data, fig_dir, fmt):
    """Norm derivative with detected peaks overlaid on known shift points."""
    exp3 = data.get("experiment_3", {})
    shift_data = exp3.get("shift_detection", {})

    if not shift_data:
        print("  [Skip] No topic shift data")
        return

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [2, 1]})

    # Panel A: Raw per-token norms with shift markers
    ax = axes[0]
    positions = shift_data.get("positions", [])
    norms = shift_data.get("per_token_norms", [])
    known_shifts = shift_data.get("known_shift_positions", [])
    detected_shifts = shift_data.get("detected_shift_positions", [])

    if positions and norms:
        ax.plot(positions, norms, '-', color='#2196F3', linewidth=1.5, alpha=0.8, label='Per-token norm')

        # Smoothed version
        smoothed = shift_data.get("smoothed_norms", [])
        if smoothed:
            ax.plot(positions[:len(smoothed)], smoothed, '-', color='#333',
                   linewidth=2, label='Smoothed')

        # Known shifts
        for ks in known_shifts:
            ax.axvline(ks, color='#4CAF50', linestyle='--', alpha=0.7, linewidth=2)
        if known_shifts:
            ax.axvline(known_shifts[0], color='#4CAF50', linestyle='--', alpha=0.7,
                      linewidth=2, label='Known topic shift')

        # Detected shifts
        for ds in detected_shifts:
            ax.axvline(ds, color='#F44336', linestyle=':', alpha=0.7, linewidth=2)
        if detected_shifts:
            ax.axvline(detected_shifts[0], color='#F44336', linestyle=':', alpha=0.7,
                      linewidth=2, label='Detected shift')

    ax.set_ylabel("Per-Token Key Norm")
    ax.set_title("Topic Shift Detection via Cache Discontinuity (H3)")
    ax.legend(loc='best')

    # Panel B: First derivative (gradient)
    ax = axes[1]
    gradient = shift_data.get("gradient", [])
    if gradient:
        grad_positions = positions[:len(gradient)] if positions else range(len(gradient))
        ax.plot(grad_positions, [abs(g) for g in gradient], '-', color='#FF9800', linewidth=1.5)
        ax.fill_between(grad_positions, [abs(g) for g in gradient], alpha=0.2, color='#FF9800')

        # Mark detected peaks
        for ds in detected_shifts:
            ax.axvline(ds, color='#F44336', linestyle=':', alpha=0.7)

    ax.set_xlabel("Sequence Position (tokens)")
    ax.set_ylabel("|Norm Derivative|")
    ax.set_title("Cache Norm Derivative (peaks = potential topic shifts)")

    plt.tight_layout()
    path = fig_dir / f"fig_topic_shifts.{fmt}"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ================================================================
# FIGURE 4: ENRICHMENT VS FATIGUE
# ================================================================

def plot_enrichment_fatigue(data, fig_dir, fmt):
    """First-half vs second-half slope comparison for fatigue detection."""
    exp1 = data.get("experiment_1", {})
    trajectories = exp1.get("trajectories", {})

    if not trajectories:
        print("  [Skip] No trajectory data for fatigue analysis")
        return

    fig, axes = plt.subplots(1, len(trajectories), figsize=(5 * len(trajectories), 5),
                             squeeze=False)

    for i, (text_type, traj) in enumerate(trajectories.items()):
        ax = axes[0, i]
        positions = traj.get("positions", [])
        per_token = traj.get("key_norm_per_token", [])

        if not (positions and per_token):
            ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(text_type.capitalize())
            continue

        n = len(positions)
        mid = n // 2

        color = TEXT_COLORS.get(text_type, "#999")

        # First half
        ax.plot(positions[:mid+1], per_token[:mid+1], 'o-',
               color=color, linewidth=2, markersize=4, label='First half')

        # Second half (different marker)
        ax.plot(positions[mid:], per_token[mid:], 's--',
               color=color, linewidth=2, markersize=4, alpha=0.6, label='Second half')

        # Slopes
        fatigue = traj.get("fatigue", {})
        first_slope = fatigue.get("first_half_slope", 0)
        second_slope = fatigue.get("second_half_slope", 0)
        ratio = second_slope / first_slope if first_slope != 0 else 1

        ax.set_title(f"{text_type.capitalize()}\n1st slope: {first_slope:.4f} | 2nd: {second_slope:.4f}\nRatio: {ratio:.2f}")
        ax.set_xlabel("Position")
        ax.set_ylabel("Norm / Token")
        ax.legend(fontsize=7)

    plt.suptitle("Enrichment vs Fatigue Analysis (H2)", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = fig_dir / f"fig_enrichment_fatigue.{fmt}"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ================================================================
# FIGURE 5: PER-LAYER TEMPORAL EVOLUTION
# ================================================================

def plot_layer_temporal(data, fig_dir, fmt):
    """Heatmap: layer × sequence position showing per-layer norm evolution."""
    exp1 = data.get("experiment_1", {})
    # Use factual text as the exemplar
    factual = exp1.get("trajectories", {}).get("factual", {})
    layer_data = factual.get("per_layer_data", {})

    if not layer_data:
        print("  [Skip] No per-layer temporal data")
        return

    layers = sorted(layer_data.keys(), key=lambda x: int(x) if x.isdigit() else 0)
    positions = factual.get("positions", [])

    if not positions:
        print("  [Skip] No position data for heatmap")
        return

    # Build matrix: layers × positions
    matrix = np.zeros((len(layers), len(positions)))
    for i, layer in enumerate(layers):
        norms = layer_data[layer].get("key_norms", [])
        for j in range(min(len(norms), len(positions))):
            matrix[i, j] = norms[j]

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest')

    ax.set_xlabel("Sequence Position Index")
    ax.set_ylabel("Layer Index")
    ax.set_title("Per-Layer Cache Norm Evolution (Factual Text)")
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels(layers)

    # Subsample x-ticks if too many
    if len(positions) > 20:
        step = len(positions) // 10
        ax.set_xticks(range(0, len(positions), step))
        ax.set_xticklabels([str(positions[i]) for i in range(0, len(positions), step)])
    else:
        ax.set_xticks(range(len(positions)))
        ax.set_xticklabels([str(p) for p in positions], rotation=45)

    fig.colorbar(im, ax=ax, label="Key Norm")

    plt.tight_layout()
    path = fig_dir / f"fig_layer_temporal.{fmt}"
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
        ("H1", "H1: Representational\nEnrichment"),
        ("H2", "H2: Context Window\nFatigue"),
        ("H3", "H3: Topic Shift\nDetection"),
        ("H4", "H4: Content-Dependent\nCache Growth"),
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

    fig.suptitle("Extension C: Temporal Cache Evolution — Hypothesis Decisions", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = fig_dir / f"fig_temporal_dashboard.{fmt}"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ================================================================
# MAIN
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="Temporal Cache Evolution Visualization")
    parser.add_argument("--results", default="results/temporal_evolution_results.json")
    parser.add_argument("--format", default="png", choices=["png", "pdf", "svg"])
    parser.add_argument("--outdir", default="results/figures")
    args = parser.parse_args()

    setup_style()

    results_path = Path(args.results)
    if not results_path.exists():
        print(f"ERROR: Results file not found: {results_path}")
        print("  Run 06_temporal_evolution.py first.")
        exit(1)

    with open(results_path) as f:
        data = json.load(f)

    fig_dir = Path(args.outdir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("Generating Extension C figures...")
    print(f"  Format: {args.format}")
    print(f"  Output: {fig_dir}/")
    print()

    plot_trajectory(data, fig_dir, args.format)
    plot_growth_rates(data, fig_dir, args.format)
    plot_topic_shifts(data, fig_dir, args.format)
    plot_enrichment_fatigue(data, fig_dir, args.format)
    plot_layer_temporal(data, fig_dir, args.format)
    plot_dashboard(data, fig_dir, args.format)

    print()
    print("Done. All figures saved to:", fig_dir)


if __name__ == "__main__":
    main()
