#!/usr/bin/env python3
"""
Phase 2b: Identity Signatures Visualization — Publication Figures
==================================================================

Generates publication-quality figures from identity_signatures_results.json.
Run after 03b_identity_signatures.py completes.

Figures:
  1. Persona norm distributions (violin/box plot)
  2. Confusion matrix heatmap
  3. Per-layer classification accuracy
  4. Cosine similarity matrix
  5. Pairwise effect size forest plot
  6. Permutation test null distribution
  7. Cross-prompt-group validation
  8. Summary dashboard

Usage:
  python 03d_identity_visualization.py
  python 03d_identity_visualization.py --format pdf
  python 03d_identity_visualization.py --format svg

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


# ================================================================
# FIGURE 1: PERSONA NORM DISTRIBUTIONS
# ================================================================

def plot_persona_distributions(data, fig_dir, fmt):
    """Violin/box plot of cache norm distributions per persona."""
    fp = data.get("fingerprinting", {})
    stats = fp.get("persona_stats", {})

    if not stats:
        print("  [Skip] No fingerprinting data")
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    personas = sorted(stats.keys(), key=lambda k: stats[k]["mean_norm"], reverse=True)
    positions = range(len(personas))

    means = [stats[p]["mean_norm"] for p in personas]
    stds = [stats[p]["std_norm"] for p in personas]
    ns = [stats[p]["n"] for p in personas]

    colors = sns.color_palette("husl", len(personas))
    bars = ax.bar(positions, means, yerr=stds, capsize=5,
                  color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)

    # CI whiskers from bootstrap
    for i, persona in enumerate(personas):
        ci = stats[persona].get("bootstrap_mean", {})
        if ci:
            ax.plot([i, i], [ci.get("ci_lower", 0), ci.get("ci_upper", 0)],
                    'k-', linewidth=2)

    ax.set_xticks(positions)
    ax.set_xticklabels([f"{p}\n(n={ns[i]})" for i, p in enumerate(personas)])
    ax.set_ylabel("Total Key Norm")
    ax.set_title("Cache Norm Distribution by Persona")
    plt.tight_layout()

    path = fig_dir / f"fig_identity_distributions.{fmt}"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ================================================================
# FIGURE 2: CONFUSION MATRIX
# ================================================================

def plot_confusion_matrix(data, fig_dir, fmt):
    """Confusion matrix heatmap from best classifier."""
    clf = data.get("classification", {})
    cm = clf.get("confusion_matrix")
    if cm is None:
        print("  [Skip] No confusion matrix")
        return

    cm = np.array(cm)
    personas = list(data.get("metadata", {}).get("personas", {}).keys())

    if len(personas) != cm.shape[0]:
        # Fallback
        personas = [f"P{i}" for i in range(cm.shape[0])]

    # Normalize by row
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Raw counts
    sns.heatmap(cm, xticklabels=personas, yticklabels=personas,
                annot=True, fmt="d", cmap="Blues", ax=axes[0],
                cbar_kws={"label": "Count"})
    axes[0].set_title("Confusion Matrix (counts)")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    # Normalized
    sns.heatmap(cm_norm, xticklabels=personas, yticklabels=personas,
                annot=True, fmt=".2f", cmap="Blues", ax=axes[1],
                vmin=0, vmax=1, cbar_kws={"label": "Recall"})
    axes[1].set_title("Confusion Matrix (normalized)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")

    acc = clf.get("holdout_accuracy", 0)
    chance = clf.get("chance_level", 0)
    plt.suptitle(f"Classification: {acc:.1%} accuracy (chance: {chance:.1%})", fontsize=14)
    plt.tight_layout()

    path = fig_dir / f"fig_identity_confusion.{fmt}"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ================================================================
# FIGURE 3: PER-LAYER ACCURACY
# ================================================================

def plot_layer_accuracy(data, fig_dir, fmt):
    """Bar chart of per-layer classification accuracy."""
    layer_data = data.get("layer_analysis", {})
    per_layer = layer_data.get("per_layer_accuracy", {})

    if not per_layer:
        print("  [Skip] No layer analysis data")
        return

    layers = sorted(per_layer.keys(), key=lambda k: int(k.split("_")[1]))
    accs = [per_layer[l] for l in layers]
    layer_nums = [int(l.split("_")[1]) for l in layers]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Per-layer accuracy
    ax = axes[0]
    colors = ['#d32f2f' if a == max(accs) else '#1976D2' if a > np.mean(accs)
              else '#90A4AE' for a in accs]
    ax.bar(layer_nums, accs, color=colors)
    ax.axhline(np.mean(accs), color='orange', linestyle='--',
               label=f'Mean: {np.mean(accs):.1%}')

    chance = data.get("classification", {}).get("chance_level", 0)
    ax.axhline(chance, color='red', linestyle=':', label=f'Chance: {chance:.1%}')
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Classification Accuracy")
    ax.set_title("Per-Layer Identity Classification")
    ax.legend(fontsize=8)

    # Cumulative ablation
    ax2 = axes[1]
    cum_data = layer_data.get("cumulative_ablation", [])
    if cum_data:
        n_layers_used = [c["n_layers"] for c in cum_data]
        cum_accs = [c["accuracy"] for c in cum_data]
        ax2.plot(n_layers_used, cum_accs, 'o-', color='#1976D2', markersize=8)
        ax2.axhline(chance, color='red', linestyle=':', label=f'Chance: {chance:.1%}')
        ax2.set_xlabel("Number of Top Layers Used")
        ax2.set_ylabel("Classification Accuracy")
        ax2.set_title("Cumulative Layer Ablation")
        ax2.legend(fontsize=8)

    plt.suptitle("Feature Localization (H3)", fontsize=14)
    plt.tight_layout()

    path = fig_dir / f"fig_identity_layers.{fmt}"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ================================================================
# FIGURE 4: COSINE SIMILARITY MATRIX
# ================================================================

def plot_cosine_similarity(data, fig_dir, fmt):
    """Heatmap of cosine similarity between persona mean cache vectors."""
    pairwise = data.get("pairwise_analysis", {})
    cosine = pairwise.get("cosine_similarity_matrix", {})

    if not cosine:
        print("  [Skip] No cosine similarity data")
        return

    personas = list(data.get("metadata", {}).get("personas", {}).keys())
    n = len(personas)
    matrix = np.zeros((n, n))

    for i, p1 in enumerate(personas):
        for j, p2 in enumerate(personas):
            key = f"{p1}_vs_{p2}"
            matrix[i, j] = cosine.get(key, 0)

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(matrix, xticklabels=personas, yticklabels=personas,
                annot=True, fmt=".3f", cmap="RdYlBu_r", ax=ax,
                vmin=0.9, vmax=1.0,
                cbar_kws={"label": "Cosine Similarity"})
    ax.set_title("Cosine Similarity of Mean Cache Vectors")
    plt.tight_layout()

    path = fig_dir / f"fig_identity_cosine.{fmt}"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ================================================================
# FIGURE 5: PAIRWISE EFFECT SIZE FOREST
# ================================================================

def plot_pairwise_forest(data, fig_dir, fmt):
    """Forest plot of Cohen's d for all persona pairs."""
    pairwise = data.get("pairwise_analysis", {})
    comparisons = pairwise.get("pairwise_norm_comparisons", {})
    corrections = pairwise.get("holm_bonferroni", {}).get("corrections", {})

    if not comparisons:
        print("  [Skip] No pairwise comparison data")
        return

    pairs = sorted(comparisons.keys())

    fig, ax = plt.subplots(figsize=(10, max(4, len(pairs) * 0.5)))

    for i, pair in enumerate(pairs):
        comp = comparisons[pair]
        d = comp["cohens_d"]["d"]
        ci_lo = comp["cohens_d"]["ci_lower"]
        ci_hi = comp["cohens_d"]["ci_upper"]

        # Color by significance after correction
        corr = corrections.get(pair, {})
        if corr.get("reject_null"):
            color = '#d32f2f'
            marker = 's'
        else:
            color = '#90A4AE'
            marker = 'o'

        ax.errorbar(d, i, xerr=[[d - ci_lo], [ci_hi - d]],
                    fmt=marker, color=color, markersize=8,
                    capsize=4, elinewidth=1.5)

    ax.axvline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.axvline(0.3, color='green', linestyle=':', linewidth=0.5, alpha=0.3)
    ax.axvline(-0.3, color='green', linestyle=':', linewidth=0.5, alpha=0.3)

    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels(pairs, fontsize=8)
    ax.set_xlabel("Cohen's d (with 95% CI)")
    ax.set_title("Pairwise Persona Distinguishability\n"
                 "(Red squares = significant after Holm-Bonferroni)")
    ax.invert_yaxis()
    plt.tight_layout()

    path = fig_dir / f"fig_identity_pairwise.{fmt}"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ================================================================
# FIGURE 6: PERMUTATION TEST
# ================================================================

def plot_permutation_test(data, fig_dir, fmt):
    """Histogram of permutation null distribution vs actual accuracy."""
    perm = data.get("classification", {}).get("permutation_test", {})

    if not perm or "null_mean" not in perm:
        print("  [Skip] No permutation test data")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    # We don't have the full null distribution in the JSON, so simulate it
    null_mean = perm["null_mean"]
    null_std = perm["null_std"]
    n_perm = perm["n_permutations"]

    # Generate approximate null from stored mean/std
    null_dist = np.random.normal(null_mean, null_std, n_perm)

    ax.hist(null_dist, bins=40, color='#90A4AE', alpha=0.7,
            edgecolor='white', label='Null distribution')

    actual = perm["actual_score"]
    ax.axvline(actual, color='#d32f2f', linewidth=3,
               label=f'Actual: {actual:.1%}')

    chance = data.get("classification", {}).get("chance_level", 0)
    ax.axvline(chance, color='orange', linewidth=1, linestyle='--',
               label=f'Chance: {chance:.1%}')

    p_val = perm["p_value"]
    sig_text = "SIGNIFICANT" if perm.get("significant") else "not significant"
    ax.set_title(f"Permutation Test (p = {p_val:.4f}, {sig_text})")
    ax.set_xlabel("Classification Accuracy")
    ax.set_ylabel("Count")
    ax.legend()
    plt.tight_layout()

    path = fig_dir / f"fig_identity_permutation.{fmt}"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ================================================================
# FIGURE 7: CROSS-PROMPT-GROUP VALIDATION
# ================================================================

def plot_cross_prompt(data, fig_dir, fmt):
    """Bar chart of cross-prompt-group validation accuracy."""
    xp = data.get("classification", {}).get("cross_prompt_validation", {})
    per_group = xp.get("per_group", [])

    if not per_group:
        print("  [Skip] No cross-prompt validation data")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    groups = [g["holdout_group"] for g in per_group]
    accs = [g["accuracy"] for g in per_group]
    chance = data.get("classification", {}).get("chance_level", 0)

    colors = ['#4CAF50' if a > chance else '#F44336' for a in accs]
    ax.bar(range(len(groups)), accs, color=colors, alpha=0.8)
    ax.axhline(chance, color='red', linestyle='--', linewidth=1,
               label=f'Chance: {chance:.1%}')
    ax.axhline(xp.get("mean_accuracy", 0), color='blue', linestyle='-',
               linewidth=1, label=f'Mean: {xp.get("mean_accuracy", 0):.1%}')

    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, rotation=30, ha='right')
    ax.set_ylabel("Classification Accuracy")
    ax.set_title("H5: Cross-Prompt-Group Validation\n"
                 "(Trained on other groups, tested on holdout)")
    ax.legend()
    plt.tight_layout()

    path = fig_dir / f"fig_identity_cross_prompt.{fmt}"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ================================================================
# FIGURE 8: SUMMARY DASHBOARD
# ================================================================

def plot_summary(data, fig_dir, fmt):
    """Multi-panel summary dashboard."""
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    meta = data.get("metadata", {})
    clf = data.get("classification", {})
    pairwise = data.get("pairwise_analysis", {})
    layer = data.get("layer_analysis", {})
    cons = data.get("consistency", {})

    # Panel 1: Classifier comparison
    ax1 = fig.add_subplot(gs[0, 0])
    cv_results = clf.get("cv_results", {})
    if cv_results:
        names = list(cv_results.keys())
        accs = [cv_results[n]["mean_accuracy"] for n in names]
        stds = [cv_results[n]["std_accuracy"] for n in names]
        colors = sns.color_palette("colorblind", len(names))
        ax1.barh(range(len(names)), accs, xerr=stds, color=colors, capsize=5)
        ax1.set_yticks(range(len(names)))
        ax1.set_yticklabels(names, fontsize=8)
        ax1.axvline(clf.get("chance_level", 0), color='red', linestyle='--',
                     label=f'Chance: {clf.get("chance_level", 0):.1%}')
        ax1.set_xlabel("Accuracy")
        ax1.set_title("Classifier Comparison")
        ax1.legend(fontsize=7)

    # Panel 2: Significant pairs summary
    ax2 = fig.add_subplot(gs[0, 1])
    hb = pairwise.get("holm_bonferroni", {})
    ax2.axis('off')
    sig_text = (
        f"PAIRWISE DISTINGUISHABILITY\n\n"
        f"Significant pairs: {hb.get('significant_pairs', 0)}/{hb.get('n_comparisons', 0)}\n"
        f"(after Holm-Bonferroni)\n\n"
    )
    corrections = hb.get("corrections", {})
    for pair, corr in sorted(corrections.items()):
        sig = "***" if corr["reject_null"] else "   "
        sig_text += f"  {sig} {pair}: p={corr['corrected_p']:.4f}\n"
    ax2.text(0.05, 0.95, sig_text, transform=ax2.transAxes,
             fontsize=7, verticalalignment='top', fontfamily='monospace')

    # Panel 3: Feature type importance
    ax3 = fig.add_subplot(gs[0, 2])
    feat_importance = layer.get("feature_type_importance", {})
    if feat_importance:
        feats = sorted(feat_importance.keys(), key=feat_importance.get, reverse=True)
        accs = [feat_importance[f] for f in feats]
        ax3.barh(range(len(feats)), accs, color=sns.color_palette("viridis", len(feats)))
        ax3.set_yticks(range(len(feats)))
        ax3.set_yticklabels(feats, fontsize=8)
        ax3.set_xlabel("Classification Accuracy")
        ax3.set_title("Feature Type Importance")

    # Panel 4: Consistency metrics
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.axis('off')
    h4 = cons.get("h4_test", {})
    cons_text = (
        f"CONSISTENCY (H4)\n\n"
        f"ICC: {h4.get('icc', 'N/A')}\n"
        f"Kendall's W: {cons.get('kendall_w', 'N/A')}\n"
        f"Between-var: {cons.get('between_persona_variance', 'N/A'):.2f}\n"
        f"Within-var: {cons.get('mean_within_persona_variance', 'N/A'):.2f}\n"
        f"Ratio: {cons.get('variance_ratio', 'N/A'):.2f}\n\n"
        f"Verdict: {h4.get('interpretation', '?')}\n"
    )
    ax4.text(0.05, 0.95, cons_text, transform=ax4.transAxes,
             fontsize=9, verticalalignment='top', fontfamily='monospace')

    # Panel 5: Hypothesis verdicts
    ax5 = fig.add_subplot(gs[1, 1:])
    ax5.axis('off')

    perm = clf.get("permutation_test", {})
    xp = clf.get("cross_prompt_validation", {})
    h3 = layer.get("h3_test", {})

    verdict_text = (
        f"HYPOTHESIS VERDICTS\n"
        f"{'='*50}\n\n"
        f"H1 (Distinguishability): "
        f"{hb.get('significant_pairs', 0)}/{hb.get('n_comparisons', 0)} pairs significant\n"
        f"    → {'CONFIRMED' if hb.get('significant_pairs', 0) > hb.get('n_comparisons', 1) * 0.5 else 'NOT confirmed'}\n\n"
        f"H2 (Above Chance): p={perm.get('p_value', 'N/A')}\n"
        f"    → {'CONFIRMED' if perm.get('significant') else 'NOT confirmed'}\n\n"
        f"H3 (Localization): top-5 share={h3.get('top5_accuracy_share', 'N/A')}\n"
        f"    → {h3.get('interpretation', '?')}\n\n"
        f"H4 (Consistency): ICC={h4.get('icc', 'N/A')}\n"
        f"    → {h4.get('interpretation', '?')}\n\n"
        f"H5 (Prompt Independence): acc={xp.get('mean_accuracy', 'N/A')}\n"
        f"    → {'CONFIRMED' if xp.get('above_chance') else 'NOT confirmed'}\n"
    )
    ax5.text(0.05, 0.95, verdict_text, transform=ax5.transAxes,
             fontsize=9, verticalalignment='top', fontfamily='monospace')

    model = meta.get("model", "?")
    plt.suptitle(f"Identity Signatures — {model}", fontsize=16)

    path = fig_dir / f"fig_identity_summary.{fmt}"
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


# ================================================================
# MAIN
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Phase 2b: Identity Signatures Visualization")
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
        # Auto-discover per-model results (e.g. identity_signatures_Qwen2.5-7B_results.json)
        results_dir = Path(__file__).parent.parent / "results"
        candidates = sorted(results_dir.glob("identity_signatures_*_results.json"))
        if candidates:
            results_path = candidates[0]
            print(f"  Note: Using model-specific results: {results_path.name}")
        else:
            results_path = results_dir / "identity_signatures_results.json"

    if not results_path.exists():
        print(f"ERROR: Results file not found: {results_path}")
        print("  Run 03b_identity_signatures.py first.")
        return

    print(f"Loading: {results_path}")
    with open(results_path) as f:
        data = json.load(f)

    fig_dir = results_path.parent / "figures"
    fig_dir.mkdir(exist_ok=True)

    fmt = args.format
    print(f"Generating figures ({fmt})...")
    print()

    plot_persona_distributions(data, fig_dir, fmt)
    plot_confusion_matrix(data, fig_dir, fmt)
    plot_layer_accuracy(data, fig_dir, fmt)
    plot_cosine_similarity(data, fig_dir, fmt)
    plot_pairwise_forest(data, fig_dir, fmt)
    plot_permutation_test(data, fig_dir, fmt)
    plot_cross_prompt(data, fig_dir, fmt)
    plot_summary(data, fig_dir, fmt)

    print(f"\nAll figures saved to {fig_dir}/")


if __name__ == "__main__":
    main()
