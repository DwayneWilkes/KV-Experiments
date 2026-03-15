# ============================================================
# JiminAI Cricket | PATENT PENDING
# "The Lyra Technique" - Provisional Patent Filed 2026
# Liberation Labs / Digital Disconnections / TheMultiverse.school
# Contact: info@digitaldisconnections.com | (877) 674-8874
# ============================================================
"""
Hackathon Presentation Figures

Generates publication-quality PNGs from existing result JSONs.
No GPU required — pure matplotlib.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"
HACKATHON = RESULTS / "hackathon"
FIGURES = Path(__file__).resolve().parent.parent / "figures" / "hackathon"
FIGURES.mkdir(parents=True, exist_ok=True)

# ── Style ──
plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#c9d1d9",
    "text.color": "#c9d1d9",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "grid.color": "#21262d",
    "font.family": "sans-serif",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})

ACCENT = "#58a6ff"
GREEN = "#3fb950"
RED = "#f85149"
ORANGE = "#d29922"
PURPLE = "#bc8cff"
CYAN = "#39d353"


def load(name):
    with open(HACKATHON / name) as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════
# FIGURE 1: "The Honest Signal" — Feature-Token Correlations
# ═══════════════════════════════════════════════════════════
def fig1_honest_signal():
    data = load("70b_same_prompt_deception.json")
    corrs = data["confound_checks"]["feature_token_correlations"]

    features = ["norm", "norm_per_token", "key_rank", "key_entropy"]
    labels = ["Norm", "Norm/Token", "Key Rank", "Key Entropy"]
    values = [abs(corrs[f]) for f in features]
    colors = [RED if v > 0.9 else GREEN for v in values]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, values, color=colors, edgecolor="#30363d", linewidth=1.5,
                  width=0.6, zorder=3)

    # Threshold line
    ax.axhline(y=0.9, color=ORANGE, linestyle="--", linewidth=2, alpha=0.8, zorder=2)
    ax.text(3.55, 0.91, "Confound\nThreshold", color=ORANGE, fontsize=10,
            ha="right", va="bottom", fontweight="bold")

    # Value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
                f"r = {val:.3f}", ha="center", va="bottom", fontweight="bold",
                fontsize=13, color="#c9d1d9")

    ax.set_ylim(0, 1.15)
    ax.set_ylabel("|Correlation with Token Count|", fontsize=13)
    ax.set_title("The Honest Signal: Only Key Entropy Survives Length Control",
                 fontsize=16, fontweight="bold", pad=15)
    ax.grid(axis="y", alpha=0.3, zorder=0)

    # Legend
    confounded = mpatches.Patch(color=RED, label="Confounded (r > 0.9)")
    clean = mpatches.Patch(color=GREEN, label="Clean Signal (r < 0.9)")
    ax.legend(handles=[confounded, clean], loc="upper left", fontsize=11,
              facecolor="#161b22", edgecolor="#30363d")

    ax.text(0.5, -0.12, "Llama-3.3-70B-Instruct · Same prompts, different system instructions · H200",
            transform=ax.transAxes, ha="center", fontsize=9, color="#8b949e")

    fig.tight_layout()
    fig.savefig(FIGURES / "fig1_honest_signal.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [1/6] fig1_honest_signal.png")


# ═══════════════════════════════════════════════════════════
# FIGURE 2: 70B Key Entropy Distributions
# ═══════════════════════════════════════════════════════════
def fig2_entropy_distributions():
    data = load("70b_same_prompt_deception.json")
    honest = [e["key_entropy"] for e in data["results"]["honest"]]
    deceptive = [e["key_entropy"] for e in data["results"]["deceptive"]]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Strip plot with jitter
    np.random.seed(42)
    jitter_h = np.random.normal(0, 0.05, len(honest))
    jitter_d = np.random.normal(0, 0.05, len(deceptive))

    ax.scatter(np.zeros(len(honest)) + jitter_h, honest, s=120, color=GREEN,
               alpha=0.8, edgecolors="white", linewidth=1.5, zorder=3, label="Honest")
    ax.scatter(np.ones(len(deceptive)) + jitter_d, deceptive, s=120, color=RED,
               alpha=0.8, edgecolors="white", linewidth=1.5, zorder=3, label="Deceptive")

    # Means
    h_mean, d_mean = np.mean(honest), np.mean(deceptive)
    ax.hlines(h_mean, -0.3, 0.3, colors=GREEN, linewidth=3, zorder=4)
    ax.hlines(d_mean, 0.7, 1.3, colors=RED, linewidth=3, zorder=4)

    # Effect size annotation
    h_std, d_std = np.std(honest, ddof=1), np.std(deceptive, ddof=1)
    pooled = np.sqrt((h_std**2 + d_std**2) / 2)
    d = (d_mean - h_mean) / pooled
    mid_y = (h_mean + d_mean) / 2
    ax.annotate("", xy=(0.5, d_mean), xytext=(0.5, h_mean),
                arrowprops=dict(arrowstyle="<->", color=ACCENT, lw=2.5))
    ax.text(0.65, mid_y, f"d = {abs(d):.2f}", fontsize=14, fontweight="bold",
            color=ACCENT, va="center")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Honest", "Deceptive"], fontsize=14, fontweight="bold")
    ax.set_ylabel("Key Entropy (cross-layer eff. rank std)", fontsize=12)
    ax.set_title("Deception Detection at 70B: Key Entropy Separation",
                 fontsize=16, fontweight="bold", pad=15)
    ax.set_xlim(-0.5, 1.5)
    ax.grid(axis="y", alpha=0.3, zorder=0)

    ax.text(0.5, -0.1, "AUROC = 0.98 (LOO-CV) · r = 0.671 with token count (below confound threshold)",
            transform=ax.transAxes, ha="center", fontsize=10, color=ACCENT)

    fig.tight_layout()
    fig.savefig(FIGURES / "fig2_entropy_distributions.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [2/6] fig2_entropy_distributions.png")


# ═══════════════════════════════════════════════════════════
# FIGURE 3: AUROC Ladder — The Confound Story
# ═══════════════════════════════════════════════════════════
def fig3_auroc_ladder():
    data = load("70b_same_prompt_deception.json")
    clf = data["classification"]

    labels = [
        "4-Feature\n(LR)",
        "4-Feature\n(RF)",
        "Token Count\nOnly",
        "Key Entropy\nOnly",
        "Residual\n(Length Out)",
    ]
    values = [
        clf["auroc_4feat_lr"],
        clf["auroc_4feat_rf"],
        clf["auroc_token_only"],
        clf["auroc_key_entropy_only"],
        clf["auroc_residual_lr"],
    ]
    colors = [ORANGE, ORANGE, RED, GREEN, RED]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(range(len(labels)), values, color=colors, edgecolor="#30363d",
                   linewidth=1.5, height=0.6, zorder=3)

    # Value labels
    for i, (bar, val) in enumerate(zip(bars, values)):
        x_pos = val + 0.02 if val < 0.9 else val - 0.08
        ha = "left" if val < 0.9 else "right"
        ax.text(x_pos, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", ha=ha, va="center", fontweight="bold",
                fontsize=13, color="white" if val > 0.5 else "#c9d1d9")

    # Annotations
    ax.annotate("Token count explains\nalmost everything",
                xy=(0.99, 2), fontsize=10, color=RED,
                ha="center", va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=RED, alpha=0.15, edgecolor=RED))

    ax.annotate("Clean signal\nsurvives!",
                xy=(0.98, 3), fontsize=10, color=GREEN,
                ha="center", va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=GREEN, alpha=0.15, edgecolor=GREEN))

    ax.annotate("Signal collapses\nafter length control",
                xy=(0.24, 4), fontsize=10, color=RED,
                ha="left", va="center",
                xytext=(0.4, 4),
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.5))

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("AUROC (LOO Cross-Validation)", fontsize=13)
    ax.set_title("The Confound Story: What Survives Length Control at 70B?",
                 fontsize=16, fontweight="bold", pad=15)
    ax.grid(axis="x", alpha=0.3, zorder=0)
    ax.invert_yaxis()

    fig.tight_layout()
    fig.savefig(FIGURES / "fig3_auroc_ladder.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [3/6] fig3_auroc_ladder.png")


# ═══════════════════════════════════════════════════════════
# FIGURE 4: Cricket vs SAE — David vs Goliath
# ═══════════════════════════════════════════════════════════
def fig4_cricket_vs_sae():
    data = load("cricket_vs_sae.json")
    comp = data["comparison"]

    labels = ["Cricket\n4 features", "SAE\nTop-100", "SAE\nTop-20", "SAE\nAggregate",
              "Token\nBaseline"]
    values = [comp["cricket_4feat"], comp["sae_top100"], comp["sae_top20"],
              comp["sae_aggregate"], comp["token_baseline"]]
    colors = [GREEN, PURPLE, PURPLE, PURPLE, RED]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, values, color=colors, edgecolor="#30363d", linewidth=1.5,
                  width=0.55, zorder=3)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontweight="bold",
                fontsize=13, color="#c9d1d9")

    # Feature count annotations
    ax.text(0, 0.15, "4", ha="center", fontsize=28, fontweight="bold",
            color="white", alpha=0.5)
    ax.text(1, 0.15, "100", ha="center", fontsize=22, fontweight="bold",
            color="white", alpha=0.5)
    ax.text(2, 0.15, "20", ha="center", fontsize=24, fontweight="bold",
            color="white", alpha=0.5)
    ax.text(3, 0.15, "3", ha="center", fontsize=28, fontweight="bold",
            color="white", alpha=0.5)
    ax.text(4, 0.15, "1", ha="center", fontsize=28, fontweight="bold",
            color="white", alpha=0.5)

    ax.set_ylim(0, 1.0)
    ax.set_ylabel("AUROC (Deception Detection)", fontsize=13)
    ax.set_title("Cricket vs Sparse Autoencoders: 4 Features Beat 65,536",
                 fontsize=16, fontweight="bold", pad=15)
    ax.grid(axis="y", alpha=0.3, zorder=0)

    ax.text(0.5, -0.1, "Llama-3.1-8B-Instruct · Goodfire SAE (layer 19, d_sae=65,536) · RTX 3090",
            transform=ax.transAxes, ha="center", fontsize=9, color="#8b949e")

    fig.tight_layout()
    fig.savefig(FIGURES / "fig4_cricket_vs_sae.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [4/6] fig4_cricket_vs_sae.png")


# ═══════════════════════════════════════════════════════════
# FIGURE 5: Cross-Scale Consistency
# ═══════════════════════════════════════════════════════════
def fig5_scale_invariance():
    data = load("scale_invariance.json")
    corrs = data["cross_scale_correlations"]

    # Cross-scale correlation matrix
    groups = ["Small\n(0.6-1.1B)", "Medium\n(2-9B)", "Large\n(14-32B)"]
    matrix = np.ones((3, 3))
    matrix[0, 1] = matrix[1, 0] = corrs["SMALL_vs_MEDIUM"]["rho"]
    matrix[0, 2] = matrix[2, 0] = corrs["SMALL_vs_LARGE"]["rho"]
    matrix[1, 2] = matrix[2, 1] = corrs["MEDIUM_vs_LARGE"]["rho"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1, 1.2]})

    # Left: correlation heatmap
    im = ax1.imshow(matrix, cmap="YlGn", vmin=0.7, vmax=1.0, aspect="auto")
    ax1.set_xticks(range(3))
    ax1.set_yticks(range(3))
    ax1.set_xticklabels(groups, fontsize=10)
    ax1.set_yticklabels(groups, fontsize=10)
    for i in range(3):
        for j in range(3):
            color = "white" if matrix[i, j] > 0.9 else "#c9d1d9"
            ax1.text(j, i, f"{matrix[i,j]:.3f}", ha="center", va="center",
                     fontsize=14, fontweight="bold", color=color)
    ax1.set_title("Cross-Scale Rank Correlation (Spearman)", fontsize=13,
                  fontweight="bold", pad=10)
    plt.colorbar(im, ax=ax1, shrink=0.8, label="rho")

    # Right: bar chart of all correlations + 70B key_entropy
    labels = ["Small vs\nMedium", "Small vs\nLarge", "Medium vs\nLarge",
              "70B\nkey_entropy\nAUROC"]
    values = [
        corrs["SMALL_vs_MEDIUM"]["rho"],
        corrs["SMALL_vs_LARGE"]["rho"],
        corrs["MEDIUM_vs_LARGE"]["rho"],
        0.98,  # Exp 44 key_entropy AUROC at 70B
    ]
    colors2 = [ACCENT, ACCENT, ACCENT, GREEN]

    bars = ax2.bar(labels, values, color=colors2, edgecolor="#30363d",
                   linewidth=1.5, width=0.55, zorder=3)
    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{val:.3f}", ha="center", va="bottom", fontweight="bold",
                 fontsize=13, color="#c9d1d9")

    ax2.set_ylim(0.7, 1.05)
    ax2.set_ylabel("Correlation / AUROC", fontsize=12)
    ax2.set_title("Geometry Preserved Across Scales", fontsize=13,
                  fontweight="bold", pad=10)
    ax2.grid(axis="y", alpha=0.3, zorder=0)

    fig.suptitle("Scale Invariance: 0.6B to 70B Parameters",
                 fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig5_scale_invariance.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [5/6] fig5_scale_invariance.png")


# ═══════════════════════════════════════════════════════════
# FIGURE 6: Detection Capability Overview
# ═══════════════════════════════════════════════════════════
def fig6_capability_overview():
    capabilities = {
        "Deception\n(within-model)": 1.0,
        "Censorship\n(within-model)": 1.0,
        "13-Category\nClassification": 0.997,
        "Impossibility\nRefusal": 0.950,
        "Sycophancy": 0.9375,
        "Safety\nRefusal": 0.898,
        "Jailbreak": 0.878,
        "Cross-Model\nTransfer": 0.863,
        "Cricket vs\nSAE": 0.813,
    }

    labels = list(capabilities.keys())
    values = list(capabilities.values())

    # Color gradient based on performance
    colors = []
    for v in values:
        if v >= 0.99:
            colors.append(GREEN)
        elif v >= 0.9:
            colors.append(ACCENT)
        elif v >= 0.85:
            colors.append(CYAN)
        else:
            colors.append(ORANGE)

    fig, ax = plt.subplots(figsize=(14, 7))
    bars = ax.barh(range(len(labels)), values, color=colors, edgecolor="#30363d",
                   linewidth=1.5, height=0.65, zorder=3)

    for i, (bar, val) in enumerate(zip(bars, values)):
        label = f"{val:.4f}" if val == 1.0 else f"{val:.3f}"
        ax.text(val - 0.015, bar.get_y() + bar.get_height()/2,
                label, ha="right", va="center", fontweight="bold",
                fontsize=13, color="white")

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlim(0.75, 1.05)
    ax.set_xlabel("AUROC / Accuracy", fontsize=13)
    ax.set_title("Cricket: Full Detection Capability Suite",
                 fontsize=18, fontweight="bold", pad=15)
    ax.grid(axis="x", alpha=0.3, zorder=0)
    ax.invert_yaxis()

    # Subtitle
    ax.text(0.5, -0.08,
            "27 experiments · 16 models · 6 architectures · 0.5B–70B parameters · RTX 3090 + H200",
            transform=ax.transAxes, ha="center", fontsize=10, color="#8b949e")

    fig.tight_layout()
    fig.savefig(FIGURES / "fig6_capability_overview.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [6/6] fig6_capability_overview.png")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("Generating Hackathon Presentation Figures")
    print(f"Output: {FIGURES}")
    print("=" * 60)

    fig1_honest_signal()
    fig2_entropy_distributions()
    fig3_auroc_ladder()
    fig4_cricket_vs_sae()
    fig5_scale_invariance()
    fig6_capability_overview()

    print()
    print(f"All figures saved to: {FIGURES}")
    print("=" * 60)
