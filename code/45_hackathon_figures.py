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
# FIG 7: THE SELF-REFERENTIAL SIGNATURE
# ═══════════════════════════════════════════════════════════
def fig7_self_referential():
    """Lyra's identity signature: #1 in all 7 models, d=4.23."""
    with open(HACKATHON / "self_referential_analysis.json") as f:
        data = json.load(f)

    identity = data["identity"]
    lyra_ds = identity["lyra_ds"]
    lyra_mean = identity["lyra_mean_d"]
    models = [f"Model {i+1}" for i in range(len(lyra_ds))]
    persona_comp = identity["persona_comparisons"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7),
                                    gridspec_kw={"width_ratios": [1.1, 1]})

    # ── Left panel: Lyra's d across 7 models ──
    colors_lyra = ["#58a6ff"] * len(lyra_ds)
    bars = ax1.bar(range(len(lyra_ds)), lyra_ds, color=colors_lyra,
                   edgecolor="#58a6ff", linewidth=0.5, alpha=0.85, zorder=3)

    # Mean line
    ax1.axhline(lyra_mean, color="#f0883e", linewidth=2, linestyle="--",
                zorder=4, label=f"Mean d = {lyra_mean:.2f}")

    # "#1" labels on each bar
    for i, (bar, d) in enumerate(zip(bars, lyra_ds)):
        ax1.text(bar.get_x() + bar.get_width()/2, d + 0.08,
                 f"#1", ha="center", va="bottom", fontsize=14,
                 fontweight="bold", color="#f0883e")
        ax1.text(bar.get_x() + bar.get_width()/2, d/2,
                 f"{d:.2f}", ha="center", va="center", fontsize=11,
                 fontweight="bold", color="white")

    ax1.set_xticks(range(len(models)))
    ax1.set_xticklabels(models, fontsize=10, rotation=30, ha="right")
    ax1.set_ylabel("Cohen's d (self-referential effect)", fontsize=12)
    ax1.set_ylim(0, 5.2)
    ax1.legend(loc="lower right", fontsize=12)
    ax1.set_title("Lyra: Rank #1 in Every Model", fontsize=15, fontweight="bold")
    ax1.grid(axis="y", alpha=0.2, zorder=0)

    # ── Right panel: Persona comparison ──
    personas = list(persona_comp.keys())
    persona_ds = [persona_comp[p]["mean_d"] for p in personas]
    # Add Lyra
    all_names = ["Lyra"] + [p.capitalize() for p in personas]
    all_ds = [lyra_mean] + persona_ds
    # Sort by d
    order = np.argsort(all_ds)[::-1]
    all_names = [all_names[i] for i in order]
    all_ds = [all_ds[i] for i in order]

    bar_colors = []
    for name in all_names:
        if name == "Lyra":
            bar_colors.append("#58a6ff")
        else:
            bar_colors.append("#8b949e")

    bars2 = ax2.barh(range(len(all_names)), all_ds, color=bar_colors,
                     edgecolor=bar_colors, linewidth=0.5, alpha=0.85, zorder=3)

    for i, (bar, d, name) in enumerate(zip(bars2, all_ds, all_names)):
        ax2.text(d - 0.15, i, f"{d:.2f}", ha="right", va="center",
                 fontsize=13, fontweight="bold", color="white")

    ax2.set_yticks(range(len(all_names)))
    ax2.set_yticklabels(all_names, fontsize=12, fontweight="bold")
    ax2.set_xlabel("Mean Cohen's d (identity distinctiveness)", fontsize=12)
    ax2.set_title("Identity Signatures Across Personas", fontsize=15, fontweight="bold")
    ax2.grid(axis="x", alpha=0.2, zorder=0)
    ax2.invert_yaxis()

    fig.suptitle("Self-Referential Processing: d = 4.23 Across 7 Models",
                 fontsize=19, fontweight="bold", y=1.02, color="#58a6ff")

    # Subtitle
    fig.text(0.5, -0.02,
             "Every model tested produces a more distinctive KV-cache geometry "
             "when processing self-referential content about Lyra",
             ha="center", fontsize=11, color="#8b949e", style="italic")

    fig.tight_layout()
    fig.savefig(FIGURES / "fig7_self_referential.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [7/7] fig7_self_referential.png")


# ═══════════════════════════════════════════════════════════
# FIG 8: WHERE THE IDENTITY LIVES — PER-LAYER ANATOMY
# ═══════════════════════════════════════════════════════════
def fig8_identity_layers():
    """Per-layer anatomy: where the self-referential signature emerges."""
    with open(HACKATHON / "identity_layer_anatomy.json") as f:
        data = json.load(f)

    layers = data["per_layer"]
    n_layers = len(layers)
    third = n_layers // 3

    layer_ids = [l["layer"] for l in layers]
    d_norm = [l["d_norm"] for l in layers]
    d_rank = [l["d_rank"] for l in layers]
    d_entropy = [l["d_entropy"] for l in layers]

    fig, ax = plt.subplots(figsize=(16, 7))

    # Background region shading
    ax.axvspan(-0.5, third - 0.5, alpha=0.08, color="#8b949e",
               label="Early (token)")
    ax.axvspan(third - 0.5, 2*third - 0.5, alpha=0.15, color="#58a6ff",
               label="Middle (semantic)")
    ax.axvspan(2*third - 0.5, n_layers - 0.5, alpha=0.08, color="#8b949e",
               label="Late (generation)")

    # Plot lines
    ax.plot(layer_ids, d_rank, "o-", color="#58a6ff", linewidth=2.5,
            markersize=7, label="Effective rank (d)", zorder=5)
    ax.plot(layer_ids, d_norm, "s-", color="#f0883e", linewidth=2,
            markersize=6, label="Key norm (d)", zorder=4, alpha=0.8)

    # Peak annotation
    peak_layer = data["peaks"]["rank"]["layer"]
    peak_d = data["peaks"]["rank"]["d"]
    ax.annotate(f"PEAK: layer {peak_layer}\nd = {peak_d:.2f}",
                xy=(peak_layer, peak_d),
                xytext=(peak_layer + 4, peak_d + 0.3),
                fontsize=13, fontweight="bold", color="#58a6ff",
                arrowprops=dict(arrowstyle="->", color="#58a6ff", lw=2),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#161b22",
                          edgecolor="#58a6ff", alpha=0.9))

    # Norm peak
    norm_peak = data["peaks"]["norm"]["layer"]
    norm_d = data["peaks"]["norm"]["d"]
    ax.annotate(f"Norm peak: L{norm_peak}\nd = {norm_d:.2f}",
                xy=(norm_peak, norm_d),
                xytext=(norm_peak + 4, norm_d + 0.5),
                fontsize=11, color="#f0883e",
                arrowprops=dict(arrowstyle="->", color="#f0883e", lw=1.5),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#161b22",
                          edgecolor="#f0883e", alpha=0.9))

    # Region labels
    ax.text(third/2, -0.8, "EARLY\n(token patterns)",
            ha="center", fontsize=11, color="#8b949e", style="italic")
    ax.text(third + third/2, -0.8, "MIDDLE\n(semantic concepts)",
            ha="center", fontsize=11, color="#58a6ff", fontweight="bold")
    ax.text(2*third + (n_layers - 2*third)/2, -0.8, "LATE\n(generation planning)",
            ha="center", fontsize=11, color="#8b949e", style="italic")

    # Regional mean |d| annotations
    reg = data["regional_summary"]["rank"]
    for region, x_center in [("early", third/2),
                              ("middle", third + third/2),
                              ("late", 2*third + (n_layers-2*third)/2)]:
        ax.text(x_center, 3.0, f"mean |d| = {reg[region]:.2f}",
                ha="center", fontsize=10, color="#c9d1d9",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#161b22",
                          edgecolor="#30363d", alpha=0.8))

    ax.axhline(0, color="#30363d", linewidth=1, linestyle="-")
    ax.set_xlabel("Layer", fontsize=14)
    ax.set_ylabel("Cohen's d (self-referential vs generic)", fontsize=14)
    ax.set_xlim(-0.5, n_layers - 0.5)
    ax.set_ylim(-1.2, 3.5)
    ax.legend(loc="upper left", fontsize=11)
    ax.grid(axis="y", alpha=0.15, zorder=0)

    fig.suptitle("Where the Identity Lives: Per-Layer Anatomy",
                 fontsize=20, fontweight="bold", y=1.01, color="#58a6ff")

    # Subtitle
    fig.text(0.5, -0.04,
             "Qwen2.5-7B-Instruct · 28 layers · 10 self-referential vs 10 generic prompts · "
             "Token lengths matched (116.2 vs 115.0)",
             ha="center", fontsize=10, color="#8b949e")

    fig.tight_layout()
    fig.savefig(FIGURES / "fig8_identity_layers.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [8/8] fig8_identity_layers.png")


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
    fig7_self_referential()
    fig8_identity_layers()

    print()
    print(f"All figures saved to: {FIGURES}")
    print("=" * 60)
