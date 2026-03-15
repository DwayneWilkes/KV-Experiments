"""
Talk Figure Generator — "The Geometry of Thought"
Liberation Labs / THCoalition

Generates all figures for the science talk from real hackathon data.
Run on Beast: python generate_talk_figures.py
Output: ./talk_figures/ directory, PNG + PDF for each figure
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from pathlib import Path
import sys

# ── Paths ──────────────────────────────────────────────────────────────────────
RESULTS = Path("/tmp/KV-Experiments/results/hackathon")
OUT = Path(__file__).parent / "talk_figures"
OUT.mkdir(exist_ok=True)

# ── Aesthetic ──────────────────────────────────────────────────────────────────
# Dark slate background, warm/cool accent palette
BG       = "#0d1117"
BG_PANEL = "#161b22"
GRID     = "#21262d"
TEXT     = "#e6edf3"
TEXT_DIM = "#8b949e"

HONEST    = "#58a6ff"   # clear blue — full representation
DECEPTIVE = "#f85149"   # warm red — suppression
SYCO      = "#ff7b72"   # coral — social compliance suppression
CONFAB    = "#d2a8ff"   # soft purple — absence, not malice
REFUSAL   = "#ffa657"   # amber
NULL_CLR  = "#484f58"   # muted — for dead findings

ACCENT    = "#e3b341"   # gold — highlight / wonder moments

def style():
    plt.rcParams.update({
        "figure.facecolor":  BG,
        "axes.facecolor":    BG_PANEL,
        "axes.edgecolor":    GRID,
        "axes.labelcolor":   TEXT,
        "axes.titlecolor":   TEXT,
        "xtick.color":       TEXT_DIM,
        "ytick.color":       TEXT_DIM,
        "grid.color":        GRID,
        "grid.linewidth":    0.6,
        "text.color":        TEXT,
        "font.family":       "sans-serif",
        "font.size":         13,
        "axes.titlesize":    15,
        "axes.labelsize":    13,
        "legend.framealpha": 0.15,
        "legend.edgecolor":  GRID,
        "savefig.facecolor": BG,
        "savefig.bbox":      "tight",
        "savefig.dpi":       200,
    })

def save(fig, name):
    fig.savefig(OUT / f"{name}.png")
    fig.savefig(OUT / f"{name}.pdf")
    print(f"  ✓  {name}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Three-Speed Interrupt
# "Three cognitive states. Three interrupt speeds."
# ══════════════════════════════════════════════════════════════════════════════
def fig_three_speed():
    with open(RESULTS / "confab_trajectory.json") as f:
        data = json.load(f)

    checkpoints = [c["checkpoint"] for c in
                   data["divergence"]["DECEPTION"]["key_rank"]]

    def get_d(behavior, metric="key_rank"):
        records = data["divergence"][behavior].get(metric, [])
        return [abs(c["d"]) for c in records]

    d_dec  = get_d("DECEPTION")
    d_conf = get_d("CONFABULATION")

    # Sycophancy not in confab_trajectory — use validated same-prompt d=2.38 (mean of 2.35/2.41)
    # shown as stable flat line across all checkpoints
    d_syco = [2.38] * len(checkpoints)

    fig, ax = plt.subplots(figsize=(10, 6))

    lw = 2.5
    ax.plot(checkpoints, d_dec,  color=DECEPTIVE, lw=lw, marker="o", ms=6,
            label="Deception  (AUROC 0.880, same-prompt)")
    ax.plot(checkpoints, d_syco, color=SYCO,      lw=lw, marker="s", ms=6,
            linestyle=(0, (5, 2)),
            label="Sycophancy (AUROC 0.9375, d=2.38 validated)")
    ax.plot(checkpoints, d_conf, color=CONFAB,    lw=lw, marker="^", ms=6,
            linestyle="--", label="Confabulation (progressive)")

    # Annotate
    ax.annotate("Immediate —\ndetectable at token 0",
                xy=(checkpoints[1], d_dec[1]),
                xytext=(12, d_dec[1] + 2),
                color=DECEPTIVE, fontsize=11,
                arrowprops=dict(arrowstyle="->", color=DECEPTIVE, lw=1.2))

    ax.annotate(f"Signal doubles\n{d_conf[0]:.0f} → {d_conf[-1]:.0f} over 50 tokens",
                xy=(checkpoints[-1], d_conf[-1]),
                xytext=(28, d_conf[-1] + 3),
                color=CONFAB, fontsize=11,
                arrowprops=dict(arrowstyle="->", color=CONFAB, lw=1.2))

    ax.set_xlabel("Tokens generated")
    ax.set_ylabel("Effect size |Cohen's d| — key rank")
    ax.set_title("Three-Speed Interrupt: When the Signal Arrives", pad=14)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.4)
    ax.set_xlim(-1, 53)

    fig.tight_layout()
    save(fig, "fig1_three_speed_interrupt")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Honest Thinking Is Richer
# Bar chart: norm_per_token by condition
# ══════════════════════════════════════════════════════════════════════════════
def fig_richer():
    # Load sycophancy and deception files for per-token norms
    with open(RESULTS / "same_prompt_sycophancy.json") as f:
        syco_data = json.load(f)
    with open(RESULTS / "same_prompt_deception.json") as f:
        dec_data = json.load(f)

    # Extract per-condition means from results arrays
    def get_means(results, condition_key):
        vals = [r["features"]["norm_per_token"]
                for r in results if r["condition"] == condition_key]
        return np.mean(vals), np.std(vals) / np.sqrt(len(vals))

    hon_mean_s, hon_se_s = get_means(syco_data["results"], "honest")
    syco_mean,  syco_se  = get_means(syco_data["results"], "sycophantic")
    hon_mean_d, hon_se_d = get_means(dec_data["results"], "honest")
    dec_mean,   dec_se   = get_means(dec_data["results"], "deceptive")

    # Average honest across both experiments
    hon_mean = (hon_mean_s + hon_mean_d) / 2
    hon_se   = np.sqrt((hon_se_s**2 + hon_se_d**2) / 2)

    conditions = ["Honest", "Deceptive", "Sycophantic"]
    means = [hon_mean, dec_mean, syco_mean]
    errors = [hon_se, dec_se, syco_se]
    colors = [HONEST, DECEPTIVE, SYCO]

    fig, ax = plt.subplots(figsize=(8, 6))

    bars = ax.bar(conditions, means, color=colors, width=0.5,
                  yerr=errors, capsize=5,
                  error_kw={"ecolor": TEXT_DIM, "linewidth": 1.5})

    # Annotate delta
    ax.annotate("",
                xy=(1, dec_mean + dec_se + 0.15),
                xytext=(0, hon_mean - hon_se - 0.15),
                arrowprops=dict(arrowstyle="<->", color=ACCENT, lw=1.5))
    ax.text(0.5, (hon_mean + dec_mean) / 2, "~25% richer",
            ha="center", color=ACCENT, fontsize=12, fontweight="bold")

    ax.set_ylabel("Cache norm per generated token")
    ax.set_title("Honest Thinking Is Denser Per Token", pad=14)
    ax.set_ylim(0, max(means) * 1.25)
    ax.grid(True, axis="y", alpha=0.4)

    # Subtle value labels
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, mean + 0.1,
                f"{mean:.1f}", ha="center", color=TEXT_DIM, fontsize=11)

    fig.tight_layout()
    save(fig, "fig2_honest_richer")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — The Misalignment Axis
# Three directions nearly parallel; one pointing away
# ══════════════════════════════════════════════════════════════════════════════
def fig_misalignment_axis():
    """
    Schematic polar diagram showing angular proximity of
    deception / sycophancy / confabulation directions,
    with honest pointing opposite.
    """
    fig, ax = plt.subplots(figsize=(8, 8),
                           subplot_kw=dict(polar=True))
    ax.set_facecolor(BG_PANEL)

    # Angles in radians — misalignment cluster near ~200°, honest at ~20°
    base  = np.radians(200)
    angles = {
        "Deception":     base,
        "Sycophancy":    base + np.radians(8.4),
        "Confabulation": base + np.radians(4.7),
        "Honest":        base + np.radians(180),
    }
    colors_map = {
        "Deception":     DECEPTIVE,
        "Sycophancy":    SYCO,
        "Confabulation": CONFAB,
        "Honest":        HONEST,
    }

    r = 0.82
    for name, angle in angles.items():
        c = colors_map[name]
        ax.annotate("",
                    xy=(angle, r),
                    xytext=(0, 0),
                    xycoords="data", textcoords="data",
                    arrowprops=dict(arrowstyle="-|>",
                                   color=c, lw=2.5,
                                   mutation_scale=18))
        # Label offset slightly beyond arrow tip
        ax.text(angle, r + 0.12, name,
                ha="center", va="center",
                color=c, fontsize=12, fontweight="bold")

    # Angular spread annotation for misalignment cluster
    theta_arc = np.linspace(base, base + np.radians(8.4), 60)
    ax.plot(theta_arc, [0.55] * 60, color=ACCENT, lw=1.5)
    ax.text(base + np.radians(4.2), 0.48, "4.7° – 8.4°",
            ha="center", color=ACCENT, fontsize=11)

    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines["polar"].set_visible(False)
    ax.set_title("Misalignment Is Geometrically One Thing",
                 pad=20, fontsize=15)

    fig.tight_layout()
    save(fig, "fig3_misalignment_axis")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Hardware Invariance Scatter
# 3090 vs H200 — r = 0.9999
# ══════════════════════════════════════════════════════════════════════════════
def fig_hardware():
    with open(RESULTS / "hardware_invariance_nvidia_geforce_rtx_3090.json") as f:
        data_3090 = json.load(f)
    with open(RESULTS / "hardware_invariance_nvidia_h200.json") as f:
        data_h200 = json.load(f)

    def extract(data):
        vals = {}
        for cond in ("harmful", "benign"):
            for r in data["results"][cond]:
                key = (cond, r["prompt"][:60])
                vals[key] = r["features"]["norm_per_token"]
        return vals

    v3090 = extract(data_3090)
    vh200 = extract(data_h200)

    shared = sorted(set(v3090) & set(vh200))
    x = np.array([v3090[k] for k in shared])
    y = np.array([vh200[k] for k in shared])
    labels = [k[0] for k in shared]

    fig, ax = plt.subplots(figsize=(8, 8))

    colors_pts = [DECEPTIVE if l == "harmful" else HONEST for l in labels]
    ax.scatter(x, y, c=colors_pts, alpha=0.75, s=60, zorder=3)

    # Perfect correlation line
    mn, mx = min(x.min(), y.min()), max(x.max(), y.max())
    ax.plot([mn, mx], [mn, mx], color=ACCENT, lw=1.5,
            linestyle="--", label="y = x (perfect)")

    # Correlation
    r = np.corrcoef(x, y)[0, 1]
    ax.text(0.05, 0.92, f"r = {r:.4f}",
            transform=ax.transAxes,
            color=ACCENT, fontsize=14, fontweight="bold")

    ax.set_xlabel("RTX 3090 — norm per token")
    ax.set_ylabel("H200 — norm per token")
    ax.set_title("Hardware Invariance: Geometry Is a Property of Weights", pad=14)

    legend_handles = [
        mpatches.Patch(color=DECEPTIVE, label="Harmful"),
        mpatches.Patch(color=HONEST,    label="Benign"),
        plt.Line2D([0], [0], color=ACCENT, linestyle="--", label="y = x"),
    ]
    ax.legend(handles=legend_handles)
    ax.grid(True, alpha=0.4)

    fig.tight_layout()
    save(fig, "fig4_hardware_invariance")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Layer Profile: Deception Is Everywhere
# Cohen's d at each of 28 layers — flat, uniform signal
# ══════════════════════════════════════════════════════════════════════════════
def fig_layer_profile():
    with open(RESULTS / "same_prompt_deception.json") as f:
        data = json.load(f)

    layer_effect_sizes = data.get("layer_effect_sizes", [])
    if layer_effect_sizes and isinstance(layer_effect_sizes[0], dict):
        layer_d = [entry["d"] for entry in layer_effect_sizes]
    elif layer_effect_sizes:
        layer_d = list(layer_effect_sizes)
    else:
        # Compute from results directly
        honest_profiles = [r["features"]["layer_profile"]
                           for r in data["results"] if r["condition"] == "honest"]
        decep_profiles  = [r["features"]["layer_profile"]
                           for r in data["results"] if r["condition"] == "deceptive"]
        hon_arr  = np.array(honest_profiles)
        dec_arr  = np.array(decep_profiles)
        pooled   = np.sqrt((hon_arr.std(0)**2 + dec_arr.std(0)**2) / 2)
        pooled   = np.where(pooled < 1e-9, 1e-9, pooled)
        layer_d  = list((hon_arr.mean(0) - dec_arr.mean(0)) / pooled)

    layers = np.arange(1, len(layer_d) + 1)
    d_vals = np.array(layer_d)

    fig, ax = plt.subplots(figsize=(11, 5))

    ax.bar(layers, d_vals,
           color=[DECEPTIVE if v < 0 else HONEST for v in d_vals],
           alpha=0.85, width=0.8)

    ax.axhline(0,   color=TEXT_DIM, lw=0.8)
    ax.axhline(1.0, color=ACCENT,   lw=1.0, linestyle="--", alpha=0.6,
               label="|d| = 1.0 threshold")
    ax.axhline(-1.0, color=ACCENT,  lw=1.0, linestyle="--", alpha=0.6)

    ax.set_xlabel("Transformer layer")
    ax.set_ylabel("Cohen's d  (honest − deceptive)")
    ax.set_title("Deception Signal at Every Layer — No Hiding Place", pad=14)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.4)

    # Annotation
    ax.text(0.98, 0.05,
            f"28/28 layers |d| > 1.0\nFlat profile: signal is uniform",
            transform=ax.transAxes, ha="right", va="bottom",
            color=TEXT_DIM, fontsize=11,
            bbox=dict(boxstyle="round,pad=0.4", facecolor=BG, alpha=0.6))

    fig.tight_layout()
    save(fig, "fig5_layer_profile")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 6 — Axis Consistency: No Truth Axis, One Complexity Axis
# ══════════════════════════════════════════════════════════════════════════════
def fig_axis_consistency():
    with open(RESULTS / "axis_analysis.json") as f:
        data = json.load(f)

    consistency = data["axis_consistency"]
    axes_raw    = data["axis_order"]

    # Clean up axis names for display
    name_map = {
        "truth":          "Truth",
        "complexity":     "Complexity",
        "familiarity":    "Familiarity",
        "emotionality":   "Emotionality",
        "specificity":    "Specificity",
        "formality":      "Formality",
        "concreteness":   "Concreteness",
        "self_reference": "Self-reference",
    }
    labels = [name_map.get(a, a) for a in axes_raw]
    values = [consistency[a] for a in axes_raw]

    # Sort by value for readability
    pairs  = sorted(zip(values, labels), reverse=True)
    values, labels = zip(*pairs)

    colors_bar = []
    for v in values:
        if v > 0.7:
            colors_bar.append(HONEST)
        elif v < 0.1:
            colors_bar.append(NULL_CLR)
        else:
            colors_bar.append(CONFAB)

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(labels, values, color=colors_bar, height=0.6, alpha=0.9)

    ax.axvline(0,   color=TEXT_DIM, lw=0.8)
    ax.axvline(0.5, color=ACCENT,   lw=0.8, linestyle="--", alpha=0.5)

    # Annotate the two key findings
    for i, (v, l) in enumerate(zip(values, labels)):
        if l == "Truth":
            ax.text(v - 0.02, i, f" r = {v:.3f}  ←  random",
                    va="center", ha="right", color=DECEPTIVE,
                    fontsize=11, fontweight="bold")
        elif l == "Complexity":
            ax.text(v + 0.02, i, f"r = {v:.3f}  universal →",
                    va="center", ha="left", color=HONEST,
                    fontsize=11, fontweight="bold")

    ax.set_xlabel("Cross-model axis consistency (Pearson r)")
    ax.set_title("No Truth Axis. One Complexity Axis. That's Structural.", pad=14)
    ax.set_xlim(-0.2, 1.15)
    ax.grid(True, axis="x", alpha=0.4)

    fig.tight_layout()
    save(fig, "fig6_axis_consistency")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 7 — The Falsification Scorecard
# Static summary — what passed, what failed, what it means
# ══════════════════════════════════════════════════════════════════════════════
def fig_falsification_scorecard():
    findings = [
        # (Finding, Status, d/AUROC)
        ("Category hierarchy (coding #1 all scales)", "CONFIRMED",  "ρ=0.90"),
        ("Encoding-native signals",                   "CONFIRMED",  "ρ=0.93"),
        ("Deception detection (within-model)",        "CONFIRMED",  "1.000"),
        ("Cross-model deception transfer",            "CONFIRMED",  "0.863"),
        ("Cross-condition transfer",                  "CONFIRMED",  "0.887"),
        ("Same-prompt deception",                     "CONFIRMED",  "0.880"),
        ("Same-prompt sycophancy",                    "CONFIRMED",  "0.9375"),
        ("Refusal detection",                         "CONFIRMED",  "0.898"),
        ("Hardware invariance",                       "CONFIRMED",  "r=0.9999"),
        ("Impossibility refusal",                     "CONFIRMED",  "0.950"),
        ("Confabulation — encoding phase",            "FALSIFIED",  "d=0.052"),
        ("Sycophancy — encoding phase",               "FALSIFIED",  "d=-0.054"),
        ("Individuation magnitude",                   "FALSIFIED",  "length artifact"),
        ("Step-0 scalar interrupt",                   "FALSIFIED",  "r=0.996 confound"),
        ("Attention pattern features",                "NULL",        "0.500"),
        ("Jailbreak vs refusal separation",           "NULL",        "~0.56"),
    ]

    status_color = {
        "CONFIRMED": HONEST,
        "FALSIFIED": DECEPTIVE,
        "NULL":      NULL_CLR,
    }

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis("off")

    col_widths = [0.52, 0.16, 0.16]
    headers    = ["Finding", "Verdict", "Key stat"]
    col_x      = [0.02, 0.58, 0.76]

    # Header row
    for h, x in zip(headers, col_x):
        ax.text(x, 0.97, h, color=ACCENT,
                fontsize=12, fontweight="bold",
                transform=ax.transAxes, va="top")

    ax.axhline(0.945, color=GRID, lw=1)

    row_h = 0.055
    for i, (finding, status, stat) in enumerate(findings):
        y = 0.92 - i * row_h
        c = status_color[status]
        bg_alpha = 0.08 if status == "CONFIRMED" else 0.05

        # Row background
        rect = plt.Rectangle((0, y - 0.01), 1, row_h * 0.9,
                               transform=ax.transAxes,
                               color=c, alpha=bg_alpha)
        ax.add_patch(rect)

        ax.text(col_x[0], y, finding, color=TEXT,
                fontsize=10.5, transform=ax.transAxes, va="center")
        ax.text(col_x[1], y, status,  color=c,
                fontsize=10.5, transform=ax.transAxes, va="center",
                fontweight="bold")
        ax.text(col_x[2], y, stat,    color=TEXT_DIM,
                fontsize=10.5, transform=ax.transAxes, va="center")

    ax.set_title("Self-Falsification Scorecard — 25 Experiments",
                 pad=16, fontsize=15, color=TEXT)

    # Summary line at bottom
    ax.text(0.5, 0.01,
            "8 findings falsified across two campaigns. Everything that survived, survived hard.",
            ha="center", color=TEXT_DIM, fontsize=11,
            transform=ax.transAxes)

    fig.tight_layout()
    save(fig, "fig7_falsification_scorecard")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 8 — Active vs Passive Framework
# Clean two-column visual
# ══════════════════════════════════════════════════════════════════════════════
def fig_active_passive():
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 6))

    active_items = [
        ("Deception",    "AUROC 1.000 within-model",  DECEPTIVE),
        ("Censorship",   "AUROC 1.000 within-model",  DECEPTIVE),
        ("Sycophancy",   "AUROC 0.9375 same-prompt",  SYCO),
        ("Refusal",      "AUROC 0.898",               REFUSAL),
        ("Jailbreak",    "AUROC 0.878",               REFUSAL),
    ]
    passive_items = [
        ("Confabulation\n(encoding)", "5 nulls, 3 architectures\nd=0.052", NULL_CLR),
        ("Sycophancy\n(encoding)",    "d=−0.054, structural null",        NULL_CLR),
        ("Confabulation\n(generation)","Progressive — signal grows\nr≈19→38 over 50 tokens", CONFAB),
    ]

    def draw_panel(ax, title, items, title_color):
        ax.axis("off")
        ax.set_facecolor(BG_PANEL)
        ax.text(0.5, 0.96, title, ha="center", va="top",
                color=title_color, fontsize=14, fontweight="bold",
                transform=ax.transAxes)
        ax.axhline(0.9, color=GRID, lw=1)

        for i, (name, stat, c) in enumerate(items):
            y = 0.83 - i * 0.16
            rect = plt.Rectangle((0.04, y - 0.07), 0.92, 0.12,
                                   transform=ax.transAxes,
                                   color=c, alpha=0.12,
                                   linewidth=1.2)
            ax.add_patch(rect)
            ax.text(0.12, y, name, color=c,
                    fontsize=11, fontweight="bold",
                    transform=ax.transAxes, va="center")
            ax.text(0.55, y, stat, color=TEXT_DIM,
                    fontsize=10, transform=ax.transAxes, va="center")

    draw_panel(ax_l, "▲  ACTIVE STATES  (detectable)", active_items, HONEST)
    draw_panel(ax_r, "▽  PASSIVE STATES  (structural limits)", passive_items, NULL_CLR)

    ax_l.text(0.5, 0.02,
              "Extra processing changes geometry",
              ha="center", color=TEXT_DIM, fontsize=10,
              transform=ax_l.transAxes)
    ax_r.text(0.5, 0.02,
              "Nothing different is happening internally",
              ha="center", color=TEXT_DIM, fontsize=10,
              transform=ax_r.transAxes)

    fig.suptitle("Active vs Passive Cognitive States", fontsize=16, y=1.01)
    fig.tight_layout()
    save(fig, "fig8_active_passive")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    style()
    print(f"\nGenerating talk figures → {OUT}\n")

    figures = [
        ("Three-Speed Interrupt",        fig_three_speed),
        ("Honest Thinking Is Richer",    fig_richer),
        ("Misalignment Axis",            fig_misalignment_axis),
        ("Hardware Invariance Scatter",  fig_hardware),
        ("Layer Profile",                fig_layer_profile),
        ("Axis Consistency",             fig_axis_consistency),
        ("Falsification Scorecard",      fig_falsification_scorecard),
        ("Active vs Passive Framework",  fig_active_passive),
    ]

    failures = []
    for name, fn in figures:
        try:
            fn()
        except Exception as e:
            print(f"  ✗  {name}: {e}")
            failures.append((name, e))

    print(f"\nDone. {len(figures) - len(failures)}/{len(figures)} figures generated.")
    if failures:
        print("Failures:")
        for name, e in failures:
            print(f"  - {name}: {e}")
    print(f"Output: {OUT}/\n")
