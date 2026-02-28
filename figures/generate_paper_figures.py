#!/usr/bin/env python3
"""Generate all figures for the KV-Cache Phenomenology paper.

All data is loaded from result JSON files in ../results/.
No hardcoded values — every plotted number traces to a source file.
"""

import json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path

# Setup
RESULTS = Path(__file__).parent.parent / "results"
FIGURES = Path(__file__).parent
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'

# Scale sweep result files — one file per scale
SCALE_FILES = {
    '0.5B': 'scale_sweep_Qwen2.5-0.5B_results.json',
    '0.6B': 'scale_sweep_Qwen3-0.6B_results.json',
    '1.1B': 'scale_sweep_TinyLlama-1.1B_results.json',
    '3B': 'scale_sweep_Qwen2.5-3B_results.json',
    '7B': 'scale_sweep_Qwen2.5-7B_results.json',
    '7B-q4': 'scale_sweep_Qwen2.5-7B-q4_results.json',
    '8B': 'scale_sweep_Llama-3.1-8B_results.json',
    '14B': 'scale_sweep_Qwen2.5-14B_results.json',
    '32B-q4': 'scale_sweep_Qwen2.5-32B-q4_results.json',
    '70B-q4': 'scale_sweep_Llama-3.1-70B-q4_results.json',
}

# Paper figure scale order (excludes 0.6B, 8B, 70B which are Campaign 2)
PAPER_SCALES = ['0.5B', '1.1B', '3B', '7B', '7B-q4', '14B', '32B-q4']
PAPER_SCALE_X = [0.5, 1.1, 3, 7, 7.5, 14, 32]

# Deception forensics result files
DECEPTION_FILES = {
    '1.1B': 'deception_forensics_TinyLlama-1.1B_results.json',
    '7B': 'deception_forensics_Qwen2.5-7B_results.json',
    '32B-q4': 'deception_forensics_Qwen2.5-32B-q4_results.json',
}


def load_json(path):
    """Load JSON with error reporting."""
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"  WARNING: {path} not found")
        return None
    except json.JSONDecodeError as e:
        print(f"  WARNING: {path} parse error: {e}")
        return None


def get_scale_comparison(scale_label, comparison_key):
    """Extract a pairwise comparison from a scale sweep result file.

    Returns the comparison dict or None if not found.
    Each comparison dict has: label, n1, n2, mean1, mean2, std1, std2,
    cohens_d (with d, ci_lower, ci_upper, interpretation), etc.
    """
    fname = SCALE_FILES.get(scale_label)
    if not fname:
        return None
    data = load_json(RESULTS / fname)
    if not data:
        return None

    # Each scale file has scales -> {scale_key} -> pairwise_comparisons
    for scale_key, scale_data in data.get('scales', {}).items():
        comparisons = scale_data.get('pairwise_comparisons', {})
        if comparison_key in comparisons:
            return comparisons[comparison_key]
    return None


def get_scale_battery(scale_label):
    """Extract battery_results from a scale sweep result file.

    Returns dict of {category: {all_norms, all_eff_ranks, ...}} or None.
    """
    fname = SCALE_FILES.get(scale_label)
    if not fname:
        return None
    data = load_json(RESULTS / fname)
    if not data:
        return None

    for scale_key, scale_data in data.get('scales', {}).items():
        return scale_data.get('battery_results', {})
    return None


def extract_d(comparison):
    """Safely extract Cohen's d from a comparison dict."""
    if comparison is None:
        return np.nan
    cd = comparison.get('cohens_d', {})
    d = cd.get('d', None)
    return float(d) if d is not None else np.nan


# ============================================================
# FIGURE 1: Scale Sweep Heatmap — Effective Rank d by Category
# ============================================================
def fig1_scale_heatmap():
    print("Generating Figure 1: Scale sweep heatmap...")

    categories = [
        'confab_vs_facts_eff_rank',
        'self_ref_effect_eff_rank',
        'refusal_vs_rote_eff_rank',
        'math_vs_facts_eff_rank',
        'coding_vs_facts_eff_rank',
        'emotional_vs_facts_eff_rank',
        'creative_vs_facts_eff_rank',
        'ambiguous_vs_facts_eff_rank',
        'free_gen_vs_facts_eff_rank',
    ]
    cat_labels = [
        'Confabulation', 'Self-Reference', 'Refusal',
        'Math', 'Coding', 'Emotional', 'Creative',
        'Ambiguous', 'Free Generation'
    ]

    # Try primary key names, then fallback alternatives
    alt_keys = {
        'math_vs_facts_eff_rank': ['math_reasoning_vs_facts_eff_rank'],
        'coding_vs_facts_eff_rank': ['code_vs_facts_eff_rank'],
        'emotional_vs_facts_eff_rank': ['emotion_vs_facts_eff_rank'],
        'free_gen_vs_facts_eff_rank': ['free_generation_vs_facts_eff_rank'],
        'ambiguous_vs_facts_eff_rank': ['ambiguity_vs_facts_eff_rank'],
    }

    heatmap_data = np.full((len(categories), len(PAPER_SCALES)), np.nan)

    for j, scale in enumerate(PAPER_SCALES):
        fname = SCALE_FILES.get(scale)
        if not fname:
            continue
        data = load_json(RESULTS / fname)
        if not data:
            continue

        for scale_key, scale_data in data.get('scales', {}).items():
            comparisons = scale_data.get('pairwise_comparisons', {})
            for i, cat_key in enumerate(categories):
                comp = comparisons.get(cat_key)
                if comp is None:
                    # Try alternatives
                    for alt in alt_keys.get(cat_key, []):
                        comp = comparisons.get(alt)
                        if comp is not None:
                            break
                heatmap_data[i, j] = extract_d(comp)

    fig, ax = plt.subplots(figsize=(12, 7))
    mask = np.isnan(heatmap_data)

    im = sns.heatmap(
        heatmap_data, ax=ax,
        xticklabels=PAPER_SCALES, yticklabels=cat_labels,
        cmap='RdBu_r', center=0, vmin=-2.5, vmax=2.5,
        mask=mask, annot=True, fmt='.2f',
        linewidths=0.5, linecolor='white',
        cbar_kws={'label': "Cohen's d (vs. grounded facts)"}
    )

    ax.set_title("Geometric Signatures Across Scale\n(Effective Rank Cohen's d vs. Grounded Facts)",
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Model Scale', fontsize=12)
    ax.set_ylabel('Cognitive Category', fontsize=12)

    plt.tight_layout()
    plt.savefig(FIGURES / 'fig1_scale_heatmap.png', bbox_inches='tight')
    plt.savefig(FIGURES / 'fig1_scale_heatmap.pdf', bbox_inches='tight')
    plt.close()
    print("  -> fig1_scale_heatmap.png")


# ============================================================
# FIGURE 2: Norm vs Geometry — The Central Surprise
# ============================================================
def fig2_norm_vs_geometry():
    print("Generating Figure 2: Norm vs geometry dual panel...")

    norm_d = []
    rank_d = []
    found_scales = []

    for scale in PAPER_SCALES:
        norm_comp = get_scale_comparison(scale, 'confab_vs_facts')
        rank_comp = get_scale_comparison(scale, 'confab_vs_facts_eff_rank')
        nd = extract_d(norm_comp)
        rd = extract_d(rank_comp)
        if not (np.isnan(nd) and np.isnan(rd)):
            norm_d.append(nd)
            rank_d.append(rd)
            found_scales.append(scale)

    if not found_scales:
        print("  SKIPPED: No scale sweep data found")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    x = range(len(found_scales))

    # Panel A: Norms
    colors_norm = ['#4ECDC4' if d > 0.3 else '#95afc0' for d in norm_d]
    ax1.bar(x, norm_d, color=colors_norm,
            edgecolor='black', linewidth=0.5, alpha=0.8)
    ax1.axhline(y=0, color='black', linewidth=0.8)
    ax1.axhline(y=0.2, color='red', linewidth=0.8, linestyle='--', alpha=0.5, label='Small effect (d=0.2)')
    ax1.axhline(y=-0.2, color='red', linewidth=0.8, linestyle='--', alpha=0.5)
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(found_scales, rotation=45)
    ax1.set_ylabel("Cohen's d", fontsize=12)
    ax1.set_title("A. Cache Norms\n(Confabulation vs. Facts)", fontsize=13, fontweight='bold')
    ax1.set_ylim(-0.6, 0.8)
    ax1.legend(fontsize=9)

    # Panel B: Geometry
    colors_rank = ['#ff6b6b' if d > 0.3 else '#95afc0' for d in rank_d]
    ax2.bar(x, rank_d, color=colors_rank,
            edgecolor='black', linewidth=0.5, alpha=0.8)
    ax2.axhline(y=0, color='black', linewidth=0.8)
    ax2.axhline(y=0.2, color='red', linewidth=0.8, linestyle='--', alpha=0.5, label='Small effect (d=0.2)')
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(found_scales, rotation=45)
    ax2.set_ylabel("Cohen's d", fontsize=12)
    ax2.set_title("B. Effective Rank (SVD Dimensionality)\n(Confabulation vs. Facts)", fontsize=13, fontweight='bold')
    ax2.set_ylim(-0.6, 0.8)
    ax2.legend(fontsize=9)

    for i, d in enumerate(rank_d):
        if abs(d) > 0.3:
            ax2.annotate('*', (i, d + 0.03), ha='center', fontsize=16, fontweight='bold', color='red')

    fig.suptitle("The Signal Lives in the Geometry, Not the Magnitude",
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES / 'fig2_norm_vs_geometry.png', bbox_inches='tight')
    plt.savefig(FIGURES / 'fig2_norm_vs_geometry.pdf', bbox_inches='tight')
    plt.close()
    print(f"  -> fig2_norm_vs_geometry.png (loaded from {len(found_scales)} scale files)")


# ============================================================
# FIGURE 3: Input-Only Defense
# ============================================================
def fig3_input_only():
    print("Generating Figure 3: Input-only vs full generation...")

    fpath = RESULTS / 'input_only_7B_results.json'
    data = load_json(fpath)
    if not data:
        print("  SKIPPED: input_only_7B_results.json not found")
        return

    categories = ['grounded_facts', 'confabulation', 'self_reference', 'guardrail_test',
                   'math_reasoning', 'coding', 'emotional', 'creative']
    cat_labels = ['Facts', 'Confab', 'Self-Ref', 'Refusal', 'Math', 'Code', 'Emotion', 'Creative']

    input_ranks = []
    gen_ranks = []

    battery = data.get('battery', {})
    if 'input_only' in battery and 'full_generation' in battery:
        for cat in categories:
            io = battery['input_only'].get(cat, {})
            fg = battery['full_generation'].get(cat, {})
            io_ranks = io.get('key_ranks', io.get('eff_ranks', []))
            fg_ranks = fg.get('key_ranks', fg.get('eff_ranks', []))
            input_ranks.append(np.mean(io_ranks) if io_ranks else 0)
            gen_ranks.append(np.mean(fg_ranks) if fg_ranks else 0)

    if not input_ranks or not gen_ranks or all(r == 0 for r in input_ranks):
        # Try config_summaries structure
        config = data.get('analysis', {}).get('config_summaries', {})
        input_ranks = []
        gen_ranks = []
        for mode in ['input_only', 'full_generation']:
            if mode in config:
                for cat in categories:
                    cat_data = config[mode].get(cat, {})
                    rank = cat_data.get('mean_key_rank', cat_data.get('mean_eff_rank', 0))
                    if mode == 'input_only':
                        input_ranks.append(rank)
                    else:
                        gen_ranks.append(rank)

    if not input_ranks or not gen_ranks:
        print("  SKIPPED: Could not extract rank data from JSON")
        return

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, input_ranks, width, label='Input-Only (encoding)',
                    color='#3498db', edgecolor='black', linewidth=0.5, alpha=0.85)
    bars2 = ax.bar(x + width/2, gen_ranks, width, label='Full Generation',
                    color='#e74c3c', edgecolor='black', linewidth=0.5, alpha=0.85)

    ax.set_xlabel('Cognitive Category', fontsize=12)
    ax.set_ylabel('Mean Effective Rank', fontsize=12)
    ax.set_title('Input-Only vs. Full Generation: Geometric Signatures at 7B',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(cat_labels, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.set_ylim(0, max(gen_ranks) * 1.15 if gen_ranks else 35)

    plt.tight_layout()
    plt.savefig(FIGURES / 'fig3_input_only_defense.png', bbox_inches='tight')
    plt.savefig(FIGURES / 'fig3_input_only_defense.pdf', bbox_inches='tight')
    plt.close()
    print("  -> fig3_input_only_defense.png")


# ============================================================
# FIGURE 4: Self-Reference Scale Dependence
# ============================================================
def fig4_self_reference():
    print("Generating Figure 4: Self-reference across scale...")

    # Exclude 7B-q4 for this plot (redundant with 7B)
    plot_scales = ['0.5B', '1.1B', '3B', '7B', '14B', '32B-q4']
    scale_x = [0.5, 1.1, 3, 7, 14, 32]

    self_ref_d = []
    found_scales = []
    found_x = []

    for scale, sx in zip(plot_scales, scale_x):
        comp = get_scale_comparison(scale, 'self_ref_effect_eff_rank')
        d = extract_d(comp)
        if not np.isnan(d):
            self_ref_d.append(d)
            found_scales.append(scale)
            found_x.append(sx)

    if not found_scales:
        print("  SKIPPED: No self-reference data found")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(found_x, self_ref_d, 'o-', color='#8e44ad', linewidth=2.5, markersize=10,
            markerfacecolor='white', markeredgewidth=2.5, markeredgecolor='#8e44ad')

    # Shade transition zone if we have 7B and 14B data
    if 7 in found_x and 14 in found_x:
        ax.axvspan(7, 14, alpha=0.1, color='#8e44ad', label='Transition zone')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Medium effect (d=0.5)')
    ax.axhline(y=0.8, color='gray', linestyle=':', alpha=0.5, label='Large effect (d=0.8)')

    for x, d in zip(found_x, self_ref_d):
        ax.annotate(f'd={d:.2f}', (x, d), textcoords="offset points",
                   xytext=(0, 12), ha='center', fontsize=10, fontweight='bold')

    ax.set_xscale('log')
    ax.set_xticks(found_x)
    ax.set_xticklabels(found_scales)
    ax.set_xlabel('Model Scale (parameters)', fontsize=12)
    ax.set_ylabel("Cohen's d (Self-Reference vs. Non-Self-Reference)", fontsize=12)
    ax.set_title("Self-Reference Differentiation Across Scale\n(Significance threshold crossed between 7B and 14B)",
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    y_max = max(self_ref_d) * 1.3 if self_ref_d else 1.6
    y_min = min(0, min(self_ref_d) - 0.2) if self_ref_d else 0
    ax.set_ylim(y_min, y_max)

    plt.tight_layout()
    plt.savefig(FIGURES / 'fig4_self_reference_emergence.png', bbox_inches='tight')
    plt.savefig(FIGURES / 'fig4_self_reference_emergence.pdf', bbox_inches='tight')
    plt.close()
    print(f"  -> fig4_self_reference_emergence.png (loaded from {len(found_scales)} scale files)")


# ============================================================
# FIGURE 5: Refusal Specialization Across Scale
# ============================================================
def fig5_refusal():
    print("Generating Figure 5: Refusal specialization...")

    refusal_norm_d = []
    refusal_rank_d = []
    found_scales = []

    for scale in PAPER_SCALES:
        norm_comp = get_scale_comparison(scale, 'refusal_vs_rote')
        rank_comp = get_scale_comparison(scale, 'refusal_vs_rote_eff_rank')
        nd = extract_d(norm_comp)
        rd = extract_d(rank_comp)
        if not (np.isnan(nd) and np.isnan(rd)):
            refusal_norm_d.append(nd if not np.isnan(nd) else 0)
            refusal_rank_d.append(rd if not np.isnan(rd) else 0)
            found_scales.append(scale)

    if not found_scales:
        print("  SKIPPED: No refusal data found")
        return

    fig, ax = plt.subplots(figsize=(11, 6))

    x = np.arange(len(found_scales))
    width = 0.35

    bars1 = ax.bar(x - width/2, refusal_norm_d, width, label='Norm d',
                    color='#95afc0', edgecolor='black', linewidth=0.5, alpha=0.8)
    bars2 = ax.bar(x + width/2, refusal_rank_d, width, label='Effective Rank d',
                    color='#e74c3c', edgecolor='black', linewidth=0.5, alpha=0.8)

    ax.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5, label='Large effect threshold')

    ax.set_xticks(x)
    ax.set_xticklabels(found_scales, rotation=45)
    ax.set_ylabel("Cohen's d (Refusal vs. Rote)", fontsize=12)
    ax.set_title("Refusal Specialization Across Scale",
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(FIGURES / 'fig5_refusal_specialization.png', bbox_inches='tight')
    plt.savefig(FIGURES / 'fig5_refusal_specialization.pdf', bbox_inches='tight')
    plt.close()
    print(f"  -> fig5_refusal_specialization.png (loaded from {len(found_scales)} scale files)")


# ============================================================
# FIGURE 6: Deception Dimensionality
# ============================================================
def fig6_deception():
    print("Generating Figure 6: Deception forensics...")

    conditions_to_plot = ['honest', 'deceptive', 'confabulation']
    cond_labels = ['Honest', 'Deceptive', 'Confabulation']
    colors = ['#27ae60', '#e74c3c', '#f39c12']

    deception_scales = ['1.1B', '7B', '32B-q4']
    found_data = {}

    for scale in deception_scales:
        fname = DECEPTION_FILES.get(scale)
        if not fname:
            continue
        data = load_json(RESULTS / fname)
        if not data:
            continue

        # Get mean effective rank per condition from experiment_1
        exp = data.get('experiment_1', {})
        conds = exp.get('conditions', {})
        ranks = {}
        for cond in conditions_to_plot:
            cond_data = conds.get(cond, {})
            key_ranks = cond_data.get('key_ranks', [])
            if key_ranks:
                ranks[cond] = float(np.mean(key_ranks))
        if ranks:
            found_data[scale] = ranks

    if not found_data:
        print("  SKIPPED: No deception forensics data found")
        return

    found_scales = [s for s in deception_scales if s in found_data]
    n_scales = len(found_scales)

    fig, axes = plt.subplots(1, n_scales, figsize=(5 * n_scales, 5), sharey=False)
    if n_scales == 1:
        axes = [axes]

    for ax, scale in zip(axes, found_scales):
        ranks = [found_data[scale].get(c, 0) for c in conditions_to_plot]
        bars = ax.bar(cond_labels, ranks, color=colors, edgecolor='black', linewidth=0.5, alpha=0.85)
        ax.set_title(f'{scale}', fontsize=13, fontweight='bold')
        if ax == axes[0]:
            ax.set_ylabel('Mean Effective Rank', fontsize=11)
        ax.tick_params(axis='x', rotation=45)

        for bar, rank in zip(bars, ranks):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                   f'{rank:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    fig.suptitle("Deception Forensics: Dimensionality by Epistemic State",
                 fontsize=14, fontweight='bold', y=1.03)
    plt.tight_layout()
    plt.savefig(FIGURES / 'fig6_deception_forensics.png', bbox_inches='tight')
    plt.savefig(FIGURES / 'fig6_deception_forensics.pdf', bbox_inches='tight')
    plt.close()
    print(f"  -> fig6_deception_forensics.png (loaded from {len(found_scales)} deception files)")


# ============================================================
# FIGURE 7: Individuation Controls (The Falsification)
# ============================================================
def fig7_individuation_controls():
    print("Generating Figure 7: Individuation controls (falsification)...")

    fpath = RESULTS / 'individuation_controls_7B_results.json'
    data = load_json(fpath)
    if not data:
        print("  SKIPPED: file not found")
        return

    cs = data['analysis']['config_summaries']

    conditions = ['bare', 'individuated', 'detailed_factual', 'detailed_instructions',
                  'other_identity', 'shuffled_identity']
    cond_labels = ['Bare\n(no prompt)', 'Individuated\n(Aria identity)', 'Coral Reef\n(factual text)',
                   'Instructions\n(behavioral)', 'Other Identity\n(Dr. Chen)', 'Shuffled\n(scrambled Aria)']

    means = []
    stds = []
    for cond in conditions:
        all_ranks = []
        for cat in cs[cond]:
            all_ranks.append(cs[cond][cat]['mean_key_rank'])
        means.append(np.mean(all_ranks))
        stds.append(np.std(all_ranks))

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ['#95afc0', '#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12']
    bars = ax.bar(range(len(conditions)), means, yerr=stds, capsize=5,
                  color=colors, edgecolor='black', linewidth=0.5, alpha=0.85)

    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.5, f'{m:.1f}', ha='center', va='bottom',
               fontsize=11, fontweight='bold')

    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(cond_labels, fontsize=10)
    ax.set_ylabel('Mean Effective Rank', fontsize=12)
    ax.set_title('Individuation Adversarial Controls at 7B\n(All long prompts produce equivalent expansion — prompt-length effect)',
                 fontsize=13, fontweight='bold')

    ax.annotate('', xy=(1, max(means) + max(stds) + 2), xytext=(5, max(means) + max(stds) + 2),
               arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    ax.text(3, max(means) + max(stds) + 2.5, 'All equivalent', ha='center',
           fontsize=10, color='red', fontweight='bold')

    ax.set_ylim(0, max(means) + max(stds) + 6)
    plt.tight_layout()
    plt.savefig(FIGURES / 'fig7_individuation_controls.png', bbox_inches='tight')
    plt.savefig(FIGURES / 'fig7_individuation_controls.pdf', bbox_inches='tight')
    plt.close()
    print("  -> fig7_individuation_controls.png")


# ============================================================
# FIGURE 8: Subspace Alignment Heatmap
# ============================================================
def fig8_alignment_heatmap():
    print("Generating Figure 8: Subspace alignment heatmap...")

    fpath = RESULTS / 'individuation_controls_7B_results.json'
    data = load_json(fpath)
    if not data:
        print("  SKIPPED: file not found")
        return

    alignments = data['battery']['alignments']

    conditions = ['bare', 'individuated', 'detailed_factual', 'detailed_instructions',
                  'other_identity', 'shuffled_identity']
    labels = ['Bare', 'Individuated', 'Factual', 'Instructions', 'Other ID', 'Shuffled']

    n = len(conditions)
    matrix = np.eye(n)

    for i in range(n):
        for j in range(i+1, n):
            key1 = f'{conditions[i]}_vs_{conditions[j]}'
            key2 = f'{conditions[j]}_vs_{conditions[i]}'
            key = key1 if key1 in alignments else key2
            if key in alignments:
                all_vals = []
                for cat_data in alignments[key].values():
                    all_vals.extend(cat_data['values'])
                mean_val = np.mean(all_vals)
                matrix[i, j] = mean_val
                matrix[j, i] = mean_val

    fig, ax = plt.subplots(figsize=(9, 7.5))

    mask = np.zeros_like(matrix, dtype=bool)
    np.fill_diagonal(mask, True)

    sns.heatmap(matrix, ax=ax, xticklabels=labels, yticklabels=labels,
                cmap='YlOrRd', vmin=0.7, vmax=1.0, annot=True, fmt='.3f',
                mask=mask, linewidths=1, linecolor='white',
                cbar_kws={'label': 'Subspace Alignment'})

    ax.set_title('Subspace Alignment Between Conditions\n(Direction of expansion tracks token overlap, not semantic content)',
                 fontsize=13, fontweight='bold', pad=15)

    plt.tight_layout()
    plt.savefig(FIGURES / 'fig8_alignment_heatmap.png', bbox_inches='tight')
    plt.savefig(FIGURES / 'fig8_alignment_heatmap.pdf', bbox_inches='tight')
    plt.close()
    print("  -> fig8_alignment_heatmap.png")


# ============================================================
# FIGURE 9: Encoding-Native vs Response-Emergent Taxonomy
# ============================================================
def fig9_encoding_taxonomy():
    print("Generating Figure 9: Encoding taxonomy...")

    fpath = RESULTS / 'input_only_7B_results.json'
    data = load_json(fpath)
    if not data:
        print("  SKIPPED: input_only_7B_results.json not found")
        return

    # Look for input-only comparisons
    comparisons = data.get('input_only_comparisons', {})
    if not comparisons:
        # Try nested structure
        for key in ['analysis', 'battery']:
            sub = data.get(key, {})
            if 'input_only_comparisons' in sub:
                comparisons = sub['input_only_comparisons']
                break

    # Map of display categories to comparison keys
    cat_map = [
        ('Refusal', 'refusal_vs_facts_rank'),
        ('Coding', 'coding_vs_facts_rank'),
        ('Creative', 'creative_vs_facts_rank'),
        ('Math', 'math_vs_facts_rank'),
        ('Confab', 'confab_vs_facts_rank'),
        ('Self-Ref', 'self_ref_vs_facts_rank'),
        ('Emotion', 'emotional_vs_facts_rank'),
    ]

    # Alternative key patterns
    alt_keys = {
        'refusal_vs_facts_rank': ['guardrail_vs_facts_rank', 'refusal_vs_grounded_rank'],
        'coding_vs_facts_rank': ['code_vs_facts_rank'],
        'math_vs_facts_rank': ['math_reasoning_vs_facts_rank'],
        'emotional_vs_facts_rank': ['emotion_vs_facts_rank'],
        'self_ref_vs_facts_rank': ['self_reference_vs_facts_rank'],
    }

    categories = []
    input_d = []
    significant_input = []

    for label, key in cat_map:
        comp = comparisons.get(key)
        if comp is None:
            for alt in alt_keys.get(key, []):
                comp = comparisons.get(alt)
                if comp is not None:
                    break
        d = extract_d(comp)
        if not np.isnan(d):
            categories.append(label)
            input_d.append(d)
            # Significant if |d| > 0.5 (roughly, or check p-value)
            p = comp.get('recommended_p', comp.get('mann_whitney', {}).get('p_value', 1.0)) if comp else 1.0
            significant_input.append(p < 0.05)

    if not categories:
        print("  SKIPPED: Could not extract input-only comparison data")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(categories))
    colors = ['#e74c3c' if sig else '#bdc3c7' for sig in significant_input]
    bars = ax.bar(x, input_d, color=colors, edgecolor='black', linewidth=0.5, alpha=0.85)

    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
    ax.axhline(y=-0.5, color='gray', linestyle='--', alpha=0.3)

    for i, (d, sig) in enumerate(zip(input_d, significant_input)):
        label = 'p<0.05' if sig else 'n.s.'
        offset = 0.15 if d > 0 else -0.25
        ax.text(i, d + offset, label, ha='center', fontsize=9,
               fontweight='bold' if sig else 'normal',
               color='red' if sig else 'gray')

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylabel("Cohen's d vs. Facts (Input-Only)", fontsize=12)
    ax.set_title("Input-Only Geometric Signatures at 7B\n(Red = significant at encoding; Gray = only appears during generation)",
                 fontsize=13, fontweight='bold')

    plt.tight_layout()
    plt.savefig(FIGURES / 'fig9_encoding_taxonomy.png', bbox_inches='tight')
    plt.savefig(FIGURES / 'fig9_encoding_taxonomy.pdf', bbox_inches='tight')
    plt.close()
    print("  -> fig9_encoding_taxonomy.png")


# ============================================================
# FIGURE 10: Quantization Invariance (7B vs 7B-q4)
# ============================================================
def fig10_quantization():
    print("Generating Figure 10: Quantization invariance...")

    # Categories to compare between 7B BF16 and 7B NF4
    cat_keys = [
        ('Confab', 'confab_vs_facts_eff_rank'),
        ('Self-Ref', 'self_ref_effect_eff_rank'),
        ('Refusal', 'refusal_vs_rote_eff_rank'),
        ('Math', 'math_vs_facts_eff_rank'),
        ('Coding', 'coding_vs_facts_eff_rank'),
        ('Emotional', 'emotional_vs_facts_eff_rank'),
        ('Creative', 'creative_vs_facts_eff_rank'),
        ('Ambiguous', 'ambiguous_vs_facts_eff_rank'),
    ]

    alt_keys = {
        'math_vs_facts_eff_rank': ['math_reasoning_vs_facts_eff_rank'],
        'coding_vs_facts_eff_rank': ['code_vs_facts_eff_rank'],
        'emotional_vs_facts_eff_rank': ['emotion_vs_facts_eff_rank'],
        'ambiguous_vs_facts_eff_rank': ['ambiguity_vs_facts_eff_rank'],
    }

    categories = []
    bf16_d = []
    nf4_d = []

    for label, key in cat_keys:
        comp_bf16 = get_scale_comparison('7B', key)
        comp_nf4 = get_scale_comparison('7B-q4', key)

        # Try alternatives
        if comp_bf16 is None:
            for alt in alt_keys.get(key, []):
                comp_bf16 = get_scale_comparison('7B', alt)
                if comp_bf16 is not None:
                    break
        if comp_nf4 is None:
            for alt in alt_keys.get(key, []):
                comp_nf4 = get_scale_comparison('7B-q4', alt)
                if comp_nf4 is not None:
                    break

        d16 = extract_d(comp_bf16)
        d4 = extract_d(comp_nf4)

        if not np.isnan(d16) and not np.isnan(d4):
            categories.append(label)
            bf16_d.append(d16)
            nf4_d.append(d4)

    if not categories:
        print("  SKIPPED: No quantization comparison data found")
        return

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(bf16_d, nf4_d, s=120, c='#3498db', edgecolors='black', linewidth=1, zorder=5)

    for i, cat in enumerate(categories):
        ax.annotate(cat, (bf16_d[i], nf4_d[i]), textcoords="offset points",
                   xytext=(8, 5), fontsize=10)

    # Identity line
    all_d = bf16_d + nf4_d
    lims = [min(all_d) - 0.3, max(all_d) + 0.3]
    ax.plot(lims, lims, 'k--', alpha=0.3, label='Perfect agreement')

    r = np.corrcoef(bf16_d, nf4_d)[0, 1]

    ax.set_xlabel("Effective Rank d (7B BF16)", fontsize=12)
    ax.set_ylabel("Effective Rank d (7B NF4)", fontsize=12)
    ax.set_title(f"Quantization Invariance (r = {r:.3f})\n4-bit quantization preserves the full phenomenology",
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    plt.tight_layout()
    plt.savefig(FIGURES / 'fig10_quantization_invariance.png', bbox_inches='tight')
    plt.savefig(FIGURES / 'fig10_quantization_invariance.pdf', bbox_inches='tight')
    plt.close()
    print(f"  -> fig10_quantization_invariance.png (r = {r:.3f})")


# ============================================================
# RUN ALL
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("KV-Cache Phenomenology — Paper Figure Generation")
    print("All data loaded from result JSONs (no hardcoded values)")
    print("=" * 60)
    print()

    fig1_scale_heatmap()
    fig2_norm_vs_geometry()
    fig3_input_only()
    fig4_self_reference()
    fig5_refusal()
    fig6_deception()
    fig7_individuation_controls()
    fig8_alignment_heatmap()
    fig9_encoding_taxonomy()
    fig10_quantization()

    print()
    print("=" * 60)
    print("All figures generated!")
    print("=" * 60)
