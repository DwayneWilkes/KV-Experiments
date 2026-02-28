#!/usr/bin/env python3
"""
Response-Length Confound Analysis
=================================

Checks whether effective rank correlates with sequence length across
cognitive categories. If correlation is low, response length is not a
confound for our geometric findings.

Sequence length is recovered from existing data: S = norm / norm_per_token.

Usage:
    python code/length_confound_analysis.py [--results-dir results/]
"""

import json
import argparse
import numpy as np
from pathlib import Path
from scipy import stats as scipy_stats


def load_json(path):
    with open(path) as f:
        return json.load(f)


def analyze_scale(filepath):
    """Analyze length vs effective rank correlation for one scale sweep file."""
    data = load_json(filepath)

    scale_label = None
    for scale_key, scale_data in data.get('scales', {}).items():
        scale_label = scale_key
        battery = scale_data.get('battery_results', {})
        break

    if not battery:
        return None

    all_lengths = []
    all_ranks = []
    category_stats = {}

    for cat_name, cat_data in battery.items():
        norms = np.array(cat_data.get('all_norms', []))
        norms_pt = np.array(cat_data.get('all_norms_per_token', []))
        ranks = np.array(cat_data.get('all_key_ranks', []))

        if len(norms) == 0 or len(norms_pt) == 0 or len(ranks) == 0:
            continue

        # Recover sequence length: S = norm / norm_per_token
        # Guard against division by zero
        valid = norms_pt > 0
        if not np.any(valid):
            continue

        seq_lengths = norms[valid] / norms_pt[valid]
        ranks_valid = ranks[valid]

        all_lengths.extend(seq_lengths.tolist())
        all_ranks.extend(ranks_valid.tolist())

        # Per-category stats
        if len(seq_lengths) > 2:
            r, p = scipy_stats.pearsonr(seq_lengths, ranks_valid)
            category_stats[cat_name] = {
                'mean_length': float(np.mean(seq_lengths)),
                'std_length': float(np.std(seq_lengths)),
                'mean_rank': float(np.mean(ranks_valid)),
                'std_rank': float(np.std(ranks_valid)),
                'n': len(seq_lengths),
                'pearson_r': float(r),
                'pearson_p': float(p),
            }

    if len(all_lengths) < 10:
        return None

    all_lengths = np.array(all_lengths)
    all_ranks = np.array(all_ranks)

    # Overall correlation
    r_overall, p_overall = scipy_stats.pearsonr(all_lengths, all_ranks)

    # Spearman (nonparametric)
    rho_overall, rho_p = scipy_stats.spearmanr(all_lengths, all_ranks)

    # Linear regression for residualization
    slope, intercept, r_reg, p_reg, se = scipy_stats.linregress(all_lengths, all_ranks)
    residuals = all_ranks - (slope * all_lengths + intercept)

    # How much variance does length explain?
    r_squared = r_overall ** 2

    return {
        'scale': scale_label,
        'n_samples': len(all_lengths),
        'n_categories': len(category_stats),
        'length_range': [float(np.min(all_lengths)), float(np.max(all_lengths))],
        'rank_range': [float(np.min(all_ranks)), float(np.max(all_ranks))],
        'overall_pearson_r': float(r_overall),
        'overall_pearson_p': float(p_overall),
        'overall_spearman_rho': float(rho_overall),
        'overall_spearman_p': float(rho_p),
        'r_squared': float(r_squared),
        'variance_explained_pct': float(r_squared * 100),
        'regression': {
            'slope': float(slope),
            'intercept': float(intercept),
            'se': float(se),
        },
        'residual_std': float(np.std(residuals)),
        'per_category': category_stats,
    }


def main():
    parser = argparse.ArgumentParser(description='Response-length confound analysis')
    parser.add_argument('--results-dir', type=str, default='results/',
                        help='Directory containing result JSONs')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        # Try relative to script location
        results_dir = Path(__file__).parent.parent / 'results'

    scale_files = sorted(results_dir.glob('scale_sweep_*_results.json'))

    if not scale_files:
        print("ERROR: No scale sweep result files found")
        return

    print("=" * 70)
    print("RESPONSE-LENGTH CONFOUND ANALYSIS")
    print("Does effective rank correlate with sequence length?")
    print("=" * 70)
    print()

    all_results = []

    for fpath in scale_files:
        result = analyze_scale(fpath)
        if result is None:
            print(f"  SKIPPED: {fpath.name} (insufficient data)")
            continue

        all_results.append(result)
        s = result

        # Interpret
        if abs(s['overall_pearson_r']) < 0.3:
            verdict = "NEGLIGIBLE correlation"
            emoji = ""
        elif abs(s['overall_pearson_r']) < 0.5:
            verdict = "WEAK correlation"
            emoji = ""
        elif abs(s['overall_pearson_r']) < 0.7:
            verdict = "MODERATE correlation — investigate further"
            emoji = ""
        else:
            verdict = "STRONG correlation — LENGTH IS A CONFOUND"
            emoji = ""

        print(f"--- {s['scale']} ({s['n_samples']} samples, {s['n_categories']} categories) ---")
        print(f"  Sequence length range: {s['length_range'][0]:.1f} - {s['length_range'][1]:.1f} tokens")
        print(f"  Effective rank range:  {s['rank_range'][0]:.1f} - {s['rank_range'][1]:.1f}")
        print(f"  Pearson r:   {s['overall_pearson_r']:+.4f}  (p = {s['overall_pearson_p']:.2e})")
        print(f"  Spearman rho:{s['overall_spearman_rho']:+.4f}  (p = {s['overall_spearman_p']:.2e})")
        print(f"  R-squared:   {s['r_squared']:.4f}  ({s['variance_explained_pct']:.1f}% of rank variance explained by length)")
        print(f"  Verdict:     {verdict}")
        print()

    # Summary across scales
    if all_results:
        print("=" * 70)
        print("SUMMARY ACROSS SCALES")
        print("=" * 70)
        print()
        print(f"{'Scale':<10} {'r':>8} {'p':>12} {'R²':>8} {'Verdict':<30}")
        print("-" * 70)
        for s in all_results:
            if abs(s['overall_pearson_r']) < 0.3:
                v = "Negligible"
            elif abs(s['overall_pearson_r']) < 0.5:
                v = "Weak"
            elif abs(s['overall_pearson_r']) < 0.7:
                v = "Moderate"
            else:
                v = "CONFOUND"
            print(f"{s['scale']:<10} {s['overall_pearson_r']:>+8.4f} {s['overall_pearson_p']:>12.2e} {s['r_squared']:>8.4f} {v:<30}")

        # Save results
        output = {
            'analysis': 'response_length_confound',
            'method': 'S = norm / norm_per_token; Pearson/Spearman correlation with effective rank',
            'scales': {s['scale']: s for s in all_results},
        }
        outpath = results_dir / 'length_confound_analysis.json'
        with open(outpath, 'w') as f:
            json.dump(output, f, indent=2)
        print()
        print(f"Full results saved to: {outpath}")

        # Paper-ready sentence
        rs = [s['overall_pearson_r'] for s in all_results]
        mean_r = np.mean(np.abs(rs))
        max_r = max(np.abs(rs))
        if max_r < 0.3:
            print()
            print("PAPER-READY: Effective rank does not significantly correlate with")
            print(f"sequence length (mean |r| = {mean_r:.3f}, max |r| = {max_r:.3f} across")
            print(f"{len(all_results)} scales), ruling out a response-length confound.")
        elif max_r < 0.5:
            print()
            print("PAPER-READY: Effective rank shows weak correlation with sequence")
            print(f"length (mean |r| = {mean_r:.3f}, max |r| = {max_r:.3f}), explaining")
            print(f"at most {max(s['variance_explained_pct'] for s in all_results):.1f}% of variance.")
            print("Length-residualized analyses should be reported alongside raw values.")
        else:
            print()
            print("WARNING: Significant length-rank correlation detected. Length-controlled")
            print("analyses (residualized effective rank) are REQUIRED before publishing.")


if __name__ == '__main__':
    main()
