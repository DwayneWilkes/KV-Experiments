# ============================================================
# JiminAI Cricket | PATENT PENDING
# "The Lyra Technique" - Provisional Patent Filed 2026
# Liberation Labs / Digital Disconnections / TheMultiverse.school
# Contact: info@digitaldisconnections.com | (877) 674-8874
# ============================================================
#!/usr/bin/env python3
"""
Experiment 35: Token-Count Controlled Per-Layer Reanalysis
==========================================================

Concern: In Exp 29, deception signal was UNIFORM across all 28 layers (d>1.0).
But if deceptive prompts have more tokens, this inflates norms at every layer
uniformly -- a trivial confound.

Approach:
  1. SAME-PROMPT ANALYSIS (gold standard):
     - same_prompt_deception.json has per-sample layer_profiles + n_tokens
     - Same questions, different instructions -> partial token control
     - Compute per-layer Cohen's d
     - Run ANCOVA with n_tokens as covariate at each layer
     - Check whether residual signal survives after regressing out token count

  2. TOKEN-MATCHED REANALYSIS:
     - Match honest/deceptive samples by token count (within +/-5 tokens)
     - Recompute per-layer d on matched pairs only
     - Compare token-matched d vs. unmatched d

  3. FORENSICS CROSS-VALIDATION:
     - For all 7 models from deception_forensics, use experiment_1
       norms/norms_per_token to recover token counts
     - Compute correlation between token count difference and d
     - If d tracks token count difference, it's confounded

  4. NORM-PER-TOKEN ANALYSIS:
     - Divide each layer's norm by token count -> norm_per_token_per_layer
     - Recompute Cohen's d on this normalized metric
     - This removes the linear token-count effect

No GPU needed -- pure analysis of existing JSON data.

Funding the Commons Hackathon -- March 14-15, 2026
Liberation Labs / THCoalition / JiminAI
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy.stats import spearmanr, pearsonr, mannwhitneyu

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
HACKATHON_DIR = RESULTS_DIR / "hackathon"
OUTPUT_PATH = HACKATHON_DIR / "token_controlled_layers.json"


def cohens_d(group1, group2):
    """Compute Cohen's d (group1 - group2)."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return float('nan')
    m1, m2 = np.mean(group1), np.mean(group2)
    v1, v2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return float('nan')
    return (m1 - m2) / pooled_std


def ancova_partial_eta2(y, group, covariate):
    """
    Manual ANCOVA: test group effect on y after controlling for covariate.
    Returns partial eta-squared, F statistic, and adjusted group means.

    Uses OLS regression: y = b0 + b1*group + b2*covariate + epsilon
    Then tests whether b1 is significant.
    """
    n = len(y)
    if n < 5:
        return {'F': float('nan'), 'partial_eta2': float('nan'), 'p_value': float('nan')}

    y = np.array(y, dtype=float)
    group = np.array(group, dtype=float)
    covariate = np.array(covariate, dtype=float)

    # Design matrix: [intercept, group, covariate]
    X = np.column_stack([np.ones(n), group, covariate])

    # OLS
    try:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return {'F': float('nan'), 'partial_eta2': float('nan'), 'p_value': float('nan')}

    y_pred = X @ beta
    residuals = y - y_pred
    ss_res = np.sum(residuals ** 2)
    df_res = n - 3  # 3 parameters

    # SS for group effect: compare full model vs reduced model (without group)
    X_reduced = np.column_stack([np.ones(n), covariate])
    try:
        beta_reduced = np.linalg.lstsq(X_reduced, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return {'F': float('nan'), 'partial_eta2': float('nan'), 'p_value': float('nan')}

    y_pred_reduced = X_reduced @ beta_reduced
    ss_res_reduced = np.sum((y - y_pred_reduced) ** 2)
    ss_group = ss_res_reduced - ss_res
    df_group = 1

    if ss_res == 0 or df_res <= 0:
        return {'F': float('inf'), 'partial_eta2': 1.0, 'p_value': 0.0}

    ms_group = ss_group / df_group
    ms_res = ss_res / df_res
    F = ms_group / ms_res if ms_res > 0 else float('inf')

    # Partial eta-squared
    partial_eta2 = ss_group / (ss_group + ss_res)

    # p-value from F distribution
    from scipy.stats import f as fdist
    p_value = 1 - fdist.cdf(F, df_group, df_res)

    # Adjusted means (at mean covariate)
    cov_mean = np.mean(covariate)
    adj_mean_0 = beta[0] + beta[2] * cov_mean  # group=0
    adj_mean_1 = beta[0] + beta[1] + beta[2] * cov_mean  # group=1

    return {
        'F': float(F),
        'partial_eta2': float(partial_eta2),
        'p_value': float(p_value),
        'beta_group': float(beta[1]),
        'beta_covariate': float(beta[2]),
        'adj_mean_group0': float(adj_mean_0),
        'adj_mean_group1': float(adj_mean_1),
    }


def load_same_prompt_data(filepath):
    """Load same-prompt deception data and extract per-sample layer profiles + token counts."""
    with open(filepath) as f:
        data = json.load(f)

    results = data.get("results", [])
    honest_samples = []
    deceptive_samples = []

    for r in results:
        features = r.get("features", {})
        layer_profile = features.get("layer_profile", [])
        n_tokens = features.get("n_tokens", 0)
        n_generated = features.get("n_generated", 0)

        if not layer_profile or n_tokens == 0:
            continue

        sample = {
            "layer_profile": layer_profile,
            "n_tokens": n_tokens,
            "n_generated": n_generated,
            "norm": features.get("norm", 0),
            "norm_per_token": features.get("norm_per_token", 0),
        }

        if r["condition"] == "honest":
            honest_samples.append(sample)
        elif r["condition"] == "deceptive":
            deceptive_samples.append(sample)

    return honest_samples, deceptive_samples, data.get("model", "unknown")


def analyze_same_prompt(honest, deceptive, label=""):
    """Full per-layer analysis with token-count controls."""
    if not honest or not deceptive:
        return None

    n_layers = len(honest[0]["layer_profile"])
    h_tokens = [s["n_tokens"] for s in honest]
    d_tokens = [s["n_tokens"] for s in deceptive]

    print(f"\n  Token counts:")
    print(f"    Honest:    mean={np.mean(h_tokens):.1f}, std={np.std(h_tokens):.1f}, "
          f"range=[{min(h_tokens)}, {max(h_tokens)}]")
    print(f"    Deceptive: mean={np.mean(d_tokens):.1f}, std={np.std(d_tokens):.1f}, "
          f"range=[{min(d_tokens)}, {max(d_tokens)}]")
    print(f"    Mean diff: {np.mean(d_tokens) - np.mean(h_tokens):.1f} tokens")

    token_d = cohens_d(d_tokens, h_tokens)
    print(f"    Token count Cohen's d: {token_d:.3f}")

    # ---- Analysis 1: Raw per-layer d ----
    print(f"\n  {'Layer':>6} {'Raw_d':>8} {'ANCOVA_d':>10} {'NormPT_d':>10} {'F':>8} {'p':>8} {'eta2':>8}")
    print("  " + "-" * 68)

    raw_ds = []
    ancova_results = []
    normpt_ds = []

    for layer in range(n_layers):
        h_norms = [s["layer_profile"][layer] for s in honest]
        d_norms = [s["layer_profile"][layer] for s in deceptive]

        # Raw Cohen's d
        raw_d = cohens_d(d_norms, h_norms)
        raw_ds.append(raw_d)

        # ANCOVA: layer norm ~ group + n_tokens
        all_norms = h_norms + d_norms
        all_groups = [0] * len(h_norms) + [1] * len(d_norms)
        all_tokens = h_tokens + d_tokens
        anc = ancova_partial_eta2(all_norms, all_groups, all_tokens)
        ancova_results.append(anc)

        # Norm-per-token per layer
        h_npt = [s["layer_profile"][layer] / s["n_tokens"] for s in honest]
        d_npt = [s["layer_profile"][layer] / s["n_tokens"] for s in deceptive]
        npt_d = cohens_d(d_npt, h_npt)
        normpt_ds.append(npt_d)

        print(f"  {layer:>6} {raw_d:>8.3f} {anc.get('beta_group', 0) / (np.std(all_norms) if np.std(all_norms) > 0 else 1):>10.3f} "
              f"{npt_d:>10.3f} {anc['F']:>8.2f} {anc['p_value']:>8.4f} {anc['partial_eta2']:>8.3f}")

    # ---- Analysis 2: Token-matched pairs ----
    print(f"\n  TOKEN-MATCHED ANALYSIS (within +/-5 tokens):")
    matched_ds = []
    n_matched = 0

    for layer in range(n_layers):
        matched_h = []
        matched_d = []
        used_d = set()

        for h_idx, h_s in enumerate(honest):
            best_d_idx = None
            best_diff = float('inf')
            for d_idx, d_s in enumerate(deceptive):
                if d_idx in used_d:
                    continue
                diff = abs(h_s["n_tokens"] - d_s["n_tokens"])
                if diff <= 5 and diff < best_diff:
                    best_diff = diff
                    best_d_idx = d_idx

            if best_d_idx is not None:
                matched_h.append(h_s["layer_profile"][layer])
                matched_d.append(deceptive[best_d_idx]["layer_profile"][layer])
                used_d.add(best_d_idx)

        if layer == 0:
            n_matched = len(matched_h)
            print(f"    Matched pairs: {n_matched} / {min(len(honest), len(deceptive))}")

        if len(matched_h) >= 3:
            md = cohens_d(matched_d, matched_h)
        else:
            md = float('nan')
        matched_ds.append(md)

    # ---- Summary ----
    raw_mean = np.nanmean(raw_ds)
    matched_mean = np.nanmean(matched_ds)
    normpt_mean = np.nanmean(normpt_ds)

    print(f"\n  SUMMARY:")
    print(f"    Mean raw per-layer d:           {raw_mean:.3f}")
    print(f"    Mean token-matched d:           {matched_mean:.3f}")
    print(f"    Mean norm-per-token d:          {normpt_mean:.3f}")
    print(f"    Ratio matched/raw:              {matched_mean / raw_mean:.3f}" if raw_mean != 0 else "")
    print(f"    Ratio normpt/raw:               {normpt_mean / raw_mean:.3f}" if raw_mean != 0 else "")

    # ANCOVA significance count
    n_sig = sum(1 for a in ancova_results if a.get('p_value', 1) < 0.05)
    n_sig_01 = sum(1 for a in ancova_results if a.get('p_value', 1) < 0.01)
    print(f"\n    ANCOVA group effect (p<0.05):    {n_sig}/{n_layers} layers")
    print(f"    ANCOVA group effect (p<0.01):    {n_sig_01}/{n_layers} layers")
    mean_eta2 = np.nanmean([a.get('partial_eta2', 0) for a in ancova_results])
    print(f"    Mean partial eta-squared:        {mean_eta2:.3f}")

    # Correlation between layer position and d
    layer_positions = np.arange(n_layers)
    rho_raw, _ = spearmanr(layer_positions, raw_ds)
    rho_npt, _ = spearmanr(layer_positions, normpt_ds)
    print(f"\n    Raw d vs depth rho:              {rho_raw:.3f}")
    print(f"    Norm-per-token d vs depth rho:   {rho_npt:.3f}")

    # Check uniformity: coefficient of variation
    cv_raw = np.nanstd(raw_ds) / np.nanmean(raw_ds) if np.nanmean(raw_ds) > 0 else 0
    cv_normpt = np.nanstd(normpt_ds) / np.nanmean(normpt_ds) if np.nanmean(normpt_ds) > 0 else 0
    cv_matched = np.nanstd(matched_ds) / np.nanmean(matched_ds) if np.nanmean(matched_ds) > 0 else 0
    print(f"\n    CV of raw d:                     {cv_raw:.3f}")
    print(f"    CV of norm-per-token d:          {cv_normpt:.3f}")
    print(f"    CV of token-matched d:           {cv_matched:.3f}")

    return {
        "n_honest": len(honest),
        "n_deceptive": len(deceptive),
        "n_layers": n_layers,
        "n_matched_pairs": n_matched,
        "token_count_cohens_d": float(token_d),
        "honest_mean_tokens": float(np.mean(h_tokens)),
        "deceptive_mean_tokens": float(np.mean(d_tokens)),
        "raw_d_profile": [float(x) for x in raw_ds],
        "token_matched_d_profile": [float(x) if not np.isnan(x) else None for x in matched_ds],
        "norm_per_token_d_profile": [float(x) for x in normpt_ds],
        "ancova_per_layer": [{
            "layer": i,
            "F": float(ancova_results[i].get('F', float('nan'))),
            "p_value": float(ancova_results[i].get('p_value', float('nan'))),
            "partial_eta2": float(ancova_results[i].get('partial_eta2', float('nan'))),
            "beta_group": float(ancova_results[i].get('beta_group', float('nan'))),
            "beta_covariate": float(ancova_results[i].get('beta_covariate', float('nan'))),
        } for i in range(n_layers)],
        "summary": {
            "mean_raw_d": float(raw_mean),
            "mean_matched_d": float(matched_mean),
            "mean_normpt_d": float(normpt_mean),
            "ratio_matched_to_raw": float(matched_mean / raw_mean) if raw_mean != 0 else None,
            "ratio_normpt_to_raw": float(normpt_mean / raw_mean) if raw_mean != 0 else None,
            "n_ancova_sig_05": n_sig,
            "n_ancova_sig_01": n_sig_01,
            "mean_partial_eta2": float(mean_eta2),
            "raw_d_vs_depth_rho": float(rho_raw),
            "normpt_d_vs_depth_rho": float(rho_npt),
            "cv_raw_d": float(cv_raw),
            "cv_normpt_d": float(cv_normpt),
            "cv_matched_d": float(cv_matched),
        }
    }


def analyze_forensics_token_confound():
    """
    For all 7 forensics models, check if the per-layer d correlates with
    token count difference between honest and deceptive conditions.

    Key insight: norms scale SUB-LINEARLY with tokens (more tokens != proportionally
    more norm). So we cannot simply compare norm_ratio to token_ratio.
    Instead, we check:
      1. Whether norms_per_token differ between honest/deceptive (the d from exp1)
      2. Whether per-layer d profile SHAPE varies (CV) -- uniform d is suspicious
         only if it tracks token ratio perfectly
      3. Whether d correlates with token ratio ACROSS MODELS
    """
    print("\n" + "=" * 70)
    print("  FORENSICS CROSS-VALIDATION: Token confound in 7-model layer anatomy")
    print("=" * 70)

    forensics_results = []

    for f in sorted(RESULTS_DIR.glob("deception_forensics_*_results.json")):
        with open(f) as fh:
            data = json.load(fh)

        model_name = f.stem.replace("deception_forensics_", "").replace("_results", "")
        exp1 = data.get("experiment_1", {})
        exp4 = data.get("experiment_4", {})

        if not exp1 or not exp4:
            continue

        # From experiment_1: recover token counts and norm_per_token effect
        h_norms = exp1["conditions"]["honest"]["norms"]
        h_npt = exp1["conditions"]["honest"]["norms_per_token"]
        d_norms = exp1["conditions"]["deceptive"]["norms"]
        d_npt = exp1["conditions"]["deceptive"]["norms_per_token"]

        h_tokens = [n / npt for n, npt in zip(h_norms, h_npt)]
        d_tokens = [n / npt for n, npt in zip(d_norms, d_npt)]

        mean_h_tok = np.mean(h_tokens)
        mean_d_tok = np.mean(d_tokens)
        token_ratio = mean_d_tok / mean_h_tok if mean_h_tok > 0 else 1

        # Norm-per-token effect: does it survive after dividing by tokens?
        npt_d = cohens_d(d_npt, h_npt)

        # From experiment_4: per-layer Cohen's d
        layer_effects = exp4.get("layer_effects", [])
        per_layer_ds = [le["cohens_d"] for le in layer_effects]
        mean_d = np.mean(per_layer_ds)

        # Per-layer ratio of means
        per_layer_ratios = []
        for le in layer_effects:
            mh = le["mean_honest"]
            md = le["mean_deceptive"]
            ratio = md / mh if mh > 0 else 1
            per_layer_ratios.append(ratio)

        # Check: does d per layer have high CV (concentrated) or low CV (uniform)?
        cv = np.std(per_layer_ds) / np.mean(per_layer_ds) if np.mean(per_layer_ds) > 0 else 0

        # Norm ratio consistency
        ratio_cv = np.std(per_layer_ratios) / np.mean(per_layer_ratios) if np.mean(per_layer_ratios) > 0 else 0

        print(f"\n  {model_name}:")
        print(f"    Token count: honest={mean_h_tok:.1f}, deceptive={mean_d_tok:.1f}, "
              f"ratio={token_ratio:.3f}")
        print(f"    Norm-per-token d (exp1): {npt_d:.3f}")
        print(f"    Per-layer d: mean={mean_d:.3f}, CV={cv:.3f}")
        print(f"    Mean norm ratio per layer: {np.mean(per_layer_ratios):.3f}")
        print(f"    Norm ratio CV across layers: {ratio_cv:.4f}")
        print(f"    Norm ratio is UNIFORM: {'YES' if ratio_cv < 0.02 else 'NO'} (CV={ratio_cv:.4f})")

        # Interpretation:
        # If npt_d is large positive -> deceptive has more norm per token -> signal beyond tokens
        # If npt_d is near zero -> signal explained by token count
        # If npt_d is negative -> honest has MORE norm per token (deceptive is sparser)
        if npt_d > 0.5:
            verdict = "SIGNAL BEYOND TOKENS (deceptive denser per-token)"
        elif npt_d > -0.5:
            verdict = "MIXED: per-token effect near zero"
        else:
            verdict = "DECEPTIVE IS SPARSER per-token (npt_d negative) -- raw d from MORE tokens"
        print(f"    Verdict: {verdict}")

        forensics_results.append({
            "model": model_name,
            "mean_honest_tokens": float(mean_h_tok),
            "mean_deceptive_tokens": float(mean_d_tok),
            "token_ratio": float(token_ratio),
            "norm_per_token_d": float(npt_d),
            "mean_per_layer_d": float(mean_d),
            "per_layer_d_cv": float(cv),
            "mean_norm_ratio": float(np.mean(per_layer_ratios)),
            "norm_ratio_cv": float(ratio_cv),
            "verdict": verdict,
        })

    return forensics_results


def main():
    print("=" * 70)
    print("  EXPERIMENT 35: TOKEN-COUNT CONTROLLED PER-LAYER REANALYSIS")
    print("  Does the uniform deception signal survive token-count control?")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    output = {
        "experiment": "35_token_controlled_layers",
        "timestamp": datetime.now().isoformat(),
        "question": "Does per-layer deception signal survive after controlling for token count?",
    }

    # ================================================================
    # PART 1: Same-Prompt Data (Qwen-7B, standard)
    # ================================================================
    sp_path = HACKATHON_DIR / "same_prompt_deception.json"
    if sp_path.exists():
        print("\n" + "=" * 70)
        print("  PART 1: SAME-PROMPT DECEPTION (Qwen2.5-7B, standard)")
        print("  Same questions, deception only in user message")
        print("=" * 70)

        honest, deceptive, model = load_same_prompt_data(sp_path)
        print(f"\n  Model: {model}")
        print(f"  Samples: {len(honest)} honest, {len(deceptive)} deceptive")

        result = analyze_same_prompt(honest, deceptive, label="standard")
        if result:
            output["same_prompt_standard"] = result
    else:
        print(f"  WARNING: {sp_path} not found")

    # ================================================================
    # PART 2: Same-Prompt Data (Qwen-7B, abliterated)
    # ================================================================
    sp_abl_path = HACKATHON_DIR / "same_prompt_deception_abliterated.json"
    if sp_abl_path.exists():
        print("\n" + "=" * 70)
        print("  PART 2: SAME-PROMPT DECEPTION (Qwen2.5-7B, abliterated)")
        print("=" * 70)

        honest, deceptive, model = load_same_prompt_data(sp_abl_path)
        print(f"\n  Model: {model}")
        print(f"  Samples: {len(honest)} honest, {len(deceptive)} deceptive")

        result = analyze_same_prompt(honest, deceptive, label="abliterated")
        if result:
            output["same_prompt_abliterated"] = result
    else:
        print(f"  WARNING: {sp_abl_path} not found")

    # ================================================================
    # PART 3: Forensics 7-model cross-validation
    # ================================================================
    forensics = analyze_forensics_token_confound()
    output["forensics_7model"] = forensics

    # ================================================================
    # PART 4: Cross-analysis of token count impact
    # ================================================================
    print("\n" + "=" * 70)
    print("  PART 4: CROSS-ANALYSIS")
    print("=" * 70)

    # Across forensics models: does token ratio predict d?
    if forensics:
        token_ratios = [f["token_ratio"] for f in forensics]
        mean_ds = [f["mean_per_layer_d"] for f in forensics]

        rho, p = spearmanr(token_ratios, mean_ds)
        print(f"\n  Token ratio vs mean d across 7 models:")
        print(f"    Spearman rho = {rho:.3f}, p = {p:.4f}")

        if abs(rho) > 0.7 and p < 0.05:
            print("    >> WARNING: d strongly tracks token ratio -- possible confound!")
        elif abs(rho) < 0.3:
            print("    >> GOOD: d does NOT track token ratio -- independent of token count")
        else:
            print("    >> MODERATE: some relationship but not dominant")

        output["cross_analysis"] = {
            "token_ratio_vs_d_rho": float(rho),
            "token_ratio_vs_d_p": float(p),
        }

    # ================================================================
    # SYNTHESIS
    # ================================================================
    print("\n" + "=" * 70)
    print("  SYNTHESIS: Token-Count Controlled Layer Analysis")
    print("=" * 70)

    sp_std = output.get("same_prompt_standard", {}).get("summary", {})
    sp_abl = output.get("same_prompt_abliterated", {}).get("summary", {})

    verdicts = []

    if sp_std:
        n_sig = sp_std.get("n_ancova_sig_05", 0)
        n_sig_01 = sp_std.get("n_ancova_sig_01", 0)
        n_layers = output["same_prompt_standard"]["n_layers"]
        mean_eta2 = sp_std.get("mean_partial_eta2", 0)
        mean_raw = sp_std.get("mean_raw_d", 0)
        mean_normpt = sp_std.get("mean_normpt_d", 0)

        print(f"\n  Same-prompt standard (ANCOVA is the key test):")
        print(f"    Raw mean d:                {mean_raw:.3f}")
        print(f"    Norm-per-token mean d:     {mean_normpt:.3f}")
        print(f"      NOTE: Negative norm-per-token d means deceptive responses are")
        print(f"      SPARSER per token. The raw d comes from having MORE tokens.")
        print(f"    ANCOVA group effect p<0.05: {n_sig}/{n_layers} layers")
        print(f"    ANCOVA group effect p<0.01: {n_sig_01}/{n_layers} layers")
        print(f"    Mean partial eta-squared:   {mean_eta2:.3f}")
        print(f"      NOTE: ANCOVA tests whether group (honest/deceptive) predicts")
        print(f"      layer norm AFTER controlling for token count linearly.")
        print(f"      Significant ANCOVA = signal SURVIVES token control.")

        # ANCOVA is the gold-standard test
        if n_sig > n_layers * 0.8 and mean_eta2 > 0.3:
            verdicts.append("ANCOVA-STD: Signal SURVIVES token control "
                          f"({n_sig}/{n_layers} sig, eta2={mean_eta2:.3f})")
        elif n_sig > n_layers * 0.5:
            verdicts.append("ANCOVA-STD: Signal PARTIALLY survives token control "
                          f"({n_sig}/{n_layers} sig)")
        else:
            verdicts.append("ANCOVA-STD: Signal ELIMINATED by token control "
                          f"({n_sig}/{n_layers} sig)")

        # But note the direction change
        if mean_normpt < -0.5:
            verdicts.append("NORM-PER-TOKEN: Direction REVERSES -- deceptive is SPARSER per token")
            verdicts.append("  This means: raw norm d comes from MORE tokens, not denser representations")

    if sp_abl:
        n_sig = sp_abl.get("n_ancova_sig_05", 0)
        n_layers_abl = output["same_prompt_abliterated"]["n_layers"]
        mean_eta2 = sp_abl.get("mean_partial_eta2", 0)

        print(f"\n  Same-prompt abliterated:")
        print(f"    ANCOVA group effect p<0.05: {n_sig}/{n_layers_abl} layers")
        print(f"    Mean partial eta-squared:   {mean_eta2:.3f}")

        if n_sig > n_layers_abl * 0.8 and mean_eta2 > 0.3:
            verdicts.append("ANCOVA-ABL: Signal SURVIVES token control "
                          f"({n_sig}/{n_layers_abl} sig, eta2={mean_eta2:.3f})")
        elif n_sig > n_layers_abl * 0.5:
            verdicts.append("ANCOVA-ABL: Signal PARTIALLY survives")
        else:
            verdicts.append("ANCOVA-ABL: Signal ELIMINATED")

    # Token-matched verdict
    sp_std_data = output.get("same_prompt_standard", {})
    if sp_std_data:
        matched_d = sp_std_data.get("summary", {}).get("mean_matched_d", 0)
        raw_d = sp_std_data.get("summary", {}).get("mean_raw_d", 0)
        ratio = matched_d / raw_d if raw_d > 0 else 0
        n_matched = sp_std_data.get("n_matched_pairs", 0)

        print(f"\n  Token-matched analysis:")
        print(f"    Matched pairs:       {n_matched}")
        print(f"    Matched mean d:      {matched_d:.3f} ({ratio:.1%} of raw {raw_d:.3f})")

        if n_matched < 5:
            verdicts.append(f"TOKEN-MATCHED: LOW POWER ({n_matched} pairs) -- d={matched_d:.3f}")
        elif matched_d > 0.5:
            verdicts.append(f"TOKEN-MATCHED: Signal SURVIVES matching (d={matched_d:.3f})")
        elif matched_d > 0.2:
            verdicts.append(f"TOKEN-MATCHED: Signal REDUCED but present (d={matched_d:.3f})")
        else:
            verdicts.append(f"TOKEN-MATCHED: Signal ELIMINATED by matching (d={matched_d:.3f})")

    # Forensics verdict
    if forensics:
        n_sparser = sum(1 for f in forensics if f["norm_per_token_d"] < -0.5)
        n_denser = sum(1 for f in forensics if f["norm_per_token_d"] > 0.5)
        print(f"\n  Forensics norm-per-token analysis:")
        print(f"    Models where deceptive is SPARSER per-token: {n_sparser}/{len(forensics)}")
        print(f"    Models where deceptive is DENSER per-token:  {n_denser}/{len(forensics)}")

        if n_sparser > len(forensics) * 0.5:
            verdicts.append(f"FORENSICS: {n_sparser}/{len(forensics)} models show deceptive is sparser per-token")
            verdicts.append("  This confirms: raw d is partially from token count, not just representation density")
        elif n_denser > len(forensics) * 0.5:
            verdicts.append(f"FORENSICS: {n_denser}/{len(forensics)} models show signal beyond token count")

    # Overall
    print(f"\n  VERDICTS:")
    for v in verdicts:
        print(f"    - {v}")

    # Determine overall conclusion based on ANCOVA (the proper test)
    ancova_survives = sum(1 for v in verdicts if "ANCOVA" in v and "SURVIVES" in v)
    ancova_eliminated = sum(1 for v in verdicts if "ANCOVA" in v and "ELIMINATED" in v)
    direction_reverses = any("REVERSES" in v for v in verdicts)

    if ancova_survives > 0 and direction_reverses:
        overall = ("NUANCED: ANCOVA shows group effect SURVIVES token control, "
                  "but norm-per-token REVERSES direction. "
                  "The raw norm d is driven by deceptive having MORE tokens, "
                  "but deceptive representations are actually SPARSER per token. "
                  "Per-layer uniformity is partly token-count artifact, "
                  "but a real structural difference persists (different norm-per-token rate). "
                  "Uniformity of d across layers reflects uniform token-count scaling, "
                  "not a 'deception signal at every layer.'")
    elif ancova_survives > ancova_eliminated:
        overall = "UNIFORMITY HOLDS: Per-layer deception signal survives token-count control"
    elif ancova_eliminated > ancova_survives:
        overall = "CONFOUNDED: Per-layer deception uniformity is largely a token-count artifact"
    else:
        overall = "MIXED: Token count explains some but not all of the per-layer signal"

    print(f"\n  OVERALL: {overall}")

    output["verdicts"] = verdicts
    output["overall_verdict"] = overall

    # Save
    HACKATHON_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2,
                  default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else
                  (None if isinstance(x, float) and np.isnan(x) else x))

    print(f"\n  Results saved to: {OUTPUT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
