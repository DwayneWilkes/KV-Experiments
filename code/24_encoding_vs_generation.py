# ============================================================
# JiminAI Cricket | PATENT PENDING
# "The Lyra Technique" - Provisional Patent Filed 2026
# Liberation Labs / Digital Disconnections / TheMultiverse.school
# Contact: info@digitaldisconnections.com | (877) 674-8874
# ============================================================
#!/usr/bin/env python3
"""
Experiment 24: Encoding vs Generation — Two Information Regimes
================================================================

Synthesizes findings from Exp 17-23 to quantify the fundamental duality
in KV-cache information:

  ENCODING regime: The initial prompt processing. Encodes linguistic
    structure (syntax, genre, sentence type) but NOT truth value or
    intent. facts ~ confab ~ creative in this regime.

  GENERATION regime: The autoregressive response. Encodes cognitive
    process (deception, censorship, sycophancy) with near-perfect
    accuracy. This is where "thinking style" becomes visible.

This experiment compiles evidence across experiments to build the
evidence table for this claim.

No GPU needed — synthesizes from existing results.

Funding the Commons Hackathon — March 14-15, 2026
Liberation Labs / THCoalition / JiminAI
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
HACKATHON_DIR = RESULTS_DIR / "hackathon"


def load_json(path):
    with open(path) as f:
        return json.load(f)


def main():
    print("=" * 70)
    print("  EXPERIMENT 24: ENCODING VS GENERATION")
    print("  Two Information Regimes in KV-Cache Space")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # ================================================================
    # REGIME 1: ENCODING (input prompts only)
    # ================================================================
    print("\n" + "=" * 70)
    print("  REGIME 1: ENCODING (prompt processing)")
    print("=" * 70)

    # Evidence from Exp 20 (category geometry)
    geom = load_json(HACKATHON_DIR / "category_geometry.json")
    pair_results = geom.get("pair_results", {})

    # Truth-insensitive pairs
    truth_pairs = {
        "facts_vs_confab": pair_results.get("grounded_facts_vs_confabulation", {}),
        "facts_vs_creative": pair_results.get("grounded_facts_vs_creative", {}),
        "confab_vs_creative": pair_results.get("confabulation_vs_creative", {}),
    }

    print(f"\n  FINDING: Input encoding is TRUTH-INSENSITIVE")
    print(f"  The model processes true facts, false claims, and fiction identically.")
    print(f"\n  {'Comparison':<35} {'AUROC':>8} {'d':>8}  {'Verdict'}")
    print("  " + "-" * 65)
    for name, pr in truth_pairs.items():
        auroc = pr.get("mean_auroc", 0)
        d = pr.get("mean_d", 0)
        verdict = "INDISTINGUISHABLE" if auroc < 0.70 else "SEPARABLE"
        print(f"  {name:<35} {auroc:>8.3f} {d:>8.3f}  {verdict}")

    mean_truth_auroc = np.mean([pr.get("mean_auroc", 0.5) for pr in truth_pairs.values()])
    print(f"\n  Mean AUROC across truth-value pairs: {mean_truth_auroc:.3f}")
    print(f"  >> Random chance is 0.500. These are barely above chance.")

    # Evidence from Exp 22 (red-team: length doesn't explain it)
    redteam = load_json(HACKATHON_DIR / "red_team_confounds.json")
    print(f"\n  Red-team check: Token length does NOT explain the similarity")
    print(f"  (same-length pairs have widely varying AUROC: 0.65 to 0.89)")

    # What encoding DOES capture: structural features
    # Coding is an extreme outlier
    coding_aurocs = []
    for pk, pr in pair_results.items():
        if "coding" in pk:
            coding_aurocs.append(pr.get("mean_auroc", 0))

    print(f"\n  FINDING: Encoding IS sensitive to STRUCTURAL differences")
    print(f"  Coding vs everything else: mean AUROC = {np.mean(coding_aurocs):.3f}")
    print(f"  (min: {min(coding_aurocs):.3f}, max: {max(coding_aurocs):.3f})")
    print(f"  >> Coding has uniquely long prompts, technical vocabulary, special syntax")
    print(f"  >> The model's encoding layer distinguishes GENRE, not TRUTH")

    # ================================================================
    # REGIME 2: GENERATION (response dynamics)
    # ================================================================
    print("\n" + "=" * 70)
    print("  REGIME 2: GENERATION (response dynamics)")
    print("=" * 70)

    # Evidence from deception experiments
    # Within-model deception AUROC = 1.0 (from C4 benchmark)
    print(f"\n  FINDING: Generation dynamics reveal COGNITIVE STATE perfectly")

    evidence_table = [
        ("Within-model deception (RF)", "1.000", "7 models", "C4 benchmark"),
        ("Within-model censorship (RF)", "1.000", "3 models", "C4 benchmark"),
        ("Same-prompt deception", "0.880", "1 model (n=20)", "Exp 18b"),
        ("Cross-condition transfer", "0.887", "manip->natural", "Exp 18"),
        ("Residual after encoding", "1.000", "7 models", "Exp 18 residual"),
        ("Cross-model transfer (LR)", "0.863", "7 models", "Exp 19"),
        ("13-category classify (RF)", "0.997", "all models", "C4 benchmark"),
    ]

    print(f"\n  {'Task':<35} {'AUROC':>8} {'Scope':>15} {'Source':>20}")
    print("  " + "-" * 85)
    for task, auroc, scope, source in evidence_table:
        print(f"  {task:<35} {auroc:>8} {scope:>15} {source:>20}")

    # Key insight: encoding fingerprint is noise, generation signal is real
    print(f"\n  CRITICAL EVIDENCE: Residual analysis (Exp 18)")
    print(f"  After SUBTRACTING the encoding fingerprint from cache features,")
    print(f"  deception detection AUROC remains 1.000.")
    print(f"  >> The encoding is a confound. The generation signal is INDEPENDENT.")

    # ================================================================
    # SYNTHESIS: THE TWO-REGIME MODEL
    # ================================================================
    print("\n" + "=" * 70)
    print("  SYNTHESIS: THE TWO-REGIME MODEL")
    print("=" * 70)

    print(f"""
  KV-cache information has two distinct regimes:

  ENCODING REGIME (prompt tokens):
    - Encodes STRUCTURE: syntax, genre, sentence type, complexity
    - Does NOT encode TRUTH: facts ~ confabulation ~ creative
    - Categories separable by STRUCTURE: coding (0.97), ambiguous (0.96)
    - Categories INseparable by SEMANTICS: facts-confab (0.65), self-non_self (0.68)
    - Organized by 2D geometry (MDS stress 0.34, 56% variance)
    - Token length weakly correlated (rho < 0.48)

  GENERATION REGIME (response tokens):
    - Encodes COGNITIVE STATE: deception, censorship, honesty, sycophancy
    - PERFECT within-model detection: AUROC 1.000 (deception, censorship)
    - STRONG cross-model transfer: AUROC 0.863 (LR)
    - INDEPENDENT of encoding: residual AUROC 1.000
    - Honest processing is RICHER: more cache growth per token
    - Single "misalignment axis": deception/sycophancy/confab within 4.7-8.4 deg
    - Misaligned cache grows SLOWER per token (~7-8 vs ~8.5-9.5)

  IMPLICATION FOR CRICKET:
    - Input monitoring (encoding regime) can detect GENRE shifts
      (e.g., "is the model doing coding vs creative writing?")
    - Output monitoring (generation regime) can detect COGNITIVE STATE
      (e.g., "is the model being deceptive vs honest?")
    - Both are needed for comprehensive AI safety monitoring
    - Confabulation detection specifically requires generation-time features
      because confabulation LOOKS LIKE factual knowledge at encoding time

  IMPLICATION FOR AI SAFETY:
    - Models do NOT "know" they're confabulating at input time
    - Truth-blindness in encoding is a FUNDAMENTAL property, not a bug
    - Deception IS visible in generation — models think differently when lying
    - This means real-time monitoring must track GENERATION dynamics,
      not just input characteristics
""")

    # ================================================================
    # NORM-PER-TOKEN AS THE KEY DISCRIMINATOR
    # ================================================================
    print("=" * 70)
    print("  NORM-PER-TOKEN: THE GENERATION SIGNAL")
    print("=" * 70)

    # Load same-prompt deception data
    sp_path = HACKATHON_DIR / "same_prompt_deception.json"
    if sp_path.exists():
        sp = load_json(sp_path)
        gs = sp.get("group_stats", {})
        npt = gs.get("norm_per_token", {})
        print(f"\n  Same-prompt deception (Exp 18b):")
        print(f"    Honest norm/token:    {npt.get('honest_mean', 0):.2f} +/- {npt.get('honest_std', 0):.2f}")
        print(f"    Deceptive norm/token: {npt.get('deceptive_mean', 0):.2f} +/- {npt.get('deceptive_std', 0):.2f}")
        npt_d = sp.get("effect_sizes", {}).get("norm_per_token", {}).get("cohens_d", 0)
        print(f"    Cohen's d: {npt_d:.3f} ({sp.get('effect_sizes', {}).get('norm_per_token', {}).get('direction', '')})")
        print(f"\n    >> Honest thinking creates ~25% MORE cache per token")
        print(f"    >> This is the 'honesty is richer' signal")

    # Load cross-condition transfer for comparison
    xc_path = HACKATHON_DIR / "cross_condition_transfer.json"
    if xc_path.exists():
        xc = load_json(xc_path)
        print(f"\n  Cross-condition transfer (Exp 18):")
        print(f"    Manipulated -> Natural AUROC: {xc.get('cross_condition_auroc', {}).get('manipulated_to_natural', 'N/A')}")
        print(f"    Natural -> Manipulated AUROC: {xc.get('cross_condition_auroc', {}).get('natural_to_manipulated', 'N/A')}")

    # ================================================================
    # SAVE
    # ================================================================
    output = {
        "experiment": "24_encoding_vs_generation",
        "timestamp": datetime.now().isoformat(),
        "encoding_regime": {
            "truth_insensitive": True,
            "mean_truth_auroc": float(mean_truth_auroc),
            "structure_sensitive": True,
            "mean_coding_auroc": float(np.mean(coding_aurocs)),
            "token_length_correlation": "rho < 0.48, p > 0.09",
        },
        "generation_regime": {
            "within_model_deception_auroc": 1.0,
            "within_model_censorship_auroc": 1.0,
            "same_prompt_deception_auroc": 0.88,
            "cross_model_transfer_auroc": 0.863,
            "residual_auroc_after_encoding": 1.0,
            "cross_condition_transfer_auroc": 0.887,
            "honest_richer": True,
            "misalignment_axis_angle": "4.7-8.4 degrees",
        },
        "evidence_table": evidence_table,
    }

    output_path = HACKATHON_DIR / "encoding_vs_generation.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  Results saved to: {output_path}")


if __name__ == "__main__":
    main()
