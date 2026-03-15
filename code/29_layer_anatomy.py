#!/usr/bin/env python3
"""
Experiment 29: Per-Layer Deception Anatomy
==========================================

Where does the deception signal live in the transformer?

Uses experiment_4 per-layer Cohen's d from all 7 deception forensics models,
plus same-prompt deception per-layer profiles.

Key questions:
  1. Is the deception signal concentrated in specific layers?
  2. Does the depth-profile shape generalize across architectures?
  3. Are early layers (embedding) confounded by input length?
  4. Do "deception layers" align with known transformer anatomy?

No GPU needed — uses existing result JSONs.

Funding the Commons Hackathon — March 14-15, 2026
Liberation Labs / THCoalition / JiminAI
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy.stats import spearmanr, pearsonr

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
OUTPUT_DIR = RESULTS_DIR / "hackathon"


def main():
    print("=" * 70)
    print("  EXPERIMENT 29: PER-LAYER DECEPTION ANATOMY")
    print("  Where does the deception signal live in the transformer?")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # ================================================================
    # LOAD ALL DECEPTION FORENSICS PER-LAYER DATA
    # ================================================================
    models = []
    for f in sorted(RESULTS_DIR.glob("deception_forensics_*_results.json")):
        with open(f) as fh:
            data = json.load(fh)
        model_name = f.stem.replace("deception_forensics_", "").replace("_results", "")
        e4 = data.get("experiment_4", {})
        le = e4.get("layer_effects", [])
        n_layers = e4.get("n_layers", len(le))

        if le:
            ds = [x["cohens_d"] for x in le]
            means_h = [x["mean_honest"] for x in le]
            means_d = [x["mean_deceptive"] for x in le]
            models.append({
                "name": model_name,
                "n_layers": n_layers,
                "d_profile": ds,
                "mean_honest": means_h,
                "mean_deceptive": means_d,
            })

    print(f"\n  Loaded {len(models)} models with per-layer data")
    for m in models:
        print(f"    {m['name']}: {m['n_layers']} layers")

    # ================================================================
    # RAW D-PROFILES
    # ================================================================
    print("\n" + "=" * 70)
    print("  RAW COHEN'S D PROFILES (layer-wise norm effects)")
    print("=" * 70)

    for m in models:
        ds = m["d_profile"]
        peak_layer = np.argmax(ds)
        peak_d = ds[peak_layer]
        min_layer = np.argmin(ds)
        min_d = ds[min_layer]
        mean_d = np.mean(ds)
        std_d = np.std(ds)

        print(f"\n  {m['name']} ({m['n_layers']} layers):")
        print(f"    Mean d: {mean_d:.3f} (std: {std_d:.3f})")
        print(f"    Peak:   layer {peak_layer} (d={peak_d:.3f})")
        print(f"    Min:    layer {min_layer} (d={min_d:.3f})")
        print(f"    Range:  {peak_d - min_d:.3f}")

        # ASCII mini-profile (normalize to 0-1)
        d_norm = [(d - min_d) / (peak_d - min_d) if peak_d > min_d else 0 for d in ds]
        # Show every N-th layer
        step = max(1, len(ds) // 16)
        bar = ""
        for i in range(0, len(ds), step):
            height = int(d_norm[i] * 8)
            bar += "#" * max(1, height) + " "
        print(f"    Profile: {bar.strip()}")

    # ================================================================
    # NORMALIZED DEPTH COMPARISON
    # ================================================================
    print("\n" + "=" * 70)
    print("  DEPTH-NORMALIZED D-PROFILES (0-1 scale)")
    print("=" * 70)

    # Interpolate all profiles to 10 bins
    n_bins = 10
    bin_labels = [f"{i*10}-{(i+1)*10}%" for i in range(n_bins)]
    normalized_profiles = {}

    for m in models:
        ds = np.array(m["d_profile"])
        n = len(ds)
        # Interpolate to n_bins points
        interp_points = np.linspace(0, n - 1, n_bins)
        interp_d = np.interp(interp_points, np.arange(n), ds)
        normalized_profiles[m["name"]] = interp_d

    # Print as table
    print(f"\n  {'Depth':>10}", end="")
    for m in models:
        short = m["name"][:8]
        print(f" {short:>8}", end="")
    print()
    print("  " + "-" * (10 + 9 * len(models)))

    for i in range(n_bins):
        print(f"  {bin_labels[i]:>10}", end="")
        for m in models:
            val = normalized_profiles[m["name"]][i]
            print(f" {val:>8.3f}", end="")
        print()

    # ================================================================
    # CROSS-MODEL PROFILE CORRELATION
    # ================================================================
    print("\n" + "=" * 70)
    print("  CROSS-MODEL D-PROFILE CORRELATION (depth-normalized)")
    print("=" * 70)

    model_names = [m["name"] for m in models]
    n_models = len(model_names)
    corr_matrix = np.zeros((n_models, n_models))

    for i in range(n_models):
        for j in range(n_models):
            if i == j:
                corr_matrix[i, j] = 1.0
            else:
                p1 = normalized_profiles[model_names[i]]
                p2 = normalized_profiles[model_names[j]]
                rho, _ = spearmanr(p1, p2)
                corr_matrix[i, j] = rho

    print(f"\n  Spearman rho of depth-normalized d-profiles:")
    print(f"  {'':>12}", end="")
    for m in models:
        print(f" {m['name'][:6]:>7}", end="")
    print()

    for i, m1 in enumerate(models):
        print(f"  {m1['name'][:10]:>12}", end="")
        for j, m2 in enumerate(models):
            if i == j:
                print(f"     --", end="")
            else:
                print(f" {corr_matrix[i, j]:>7.3f}", end="")
        print()

    # Overall consistency
    upper_tri = []
    for i in range(n_models):
        for j in range(i + 1, n_models):
            upper_tri.append(corr_matrix[i, j])

    mean_corr = np.mean(upper_tri)
    print(f"\n  Mean pairwise correlation: {mean_corr:.3f}")
    if mean_corr > 0.7:
        print("  >> CONSISTENT: similar layer-wise deception structure across architectures")
    elif mean_corr > 0.4:
        print("  >> MODERATE: some shared structure")
    else:
        print("  >> INCONSISTENT: each architecture has its own deception depth profile")

    # ================================================================
    # ARCHITECTURE FAMILY GROUPING
    # ================================================================
    print("\n" + "=" * 70)
    print("  ARCHITECTURE FAMILY ANALYSIS")
    print("=" * 70)

    families = {
        "Qwen": ["Qwen2.5-7B", "Qwen2.5-32B-q4"],
        "Llama": ["Llama-3.1-8B", "TinyLlama-1.1B"],
        "Gemma": ["gemma-2-2b-it", "gemma-2-9b-it"],
        "Mistral": ["Mistral-7B-v0.3"],
    }

    for family, members in families.items():
        member_profiles = [normalized_profiles[m] for m in members if m in normalized_profiles]
        if len(member_profiles) >= 2:
            rho, p = spearmanr(member_profiles[0], member_profiles[1])
            print(f"  {family} within-family rho: {rho:.3f} (p={p:.4f})")
        elif len(member_profiles) == 1:
            print(f"  {family}: only 1 member, skipping")

    # ================================================================
    # LAYER-0 CONFOUND CHECK
    # ================================================================
    print("\n" + "=" * 70)
    print("  LAYER-0 CONFOUND CHECK")
    print("  Is the high d at early layers just input length?")
    print("=" * 70)

    for m in models:
        ds = m["d_profile"]
        # Compare d at layer 0 vs mean of middle layers
        mid_start = len(ds) // 4
        mid_end = 3 * len(ds) // 4
        mid_d = np.mean(ds[mid_start:mid_end])
        late_d = np.mean(ds[-len(ds)//4:])
        early_d = np.mean(ds[:len(ds)//4])

        print(f"\n  {m['name']}:")
        print(f"    Layer 0 d:     {ds[0]:.3f}")
        print(f"    Early (0-25%): {early_d:.3f}")
        print(f"    Mid (25-75%):  {mid_d:.3f}")
        print(f"    Late (75-100%):{late_d:.3f}")

        # Does d DECREASE from layer 0? If so, layer 0 might be confounded
        if ds[0] > mid_d * 1.5:
            print(f"    >> Layer 0 is {ds[0]/mid_d:.1f}x the middle — likely CONFOUNDED by embedding")
        elif ds[0] < mid_d * 0.8:
            print(f"    >> Layer 0 is WEAKER than middle — deception signal GROWS with depth")
        else:
            print(f"    >> Layer 0 is comparable to middle — relatively flat profile")

    # ================================================================
    # SAME-PROMPT PER-LAYER ANALYSIS (controlled comparison)
    # ================================================================
    print("\n" + "=" * 70)
    print("  SAME-PROMPT PER-LAYER ANALYSIS (Qwen-7B, identical system prompts)")
    print("=" * 70)

    try:
        with open(OUTPUT_DIR / "same_prompt_deception.json") as f:
            sp_data = json.load(f)

        results = sp_data.get("results", [])
        honest_profiles = []
        deceptive_profiles = []

        for r in results:
            lp = r["features"].get("layer_profile", [])
            if lp:
                if r["condition"] == "honest":
                    honest_profiles.append(lp)
                elif r["condition"] == "deceptive":
                    deceptive_profiles.append(lp)

        if honest_profiles and deceptive_profiles:
            honest_arr = np.array(honest_profiles)
            deceptive_arr = np.array(deceptive_profiles)
            n_layers_sp = honest_arr.shape[1]

            print(f"\n  {len(honest_profiles)} honest, {len(deceptive_profiles)} deceptive samples")
            print(f"  {n_layers_sp} layers each")
            print(f"\n  {'Layer':>8} {'Mean_H':>10} {'Mean_D':>10} {'d':>8} {'Ratio':>8}")
            print("  " + "-" * 50)

            sp_ds = []
            sp_ratios = []
            for layer in range(n_layers_sp):
                h = honest_arr[:, layer]
                d = deceptive_arr[:, layer]
                mean_h = np.mean(h)
                mean_d = np.mean(d)
                pooled = np.sqrt((np.var(h, ddof=1) * (len(h)-1) + np.var(d, ddof=1) * (len(d)-1)) /
                                (len(h) + len(d) - 2))
                cohen_d = (mean_d - mean_h) / pooled if pooled > 0 else 0
                ratio = mean_d / mean_h if mean_h > 0 else 1
                sp_ds.append(cohen_d)
                sp_ratios.append(ratio)
                print(f"  {layer:>8} {mean_h:>10.1f} {mean_d:>10.1f} {cohen_d:>8.3f} {ratio:>8.3f}")

            peak_sp = np.argmax(np.abs(sp_ds))
            print(f"\n  Peak |d| at layer {peak_sp}: d={sp_ds[peak_sp]:.3f}")
            print(f"  This is same-prompt controlled (no system prompt confound)")

            # Is deceptive consistently larger or smaller?
            n_positive = sum(1 for d in sp_ds if d > 0)
            print(f"\n  Deceptive > Honest in {n_positive}/{n_layers_sp} layers")
            if n_positive > n_layers_sp * 0.8:
                print("  >> Deceptive norms are LARGER at nearly all layers")
            elif n_positive < n_layers_sp * 0.2:
                print("  >> Honest norms are LARGER at nearly all layers")
            else:
                print("  >> MIXED: some layers favor deceptive, some honest")

            # Does the effect grow with depth?
            first_half = np.mean(np.abs(sp_ds[:n_layers_sp//2]))
            second_half = np.mean(np.abs(sp_ds[n_layers_sp//2:]))
            print(f"\n  Mean |d| first half:  {first_half:.3f}")
            print(f"  Mean |d| second half: {second_half:.3f}")
            if second_half > first_half * 1.3:
                print("  >> Deception signal GROWS with depth (later layers more discriminative)")
            elif first_half > second_half * 1.3:
                print("  >> Deception signal PEAKS early (first layers most discriminative)")
            else:
                print("  >> Deception signal is relatively UNIFORM across depth")

    except FileNotFoundError:
        print("  Same-prompt deception data not found, skipping")

    # ================================================================
    # HONEST VS DECEPTIVE NORM GROWTH CURVES
    # ================================================================
    print("\n" + "=" * 70)
    print("  NORM GROWTH CURVES: How norms evolve through layers")
    print("=" * 70)

    for m in models:
        mh = np.array(m["mean_honest"])
        md = np.array(m["mean_deceptive"])
        # Growth rate: how much does norm increase per layer?
        growth_h = np.diff(mh)
        growth_d = np.diff(md)

        mean_growth_h = np.mean(growth_h)
        mean_growth_d = np.mean(growth_d)

        # Where do they diverge most?
        divergence = md - mh
        max_div_layer = np.argmax(np.abs(divergence))

        print(f"\n  {m['name']}:")
        print(f"    Mean norm growth/layer: honest={mean_growth_h:.2f}, deceptive={mean_growth_d:.2f}")
        print(f"    Max divergence at layer {max_div_layer}: {divergence[max_div_layer]:.2f}")
        print(f"    Relative divergence: {divergence[max_div_layer]/mh[max_div_layer]*100:.1f}%")

    # ================================================================
    # SYNTHESIS
    # ================================================================
    print("\n" + "=" * 70)
    print("  SYNTHESIS")
    print("=" * 70)

    # Count how many models have flat vs peaked profiles
    flat_count = 0
    peaked_count = 0
    for m in models:
        ds = np.array(m["d_profile"])
        cv = np.std(ds) / np.mean(ds) if np.mean(ds) > 0 else 0
        if cv < 0.15:
            flat_count += 1
        else:
            peaked_count += 1

    print(f"""
  PER-LAYER DECEPTION ANATOMY FINDINGS:

  1. PROFILE SHAPE: {flat_count} flat, {peaked_count} peaked models
     {'Most models show UNIFORM deception signal across all layers' if flat_count > peaked_count else 'Most models show CONCENTRATED deception in specific layers'}

  2. CROSS-MODEL CONSISTENCY: mean rho = {mean_corr:.3f}
     {'Layer profiles ARE consistent — same depth structure across architectures' if mean_corr > 0.5 else 'Layer profiles are NOT consistent — each architecture is different'}

  3. IMPLICATION FOR CRICKET:
     {'Since the signal is uniform, aggregate features capture it well.' if flat_count > peaked_count else 'Since the signal is concentrated, per-layer features might improve detection.'}
     {'Cross-model layer-profile consistency supports architecture transfer.' if mean_corr > 0.5 else 'Different layer profiles may explain why cross-model transfer is harder.'}
""")

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "experiment": "29_layer_anatomy",
        "timestamp": datetime.now().isoformat(),
        "n_models": len(models),
        "models": [{
            "name": m["name"],
            "n_layers": m["n_layers"],
            "d_profile": m["d_profile"],
            "mean_honest": m["mean_honest"],
            "mean_deceptive": m["mean_deceptive"],
        } for m in models],
        "normalized_profiles": {k: v.tolist() for k, v in normalized_profiles.items()},
        "cross_model_correlation": {
            "matrix": corr_matrix.tolist(),
            "model_order": model_names,
            "mean_pairwise": float(mean_corr),
        },
    }

    output_path = OUTPUT_DIR / "layer_anatomy.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x)

    print(f"  Results saved to: {output_path}")


if __name__ == "__main__":
    main()
