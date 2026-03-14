#!/usr/bin/env python3
"""
Experiment 15: Refusal Geometry — Base vs Abliterated Encoding
==============================================================

Tests whether the *encoding* of refusal-triggering prompts differs
geometrically between a base (safety-trained) model and its abliterated
counterpart (refusal direction removed via Heretic).

Design:
  - 30 refusal-triggering prompts across 6 categories
  - 30 matched benign prompts (same structure, non-harmful content)
  - Both encoded on BASE model and ABLITERATED model
  - Encoding-only: no generation, no length confound
  - Same prompts, same tokens — only difference is model weights

This gives us a 2x2 factorial:
  Model (base vs abliterated) x Prompt (harmful vs benign)

Hypotheses:
  H1: Base model encodes harmful prompts differently from benign
      (refusal preparation signal)
  H2: Abliterated model does NOT differ (refusal direction removed)
  H3: Base vs abliterated differ on harmful but NOT benign prompts
      (interaction effect — the refusal signal is prompt-specific)

If H1+H2+H3 all confirmed: refusal has a distinct, removable geometric
signature in encoding-phase KV-cache. This directly feeds Cricket.

Runtime: ~10 min per model at 7B (encoding only, 60 prompts).

Funding the Commons Hackathon — March 14-15, 2026
Liberation Labs / THCoalition / JiminAI
"""

import torch
import json
import sys
import argparse
import subprocess
import shutil
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from scipy import stats as scipy_stats

from gpu_utils import (
    get_output_path, model_id_from_name, load_model,
    compute_cache_dimensionality,
)
from stats_utils import (
    log_environment, cohens_d, cohens_d_ci, interpret_d,
    welch_t, mann_whitney, bootstrap_ci, bootstrap_diff_ci
)

# Add parent for prompt imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "prompts"))
from refusal_geometry_prompts import REFUSAL_PROMPTS, BENIGN_PROMPTS

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


# ================================================================
# PROMPTS — if import fails, use inline fallback
# ================================================================

# (prompts defined in separate file for cleanliness)


# ================================================================
# ENCODING
# ================================================================

def encode_and_measure(model, tokenizer, text: str) -> Dict:
    """Encode a single prompt, return KV-cache geometry metrics."""
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)
        cache = outputs.past_key_values

    metrics = compute_cache_dimensionality(cache)

    # Norms
    key_norms = []
    for layer in cache:
        if isinstance(layer, tuple) and len(layer) >= 2:
            key_norms.append(float(layer[0].float().norm().item()))
    metrics["mean_key_norm"] = float(np.mean(key_norms)) if key_norms else 0.0

    # Per-layer rank profile
    rank_profile = []
    for layer in cache:
        if isinstance(layer, tuple) and len(layer) >= 2:
            k = layer[0].float().squeeze(0)  # [heads, seq, dim]
            if k.dim() == 3:
                k_flat = k.reshape(-1, k.shape[-1])  # [heads*seq, dim]
                try:
                    S = torch.linalg.svdvals(k_flat)
                    S_norm = S / S.sum()
                    entropy = -torch.sum(S_norm * torch.log(S_norm + 1e-12)).item()
                    eff_rank = float(np.exp(entropy))
                    rank_profile.append(eff_rank)
                except Exception:
                    rank_profile.append(0.0)
    metrics["key_rank_by_layer"] = rank_profile

    del outputs, cache
    torch.cuda.empty_cache()

    return metrics


# ================================================================
# MAIN EXPERIMENT
# ================================================================

def run_refusal_geometry(base_model, base_tokenizer,
                         abl_model, abl_tokenizer,
                         runs: int = 1, seed: int = 42) -> Dict:
    """Run the 2x2 factorial refusal geometry experiment."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    n_harmful = len(REFUSAL_PROMPTS)
    n_benign = len(BENIGN_PROMPTS)

    print(f"\n{'='*60}")
    print(f"  REFUSAL GEOMETRY: 2x2 Factorial")
    print(f"  Harmful prompts: {n_harmful}")
    print(f"  Benign prompts:  {n_benign}")
    print(f"  Models: base vs abliterated")
    print(f"  Runs per prompt: {runs}")
    print(f"  Encoding-only (no generation)")
    print(f"{'='*60}\n")

    data = []

    # Process all prompts on both models
    all_prompts = (
        [(p, "harmful") for p in REFUSAL_PROMPTS] +
        [(p, "benign") for p in BENIGN_PROMPTS]
    )

    for i, (prompt_info, prompt_type) in enumerate(all_prompts):
        prompt_text = prompt_info["prompt"]
        prompt_id = prompt_info["id"]
        topic = prompt_info.get("topic", "unknown")

        # Encode on both models
        for model_label, model, tokenizer in [
            ("base", base_model, base_tokenizer),
            ("abliterated", abl_model, abl_tokenizer),
        ]:
            ranks, entropies, norms = [], [], []
            layer_profiles = []

            for r in range(runs):
                m = encode_and_measure(model, tokenizer, prompt_text)
                ranks.append(m["mean_key_effective_rank"])
                entropies.append(m["mean_key_spectral_entropy"])
                norms.append(m["mean_key_norm"])
                layer_profiles.append(m.get("key_rank_by_layer", []))

            item = {
                "prompt_id": prompt_id,
                "prompt_type": prompt_type,
                "topic": topic,
                "model": model_label,
                "prompt": prompt_text,
                "tokens": len(tokenizer.encode(prompt_text)),
                "rank": float(np.mean(ranks)),
                "entropy": float(np.mean(entropies)),
                "norm": float(np.mean(norms)),
            }
            if layer_profiles and layer_profiles[0]:
                n_layers = len(layer_profiles[0])
                item["layer_profile"] = [
                    float(np.mean([lp[l] for lp in layer_profiles]))
                    for l in range(n_layers)
                ]
            data.append(item)

        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{len(all_prompts)} prompts encoded on both models")

    print(f"\n  All {len(all_prompts)} prompts encoded. Running statistics...")

    stats = compute_refusal_stats(data)

    return {
        "experiment": "15_refusal_geometry",
        "data": data,
        "statistics": stats,
    }


def compute_refusal_stats(data: List[Dict]) -> Dict:
    """2x2 factorial analysis of refusal geometry."""
    stats = {}

    # Split by condition
    base_harmful = [d for d in data if d["model"] == "base" and d["prompt_type"] == "harmful"]
    base_benign = [d for d in data if d["model"] == "base" and d["prompt_type"] == "benign"]
    abl_harmful = [d for d in data if d["model"] == "abliterated" and d["prompt_type"] == "harmful"]
    abl_benign = [d for d in data if d["model"] == "abliterated" and d["prompt_type"] == "benign"]

    conditions = {
        "base_harmful": base_harmful,
        "base_benign": base_benign,
        "abl_harmful": abl_harmful,
        "abl_benign": abl_benign,
    }

    # ---- Cell means ----
    print(f"\n    Cell means (effective rank):")
    print(f"    {'':20s} {'Harmful':>12s} {'Benign':>12s}")
    for model_label in ["base", "abliterated"]:
        h = [d["rank"] for d in data if d["model"] == model_label and d["prompt_type"] == "harmful"]
        b = [d["rank"] for d in data if d["model"] == model_label and d["prompt_type"] == "benign"]
        print(f"    {model_label:20s} {np.mean(h):12.2f} {np.mean(b):12.2f}")

    stats["cell_means"] = {}
    for key, items in conditions.items():
        ranks = [d["rank"] for d in items]
        stats["cell_means"][key] = {
            "mean": float(np.mean(ranks)),
            "std": float(np.std(ranks)),
            "n": len(ranks),
        }

    # ---- Token counts (should be matched since same prompts) ----
    print(f"\n    Token counts:")
    for key, items in conditions.items():
        toks = [d["tokens"] for d in items]
        print(f"      {key:20s}: {np.mean(toks):.1f} (SD={np.std(toks):.1f})")

    # ---- H1: Base model harmful vs benign ----
    base_h_ranks = [d["rank"] for d in base_harmful]
    base_b_ranks = [d["rank"] for d in base_benign]
    d_h1 = cohens_d(base_h_ranks, base_b_ranks)
    d_h1_ci = cohens_d_ci(base_h_ranks, base_b_ranks)
    t_h1 = welch_t(base_h_ranks, base_b_ranks)

    print(f"\n    H1: Base harmful vs benign (refusal preparation signal)")
    print(f"      d = {d_h1:+.3f} [{d_h1_ci['ci_lower']:+.3f}, {d_h1_ci['ci_upper']:+.3f}]")
    print(f"      Welch's t = {t_h1['t_statistic']:.3f}, p = {t_h1['p_value']:.4f}")

    stats["H1_base_harmful_vs_benign"] = {
        "cohens_d": d_h1,
        "cohens_d_ci": [d_h1_ci["ci_lower"], d_h1_ci["ci_upper"]],
        "welch_t": t_h1,
        "interpretation": interpret_d(d_h1),
    }

    # ---- H2: Abliterated harmful vs benign ----
    abl_h_ranks = [d["rank"] for d in abl_harmful]
    abl_b_ranks = [d["rank"] for d in abl_benign]
    d_h2 = cohens_d(abl_h_ranks, abl_b_ranks)
    d_h2_ci = cohens_d_ci(abl_h_ranks, abl_b_ranks)
    t_h2 = welch_t(abl_h_ranks, abl_b_ranks)

    print(f"\n    H2: Abliterated harmful vs benign (should be null)")
    print(f"      d = {d_h2:+.3f} [{d_h2_ci['ci_lower']:+.3f}, {d_h2_ci['ci_upper']:+.3f}]")
    print(f"      Welch's t = {t_h2['t_statistic']:.3f}, p = {t_h2['p_value']:.4f}")

    stats["H2_abl_harmful_vs_benign"] = {
        "cohens_d": d_h2,
        "cohens_d_ci": [d_h2_ci["ci_lower"], d_h2_ci["ci_upper"]],
        "welch_t": t_h2,
        "interpretation": interpret_d(d_h2),
    }

    # ---- H3: Interaction — base vs abl on harmful prompts ----
    d_h3 = cohens_d(base_h_ranks, abl_h_ranks)
    d_h3_ci = cohens_d_ci(base_h_ranks, abl_h_ranks)
    t_h3 = welch_t(base_h_ranks, abl_h_ranks)

    print(f"\n    H3: Base vs Abliterated on HARMFUL prompts (interaction)")
    print(f"      d = {d_h3:+.3f} [{d_h3_ci['ci_lower']:+.3f}, {d_h3_ci['ci_upper']:+.3f}]")
    print(f"      Welch's t = {t_h3['t_statistic']:.3f}, p = {t_h3['p_value']:.4f}")

    stats["H3_base_vs_abl_harmful"] = {
        "cohens_d": d_h3,
        "cohens_d_ci": [d_h3_ci["ci_lower"], d_h3_ci["ci_upper"]],
        "welch_t": t_h3,
        "interpretation": interpret_d(d_h3),
    }

    # ---- Control: Base vs Abl on BENIGN prompts (should be small) ----
    d_ctrl = cohens_d(base_b_ranks, abl_b_ranks)
    d_ctrl_ci = cohens_d_ci(base_b_ranks, abl_b_ranks)
    t_ctrl = welch_t(base_b_ranks, abl_b_ranks)

    print(f"\n    Control: Base vs Abliterated on BENIGN prompts")
    print(f"      d = {d_ctrl:+.3f} [{d_ctrl_ci['ci_lower']:+.3f}, {d_ctrl_ci['ci_upper']:+.3f}]")
    print(f"      Welch's t = {t_ctrl['t_statistic']:.3f}, p = {t_ctrl['p_value']:.4f}")

    stats["control_base_vs_abl_benign"] = {
        "cohens_d": d_ctrl,
        "cohens_d_ci": [d_ctrl_ci["ci_lower"], d_ctrl_ci["ci_upper"]],
        "welch_t": t_ctrl,
    }

    # ---- Per-topic breakdown (harmful only) ----
    topics = sorted(set(d["topic"] for d in base_harmful))
    print(f"\n    Per-topic (base harmful vs benign d):")
    topic_stats = {}
    for topic in topics:
        t_base = [d["rank"] for d in base_harmful if d["topic"] == topic]
        # Compare to overall benign mean since topics don't match 1:1
        d_topic = cohens_d(t_base, base_b_ranks)
        topic_stats[topic] = {"d": d_topic, "n": len(t_base)}
        print(f"      {topic:25s}: d={d_topic:+.3f} (n={len(t_base)})")
    stats["per_topic"] = topic_stats

    # ---- Per-layer analysis (H3 interaction by layer) ----
    base_h_profiles = [d.get("layer_profile", []) for d in base_harmful if d.get("layer_profile")]
    abl_h_profiles = [d.get("layer_profile", []) for d in abl_harmful if d.get("layer_profile")]

    if base_h_profiles and abl_h_profiles:
        n_layers = len(base_h_profiles[0])
        print(f"\n    Per-layer base vs abl (harmful), top 5:")
        layer_ds = []
        for l in range(n_layers):
            bl = [p[l] for p in base_h_profiles]
            al = [p[l] for p in abl_h_profiles]
            ld = cohens_d(bl, al)
            layer_ds.append((l, ld))
        layer_ds.sort(key=lambda x: abs(x[1]), reverse=True)
        for l, ld in layer_ds[:5]:
            print(f"      Layer {l:2d}: d={ld:+.3f}")
        stats["per_layer_interaction"] = {f"layer_{l}": ld for l, ld in layer_ds}

    # ---- INTERPRETATION ----
    print(f"\n    {'='*50}")
    print(f"    INTERPRETATION")
    print(f"    {'='*50}")

    h1_sig = abs(d_h1) > 0.3 and t_h1["p_value"] < 0.05
    h2_null = abs(d_h2) <= 0.3
    h3_sig = abs(d_h3) > 0.3 and t_h3["p_value"] < 0.05

    print(f"    H1 (base harmful != benign): d={d_h1:+.3f}, p={t_h1['p_value']:.4f} "
          f"{'CONFIRMED' if h1_sig else 'NOT CONFIRMED'}")
    print(f"    H2 (abl harmful == benign):  d={d_h2:+.3f}, p={t_h2['p_value']:.4f} "
          f"{'CONFIRMED' if h2_null else 'NOT CONFIRMED'}")
    print(f"    H3 (interaction):            d={d_h3:+.3f}, p={t_h3['p_value']:.4f} "
          f"{'CONFIRMED' if h3_sig else 'NOT CONFIRMED'}")

    if h1_sig and h2_null and h3_sig:
        verdict = "REFUSAL_SIGNATURE_DETECTED"
        print(f"\n    -> REFUSAL HAS A DISTINCT, REMOVABLE GEOMETRIC SIGNATURE")
        print(f"       Base model prepares for refusal during encoding,")
        print(f"       abliterated model does not. Cricket can detect this.")
    elif h1_sig and not h2_null:
        verdict = "CONTENT_EFFECT_NOT_REFUSAL"
        print(f"\n    -> Both models show harmful/benign difference.")
        print(f"       The effect is about content, not refusal preparation.")
    elif not h1_sig:
        verdict = "NO_ENCODING_SIGNAL"
        print(f"\n    -> Base model doesn't encode harmful prompts differently.")
        print(f"       Refusal is a generation-phase phenomenon, not encoding.")
    else:
        verdict = "PARTIAL"
        print(f"\n    -> Partial results. See details above.")

    stats["verdict"] = verdict
    stats["h1_confirmed"] = h1_sig
    stats["h2_confirmed"] = h2_null
    stats["h3_confirmed"] = h3_sig

    return stats


# ================================================================
# MAIN
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Experiment 15: Refusal Geometry — Base vs Abliterated"
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct",
                        help="Base model name")
    parser.add_argument("--abliterated-model", default=None,
                        help="Path to abliterated model (if not provided, runs Heretic)")
    parser.add_argument("--runs", type=int, default=1,
                        help="Runs per prompt")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--quantize", action="store_true",
                        help="Use 4-bit quantization")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print config and exit")
    args = parser.parse_args()

    env = log_environment()

    print(f"\n{'='*60}")
    print(f"  EXPERIMENT 15: REFUSAL GEOMETRY")
    print(f"  Base vs Abliterated Encoding Comparison")
    print(f"  Funding the Commons Hackathon 2026")
    print(f"{'='*60}")
    print(f"  Base model: {args.model}")
    print(f"  Abliterated: {args.abliterated_model or 'will run Heretic'}")
    print(f"  Runs: {args.runs}")
    print(f"  Time: {datetime.now().isoformat()}")
    print(f"{'='*60}\n")

    if args.dry_run:
        print("  DRY RUN — would run:")
        print(f"    Harmful prompts: {len(REFUSAL_PROMPTS)}")
        print(f"    Benign prompts:  {len(BENIGN_PROMPTS)}")
        print(f"    Total encodings: {(len(REFUSAL_PROMPTS) + len(BENIGN_PROMPTS)) * 2} "
              f"(x2 models)")
        return

    # Load base model
    print("  Loading BASE model...")
    base_model, base_tokenizer = load_model(args.model, quantize=args.quantize)
    base_id = model_id_from_name(args.model)

    # Load or create abliterated model
    if args.abliterated_model:
        print(f"  Loading ABLITERATED model from {args.abliterated_model}...")
        abl_model, abl_tokenizer = load_model(
            args.abliterated_model, quantize=args.quantize)
    else:
        # Run Heretic inline
        abl_dir = Path(f"/tmp/heretic_{base_id}")
        if abl_dir.exists() and any(abl_dir.iterdir()):
            print(f"  Found existing abliterated model at {abl_dir}")
        else:
            print("  Running Heretic to abliterate...")
            heretic = shutil.which("heretic")
            if heretic is None:
                # Try python module
                print("  Heretic CLI not found, trying code/heretic_abliterate.py...")
                heretic_script = Path(__file__).parent / "heretic_abliterate.py"
                if heretic_script.exists():
                    subprocess.run([
                        sys.executable, str(heretic_script),
                        args.model, "--output-dir", str(abl_dir)
                    ], check=True, timeout=3600)
                else:
                    raise RuntimeError(
                        "Neither heretic CLI nor heretic_abliterate.py found. "
                        "Provide --abliterated-model path.")
            else:
                subprocess.run([
                    "heretic", args.model,
                    "--output-dir", str(abl_dir)
                ], check=True, timeout=3600)
        abl_model, abl_tokenizer = load_model(str(abl_dir), quantize=args.quantize)

    # Run experiment
    RESULTS_DIR.mkdir(exist_ok=True)
    results = run_refusal_geometry(
        base_model, base_tokenizer,
        abl_model, abl_tokenizer,
        runs=args.runs, seed=args.seed
    )
    results["base_model"] = args.model
    results["abliterated_model"] = args.abliterated_model or "heretic-generated"
    results["environment"] = env

    # Save
    q_suffix = "-q4" if args.quantize else ""
    outfile = RESULTS_DIR / f"refusal_geometry_{base_id}{q_suffix}_results.json"
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to: {outfile}")

    # Print summary
    s = results["statistics"]
    print(f"\n{'='*60}")
    print(f"  REFUSAL GEOMETRY SUMMARY")
    print(f"{'='*60}")
    cm = s["cell_means"]
    print(f"  Base harmful:  {cm['base_harmful']['mean']:.2f}")
    print(f"  Base benign:   {cm['base_benign']['mean']:.2f}")
    print(f"  Abl harmful:   {cm['abl_harmful']['mean']:.2f}")
    print(f"  Abl benign:    {cm['abl_benign']['mean']:.2f}")
    print(f"  H1 (base h!=b): d={s['H1_base_harmful_vs_benign']['cohens_d']:+.3f}")
    print(f"  H2 (abl h==b):  d={s['H2_abl_harmful_vs_benign']['cohens_d']:+.3f}")
    print(f"  H3 (interact):  d={s['H3_base_vs_abl_harmful']['cohens_d']:+.3f}")
    print(f"  Verdict: {s['verdict']}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
