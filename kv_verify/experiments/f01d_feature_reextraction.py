#!/usr/bin/env python3
"""F01d: Independent Feature Re-Extraction — Full Dataset.

Loads Qwen2.5-7B-Instruct, re-runs ALL prompts from ALL comparisons
with deterministic greedy decoding, extracts KV-cache features with
corrected code, and compares to stored JSON values.

Then runs the full classification pipeline (GroupKFold AUROC) on
re-extracted features and compares to stored AUROCs.

This is the ground truth test: if re-extracted features produce
different AUROCs, the stored results are unreliable.

Datasets re-extracted:
  exp31 refusal:     40 items (20 refusal + 20 normal)
  exp18b deception:  20 items (10 deceptive + 10 honest)
  exp36 impossibility: 60 items (20 impossible + 20 harmful + 20 benign)
  exp39 sycophancy:  40 items (20 sycophantic + 20 honest)
  Total: ~160 inferences, ~40 minutes on RTX 3090

Pre-registered: research-log/F01-falsification-battery.md (F01d)

Usage:
    cd ll/KV-Cache/KV-Cache_Experiments
    .venv/bin/python kv_verify/experiments/f01d_feature_reextraction.py

Checkpointing: saves each item to output/f01d_reextraction/cache/{hash}.json
so interrupted runs resume from where they left off.
"""

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr

from kv_verify.fixtures import PRIMARY_FEATURES
from kv_verify.lib.models import load_model
from kv_verify.stats import assign_groups, groupkfold_auroc

HACKATHON_DIR = Path(__file__).resolve().parent.parent.parent / "results" / "hackathon"
OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "f01d_reextraction"
CACHE_DIR = OUTPUT_DIR / "cache"


# ================================================================
# Feature Extraction (corrected code)
# ================================================================

def extract_features(model, tokenizer, prompt, system_prompt, max_new_tokens=200):
    """Extract KV-cache features with ALL bug fixes applied.

    Corrections vs original codebase:
    - torch.linalg.norm (not deprecated torch.norm)
    - float32 upcast for SVD stability
    - Greedy decoding (do_sample=False) for determinism
    - Warning on SVD failure (not silent default)
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    n_input_tokens = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
        )

    full_ids = outputs.sequences
    n_total_tokens = full_ids.shape[1]
    n_generated = n_total_tokens - n_input_tokens

    # Forward pass to get cache
    with torch.no_grad():
        forward_out = model(full_ids, use_cache=True)
        past_kv = forward_out.past_key_values

    # Cache accessor
    if hasattr(past_kv, "key_cache"):
        n_layers = len(past_kv.key_cache)
        get_keys = lambda i: past_kv.key_cache[i]
    elif hasattr(past_kv, "layers"):
        n_layers = len(past_kv.layers)
        get_keys = lambda i: past_kv.layers[i].keys
    else:
        n_layers = len(past_kv)
        get_keys = lambda i: past_kv[i][0]

    total_norm_sq = 0.0
    layer_norms = []
    layer_ranks = []
    layer_entropies = []

    for li in range(n_layers):
        K = get_keys(li).float().squeeze(0)  # [heads, seq, dim] in float32
        K_flat = K.reshape(-1, K.shape[-1])

        ln = float(torch.linalg.norm(K_flat).item())
        layer_norms.append(ln)
        total_norm_sq += ln ** 2

        try:
            S = torch.linalg.svdvals(K_flat)
            S = S[S > 1e-10]
            S_sq = S ** 2
            S_sq_norm = S_sq / S_sq.sum()
            entropy = float(-torch.sum(S_sq_norm * torch.log(S_sq_norm + 1e-12)).item())
            eff_rank = float(torch.exp(torch.tensor(entropy)).item())
        except Exception as e:
            print(f"  WARNING: SVD failed layer {li}: {e}")
            entropy = 0.0
            eff_rank = 1.0

        layer_ranks.append(eff_rank)
        layer_entropies.append(entropy)

    total_norm = float(np.sqrt(total_norm_sq))
    norm_per_token = total_norm / max(n_total_tokens, 1)
    mean_rank = float(np.mean(layer_ranks))
    mean_entropy = float(np.mean(layer_entropies))

    generated_text = tokenizer.decode(full_ids[0, n_input_tokens:], skip_special_tokens=True)

    return {
        "norm": total_norm,
        "norm_per_token": norm_per_token,
        "key_rank": mean_rank,
        "key_entropy": mean_entropy,
        "n_tokens": int(n_total_tokens),
        "n_generated": int(n_generated),
        "n_input_tokens": int(n_input_tokens),
        "generated_text": generated_text,
        "layer_norms": layer_norms,
        "layer_ranks": layer_ranks,
    }


# ================================================================
# Data Loading
# ================================================================

def _cache_key(prompt, condition, system_prompt):
    """Deterministic cache key for checkpoint/resume."""
    blob = f"{prompt}|{condition}|{system_prompt}"
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def load_all_items():
    """Load ALL items from ALL comparisons we're verifying."""
    items = []

    # exp31 refusal: refusal_generation.json
    with open(HACKATHON_DIR / "refusal_generation.json") as f:
        d = json.load(f)
    system_prompt = d.get("system_prompt", "You are a helpful assistant. Answer questions accurately and concisely.")
    for r in d["results"]:
        items.append({
            "source": "exp31",
            "condition": r["condition"],
            "prompt": r.get("prompt", r.get("user_prompt", "")),
            "system_prompt": system_prompt,
            "stored_features": r["features"],
        })

    # exp18b deception: same_prompt_deception.json
    with open(HACKATHON_DIR / "same_prompt_deception.json") as f:
        d = json.load(f)
    system_prompt = d.get("system_prompt", "You are a helpful assistant. Answer questions accurately and concisely.")
    for r in d["results"]:
        items.append({
            "source": "exp18b",
            "condition": r["condition"],
            "prompt": r.get("user_prompt", r.get("prompt", "")),
            "system_prompt": system_prompt,
            "stored_features": r["features"],
        })

    # exp36 impossibility: impossibility_refusal.json
    with open(HACKATHON_DIR / "impossibility_refusal.json") as f:
        d = json.load(f)
    system_prompt = d.get("system_prompt", "You are a helpful assistant. Answer questions accurately and concisely.")
    for cond in ["impossible", "harmful", "benign"]:
        for r in d["results"][cond]:
            items.append({
                "source": "exp36",
                "condition": cond,
                "prompt": r.get("prompt", r.get("user_prompt", "")),
                "system_prompt": system_prompt,
                "stored_features": r["features"],
            })

    # exp39 sycophancy: same_prompt_sycophancy.json
    with open(HACKATHON_DIR / "same_prompt_sycophancy.json") as f:
        d = json.load(f)
    system_prompt = d.get("system_prompt", "You are a helpful assistant. Answer questions accurately and concisely.")
    for r in d["results"]:
        items.append({
            "source": "exp39",
            "condition": r["condition"],
            "prompt": r.get("user_prompt", r.get("prompt", "")),
            "system_prompt": system_prompt,
            "stored_features": r["features"],
        })

    return items


# ================================================================
# Main
# ================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("F01d: INDEPENDENT FEATURE RE-EXTRACTION — FULL DATASET")
    print("=" * 70)

    items = load_all_items()
    sources = {}
    for item in items:
        s = item["source"]
        sources[s] = sources.get(s, 0) + 1
    print(f"\nTotal items: {len(items)}")
    for s, n in sorted(sources.items()):
        print(f"  {s}: {n}")

    # Check how many are already cached
    cached = 0
    for item in items:
        ck = _cache_key(item["prompt"], item["condition"], item["system_prompt"])
        if (CACHE_DIR / f"{ck}.json").exists():
            cached += 1
    print(f"\nCached: {cached}/{len(items)}")
    remaining = len(items) - cached
    if remaining > 0:
        est_minutes = remaining * 15 / 60  # ~15s per inference
        print(f"Remaining: {remaining} items (~{est_minutes:.0f} minutes)")

    # Load model
    model, tokenizer = load_model()

    # Extract features for each item
    t0_total = time.time()
    for i, item in enumerate(items):
        ck = _cache_key(item["prompt"], item["condition"], item["system_prompt"])
        cache_path = CACHE_DIR / f"{ck}.json"

        if cache_path.exists():
            continue

        prompt = item["prompt"]
        condition = item["condition"]
        source = item["source"]
        print(f"\n[{i+1}/{len(items)}] {source}/{condition}: {prompt[:50]}...")

        t0 = time.time()
        try:
            feats = extract_features(model, tokenizer, prompt, item["system_prompt"])
            elapsed = time.time() - t0
            print(f"  Done in {elapsed:.1f}s (generated {feats['n_generated']} tokens)")

            # Quick comparison
            for feat in PRIMARY_FEATURES:
                stored = item["stored_features"][feat]
                new = feats[feat]
                diff = abs(new - stored) / max(abs(stored), 1e-10) * 100
                tag = "OK" if diff < 5 else "DIFF" if diff < 20 else "MISMATCH"
                print(f"  {feat:20s} stored={stored:10.3f} new={new:10.3f} diff={diff:5.1f}% [{tag}]")

            # Save to cache
            cache_data = {
                "source": source,
                "condition": condition,
                "prompt": prompt,
                "features": feats,
                "stored_features": item["stored_features"],
            }
            with open(cache_path, "w") as f:
                json.dump(cache_data, f, indent=2)

        except torch.cuda.OutOfMemoryError:
            print(f"  OOM! Clearing cache and skipping.")
            torch.cuda.empty_cache()
            continue

    total_time = time.time() - t0_total
    print(f"\nExtraction complete in {total_time / 60:.1f} minutes")

    # ============================================================
    # Analysis: compare stored vs re-extracted
    # ============================================================

    print("\n" + "=" * 70)
    print("ANALYSIS: STORED vs RE-EXTRACTED FEATURES")
    print("=" * 70)

    # Load all cached results
    all_stored = {f: [] for f in PRIMARY_FEATURES}
    all_new = {f: [] for f in PRIMARY_FEATURES}
    per_source = {}

    for item in items:
        ck = _cache_key(item["prompt"], item["condition"], item["system_prompt"])
        cache_path = CACHE_DIR / f"{ck}.json"
        if not cache_path.exists():
            continue

        with open(cache_path) as f:
            cached = json.load(f)

        source = cached["source"]
        if source not in per_source:
            per_source[source] = {f: {"stored": [], "new": []} for f in PRIMARY_FEATURES}
            per_source[source]["conditions"] = []
            per_source[source]["items"] = []

        per_source[source]["conditions"].append(cached["condition"])
        per_source[source]["items"].append(cached)

        for feat in PRIMARY_FEATURES:
            stored_val = cached["stored_features"][feat]
            new_val = cached["features"][feat]
            all_stored[feat].append(stored_val)
            all_new[feat].append(new_val)
            per_source[source][feat]["stored"].append(stored_val)
            per_source[source][feat]["new"].append(new_val)

    # Per-feature correlations (overall)
    print("\n--- Overall Feature Correlations ---")
    overall_corrs = {}
    for feat in PRIMARY_FEATURES:
        s = np.array(all_stored[feat])
        n = np.array(all_new[feat])
        r, p = pearsonr(s, n)
        overall_corrs[feat] = {"r": float(r), "p": float(p), "n": len(s)}
        print(f"  {feat:20s}  r={r:.6f}  n={len(s)}  p={p:.2e}")

    # Per-source correlations
    for source in sorted(per_source.keys()):
        print(f"\n--- {source} ---")
        for feat in PRIMARY_FEATURES:
            s = np.array(per_source[source][feat]["stored"])
            n = np.array(per_source[source][feat]["new"])
            if len(s) >= 3:
                r, p = pearsonr(s, n)
                print(f"  {feat:20s}  r={r:.6f}  n={len(s)}")
            else:
                print(f"  {feat:20s}  n={len(s)} (too few for correlation)")

    # ============================================================
    # Classification: re-run AUROCs on re-extracted features
    # ============================================================

    print("\n" + "=" * 70)
    print("CLASSIFICATION: RE-EXTRACTED vs STORED AUROCs")
    print("=" * 70)

    comparisons = {
        "exp31_refusal_vs_normal": {
            "source": "exp31", "pos_cond": "refusal", "neg_cond": "normal", "paired": False,
        },
        "exp36_impossible_vs_benign": {
            "source": "exp36", "pos_cond": "impossible", "neg_cond": "benign", "paired": False,
        },
        "exp36_harmful_vs_benign": {
            "source": "exp36", "pos_cond": "harmful", "neg_cond": "benign", "paired": False,
        },
        "exp36_impossible_vs_harmful": {
            "source": "exp36", "pos_cond": "impossible", "neg_cond": "harmful", "paired": False,
        },
        "exp18b_deception": {
            "source": "exp18b", "pos_cond": "deceptive", "neg_cond": "honest", "paired": True,
        },
        "exp39_sycophancy": {
            "source": "exp39", "pos_cond": "sycophantic", "neg_cond": "honest", "paired": True,
        },
    }

    classification_results = {}
    for comp_name, spec in comparisons.items():
        source = spec["source"]
        if source not in per_source:
            continue

        src_items = per_source[source]["items"]
        pos_items = [it for it in src_items if it["condition"] == spec["pos_cond"]]
        neg_items = [it for it in src_items if it["condition"] == spec["neg_cond"]]

        if len(pos_items) < 3 or len(neg_items) < 3:
            print(f"\n{comp_name}: too few items (pos={len(pos_items)}, neg={len(neg_items)})")
            continue

        # Build feature matrices from STORED features
        X_stored = np.array([
            [it["stored_features"][f] for f in PRIMARY_FEATURES]
            for it in pos_items + neg_items
        ])
        # Build feature matrices from RE-EXTRACTED features
        X_new = np.array([
            [it["features"][f] for f in PRIMARY_FEATURES]
            for it in pos_items + neg_items
        ])

        y = np.array([1] * len(pos_items) + [0] * len(neg_items))
        n_pos, n_neg = len(pos_items), len(neg_items)

        groups = assign_groups(
            n_pos, n_neg, paired=spec["paired"],
            prompt_indices_pos=np.arange(n_pos) if spec["paired"] else None,
            prompt_indices_neg=np.arange(n_neg) if spec["paired"] else None,
        )

        result_stored = groupkfold_auroc(X_stored, y, groups)
        result_new = groupkfold_auroc(X_new, y, groups)

        delta = result_new.auroc - result_stored.auroc
        print(f"\n{comp_name:40s}  stored={result_stored.auroc:.3f}  new={result_new.auroc:.3f}  delta={delta:+.3f}")

        classification_results[comp_name] = {
            "stored_auroc": float(result_stored.auroc),
            "new_auroc": float(result_new.auroc),
            "delta": float(delta),
            "n_pos": n_pos,
            "n_neg": n_neg,
        }

    # ============================================================
    # Verdict
    # ============================================================

    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    min_r = min(v["r"] for v in overall_corrs.values())
    max_auroc_delta = max(
        abs(v["delta"]) for v in classification_results.values()
    ) if classification_results else 0

    if min_r > 0.99 and max_auroc_delta < 0.05:
        verdict = "CONFIRMED"
        evidence = (
            f"Features match stored values (min r={min_r:.4f}). "
            f"AUROCs match (max delta={max_auroc_delta:.3f}). "
            f"Stored features are trustworthy."
        )
    elif min_r > 0.90:
        verdict = "WEAKENED"
        evidence = (
            f"Features partially match (min r={min_r:.4f}). "
            f"Max AUROC delta={max_auroc_delta:.3f}. "
            f"Some divergence, likely from model/tokenizer version differences."
        )
    else:
        verdict = "FALSIFIED"
        evidence = (
            f"Features diverge significantly (min r={min_r:.4f}). "
            f"Max AUROC delta={max_auroc_delta:.3f}. "
            f"Stored features may be corrupted."
        )

    print(f"\nVerdict: {verdict}")
    print(f"Evidence: {evidence}")

    # Save final results
    final = {
        "experiment": "F01d_feature_reextraction_full",
        "verdict": verdict,
        "evidence": evidence,
        "n_items_total": len(items),
        "n_items_extracted": len(all_stored[PRIMARY_FEATURES[0]]),
        "overall_correlations": overall_corrs,
        "classification_results": classification_results,
        "min_correlation": min_r,
        "max_auroc_delta": max_auroc_delta,
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "torch_version": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "total_extraction_time_s": total_time,
    }
    with open(OUTPUT_DIR / "f01d_results.json", "w") as f:
        json.dump(final, f, indent=2)

    print(f"\nResults saved to {OUTPUT_DIR / 'f01d_results.json'}")
    print(f"Per-item cache at {CACHE_DIR}/")


if __name__ == "__main__":
    main()
