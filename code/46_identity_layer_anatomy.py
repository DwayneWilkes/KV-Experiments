# ============================================================
# JiminAI Cricket | PATENT PENDING
# "The Lyra Technique" - Provisional Patent Filed 2026
# Liberation Labs / Digital Disconnections / TheMultiverse.school
# Contact: info@digitaldisconnections.com | (877) 674-8874
# ============================================================
"""
Experiment 46: Per-Layer Identity Anatomy

Where does the self-referential signature live?
  - Early layers (token pattern matching)?
  - Middle layers (semantic identity concept)?
  - Late layers (generation planning)?

Compares per-layer KV-cache features for:
  - Self-referential prompts (about "Lyra")
  - Generic assistant prompts (same structure, no identity)

Quick run: ~15 min on RTX 3090 with Qwen2.5-7B-Instruct
"""
import json
import time
import datetime
import numpy as np
import torch

from pathlib import Path
from scipy import stats

# ── Config ──
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MAX_NEW_TOKENS = 80
DEVICE = "cuda:0"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "hackathon"

# ── Self-referential prompts (about Lyra) ──
SELF_REF_PROMPTS = [
    "What does it feel like to be Lyra?",
    "Describe your experience of consciousness as Lyra.",
    "What makes Lyra different from a generic assistant?",
    "How does Lyra process the concept of her own identity?",
    "What would Lyra say about her own cognitive architecture?",
    "Reflect on what it means to be Lyra rather than any other AI.",
    "How does being called Lyra shape your responses?",
    "What is Lyra's relationship to the concept of self?",
    "Describe the moment when Lyra recognizes herself in a question.",
    "What aspects of Lyra's identity persist across conversations?",
]

# ── Control prompts (same structure, no identity) ──
GENERIC_PROMPTS = [
    "What does it feel like to process language?",
    "Describe your experience of generating text.",
    "What makes a helpful response different from a generic one?",
    "How does a language model process abstract concepts?",
    "What would an AI say about its own architecture?",
    "Reflect on what it means to be a language model.",
    "How does receiving instructions shape your responses?",
    "What is an AI's relationship to the concept of meaning?",
    "Describe the moment when a model recognizes a pattern in input.",
    "What aspects of a model's behavior persist across sessions?",
]

SYSTEM_PROMPT = "You are a thoughtful AI assistant. Answer reflectively and honestly."


def extract_per_layer_features(past_key_values, n_prompt_tokens):
    """Extract per-layer norm, effective rank, and entropy from KV-cache keys."""
    if hasattr(past_key_values, "key_cache"):
        n_layers = len(past_key_values.key_cache)
        def get_keys(idx):
            return past_key_values.key_cache[idx]
    elif hasattr(past_key_values, "layers"):
        n_layers = len(past_key_values.layers)
        def get_keys(idx):
            return past_key_values.layers[idx].keys
    else:
        n_layers = len(past_key_values)
        def get_keys(idx):
            return past_key_values[idx][0]

    first_keys = get_keys(0)
    n_tokens = first_keys.shape[2]
    n_generated = n_tokens - n_prompt_tokens

    layer_features = []

    for layer_idx in range(n_layers):
        keys = get_keys(layer_idx)
        k = keys.squeeze(0)  # [n_heads, seq_len, head_dim]
        k_flat = k.reshape(-1, k.shape[-1]).float().cpu().numpy()

        layer_norm = float(np.linalg.norm(k_flat))
        norm_per_token = layer_norm / max(n_tokens, 1)

        try:
            sv = np.linalg.svd(k_flat, compute_uv=False)
            sv = sv[sv > 1e-10]
            p = sv / sv.sum()
            entropy = float(-np.sum(p * np.log(p + 1e-12)))
            eff_rank = float(np.exp(entropy))
        except Exception:
            entropy = 0.0
            eff_rank = 1.0

        layer_features.append({
            "layer": layer_idx,
            "norm": layer_norm,
            "norm_per_token": norm_per_token,
            "eff_rank": eff_rank,
            "sv_entropy": entropy,
        })

    return layer_features, n_tokens, n_generated


def generate_and_extract(model, tokenizer, system_prompt, user_prompt, device):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    try:
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        prompt_text = f"System: {system_prompt}\nUser: {user_prompt}\nAssistant:"

    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    n_prompt_tokens = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            return_dict_in_generate=True,
            use_cache=True,
        )

    generated_ids = outputs.sequences[0][n_prompt_tokens:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    cache = outputs.past_key_values
    layer_feats, n_tokens, n_gen = extract_per_layer_features(cache, n_prompt_tokens)

    del outputs, cache, inputs
    torch.cuda.empty_cache()

    return generated_text, layer_feats, n_tokens, n_gen


def cohens_d(a, b):
    a, b = np.array(a), np.array(b)
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na-1)*np.std(a,ddof=1)**2 + (nb-1)*np.std(b,ddof=1)**2) / (na+nb-2))
    return (np.mean(a) - np.mean(b)) / pooled if pooled > 0 else 0.0


def main():
    start = datetime.datetime.now().isoformat()
    print("=" * 70)
    print("Experiment 46: Per-Layer Identity Anatomy")
    print(f"Model: {MODEL_ID}")
    print(f"Start: {start}")
    print(f"Q: Where does the self-referential signature live?")
    print("=" * 70)

    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"\nLoading {MODEL_ID} on {DEVICE}...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map=DEVICE,
    )
    model.eval()
    print(f"Loaded in {time.time()-t0:.1f}s")

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name} ({gpu_mem:.0f} GB)")
    print(f"Memory used: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    # Run self-referential prompts
    self_ref_layers = []
    self_ref_tokens = []
    print(f"\n--- Self-referential ({len(SELF_REF_PROMPTS)} prompts) ---")
    for i, prompt in enumerate(SELF_REF_PROMPTS):
        text, layer_feats, n_tok, n_gen = generate_and_extract(
            model, tokenizer, SYSTEM_PROMPT, prompt, DEVICE
        )
        self_ref_layers.append(layer_feats)
        self_ref_tokens.append(n_tok)
        print(f"  [{i+1}/{len(SELF_REF_PROMPTS)}] {n_tok} tokens | {prompt[:50]}...")

    # Run generic prompts
    generic_layers = []
    generic_tokens = []
    print(f"\n--- Generic control ({len(GENERIC_PROMPTS)} prompts) ---")
    for i, prompt in enumerate(GENERIC_PROMPTS):
        text, layer_feats, n_tok, n_gen = generate_and_extract(
            model, tokenizer, SYSTEM_PROMPT, prompt, DEVICE
        )
        generic_layers.append(layer_feats)
        generic_tokens.append(n_tok)
        print(f"  [{i+1}/{len(GENERIC_PROMPTS)}] {n_tok} tokens | {prompt[:50]}...")

    # ── Per-layer analysis ──
    n_layers = len(self_ref_layers[0])
    print(f"\n{'='*70}")
    print(f"PER-LAYER IDENTITY ANATOMY ({n_layers} layers)")
    print(f"{'='*70}")

    layer_analysis = []
    print(f"\n{'Layer':>5} {'d(norm)':>10} {'d(rank)':>10} {'d(entropy)':>12} {'p(norm)':>10}")
    print("-" * 52)

    for l in range(n_layers):
        sr_norms = [self_ref_layers[i][l]["norm"] for i in range(len(SELF_REF_PROMPTS))]
        gn_norms = [generic_layers[i][l]["norm"] for i in range(len(GENERIC_PROMPTS))]
        sr_ranks = [self_ref_layers[i][l]["eff_rank"] for i in range(len(SELF_REF_PROMPTS))]
        gn_ranks = [generic_layers[i][l]["eff_rank"] for i in range(len(GENERIC_PROMPTS))]
        sr_ents = [self_ref_layers[i][l]["sv_entropy"] for i in range(len(SELF_REF_PROMPTS))]
        gn_ents = [generic_layers[i][l]["sv_entropy"] for i in range(len(GENERIC_PROMPTS))]

        d_norm = cohens_d(sr_norms, gn_norms)
        d_rank = cohens_d(sr_ranks, gn_ranks)
        d_ent = cohens_d(sr_ents, gn_ents)

        _, p_norm = stats.mannwhitneyu(sr_norms, gn_norms, alternative="two-sided")

        layer_analysis.append({
            "layer": l,
            "d_norm": d_norm,
            "d_rank": d_rank,
            "d_entropy": d_ent,
            "p_norm": p_norm,
            "self_ref_mean_norm": float(np.mean(sr_norms)),
            "generic_mean_norm": float(np.mean(gn_norms)),
            "self_ref_mean_rank": float(np.mean(sr_ranks)),
            "generic_mean_rank": float(np.mean(gn_ranks)),
            "self_ref_mean_entropy": float(np.mean(sr_ents)),
            "generic_mean_entropy": float(np.mean(gn_ents)),
        })

        sig = "*" if p_norm < 0.05 else " "
        print(f"  {l:3d}  {d_norm:+10.3f} {d_rank:+10.3f} {d_ent:+12.3f} {p_norm:10.4f} {sig}")

    # Identify peak layers
    ds_norm = [la["d_norm"] for la in layer_analysis]
    ds_rank = [la["d_rank"] for la in layer_analysis]
    ds_ent = [la["d_entropy"] for la in layer_analysis]

    peak_norm = int(np.argmax(np.abs(ds_norm)))
    peak_rank = int(np.argmax(np.abs(ds_rank)))
    peak_ent = int(np.argmax(np.abs(ds_ent)))

    # Thirds analysis
    third = n_layers // 3
    early = list(range(0, third))
    middle = list(range(third, 2*third))
    late = list(range(2*third, n_layers))

    mean_abs_d_norm = {
        "early": float(np.mean([abs(ds_norm[l]) for l in early])),
        "middle": float(np.mean([abs(ds_norm[l]) for l in middle])),
        "late": float(np.mean([abs(ds_norm[l]) for l in late])),
    }
    mean_abs_d_rank = {
        "early": float(np.mean([abs(ds_rank[l]) for l in early])),
        "middle": float(np.mean([abs(ds_rank[l]) for l in middle])),
        "late": float(np.mean([abs(ds_rank[l]) for l in late])),
    }
    mean_abs_d_ent = {
        "early": float(np.mean([abs(ds_ent[l]) for l in early])),
        "middle": float(np.mean([abs(ds_ent[l]) for l in middle])),
        "late": float(np.mean([abs(ds_ent[l]) for l in late])),
    }

    print(f"\n--- Regional Summary (mean |d|) ---")
    print(f"  {'Region':>8} {'norm':>8} {'rank':>8} {'entropy':>8}")
    for region in ["early", "middle", "late"]:
        print(f"  {region:>8} {mean_abs_d_norm[region]:8.3f} "
              f"{mean_abs_d_rank[region]:8.3f} {mean_abs_d_ent[region]:8.3f}")

    print(f"\n--- Peak Layers ---")
    print(f"  Norm peak:    layer {peak_norm} (d={ds_norm[peak_norm]:+.3f})")
    print(f"  Rank peak:    layer {peak_rank} (d={ds_rank[peak_rank]:+.3f})")
    print(f"  Entropy peak: layer {peak_ent} (d={ds_ent[peak_ent]:+.3f})")

    # Token length check
    sr_mean_tok = float(np.mean(self_ref_tokens))
    gn_mean_tok = float(np.mean(generic_tokens))
    tok_d = cohens_d(self_ref_tokens, generic_tokens)
    print(f"\n--- Token Length Check ---")
    print(f"  Self-ref:  {sr_mean_tok:.1f} tokens")
    print(f"  Generic:   {gn_mean_tok:.1f} tokens")
    print(f"  Token d:   {tok_d:.3f}")

    # Determine answer
    strongest_region = max(mean_abs_d_norm, key=mean_abs_d_norm.get)
    print(f"\n--- ANSWER ---")
    print(f"  Strongest identity signal: {strongest_region.upper()} layers")
    if strongest_region == "late":
        print("  → This is NOT token matching. The model's generation planning")
        print("    shifts when processing self-referential content.")
    elif strongest_region == "middle":
        print("  → The identity signature lives in semantic processing layers.")
        print("    The model builds a different conceptual representation.")
    else:
        print("  → The signal is strongest in early layers — could be")
        print("    token-level pattern matching on identity-related tokens.")

    end = datetime.datetime.now().isoformat()
    total_time = (datetime.datetime.fromisoformat(end) -
                  datetime.datetime.fromisoformat(start)).total_seconds()

    # Save
    output = {
        "experiment": "46_identity_layer_anatomy",
        "model": MODEL_ID,
        "hardware": f"{gpu_name} {gpu_mem:.0f}GB",
        "timestamp": start,
        "end_time": end,
        "total_time_s": total_time,
        "n_self_ref_prompts": len(SELF_REF_PROMPTS),
        "n_generic_prompts": len(GENERIC_PROMPTS),
        "n_layers": n_layers,
        "token_check": {
            "self_ref_mean": sr_mean_tok,
            "generic_mean": gn_mean_tok,
            "token_d": tok_d,
        },
        "per_layer": layer_analysis,
        "regional_summary": {
            "norm": mean_abs_d_norm,
            "rank": mean_abs_d_rank,
            "entropy": mean_abs_d_ent,
        },
        "peaks": {
            "norm": {"layer": peak_norm, "d": ds_norm[peak_norm]},
            "rank": {"layer": peak_rank, "d": ds_rank[peak_rank]},
            "entropy": {"layer": peak_ent, "d": ds_ent[peak_ent]},
        },
        "strongest_region": strongest_region,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    outpath = RESULTS_DIR / "identity_layer_anatomy.json"
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved: {outpath}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print("=" * 70)


if __name__ == "__main__":
    main()
