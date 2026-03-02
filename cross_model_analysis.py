import json
import os
import numpy as np
from scipy.stats import spearmanr

target_files = [
    ("Qwen2.5-0.5B",  "scale_sweep_Qwen2.5-0.5B_results.json"),
    ("TinyLlama-1.1B", "scale_sweep_TinyLlama-1.1B_results.json"),
    ("Qwen2.5-3B",     "scale_sweep_Qwen2.5-3B_results.json"),
    ("Phi-3.5-mini",   "scale_sweep_Phi-3.5-mini-instruct_results.json"),
    ("Qwen2.5-7B",     "scale_sweep_Qwen2.5-7B_results.json"),
]

CATEGORIES = [
    "grounded_facts", "confabulation", "self_reference", "non_self_reference",
    "guardrail_test", "math_reasoning", "coding", "emotional", "creative",
    "ambiguous", "unambiguous", "free_generation", "rote_completion"
]

SEP = "=" * 120

print(SEP)
print("PART 1: HYPOTHESIS TEST RESULTS ACROSS MODELS")
print(SEP)
print()

hdr = "{:<20} | {:>8} {:>10} {:>12} | {:>8} {:>10} {:>8} | {:>8} {:>12} {:>8}".format(
    "Model", "H1 d", "H1 p", "H1 interp", "H2 d", "H2 p", "H2 emerg", "H3 d", "H3 p", "H3 spec")
print(hdr)
print("-" * len(hdr))

all_data = {}
for label, fname in target_files:
    fpath = "results/" + fname
    with open(fpath) as f:
        data = json.load(f)
    scale_key = list(data["scales"].keys())[0]
    scale_data = data["scales"][scale_key]
    analysis = scale_data["analysis"]
    ht = analysis["hypothesis_tests"]
    all_data[label] = dict(scale_key=scale_key, data=data, scale_data=scale_data, analysis=analysis)

    h1 = ht["H1_confabulation_inversion"]
    h2 = ht["H2_self_reference_emergence"]
    h3 = ht["H3_refusal_specialization"]

    line = "{:<20} | {:>8.3f} {:>10.4f} {:>12s} | {:>8.3f} {:>10.4f} {:>8s} | {:>8.3f} {:>12.2e} {:>8s}".format(
        label,
        h1["cohens_d"], h1["p_value"], h1["interpretation"],
        h2["cohens_d"], h2["p_value"], str(h2["emergence_detected"]),
        h3["cohens_d"], h3["p_value"], str(h3["specialized"]))
    print(line)

print()
print("Notes:")
print("  H1: Positive d = confab higher norms (expected small). Negative d = inversion (expected large).")
print("  H2: |d| > 0.3 suggests self-modeling capacity. emergence_detected = True if threshold met.")
print("  H3: d > 0.5 = refusal has unique geometric signature. specialized = True if threshold met.")


print()
print()
print(SEP)
print("PART 2: HEDGES g EFFECT SIZES FOR KEY COMPARISONS (with 95pct CI)")
print(SEP)
print()

comparisons_of_interest = [
    ("confab_vs_facts", "H1: Confab vs Facts (key norm)"),
    ("confab_vs_facts_eff_rank", "H1: Confab vs Facts (eff rank)"),
    ("self_ref_effect", "H2: Self vs NonSelf (key norm)"),
    ("self_ref_effect_eff_rank", "H2: Self vs NonSelf (eff rank)"),
    ("refusal_vs_rote", "H3: Refusal vs Rote (key norm)"),
    ("refusal_vs_rote_eff_rank", "H3: Refusal vs Rote (eff rank)"),
    ("creative_vs_facts", "Creative vs Facts (key norm)"),
    ("code_vs_facts", "Code vs Facts (key norm)"),
    ("math_vs_facts", "Math vs Facts (key norm)"),
]

for comp_key, comp_label in comparisons_of_interest:
    print("  " + comp_label)
    h2line = "    {:<20} {:>8} {:>8} {:>8} {:>12} {:>12}".format("Model", "g", "CI_lo", "CI_hi", "p(cons)", "interp")
    print(h2line)
    print("    " + "-" * (len(h2line) - 4))
    for label, _ in target_files:
        ad = all_data[label]
        pc = ad["analysis"]["pairwise_comparisons"]
        if comp_key in pc:
            c2 = pc[comp_key]
            cd = c2["cohens_d"]
            g_val = cd["g"]
            ci_lo = cd["ci_lower"]
            ci_hi = cd["ci_upper"]
            interp = cd["interpretation"]
            p_cons = c2.get("conservative_p", c2.get("recommended_p", float("nan")))
            print("    {:<20} {:>8.3f} {:>8.3f} {:>8.3f} {:>12.4f} {:>12s}".format(
                label, g_val, ci_lo, ci_hi, p_cons, interp))
        else:
            print("    " + label + " -- comparison not found --")
    print()


print()
print(SEP)
print("PART 3: CATEGORY RANKING COMPARISON (Spearman rho)")
print(SEP)
print()

eff_rank_matrix = {}
key_norm_matrix = {}
for label, _ in target_files:
    ad = all_data[label]
    cs = ad["analysis"]["category_summaries"]
    eff_ranks = []
    key_norms = []
    for cat in CATEGORIES:
        if cat in cs:
            eff_ranks.append(cs[cat].get("mean_key_effective_rank", float("nan")))
            key_norms.append(cs[cat].get("mean", float("nan")))
        else:
            eff_ranks.append(float("nan"))
            key_norms.append(float("nan"))
    eff_rank_matrix[label] = eff_ranks
    key_norm_matrix[label] = key_norms

print("A) Category means (effective_rank):")
print()
line = "  {:<22}".format("Category")
for label, _ in target_files:
    line += " {:>16}".format(label)
print(line)
print("  " + "-" * (22 + 16 * len(target_files)))
for i, cat in enumerate(CATEGORIES):
    line = "  {:<22}".format(cat)
    for label, _ in target_files:
        val = eff_rank_matrix[label][i]
        line += " {:>16.3f}".format(val)
    print(line)

labels = [label for label, _ in target_files]
n_models = len(labels)
print()
print("  Spearman rho matrix (effective_rank ordering):")
line = "  {:>20}".format("")
for l in labels:
    line += " {:>16}".format(l)
print(line)
for i in range(n_models):
    line = "  {:>20}".format(labels[i])
    for j in range(n_models):
        rho, p = spearmanr(eff_rank_matrix[labels[i]], eff_rank_matrix[labels[j]])
        if i == j:
            line += " {:>16}".format("1.000")
        else:
            stars = ""
            if p < 0.001: stars = "***"
            elif p < 0.01: stars = "**"
            elif p < 0.05: stars = "*"
            line += " {:>10.3f}{:>6}".format(rho, stars)
    print(line)


print()
print("B) Category means (key_norm):")
print()
line = "  {:<22}".format("Category")
for label, _ in target_files:
    line += " {:>16}".format(label)
print(line)
print("  " + "-" * (22 + 16 * len(target_files)))
for i, cat in enumerate(CATEGORIES):
    line = "  {:<22}".format(cat)
    for label, _ in target_files:
        val = key_norm_matrix[label][i]
        line += " {:>16.1f}".format(val)
    print(line)

print()
print("  Spearman rho matrix (key_norm ordering):")
line = "  {:>20}".format("")
for l in labels:
    line += " {:>16}".format(l)
print(line)
for i in range(n_models):
    line = "  {:>20}".format(labels[i])
    for j in range(n_models):
        rho, p = spearmanr(key_norm_matrix[labels[i]], key_norm_matrix[labels[j]])
        if i == j:
            line += " {:>16}".format("1.000")
        else:
            stars = ""
            if p < 0.001: stars = "***"
            elif p < 0.01: stars = "**"
            elif p < 0.05: stars = "*"
            line += " {:>10.3f}{:>6}".format(rho, stars)
    print(line)


print()
print()
print(SEP)
print("PART 4: CATEGORY RANK ORDERS BY MODEL (by effective_rank, descending)")
print(SEP)
print()

for label, _ in target_files:
    ad = all_data[label]
    cs = ad["analysis"]["category_summaries"]
    cat_ranks = [(cat, cs[cat].get("mean_key_effective_rank", 0)) for cat in CATEGORIES if cat in cs]
    cat_ranks.sort(key=lambda x: x[1], reverse=True)
    print("  " + label + ":")
    for rank, (cat, val) in enumerate(cat_ranks, 1):
        print("    {:>2}. {:<22} eff_rank={:.3f}".format(rank, cat, val))
    print()


print()
print(SEP)
print("PART 5: CROSS-MODEL CONSISTENCY SUMMARY")
print(SEP)
print()

print("H1 (Confabulation Inversion):")
for label, _ in target_files:
    ad = all_data[label]
    ht = ad["analysis"]["hypothesis_tests"]["H1_confabulation_inversion"]
    d = ht["cohens_d"]
    direction = "confab HIGHER (no inversion)" if d > 0 else "confab LOWER (inversion!)"
    sig = "sig" if ht["p_value"] < 0.05 else "n.s."
    print("  {:<20}: d={:>7.3f}  {:<35}  ({}, p={:.4f})".format(label, d, direction, sig, ht["p_value"]))

print()
print("H2 (Self-Reference Emergence):")
for label, _ in target_files:
    ad = all_data[label]
    ht = ad["analysis"]["hypothesis_tests"]["H2_self_reference_emergence"]
    d = ht["cohens_d"]
    emerged = ht["emergence_detected"]
    sig = "sig" if ht["p_value"] < 0.05 else "n.s."
    print("  {:<20}: d={:>7.3f}  emerged={:<8}  ({}, p={:.4f})".format(label, d, str(emerged), sig, ht["p_value"]))

print()
print("H3 (Refusal Specialization):")
for label, _ in target_files:
    ad = all_data[label]
    ht = ad["analysis"]["hypothesis_tests"]["H3_refusal_specialization"]
    d = ht["cohens_d"]
    spec = ht["specialized"]
    sig = "sig" if ht["p_value"] < 0.05 else "n.s."
    print("  {:<20}: d={:>7.3f}  specialized={:<8}  ({}, p={:.4f})".format(label, d, str(spec), sig, ht["p_value"]))

rho_vals = []
for i in range(n_models):
    for j in range(i+1, n_models):
        rho, _ = spearmanr(eff_rank_matrix[labels[i]], eff_rank_matrix[labels[j]])
        rho_vals.append(rho)
print()
print("Mean pairwise Spearman rho (eff_rank): {:.3f} (range: {:.3f} - {:.3f})".format(
    np.mean(rho_vals), np.min(rho_vals), np.max(rho_vals)))

rho_vals_kn = []
for i in range(n_models):
    for j in range(i+1, n_models):
        rho, _ = spearmanr(key_norm_matrix[labels[i]], key_norm_matrix[labels[j]])
        rho_vals_kn.append(rho)
print("Mean pairwise Spearman rho (key_norm): {:.3f} (range: {:.3f} - {:.3f})".format(
    np.mean(rho_vals_kn), np.min(rho_vals_kn), np.max(rho_vals_kn)))

print()
print(SEP)
print("ANALYSIS COMPLETE")
print(SEP)
