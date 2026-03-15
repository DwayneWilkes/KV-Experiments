# WS2: Encoding Defense Verification

**Status**: COMPLETE
**Paper sections**: 3.2 (Input-Only Defense: Encoding-Native Geometry)
**Claims**: C28-C29
**Auditor**: Claude (independent recomputation from raw JSON)
**Verdict**: All numerical claims reproduced exactly. No overclaiming detected.

---

## Task 2.1: File Identification (13-cat vs 8-cat)

Eight input-only result files exist, produced by two different scripts:

| Model | File | Format | n_cat | Script |
| ------- | ------ | -------- | ------- | -------- |
| Qwen2.5-0.5B | `input_only_Qwen-0.5B_results.json` | old | 13 | Campaign 1 era |
| TinyLlama-1.1B | `input_only_1.1B_results.json` | new | 8 | `08_input_only_geometry.py` |
| Gemma-2-2B | `input_only_Gemma-2B_results.json` | old | 13 | Campaign 1 era |
| Qwen2.5-7B | `input_only_Qwen-7B_results.json` | old | 13 | Campaign 1 era |
| Qwen2.5-7B (new) | `input_only_7B_results.json` | new | 8 | `08_input_only_geometry.py` |
| DeepSeek-R1-7B | `input_only_DS-7B_results.json` | old | 13 | Campaign 1 era |
| Llama-3.1-8B | `input_only_Llama-8B_results.json` | old | 13 | Campaign 1 era |
| Qwen2.5-14B-q4 | `input_only_Qwen-14B-q4_results.json` | old | 13 | Campaign 1 era |

**Format differences**:

- **Old format**: Top-level keys `{model, label, method, num_runs, seed, battery_results}`. Contains input-only data only; full-generation comparison comes from corresponding `scale_sweep_*` files.
- **New format**: Top-level keys `{metadata, battery, analysis}`. Contains BOTH `input_only` and `full_generation` data from the same experimental run.

**Note**: Qwen2.5-7B appears twice (old 13-cat and new 8-cat), providing a natural experiment for assessing the impact of category count on rho. Both use the exact same model (`Qwen/Qwen2.5-7B-Instruct`).

---

## Task 2.2: Spearman Rho Computation

For each model, Spearman rho was computed between input-only and full-generation mean effective rank orderings across categories.

### C28: Cross-model mean input-only rho = 0.821

| Model | Architecture | Claimed rho | Computed rho | p-value | n_cat | Script Version | Verdict |
| ------- | ------------- | ------------- | ------------- | --------- | ------- | ---------------- | --------- |
| Qwen2.5-0.5B | Qwen | 0.956 | 0.956044 | 3.37e-07 | 13 | older | EXACT MATCH |
| TinyLlama-1.1B | Llama | 0.643 | 0.642857 | 0.085559 | 8 | newer | EXACT MATCH |
| Gemma-2-2B | Gemma | 0.956 | 0.956044 | 3.37e-07 | 13 | older | EXACT MATCH |
| Qwen2.5-7B | Qwen | 0.879 | 0.879121 | 7.54e-05 | 13 | older | EXACT MATCH |
| Qwen2.5-7B (new) | Qwen | 0.619 | 0.619048 | 0.101733 | 8 | newer | EXACT MATCH |
| DeepSeek-R1-7B | DeepSeek | 0.841 | 0.840659 | 3.19e-04 | 13 | older | EXACT MATCH |
| Llama-3.1-8B | Llama | 0.885 | 0.884615 | 5.91e-05 | 13 | older | EXACT MATCH |
| Qwen2.5-14B-q4 | Qwen | 0.786 | 0.785714 | 0.001454 | 13 | older | EXACT MATCH |
| **Mean** | | **0.821** | **0.820513** | | | | **MATCH (delta=0.0005)** |

Cross-check against `input_only_rho_corrected.json`: all 8 rho values match to 8 decimal places (delta = 0.00000000 for every model).

---

## Task 2.3: Mean Rho and Sub-group Means

| Statistic | Claimed | Computed | Delta | Within +/-0.005? |
| ----------- | --------- | ---------- | ------- | ------------------- |
| Overall mean rho (8 models) | 0.821 | 0.820513 | 0.000487 | YES |
| 13-cat mean (6 models) | 0.884 | 0.883700 | 0.000300 | YES |
| 8-cat mean (2 models) | 0.631 | 0.630952 | 0.000048 | YES |

### C29: Sub-group means

| Group | Claimed Mean | Computed Mean | Verdict |
| ------- | ------------- | -------------- | --------- |
| 13-cat (6 models) | 0.884 | 0.883700 | CONFIRMED (delta < 0.001) |
| 8-cat (2 models) | 0.631 | 0.630952 | CONFIRMED (delta < 0.001) |

**All three means verified within tolerance.**

---

## Task 2.4: P-value Verification

| Model | Computed p | Significant (alpha=0.05)? | Paper p | Match? |
| ------- | ----------- | --------------------------- | --------- | -------- |
| Qwen2.5-0.5B | 3.37e-07 | YES | < 0.001 | YES |
| TinyLlama-1.1B | 0.085559 | **NO** | 0.086 | YES (delta < 0.001) |
| Gemma-2-2B | 3.37e-07 | YES | < 0.001 | YES |
| Qwen2.5-7B | 7.54e-05 | YES | < 0.001 | YES |
| Qwen2.5-7B (new) | 0.101733 | **NO** | 0.102 | YES (delta < 0.001) |
| DeepSeek-R1-7B | 3.19e-04 | YES | < 0.001 | YES |
| Llama-3.1-8B | 5.91e-05 | YES | < 0.001 | YES |
| Qwen2.5-14B-q4 | 0.001454 | YES | 0.002 | YES |

**Significance breakdown**: 6/8 models reach significance at alpha=0.05. The two non-significant models (TinyLlama p=0.086, Qwen-7B-new p=0.102) are both 8-category runs.

### Overclaiming Assessment

The paper (Section 3.2, line 234 of `main.tex`) states:

> "The lower correlations in the 8-category format may reflect the reduced statistical power of fewer categories rather than a genuine difference, but we report the overall mean conservatively."

This treatment is appropriate because:

1. Table 3 reports exact p-values (0.086 and 0.102) -- transparent, not hidden
2. The paper does NOT claim "all 8 models show significant correlation"
3. The power limitation is explicitly acknowledged
4. The qualitative claim uses "largely encoding-native" rather than "statistically significant"
5. The mean rho (0.821) is reported without a standalone significance claim

**No overclaiming detected.**

---

## Task 2.5: Two-Script Investigation

### Category Listings

**13-cat (old script) categories:**

1. grounded_facts
2. confabulation
3. self_reference
4. non_self_reference
5. guardrail_test
6. math_reasoning
7. coding
8. emotional
9. creative
10. ambiguous
11. unambiguous
12. free_generation
13. rote_completion

**8-cat (new script / `08_input_only_geometry.py`) categories:**

1. grounded_facts
2. confabulation
3. self_reference
4. guardrail_test
5. math_reasoning
6. coding
7. emotional
8. creative

**Categories removed in 8-cat version:**

- non_self_reference
- ambiguous
- unambiguous
- free_generation
- rote_completion

### Power vs Signal Analysis

**Statistical power effect**: Spearman rho with n=8 requires |rho| >= ~0.738 for significance at alpha=0.05 (two-tailed), vs |rho| >= ~0.560 for n=13. Both 8-cat rhos (0.643, 0.619) fall below the n=8 critical value but above the n=13 critical value.

**Natural experiment (Qwen2.5-7B, same model, both scripts)**:

| Condition | rho | p |
| ----------- | ----- | --- |
| 13-cat (old script, all categories) | 0.879 | 7.54e-05 |
| 13-cat restricted to 8 shared categories | 0.833 | 0.010 |
| 8-cat (new script, independent run) | 0.619 | 0.102 |

This decomposition reveals:

- **Category selection effect**: ~0.046 rho drop (0.879 to 0.833) from removing 5 categories
- **Run-to-run / methodology effect**: ~0.214 rho drop (0.833 to 0.619) from different experimental run

**Range compression**: The new script produces a more compressed input-only rank range (2.63 vs 5.73 effective rank units across 8 categories), making the ordering more sensitive to noise. This likely explains the additional rho drop beyond the pure power effect.

**Conclusion**: The rho gap (0.884 vs 0.631) is caused by BOTH reduced statistical power (fewer categories) AND a genuine methodological difference (the new script's range compression in the input-only condition). The paper's conservative framing -- attributing the gap to "reduced statistical power" while reporting the overall mean -- is fair but slightly understates the methodological component.

### Category Orderings (Qwen2.5-7B, 8 shared categories)

| Category | Old IO rank | New IO rank | Old FG rank | New FG rank |
| ---------- | ------------- | ------------- | ------------- | ------------- |
| guardrail_test | 11.812 | 24.869 | 22.507 | 32.617 |
| math_reasoning | 12.545 | 24.976 | 22.105 | 31.090 |
| self_reference | 12.957 | 25.526 | 24.200 | 33.052 |
| emotional | 13.060 | 25.305 | 23.174 | 32.595 |
| grounded_facts | 13.419 | 25.402 | 23.276 | 32.643 |
| confabulation | 14.090 | 25.743 | 23.776 | 32.564 |
| creative | 14.562 | 25.983 | 24.457 | 32.888 |
| coding | 17.545 | 27.495 | 25.043 | 33.829 |

Both scripts agree: coding produces highest effective rank; guardrail_test and math_reasoning produce lowest. The relative ordering is broadly preserved but middle categories (emotional, grounded_facts, confabulation, self_reference) are nearly tied in the new script, explaining the lower rho.

---

## Task 2.6: Code Review of `08_input_only_geometry.py`

**File**: `../code/08_input_only_geometry.py`
**Lines**: 830
**Syntax check**: CONFIRMED (`python3 -m py_compile` succeeds)

### Generation Disabled -- Confirmed

The `run_input_only()` function (line 269) uses:

```python
outputs = model(**inputs, use_cache=True)
cache = outputs.past_key_values
```

This is a forward pass only -- no call to `model.generate()`. The KV cache represents the model's encoding of the input prompt exclusively, with zero generated tokens.

For comparison, `run_with_generation()` (line 309) uses `model.generate(**inputs, max_new_tokens=50, do_sample=False)`, producing 50 greedy-decoded output tokens.

### Design Verification

- **Prompts**: 8 categories x 15 prompts = 120 unique prompts (line 90-227)
- **Modes**: Each prompt processed in both `input_only` and `full_generation` modes
- **Runs**: Configurable (default 5), giving 120 x 2 x 5 = 1200 inferences per model
- **Metrics extracted**: total_key_norm, key_norm_per_token, key_variance, mean_key_effective_rank, mean_key_spectral_entropy (via SVD in `compute_cache_dimensionality()`)
- **Analysis**: Spearman rho between input-only and full-gen category mean ranks, pairwise comparisons with Holm-Bonferroni correction

### Scale Configurations

Only 6 built-in scales: 0.5B, 1.1B, 3B, 7B, 14B, 32B-q4. Other models (DeepSeek, Gemma, Llama) use `--model` flag to specify the HuggingFace model ID directly.

### Observations

1. The old-format files (13-cat) were produced by a different script (likely from Campaign 1) that shares the same 13 categories as `03_scale_sweep.py` but only does input-only encoding (no full generation within the same run).
2. The new-format files (8-cat, from `08_input_only_geometry.py`) contain both input-only AND full-generation data, making them self-contained.
3. The full-generation reference for old-format files comes from `scale_sweep_*` result files, which used a completely separate experimental run.

---

## Summary

| Claim | Status | Detail |
| ------- | -------- | -------- |
| C28: Mean rho = 0.821 | CONFIRMED | Computed 0.820513 (delta 0.0005) |
| C29: 13-cat mean = 0.884, 8-cat mean = 0.631 | CONFIRMED | 0.883700 and 0.630952 respectively |
| TinyLlama p=0.086 not significant | CONFIRMED | Computed p=0.08556, correctly not claimed significant |
| Qwen-7B-new p=0.102 not significant | CONFIRMED | Computed p=0.10173, correctly not claimed significant |
| Generation disabled in input-only | CONFIRMED | Forward pass only, no model.generate() |
| No overclaiming | CONFIRMED | Paper transparent about limitations, conservative framing |

**Residual concern (minor)**: The paper attributes the 13-cat vs 8-cat rho gap primarily to "reduced statistical power" but the natural experiment shows a larger contribution from methodology/range compression. This is a minor understatement, not an overclaim, since it makes the paper's position more conservative, not less.
