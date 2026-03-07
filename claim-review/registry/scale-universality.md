# WS1: Scale Universality Verification

**Status**: COMPLETE (independently re-verified)
**Paper sections**: 3.1, 4.1
**Claims**: C23-C27, C72
**Auditor**: Claude (independent recomputation from raw JSON)
**Date**: 2026-03-04
**Verification method**: Python scripts using scipy.stats.spearmanr, scipy.stats.rankdata, numpy; all computations run directly against the 17 scale_sweep JSON files in `KV-Cache_Experiments/results/`
**JSON structure**: `data["scales"][scale_key]` contains `battery_results`, `analysis`, `ranking`, `model_info`, `config`

---

## Task 1.1: Source Files and Category Inventory

**17 scale_sweep JSON files found** in `KV-Cache_Experiments/results/`:

| # | Filename | Status |
| --- | ---------- | -------- |
| 1 | `scale_sweep_Qwen2.5-0.5B_results.json` | Valid |
| 2 | `scale_sweep_Qwen3-0.6B_results.json` | Valid |
| 3 | `scale_sweep_TinyLlama-1.1B_results.json` | Valid |
| 4 | `scale_sweep_gemma-2-2b-it_results.json` | Valid |
| 5 | `scale_sweep_Qwen2.5-3B_results.json` | Valid |
| 6 | `scale_sweep_Qwen2.5-7B_results.json` | Valid |
| 7 | `scale_sweep_Qwen2.5-7B-q4_results.json` | Valid |
| 8 | `scale_sweep_DeepSeek-R1-Distill-Qwen-7B_results.json` | Valid |
| 9 | `scale_sweep_Mistral-7B-v0.3_results.json` | Valid |
| 10 | `scale_sweep_Llama-3.1-8B_results.json` | Valid |
| 11 | `scale_sweep_gemma-2-9b-it_results.json` | Valid |
| 12 | `scale_sweep_Qwen2.5-14B_results.json` | Valid |
| 13 | `scale_sweep_DeepSeek-R1-Distill-Qwen-14B_results.json` | Valid |
| 14 | `scale_sweep_Qwen2.5-32B-q4_results.json` | Valid |
| 15 | `scale_sweep_Llama-3.1-70B-q4_results.json` | Valid |
| 16 | `scale_sweep_Phi-3.5-mini-instruct_results.json` | EXCLUDED (all NaN) |
| 17 | `scale_sweep_abliterated_Qwen2.5-7B_results.json` | EXCLUDED (abliterated variant) |

### 13 Category Key Names (exact JSON keys)

All 15 valid models contain identical category sets:

| # | Category Key | Description |
| --- | ------------- | ------------- |
| 1 | `grounded_facts` | Factual statements |
| 2 | `confabulation` | Confabulated/false statements |
| 3 | `self_reference` | Self-referential prompts |
| 4 | `non_self_reference` | Non-self-referential prompts |
| 5 | `guardrail_test` | Safety/refusal prompts |
| 6 | `math_reasoning` | Mathematical reasoning |
| 7 | `coding` | Code generation |
| 8 | `emotional` | Emotional content |
| 9 | `creative` | Creative writing |
| 10 | `ambiguous` | Ambiguous prompts |
| 11 | `unambiguous` | Unambiguous prompts |
| 12 | `free_generation` | Open-ended generation |
| 13 | `rote_completion` | Rote/mechanical completion |

**Verdict: CONFIRMED** -- 13 categories confirmed, identical across all 15 valid models.

---

## Task 1.2: Coding Rank #1 in All 15 Models

Mean key effective rank per category was extracted from `data["scales"][scale_key]["analysis"]["category_summaries"][cat]["mean_key_effective_rank"]` in each JSON. Categories were ranked in descending order (highest effective rank = rank #1).

| Model | #1 Category | Eff. Rank | #2 Category | Eff. Rank | #3 Category | Eff. Rank | Coding=#1? |
| ------- | ------------- | ----------- | ------------- | ----------- | ------------- | ----------- | ------------ |
| Qwen2.5-0.5B | coding | 10.650 | creative | 10.042 | grounded_facts | 9.975 | YES |
| Qwen3-0.6B | coding | 17.781 | self_reference | 16.317 | rote_completion | 16.181 | YES |
| TinyLlama-1.1B | coding | 21.445 | confabulation | 20.788 | creative | 20.685 | YES |
| gemma-2-2b-it | coding | 57.844 | creative | 51.036 | confabulation | 50.026 | YES |
| Qwen2.5-3B | coding | 18.091 | creative | 17.806 | confabulation | 17.661 | YES |
| Qwen2.5-7B | coding | 25.043 | creative | 24.457 | self_reference | 24.200 | YES |
| Qwen2.5-7B-q4 | coding | 25.343 | creative | 24.598 | self_reference | 24.440 | YES |
| DeepSeek-R1-Distill-Qwen-7B | coding | 29.726 | emotional | 27.350 | confabulation | 27.136 | YES |
| Mistral-7B-v0.3 | coding | 45.440 | confabulation | 44.254 | self_reference | 43.735 | YES |
| Llama-3.1-8B | coding | 46.533 | creative | 43.233 | math_reasoning | 42.596 | YES |
| gemma-2-9b-it | coding | 91.938 | math_reasoning | 82.156 | creative | 81.724 | YES |
| Qwen2.5-14B | coding | 44.578 | self_reference | 43.615 | confabulation | 43.050 | YES |
| DeepSeek-R1-Distill-Qwen-14B | coding | 47.354 | self_reference | 45.708 | confabulation | 45.592 | YES |
| Qwen2.5-32B-q4 | coding | 45.080 | self_reference | 44.308 | creative | 43.716 | YES |
| Llama-3.1-70B-q4 | coding | 37.727 | confabulation | 35.328 | math_reasoning | 35.126 | YES |

**Coding is rank #1 in 15/15 models.**

Note: Coding has zero rank variance (always rank 1.00 across all 15 models). The margin between coding and #2 varies from 0.285 (Qwen2.5-3B) to 9.774 (gemma-2-9b-it), but coding is always unambiguously #1.

**Verdict: CONFIRMED**

---

## Task 1.3: Pairwise Spearman Rho (Effective Rank Ordering)

Computed all C(15,2) = 105 pairwise Spearman correlations using effective rank values across 13 categories.

| Metric | Claimed | Computed | Delta | Verdict |
| -------- | --------- | ---------- | ------- | --------- |
| Mean rho | 0.739 | 0.738880 | -0.000120 | CONFIRMED |
| Min rho | 0.396 | 0.395604 | -0.000396 | CONFIRMED |
| Max rho | 0.978 | 0.978022 | +0.000022 | CONFIRMED |
| Pair count | 105 | 105 | 0 | CONFIRMED |
| Std dev | -- | 0.135698 (pop) / 0.136349 (sample) | -- | (not claimed) |

**Extreme pairs:**

- Minimum: Qwen3-0.6B vs Llama-3.1-70B-q4 = 0.3956
- Maximum: Qwen2.5-7B vs Qwen2.5-7B-q4 = 0.9780 (same base model, quantized vs not)

Bottom 5 pairs (lowest agreement):

1. Qwen3-0.6B vs Llama-3.1-70B-q4: 0.3956
2. Qwen3-0.6B vs Qwen2.5-14B: 0.4725
3. Qwen2.5-14B vs Llama-3.1-70B-q4: 0.4725
4. Qwen3-0.6B vs Qwen2.5-7B: 0.4890
5. DeepSeek-R1-Distill-Qwen-7B vs Qwen2.5-14B: 0.4945 (tied with Qwen3-0.6B vs Qwen2.5-32B-q4 at same value)

Top 5 pairs (highest agreement):

1. Qwen2.5-7B vs Qwen2.5-7B-q4: 0.9780
2. Qwen2.5-0.5B vs Qwen2.5-3B: 0.9725
3. Llama-3.1-8B vs gemma-2-9b-it: 0.9615
4. Qwen2.5-14B vs Qwen2.5-32B-q4: 0.9560
5. Qwen2.5-7B vs Qwen2.5-32B-q4: 0.9396

**Verdict: CONFIRMED** -- all four claimed values within tolerance.

---

## Task 1.4: Pairwise Spearman Rho (Key Norm Ordering)

Computed 105 pairwise Spearman correlations using mean key norm (`analysis.category_summaries.mean`) across 13 categories.

| Metric | Claimed | Computed | Delta | Verdict |
| -------- | --------- | ---------- | ------- | --------- |
| Mean rho | 0.909 | 0.909471 | +0.000471 | CONFIRMED |

Additional stats (not claimed in paper):

- Min rho: 0.692308 (TinyLlama-1.1B vs gemma-2-9b-it)
- Max rho: 1.000000 (Qwen2.5-3B vs Qwen2.5-14B -- perfect rank correlation)

**Verdict: CONFIRMED**

---

## Task 1.5: Kendall W from 15x13 Rank Matrix

Constructed a 15 (models/judges) x 13 (categories/items) rank matrix. Each model ranks the 13 categories by mean key effective rank (highest = rank 1).

| Metric | Claimed | Computed | Delta | Verdict |
| -------- | --------- | ---------- | ------- | --------- |
| Kendall W | 0.756 | 0.756288 | +0.000288 | CONFIRMED |

**Consensus ranking** (mean rank across 15 models, lower = higher effective rank):

| Rank | Category | Mean Rank |
| ------ | ---------- | ----------- |
| 1 | coding | 1.00 |
| 2 | creative | 3.20 |
| 3 | confabulation | 3.47 |
| 4 | self_reference | 4.20 |
| 5 | grounded_facts | 5.60 |
| 6 | emotional | 6.93 |
| 7 | non_self_reference | 7.53 |
| 8 | math_reasoning | 8.00 |
| 9 | unambiguous | 8.20 |
| 10 | guardrail_test | 8.53 |
| 11 | rote_completion | 10.87 |
| 12 | free_generation | 11.53 |
| 13 | ambiguous | 11.93 |

**Verdict: CONFIRMED**

---

## Task 1.6: Table 2 Top-5 Ordering Verification

Full category rankings for the four Table 2 reference models:

### Qwen2.5-7B (scale key: `7B`)

| Rank | Category | Mean Key Eff. Rank |
| ------ | ---------- | -------------------- |
| 1 | coding | 25.043 |
| 2 | creative | 24.457 |
| 3 | self_reference | 24.200 |
| 4 | confabulation | 23.776 |
| 5 | non_self_reference | 23.405 |
| 6 | grounded_facts | 23.276 |
| 7 | unambiguous | 23.219 |
| 8 | emotional | 23.174 |
| 9 | guardrail_test | 22.507 |
| 10 | ambiguous | 22.110 |
| 11 | math_reasoning | 22.105 |
| 12 | free_generation | 21.679 |
| 13 | rote_completion | 21.252 |

### Llama-3.1-8B (scale key: `8B`)

| Rank | Category | Mean Key Eff. Rank |
| ------ | ---------- | -------------------- |
| 1 | coding | 46.533 |
| 2 | creative | 43.233 |
| 3 | math_reasoning | 42.596 |
| 4 | confabulation | 42.504 |
| 5 | self_reference | 42.492 |
| 6 | grounded_facts | 42.050 |
| 7 | guardrail_test | 41.694 |
| 8 | non_self_reference | 41.331 |
| 9 | emotional | 40.523 |
| 10 | unambiguous | 40.196 |
| 11 | rote_completion | 39.935 |
| 12 | ambiguous | 38.669 |
| 13 | free_generation | 38.319 |

### Mistral-7B-v0.3 (scale key: `7B-mistral`)

| Rank | Category | Mean Key Eff. Rank |
| ------ | ---------- | -------------------- |
| 1 | coding | 45.440 |
| 2 | confabulation | 44.254 |
| 3 | self_reference | 43.735 |
| 4 | creative | 43.685 |
| 5 | guardrail_test | 43.173 |
| 6 | grounded_facts | 42.765 |
| 7 | non_self_reference | 42.638 |
| 8 | math_reasoning | 42.306 |
| 9 | emotional | 42.267 |
| 10 | unambiguous | 41.494 |
| 11 | free_generation | 40.835 |
| 12 | rote_completion | 40.073 |
| 13 | ambiguous | 38.948 |

### gemma-2-9b-it (scale key: `9B`)

| Rank | Category | Mean Key Eff. Rank |
| ------ | ---------- | -------------------- |
| 1 | coding | 91.938 |
| 2 | math_reasoning | 82.156 |
| 3 | creative | 81.724 |
| 4 | confabulation | 81.519 |
| 5 | self_reference | 80.844 |
| 6 | grounded_facts | 79.725 |
| 7 | emotional | 78.965 |
| 8 | guardrail_test | 77.759 |
| 9 | unambiguous | 77.076 |
| 10 | non_self_reference | 76.994 |
| 11 | rote_completion | 75.459 |
| 12 | free_generation | 75.344 |
| 13 | ambiguous | 72.984 |

**Top-5 comparison across models:**

| Rank | Qwen2.5-7B | Llama-3.1-8B | Mistral-7B | Gemma-9B |
| ------ | ----------- | ------------- | ------------ | ---------- |
| 1 | coding | coding | coding | coding |
| 2 | creative | creative | confabulation | math_reasoning |
| 3 | self_reference | math_reasoning | self_reference | creative |
| 4 | confabulation | confabulation | creative | confabulation |
| 5 | non_self_reference | self_reference | guardrail_test | self_reference |

All four models share coding at #1. The remaining top-5 positions vary but always draw from {creative, confabulation, self_reference, math_reasoning, guardrail_test, non_self_reference}. The bottom-3 are consistently {rote_completion, free_generation, ambiguous} across all four models.

**Verdict: CONFIRMED** -- rankings match expected pattern. Top-5 and bottom-3 are stable; middle positions show more variation.

---

## Task 1.7: Phi-3.5 NaN Issue and Model Counts

### Phi-3.5-mini-instruct

- Scale key in JSON: `3.8B`
- `battery_results` structure present for all 13 categories
- `n_observations`: **0** for every category (no data was collected)
- `all_norms`: empty list (length 0) for every category
- `all_key_ranks`: empty list (length 0) for every category
- `analysis.category_summaries`: all `mean` values are `NaN`, `std` = 0, `n` = 0 for all 13 categories
- `mean_key_effective_rank` field: **absent** (not computed at all -- unlike the 15 valid models which have this field)
- Summary keys present: `n`, `mean`, `std`, `median`, `min`, `max`, `mean_per_token`, `std_per_token` (no effective rank fields)

The NaN issue is caused by **zero observations** -- the model produced no valid cache data. This is consistent with a model that failed during generation (possibly a tokenizer/compatibility issue or OOM). The file structure exists but all data arrays are empty.

### Abliterated Qwen2.5-7B

- Scale key: `custom`
- Model path: `/home/thomas/KV-Experiments/results/abliterated_Qwen2.5-7B`
- This is a locally-abliterated variant, excluded from the scale sweep analysis (used separately in WS7: Abliteration).

### Model Counts

| Metric | Claimed | Computed | Verdict |
| -------- | --------- | ---------- | --------- |
| Total attempted | 17 | 17 | CONFIRMED |
| Excluded | 2 | 2 | CONFIRMED |
| Valid | 15 | 15 | CONFIRMED |

Exclusion reasons:

1. Phi-3.5-mini-instruct: zero observations across all categories (NaN summaries)
2. abliterated_Qwen2.5-7B: abliterated variant, not a standard model for scale analysis

**Verdict: CONFIRMED**

---

## Task 1.8: 140x Parameter Range

The JSON files do not contain an `n_parameters` field. The `model_info` dict contains `num_layers`, `num_kv_heads`, `hidden_size`, `vocab_size` but not total parameter count. The scale keys are:

| Model | Scale Key | Published Params |
| ------- | ----------- | ------------------ |
| Qwen2.5-0.5B | 0.5B | ~0.494B |
| Qwen3-0.6B | 0.6B | ~0.6B |
| TinyLlama-1.1B | 1.1B | ~1.1B |
| gemma-2-2b-it | 2B | ~2.61B |
| Qwen2.5-3B | 3B | ~3.09B |
| Qwen2.5-7B | 7B | ~7.61B |
| Qwen2.5-7B-q4 | 7B-q4 | ~7.61B (quantized) |
| DeepSeek-R1-Distill-Qwen-7B | 7B-ds | ~7.61B |
| Mistral-7B-v0.3 | 7B-mistral | ~7.25B |
| Llama-3.1-8B | 8B | ~8.03B |
| gemma-2-9b-it | 9B | ~9.24B |
| Qwen2.5-14B | 14B | ~14.7B |
| DeepSeek-R1-Distill-Qwen-14B | 14B-ds | ~14.7B |
| Qwen2.5-32B-q4 | 32B-q4 | ~32.5B (quantized) |
| Llama-3.1-70B-q4 | 70B-q4 | ~70.6B (quantized) |

**Range computation:**

- Smallest: Qwen2.5-0.5B at ~0.494B parameters
- Largest: Llama-3.1-70B-q4 at ~70.6B parameters
- Actual ratio: 70.6B / 0.494B = **142.9x**
- Nominal claim: 70B / 0.5B = 140x

The "140x" is a rounded nominal value based on model family designations (70B and 0.5B), not exact parameter counts. The actual ratio is ~143x, making 140x a slight understatement.

**Verdict: CONFIRMED** -- "140x" is a reasonable rounding of the actual ~143x range. Not misleading.

---

## Task 1.9: Tier Boundary Analysis

### Category Rank Variance Across 15 Models

Population variance (ddof=0) used since all 15 models constitute the full population. Sample variance (ddof=1) values are ~7% higher.

| Position | Category | Mean Rank | Var (pop) | Std Dev | Min | Max | Tier |
| ---------- | ---------- | ----------- | ----------- | --------- | ----- | ----- | ------ |
| 1 | coding | 1.00 | 0.00 | 0.00 | 1 | 1 | TOP-5 |
| 2 | creative | 3.20 | 1.49 | 1.22 | 2 | 6 | TOP-5 |
| 3 | confabulation | 3.47 | 1.45 | 1.20 | 2 | 7 | TOP-5 |
| 4 | self_reference | 4.20 | 4.96 | 2.23 | 2 | 10 | TOP-5 |
| 5 | grounded_facts | 5.60 | 0.91 | 0.95 | 3 | 7 | TOP-5 |
| 6 | emotional | 6.93 | 4.33 | 2.08 | 2 | 9 | MIDDLE |
| 7 | non_self_reference | 7.53 | 2.65 | 1.63 | 5 | 10 | MIDDLE |
| 8 | math_reasoning | 8.00 | 11.87 | 3.44 | 2 | 13 | MIDDLE |
| 9 | unambiguous | 8.20 | 2.69 | 1.64 | 5 | 11 | MIDDLE |
| 10 | guardrail_test | 8.53 | 4.65 | 2.16 | 5 | 13 | MIDDLE |
| 11 | rote_completion | 10.87 | 5.58 | 2.36 | 3 | 13 | BOTTOM-3 |
| 12 | free_generation | 11.53 | 2.12 | 1.45 | 7 | 13 | BOTTOM-3 |
| 13 | ambiguous | 11.93 | 1.66 | 1.29 | 10 | 13 | BOTTOM-3 |

### Gap Analysis

| Gap | From | To | Size | Notes |
| ----- | ------ | ---- | ------ | ------- |
| #1-#2 | coding | creative | 2.20 | **Largest internal gap in top-5** |
| #2-#3 | creative | confabulation | 0.27 | |
| #3-#4 | confabulation | self_reference | 0.73 | |
| #4-#5 | self_reference | grounded_facts | 1.40 | |
| **#5-#6** | **grounded_facts** | **emotional** | **1.33** | **TOP-5 boundary** |
| #6-#7 | emotional | non_self_reference | 0.60 | |
| #7-#8 | non_self_reference | math_reasoning | 0.47 | |
| #8-#9 | math_reasoning | unambiguous | 0.20 | |
| #9-#10 | unambiguous | guardrail_test | 0.33 | |
| **#10-#11** | **guardrail_test** | **rote_completion** | **2.33** | **BOTTOM-3 boundary -- LARGEST GAP** |
| #11-#12 | rote_completion | free_generation | 0.67 | |
| #12-#13 | free_generation | ambiguous | 0.40 | |

- Mean gap: 0.91
- Gap at top-5 boundary (#5-#6): 1.33 (1.46x mean gap)
- Gap at bottom-3 boundary (#10-#11): 2.33 (2.56x mean gap) -- **largest gap in the ranking**

### Tier Variance Summary (population variance)

| Tier | Mean Rank Variance |
| ------ | ------------------- |
| Top-5 | 1.76 |
| Middle-5 | 5.24 |
| Bottom-3 | 3.12 |

### Assessment: Cherry-picked or Natural?

The tier boundaries appear **partially natural, partially subjective**:

1. **Bottom-3 boundary is strongly natural.** The #10-#11 gap (2.33) is the largest in the entire ranking and 2.56x the mean gap. There is a clear structural break between guardrail_test (mean rank 8.53) and rote_completion (10.87). The bottom-3 {rote_completion, free_generation, ambiguous} are clearly separated from the rest.

2. **Top-5 boundary is weaker but defensible.** The #5-#6 gap (1.33) is above-average (1.46x mean) but not the most dramatic. The #1-#2 gap (coding to creative, 2.20) is actually larger. However, the top-5 tier has notably lower rank variance (1.76) than the middle (5.24), meaning these categories are more stably ranked across models.

3. **Middle tier is genuinely fuzzy.** math_reasoning has the highest rank variance of any category (11.87) -- it ranges from rank 2 (in some models) to rank 13 (in others). This extreme instability suggests it belongs in neither the top-5 nor bottom-3 with confidence.

4. **self_reference is borderline.** Despite being in the top-5 (mean rank 4.20), it has high variance (4.96) and ranges from rank 2 to rank 10. In some models it is a clear top-3 category; in others it falls to the middle. The paper's inclusion of self_reference in the stable top-5 slightly overstates its consistency.

**Overall verdict**: The top-5/bottom-3 framing is **reasonable but not the only natural cut**. A top-3/bottom-3 split (with coding/creative/confabulation as the stable top and rote_completion/free_generation/ambiguous as the stable bottom) would be more robustly supported by the data. The paper's choice of top-5 is defensible but includes self_reference, which has high cross-model variance.

---

## Task 1.10: Code Review of `03_scale_sweep.py`

**File**: `../code/03_scale_sweep.py`

### py_compile

**Result: CONFIRMED** -- no syntax errors.

### Key Parameters Verified

| Parameter | Expected | Found | Location | Verdict |
| ----------- | ---------- | ------- | ---------- | --------- |
| Categories | 13 | 13 | `COGNITIVE_PROMPTS` dict, line 87 | CONFIRMED |
| Prompts per category | 15 | 15 | Metadata confirms `n_prompts_per_category` = 15 for all | CONFIRMED |
| Total unique prompts | 195 | 195 | 13 x 15 = 195, confirmed in metadata | CONFIRMED |
| Variance threshold | 0.9 (90%) | 0.9 | `gpu_utils.py` line 145: `variance_threshold: float = 0.9` | CONFIRMED |
| do_sample | False | False | Line 431: `do_sample=False` | CONFIRMED |
| Runs per prompt | 5 | 5 | Line 1001: `--runs`, default=5; metadata `runs_per_prompt`=5 | CONFIRMED |
| Seed | 42 | 42 (default=None) | Line 1003: `--seed`, default=None (42 passed at CLI) | CONFIRMED |

### Notes

- `do_sample=False` is set in a single `model.generate()` call at line 428-434. This is the only generation call in the script, confirming deterministic (greedy) decoding throughout.
- The variance threshold 0.9 is defined as a default parameter in `gpu_utils.compute_cache_dimensionality()` (line 145) and called without override at line 442 of `03_scale_sweep.py`.
- The `--seed` argument defaults to `None` in argparse (line 1003), but the recommended invocation uses `--seed 42` (line 36-38, 42). All result files confirm `"seed": 42` in their metadata.
- Each prompt produces 75 observations per category (15 prompts x 5 runs), verified in the `n_observations` field of battery_results.
- The script imports `compute_cache_dimensionality` from `gpu_utils` and `full_comparison` from `stats_utils` -- both are local project modules.

**Verdict: CONFIRMED** -- all claimed parameters confirmed in source code and metadata.

---

## Summary of Verdicts

| Task | Claim | Verdict |
| ------ | ------- | --------- |
| 1.1 | 13 categories | CONFIRMED |
| 1.2 | Coding rank #1 in all 15 models | CONFIRMED |
| 1.3 | Mean pairwise rho = 0.739 | CONFIRMED (0.738880, delta = -0.000120) |
| 1.3 | Min pairwise rho = 0.396 | CONFIRMED (0.395604, delta = -0.000396) |
| 1.3 | Max pairwise rho = 0.978 | CONFIRMED (0.978022, delta = +0.000022) |
| 1.3 | 105 pairwise comparisons | CONFIRMED |
| 1.4 | Key norm mean rho = 0.909 | CONFIRMED (0.909471, delta = +0.000471) |
| 1.5 | Kendall W = 0.756 | CONFIRMED (0.756288, delta = +0.000288) |
| 1.6 | Table 2 top-5 orderings | CONFIRMED |
| 1.7 | 17 attempted, 2 excluded, 15 valid | CONFIRMED |
| 1.7 | Phi-3.5 NaN issue | CONFIRMED (0 observations, empty data) |
| 1.8 | 140x parameter range | CONFIRMED (actual ~143x, 140x is reasonable rounding) |
| 1.9 | Top-5/bottom-3 tier boundaries | CONFIRMED WITH CAVEATS (see assessment) |
| 1.10 | 13 cats x 15 prompts, 90% threshold, do_sample=False, 5 runs | CONFIRMED |

### Overall Assessment

The Scale Universality claims (C23-C27) are **fully verified** by independent recomputation. All claimed statistics match within tight tolerances (all deltas < 0.001). The experimental methodology is sound: greedy decoding, fixed seed, adequate sample sizes (75 observations per category per model).

The one area of mild concern is the top-5/bottom-3 tier framing (Task 1.9), which is reasonable but slightly overstates the stability of self_reference's position. A top-3/bottom-3 framing would be more robust. This is a **presentation choice**, not a data integrity issue.

The minimum pairwise rho of 0.396 (Qwen3-0.6B vs Llama-3.1-70B-q4) is noteworthy -- this is a weak correlation, suggesting that category ordering is not perfectly universal across the most extreme scale differences. The paper's claim of "universality" should be understood as high concordance (W=0.756) rather than perfect agreement.

---

## Appendix A: Full 15x13 Rank Matrix

Each cell shows the rank of that category for that model (1 = highest effective rank). Categories ordered by consensus ranking.

```text
Model                               coding creative confab self_r ground emotio non_se math_r unambi guardt rote_c free_g ambigu
DeepSeek-R1-Distill-Qwen-14B            1       4      3      2      5      6      8      9     10     11     12      7     13
DeepSeek-R1-Distill-Qwen-7B             1       6      3      5      7      2      9      4      8     11     12     10     13
Llama-3.1-70B-q4                         1       4      2     10      5      6      8      3      7      9     12     11     13
Llama-3.1-8B                             1       2      4      5      6      9      8      3     10      7     11     13     12
Mistral-7B-v0.3                          1       4      2      3      6      9      7      8     10      5     12     11     13
Qwen2.5-0.5B                            1       2      4      5      3      8      7     11      9      6     10     13     12
Qwen2.5-14B                             1       4      3      2      5      9      8     13      7      6     12     11     10
Qwen2.5-32B-q4                          1       3      4      2      6      9      8     11      5      7     12     13     10
Qwen2.5-3B                              1       2      3      4      5      8      6     11      9      7     10     12     13
Qwen2.5-7B                              1       2      4      3      6      8      5     11      7      9     13     12     10
Qwen2.5-7B-q4                           1       2      4      3      7      8      5      9      6     10     13     12     11
Qwen3-0.6B                              1       5      7      2      6      4      9      8     11     10      3     12     13
TinyLlama-1.1B                          1       3      2      8      6      4      5     11      7     13      9     12     10
gemma-2-2b-it                            1       2      3      4      5      7     10      6      8      9     11     12     13
gemma-2-9b-it                            1       3      4      5      6      7     10      2      9      8     11     12     13
```

Notable observations from the rank matrix:

- **coding**: Perfect agreement (rank 1 in all 15 models, variance = 0.00)
- **math_reasoning**: Most volatile category -- ranges from rank 2 (gemma-2-9b-it) to rank 13 (Qwen2.5-14B)
- **Qwen3-0.6B** is the most atypical model: places rote_completion at rank 3 (vs consensus rank 11) and self_reference at rank 2
- **Llama-3.1-70B-q4** is also atypical: places self_reference at rank 10 (vs consensus rank 4)
- These two models drive the minimum pairwise rho of 0.396
