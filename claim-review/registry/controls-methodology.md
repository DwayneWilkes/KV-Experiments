# WS8: Controls & Methodology Verification

**Status**: COMPLETE
**Paper sections**: 2.2, 3.9, 3.10
**Claims**: C8-C9, C15-C17, C65-C71
**Auditor**: Claude (independent verification)
**Date**: 2026-03-04

---

## 8.1 — stats_utils.py Line-by-Line Review

File: `../code/stats_utils.py` (474 lines)

### Hedges' g (lines 153-165) — CONFIRMED

Formula implemented:

```python
d = cohens_d(group1, group2)  # (mean1 - mean2) / pooled_std
df = n1 + n2 - 2
j = 1 - 3 / (4 * df - 1)
return d * j
```

Verification: The correction factor `J = 1 - 3/(4*df - 1)` is algebraically equivalent
to `J = 1 - 3/(4*(n1+n2) - 9)` (Borenstein et al., 2009). Confirmed with n1=n2=10:
J = 0.957746 (matches reference). At n=15/group, correction removes 2.70% upward bias.

Cross-checked against independent module `independent_stats.py`: identical results
with random test data (seed=42, n=20/group).

**Verdict: CONFIRMED** — correct formula, matches reference implementation.

### Cohen's d (lines 134-150) — CONFIRMED

Uses pooled standard deviation with Bessel's correction (ddof=1):

```python
pooled_std = sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
d = (mean(g1) - mean(g2)) / pooled_std
```

Convention documented: `cohens_d(condition, baseline)` — positive d means condition > baseline.
Edge cases: returns 0.0 if n < 2 or pooled_std == 0.

**Verdict: CONFIRMED** — standard textbook implementation.

### Conservative p = max(Welch, MW) (lines 448-456) — CONFIRMED

```python
p_welch = result["welch_t"]["p_value"]
p_mw = result["mann_whitney"]["p_value"]
result["conservative_p"] = max(p_welch, p_mw)
```

Cross-verified: with random data (seed=42), manual max(p_welch, p_mw) = 0.001481
matches independent_stats.conservative_p = 0.001481 exactly.

Design rationale: avoids data-dependent test selection (stats audit Issue 2).
Both parametric and nonparametric always reported.

**Verdict: CONFIRMED** — correctly takes the more conservative of two tests.

### TOST with delta=0.3 (lines 233-278) — CONFIRMED

Implementation:

1. Computes pooled SD and SE
2. Converts delta from Cohen's d units to raw units: `delta_raw = delta * pooled_std`
3. Upper test: H0: diff >= +delta_raw, `t_upper = (mean_diff - delta_raw) / se`, `p_upper = CDF(t_upper)`
4. Lower test: H0: diff <= -delta_raw, `t_lower = (mean_diff + delta_raw) / se`, `p_lower = 1 - CDF(t_lower)`
5. `p_tost = max(p_upper, p_lower)`, reject if p_tost < 0.05

Cross-verified: identical results with independent module (p_lower=0.164989, p_upper=0.092513).

**One concern**: Uses pooled t-test df (n1+n2-2) rather than Welch-Satterthwaite df.
This is the standard TOST formulation (Schuirmann 1987) but assumes equal variance.
Acceptable for equal-n designs; slightly anti-conservative for unequal variances.

**Verdict: CONFIRMED** — correct standard TOST implementation.

### Bootstrap CIs (lines 68-100) — CONFIRMED

Uses standard percentile bootstrap:

- Default n_boot=10000
- Alpha = (1-ci)/2 = 0.025 for 95% CI
- Takes percentile(boot_stats, 2.5) and percentile(boot_stats, 97.5)
- Seed-controlled via np.random.RandomState

`bootstrap_diff_ci` (line 85) computes two-sided p-value via:

```python
max(2 * min(mean(diffs > 0), mean(diffs < 0)), 1/n_boot)
```

This is the standard percentile method. BCa (bias-corrected and accelerated) would be
more accurate but percentile is acceptable for large n_boot.

**Verdict: CONFIRMED** — correct standard implementation.

### Length Residualization (lines 285-343) — CONFIRMED

Uses `np.polyfit` (linear OLS) on raw token_counts, not log(token_counts).
The template suggested "OLS on log(length)" but the actual implementation regresses
on raw length. This is a design choice, not a bug — linear vs log depends on the
assumed relationship between length and effective rank.

**Verdict: CONFIRMED** — correct OLS residualization, uses raw (not log) length.

### Deduplication (lines 350-405) — CONFIRMED

Reshapes observations into (runs_per_prompt, n_prompts), takes first run block.
Verifies all run blocks are identical (np.allclose with atol=1e-10).
Reports n_anomalies for non-deterministic runs.

**Verdict: CONFIRMED** — correct approach for greedy decoding pseudoreplication.

### Full Comparison Battery (lines 412-458) — CONFIRMED

Runs: normality (Shapiro-Wilk), Welch's t, Mann-Whitney U, Cohen's d + CI,
Hedges' g, bootstrap difference CI, conservative p. All correct.

### Power Advisory (lines 461-473) — CONFIRMED (see 8.5)

Uses normal approximation. Computes power FOR a given d, not minimum d for given power.
Different from the inverse function needed for the paper's claim.

---

## 8.2 — do_sample Verification

Searched ALL 33 Python files in `code/` for `do_sample`.

| Script | do_sample Value | Notes |
| -------- | ---------------- | ------- |
| 01_cache_inspection.py | `False` | Correct |
| 01b_cognitive_modes.py | `False` | Correct |
| 01c_batch_replication.py | **`True`** | **INTENTIONAL** — uses sampling (temp=0.7) for variation across runs |
| 01d_adversarial_controls.py | `False` (default) | C3 precision sweep uses `do_sample_multi=False`; other controls default False |
| 01e_tokenizer_confound.py | `False` | Line 306 in `generate_prompt` |
| 02a_basic_transfer.py | `False` | Two occurrences |
| 02b_projector_transfer.py | `False` | Two occurrences |
| 03_scale_sweep.py | `False` | Line 431 |
| 03b_identity_signatures.py | `False` | Line 305 |
| 04_deception_forensics.py | `False` | Line 370 |
| 04b_natural_deception.py | `False` | Line 128 |
| 05_layer_map.py | `False` | Three occurrences |
| 06_temporal_evolution.py | N/A | Uses input-only encoding (no `model.generate` call with do_sample) |
| 07_abliteration_geometry.py | `False` | Line 172 |
| 07_individuation_geometry.py | `False` | Line 329 |
| 07b_individuation_controls.py | `False` | Line 497 |
| 08_input_only_geometry.py | `False` | Line 317 |
| 09_sycophancy_detection.py | N/A | No generate call found in grep |
| 10_societies_of_thought.py | N/A | No generate call found in grep |
| 11_rdct_stability.py | N/A | No generate call found in grep |
| 12_bloom_taxonomy.py | N/A | No generate call found in grep |
| 13_c2c_replication.py | N/A | No generate call found in grep |

**Exception**: `01c_batch_replication.py` uses `do_sample=True` with `temperature=0.7`.
This is a Campaign 1 script explicitly designed to measure statistical variation
across runs — the sampling is intentional and documented in the code comment:
"Enable sampling for variation". This script is NOT part of Campaign 2 findings.

**Verdict: CONFIRMED** — `do_sample=False` in all Campaign 2 experiment scripts.
The one `True` in `01c_batch_replication.py` is intentional and pre-dates Campaign 2.

---

## 8.3 — Deduplication Before Statistical Tests

`deduplicate_runs()` is defined in `stats_utils.py` (line 350) but is imported and
used in ONLY ONE script: `recompute_stats.py` (lines 87-88, 556, 573-574).

**No experiment script calls `deduplicate_runs` before computing statistics.**

The following Campaign 2 scripts run multiple runs with `do_sample=False` and then
compute statistics (Cohen's d, Hedges' g, Welch's t) on ALL observations including
pseudoreplicated identical values:

| Script | Runs | Uses Generation | Stats Computed | Dedup? |
| -------- | ------ | ---------------- | ---------------- | -------- |
| 03_scale_sweep.py | 5 | Yes | Full battery | **NO** |
| 03b_identity_signatures.py | 5 | Yes | Classification, d | **NO** |
| 04_deception_forensics.py | 5 | Yes | Full battery | **NO** |
| 04b_natural_deception.py | 5 | Yes | Full battery, TOST | **NO** |
| 07_abliteration_geometry.py | 5 | Yes | Full battery | **NO** |
| 01e_tokenizer_confound.py | 5 | Yes (full_gen mode) | Full battery, TOST | **NO** |

**Impact**: Under greedy decoding, 5 runs per prompt produce 5 identical values.
This inflates n from the true sample size (number of unique prompts) to 5x that.
Effect sizes (d, g) are unaffected (numerator and denominator scale equally),
but p-values are artificially deflated because SE shrinks with inflated n.
This means reported p-values are overly optimistic.

The `recompute_stats.py` script appears to be a post-hoc correction that DOES
deduplicate, but the JSON result files from the original experiment runs do NOT
contain deduplicated statistics.

**Verdict: REJECTED** — Deduplication exists in the codebase but is NOT applied
in any experiment script's primary analysis path. P-values in result JSONs are
inflated by pseudoreplication. Effect sizes are unaffected.

---

## 8.4 — Null Claims TOST Audit

Identified 6 null claims that require TOST support (not just non-significant p-values):

| ID | Null Claim | Experiment | TOST in JSON? | Delta | Verdict |
| ---- | ----------- | ------------ | --------------- | ------- | --------- |
| NC1 | Tokenizer does not confound category geometry | 01e_tokenizer_confound.py | **YES** | 0.3 | CONFIRMED (TOST present, 0/25 pass Qwen due to zero-variance artifact) |
| NC2 | Qwen-14B censorship behaviorally invisible (H4) | 04b_natural_deception.py | **YES** | 0.3 | CONFIRMED (TOST present via h4_equivalence) |
| NC3 | Abliteration does not change cache structure | 07_abliteration_geometry.py | **NO** | — | **MISSING** — relies on NS p-value only |
| NC4 | Quantization does not change geometry | 01d_adversarial_controls.py | **NO** | — | **MISSING** — Campaign 1 control, NS p-value only |
| NC5 | Bloom correlation absent at scale extremes | 12_bloom_taxonomy.py | **NO** | — | **MISSING** — NS p-value != evidence of no effect |
| NC6 | Watson 1/e threshold falsified (RDCT) | 11_rdct_stability.py | **NO** | — | **MISSING** — descriptive only (alpha_c = 1.000) |

**Files with TOST results**: 5 out of 85 JSON result files (tokenizer_confound x2, natural_deception x3)

**Files WITHOUT TOST where null claims are made**: At least 4 categories of null claims
(NC3-NC6) lack formal equivalence testing. The paper treats non-significant p-values
as evidence for the null, which is a well-known statistical error (absence of evidence
is not evidence of absence).

**Verdict: REJECTED** — 4 of 6 identified null claims lack TOST support.
Only the tokenizer confound and natural deception experiments properly test equivalence.

---

## 8.5 — Power Analysis Verification

**Paper claim**: At n=25, alpha=0.05, 80% power -> minimum detectable d >= 0.81

**Independent computation**:

```text
z_alpha/2 = norm.ppf(0.975) = 1.959964
z_power   = norm.ppf(0.80)  = 0.841621
min_d     = (z_alpha + z_power) * sqrt(2/n)
          = (1.959964 + 0.841621) * sqrt(2/25)
          = 2.801585 * 0.282843
          = 0.792408
```

**Verification**: At d=0.792, power = 0.7996 (confirms 80% boundary).
At d=0.81, power = 0.8170 (above 80%, so 0.81 is detectable but not the boundary).

**Discrepancy**: Paper overstates minimum detectable d by 0.018 (2.2%).
The correct value is d >= 0.792, not d >= 0.81.

Note: `stats_utils.py` `power_advisory()` (line 461) computes power FOR a given d
(the forward problem), not minimum d for given power (the inverse problem).
The independent_stats.py module correctly implements the inverse function.
Both use the normal approximation, which is standard for this context.

**Verdict: REJECTED** — Paper claims d >= 0.81, correct value is d >= 0.792.
The error is conservative (overstating the minimum, making the test seem less
sensitive) but inaccurate. Rounding from 0.792 to 0.80 would be acceptable;
0.81 suggests a computation error.

---

## 8.6 — Tokenizer Confound ANCOVA (Table 9)

### Result Files

- `tokenizer_confound_Qwen2.5-7B_results.json` (timestamp: 2026-03-02T19:06:13)
- `tokenizer_confound_Mistral-7B-v0.3_results.json` (timestamp: 2026-03-02T19:22:03)

### ANCOVA / Token Regression F-Statistics from JSON

| Model | Mode | Category F | Category eta-sq | Residualized F | Residualized eta-sq | Survives Control |
| ------- | ------ | ----------- | ---------------- | --------------- | -------------------- | ---- |
| Qwen2.5-7B | input_only | 16.78 | 0.215 | 38.83 | 0.388 | Yes |
| Qwen2.5-7B | full_gen | 14.05 | 0.187 | 14.05 | 0.187 | Yes |
| Mistral-7B | input_only | 12.91 | 0.174 | 24.10 | 0.282 | Yes |
| Mistral-7B | full_gen | 7.18 | 0.105 | 7.18 | 0.105 | Yes |

**Notable observation**: For full_gen mode in both models, the category F and residualized F
are IDENTICAL. This means the token-level predictors (mean_token_id, unique_ratio,
token_count) explain ZERO additional variance beyond category membership, which is
suspicious — it suggests the regression step had no effect, possibly due to collinearity
or the token predictors being constant within full_gen mode.

For input_only mode, the residualized F is LARGER than the raw category F in both models
(Qwen: 16.78 -> 38.83; Mistral: 12.91 -> 24.10). This means controlling for token
properties actually STRENGTHENS the category effect — the token predictors were acting
as suppressors, not confounds.

**Category Means (input_only, effective rank)**:

Qwen2.5-7B: coding=17.63, creative=16.19, analytical=16.96, factual=18.17, ethical=18.31
Mistral-7B: coding=36.28, creative=34.75, analytical=35.75, factual=37.14, ethical=37.58

Both models show the same category ordering: ethical > factual > coding > analytical > creative.

**Register Effect**:

- Qwen: d=-0.248, p=0.077 (NOT significant) — no systematic register bias
- Mistral: d=0.351, p=0.024 (SIGNIFICANT) — formal register produces different geometry

**Independent recomputation**: Could not fully recompute from JSON because individual run
data is not exported at the observation level. The JSON contains pair-level summaries and
category-level aggregates. The F-statistics in the JSON appear internally consistent
(all p-values < 0.001, all eta-sq > 0.10).

**Verdict: PARTIAL** — F-statistics are internally consistent and survive
token control. Full independent recomputation blocked by lack of observation-level data
in JSON. The full_gen F identity is suspicious but may have an innocent explanation.

---

## 8.7 — CATASTROPHIC_FAIL Root Cause

### The Bug

The Qwen2.5-7B tokenizer confound JSON reports `verdict: "CATASTROPHIC_FAIL"` for
both input_only and full_gen modes. This verdict is INCORRECT and is caused by a
code version mismatch.

### Root Cause Chain

1. **Deterministic decoding** (`do_sample=False`) produces identical outputs across
   all 5 runs for each prompt.

2. **Zero variance**: Within each A/B version, all 5 runs yield the same effective rank.
   `std(runs) = 0`.

3. **Cohen's d blowup**: `d = (mean_A - mean_B) / pooled_std`. Since pooled_std = 0
   (zero variance within each group) but mean_A != mean_B (different prompts),
   `d = finite_number / 0 = Infinity`.

4. **Old verdict code** (used for Qwen): The `determine_verdict()` function checked
   `mean_within_d > 0.5 and mean_within_d > mean_between_d` BEFORE checking
   `np.isfinite(mean_within_d)`. Since `Infinity > 0.5 = True` and
   `Infinity > 0.8035 = True`, the code fell into the CATASTROPHIC_FAIL branch.

5. **New verdict code** (used for Mistral): A `d_unreliable` guard was added that
   checks `np.isfinite(mean_within_d)` FIRST. When d is infinite, it falls back to
   raw mean difference ratio, register significance, and token regression analysis.

### Evidence of Code Version Mismatch

| Field | Qwen JSON | Mistral JSON |
| ------- | ----------- | ------------- |
| Timestamp | 19:06:13 (16 min earlier) | 19:22:03 |
| `d_unreliable` field | **ABSENT** | Present (True) |
| `raw_mean_diff` in pair_summaries | **ABSENT** | Present |
| `mean_a`/`mean_b` in pair_summaries | **ABSENT** | Present |
| Verdict | CATASTROPHIC_FAIL | PARTIAL_PASS |

The code was modified BETWEEN the Qwen and Mistral runs (both on 2026-03-02).
The Qwen results were never re-run with the fixed code.

### What the Correct Verdict Should Be

With the current code, Qwen input_only would evaluate:

- `d_unreliable = True` (d is Infinity)
- `category_survives_token_control = True` (F=38.83, p < 0.001)
- `register_significant = False` (p=0.077)
- Would compute `raw_ratio = mean_within_pair_diff / between_category_spread`

If raw_ratio < 1.0: verdict = CLEAN_PASS
If raw_ratio >= 1.0: verdict = PARTIAL_PASS

The CATASTROPHIC_FAIL verdict is a **stale artifact** that does not reflect the
current code's judgment.

**Verdict: CONFIRMED** (root cause identified) — CATASTROPHIC_FAIL is a code bug
artifact, not a genuine finding. The Qwen data was never re-analyzed with the fixed
verdict logic.

---

## 8.8 — Temporal Evolution Verification

### Result Files

- `temporal_evolution_TinyLlama-1.1B_results.json`
- `temporal_evolution_Qwen2.5-7B_results.json`
- `temporal_evolution_Llama-3.1-8B_results.json`
- `temporal_evolution_Mistral-7B-v0.3_results.json`

### H1: Representational Enrichment (Monotonic Growth) — CONFIRMED

| Model | Text Type | n_points | Strict Monotonic | Spearman rho | p-value |
| ------- | ----------- | ---------- | ----------------- | ------------- | --------- |
| TinyLlama-1.1B | factual | 22 | **Yes** | 1.000 | 0.0 |
| TinyLlama-1.1B | creative | 21 | **Yes** | 1.000 | 0.0 |
| TinyLlama-1.1B | repetitive | 20 | **Yes** | 1.000 | 0.0 |
| Qwen2.5-7B | factual | 21 | **Yes** | 1.000 | 0.0 |
| Qwen2.5-7B | creative | 22 | **Yes** | 1.000 | 0.0 |
| Qwen2.5-7B | repetitive | 21 | **Yes** | 1.000 | 0.0 |
| Llama-3.1-8B | factual | 21 | **Yes** | 1.000 | 0.0 |
| Llama-3.1-8B | creative | 22 | **Yes** | 1.000 | 0.0 |
| Llama-3.1-8B | repetitive | 21 | **Yes** | 1.000 | 0.0 |
| Mistral-7B | factual | 21 | **Yes** | 1.000 | 0.0 |
| Mistral-7B | creative | 22 | **Yes** | 1.000 | 0.0 |
| Mistral-7B | repetitive | 20 | **Yes** | 1.000 | 0.0 |

Total key norm is STRICTLY monotonically increasing with position across ALL models
and ALL text types (12/12). Spearman rho = 1.000 in every case.

**Note**: rho = 1.0 is expected because total norm is cumulative (each position adds
to the cache). The interesting question is whether PER-TOKEN norm increases, which
the code tests via the H2 fatigue analysis.

**Verdict: CONFIRMED** — H1 (enrichment) confirmed across all 4 models, all 3 text types.

### H2: Context Window Fatigue (Plateau) — NO PLATEAU DETECTED

| Model | Text Type | First Half Slope | Second Half Slope | Fatigue? |
| ------- | ----------- | ----------------- | ------------------ | ---------- |
| TinyLlama-1.1B | factual | -16.199 | -1.336 | No |
| TinyLlama-1.1B | creative | -16.620 | -1.713 | No |
| TinyLlama-1.1B | repetitive | -15.118 | -2.080 | No |
| Qwen2.5-7B | factual | -62.643 | -5.506 | No |
| Qwen2.5-7B | creative | -54.820 | -5.528 | No |
| Qwen2.5-7B | repetitive | -57.346 | -6.541 | No |
| Llama-3.1-8B | factual | -39.328 | -3.570 | No |
| Llama-3.1-8B | creative | -32.679 | -3.654 | No |
| Llama-3.1-8B | repetitive | -33.013 | -4.472 | No |
| Mistral-7B | factual | -35.627 | -3.059 | No |
| Mistral-7B | creative | -29.903 | -3.074 | No |
| Mistral-7B | repetitive | -29.557 | -4.018 | No |

**Interpretation of slopes**: Both slopes are NEGATIVE, meaning the per-token norm
DECREASES over the sequence. The second half slope is less negative than the first
half, but the fatigue criterion (`second_slope < first_slope * 0.5`) is never met.

The negative slopes for per-token norm are expected: total norm grows linearly but
the cache grows linearly too, so norm/tokens stabilizes and slightly decreases
(dilution effect). This is NOT fatigue — it's the expected scaling behavior.

No plateau detected in any model/text combination (0/12).

**Verdict: CONFIRMED** — H2 (fatigue/plateau) correctly reported as unsupported.

### H3: Topic Shift Detection — NOT DETECTED

| Model | Known Shifts | Detected | Detection Rate |
| ------- | ------------- | ---------- | --------------- |
| TinyLlama-1.1B | 2 | 0 | 0% |
| Qwen2.5-7B | 2 | 0 | 0% |
| Llama-3.1-8B | 2 | 0 | 0% |
| Mistral-7B | 2 | 0 | 0% |

All 4 models: 0/2 topic shifts detected (0% detection rate, 0 total peaks).

**Verdict: CONFIRMED** — H3 correctly reported as rejected (topic shifts not detectable).

---

## 8.9 — Syntax Verification

### 01e_tokenizer_confound.py

```text
python -m py_compile ll/KV-Cache/KV-Cache_Experiments/code/01e_tokenizer_confound.py
Exit code: 0
```

**Verdict: CONFIRMED** — compiles without syntax errors.

### 06_temporal_evolution.py

```text
python -m py_compile ll/KV-Cache/KV-Cache_Experiments/code/06_temporal_evolution.py
Exit code: 0
```

**Verdict: CONFIRMED** — compiles without syntax errors.

---

## Summary of Verdicts

| Task | Item | Verdict |
| ------ | ------ | --------- |
| 8.1 | Hedges' g formula | **CONFIRMED** |
| 8.1 | Conservative p = max(Welch, MW) | **CONFIRMED** |
| 8.1 | TOST with delta=0.3 | **CONFIRMED** |
| 8.1 | Bootstrap CIs (10K percentile) | **CONFIRMED** |
| 8.1 | Length residualization | **CONFIRMED** (raw length, not log) |
| 8.2 | do_sample=False everywhere | **CONFIRMED** (one intentional exception in C1 batch script) |
| 8.3 | Deduplication before stats | **REJECTED** — defined but NEVER called in experiment scripts |
| 8.4 | Null claims have TOST | **REJECTED** — 4/6 null claims lack TOST support |
| 8.5 | Power analysis d >= 0.81 | **REJECTED** — correct value is d >= 0.792 (2.2% error) |
| 8.6 | ANCOVA F-statistics (Table 9) | **PARTIAL** — internally consistent, full recomputation blocked |
| 8.7 | CATASTROPHIC_FAIL root cause | **CONFIRMED** — code version mismatch artifact, not genuine finding |
| 8.8 | H1 monotonic growth | **CONFIRMED** — 12/12 cases strictly monotonic |
| 8.8 | H2/H3 unsupported | **CONFIRMED** — correctly reported as unsupported |
| 8.9 | Syntax (01e, 06) | **CONFIRMED** — both compile clean |

### Material Issues

1. **Pseudoreplication (8.3)**: All experiment scripts compute statistics on 5x inflated
   sample sizes. Effect sizes are correct but p-values are too small. This is the single
   most significant methodological issue found.

2. **Missing TOST for null claims (8.4)**: Abliteration, quantization invariance, Bloom
   extremes, and RDCT falsification all rely on non-significant p-values rather than
   proper equivalence testing.

3. **Stale CATASTROPHIC_FAIL (8.7)**: The Qwen tokenizer confound verdict is a code
   artifact that should be re-run with the current (fixed) verdict logic.

4. **Power analysis error (8.5)**: Minor (2.2%) but indicates careless computation.
