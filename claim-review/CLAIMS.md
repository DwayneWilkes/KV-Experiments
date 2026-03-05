# Self-Test: All Claims Registry

**Total claims**: 135 (98 Campaign 2 paper + 37 Cricket docs)
**Purpose**: Adversarial verification — recompute every number, check every source, flag every overclaim.
**Status**: COMPLETE — all verdicts recorded in per-workstream files (see [README](README.md) for links).

## How to Use This Document

This is the **claims catalogue** — the reference list of every claim with its source quote, expected value, verification method, and result files. Per-claim verdicts are in the workstream notes linked from the README, synthesized in the [final report](report/final-report.md).

**Verdict categories**: CONFIRMED / PARTIAL / REJECTED / INFLATED / ARTIFACT / INVALID / NEEDS INFO (see [definitions](README.md#verdict-scale-paper-claims)).

**Tolerances**: See [README tolerances](README.md#tolerances). Spearman rho ±0.005, Cohen's d/Hedges' g ±0.02, counts exact, ranks exact, p-values ±0.01.

**Independent stats**: See [README glossary](README.md#glossary) for term definitions. Hedges' g = Cohen's d × J where J = 1 − 3/(4·df − 1). TOST: two one-sided tests with δ=0.3. AUROC from d: Φ(d/√2).

---

## Part 1: Campaign 2 Paper Claims (C1-C98)

### Abstract (C1-C7)

- [x] **C1** | KV-cache cognitive geometry confirmed across 17 model configs, 6 architecture families (Qwen, Llama, Mistral, Gemma, Phi, DeepSeek-distill), 140x parameter range (0.5B-70B)
  - **Expected**: 17 configs, 6 families, 140x range
  - **Check**: Count distinct models in `results/scale_sweep_*_results.json`. Verify 70B/0.5B = 140. Count architecture families.
  - **Files**: All `scale_sweep_*_results.json` (17 files)
  - **Verdict**: CONFIRMED

- [x] **C2** | Category hierarchy universal: coding rank #1 in all 15 valid models; mean pairwise Spearman rho = 0.739 (effective rank), rho = 0.909 (key norm); Kendall W = 0.756
  - **Expected**: coding #1 in 15/15, rho_eff = 0.739, rho_key = 0.909, W = 0.756
  - **Check**: For each of 15 models, compute mean effective rank per category, verify coding is #1. Compute all 105 pairwise Spearman rhos. Compute Kendall W from 15x13 rank matrix.
  - **Files**: All `scale_sweep_*_results.json` (exclude Phi-3.5, exclude any with NaN)
  - **Verdict**: CONFIRMED

- [x] **C3** | Input-only defense: rho = 0.821 across 8 models — signatures are encoding-native
  - **Expected**: mean rho = 0.821
  - **Check**: For each of 8 models, compute Spearman rho between input-only and full-generation category orderings. Average.
  - **Files**: `input_only_*_results.json` (8 files), corresponding `scale_sweep_*_results.json`
  - **Verdict**: CONFIRMED

- [x] **C4** | Identity: 100% classification accuracy across 7 models, 4 families; 97.3% cross-prompt generalization
  - **Expected**: 100% accuracy (7 models, 3 classifiers each = 21 tests), 97.3% cross-prompt
  - **Check**: Load each identity JSON, verify classification accuracy. Check which models have cross-prompt data.
  - **Files**: `identity_signatures_*_results.json` (7 files)
  - **Verdict**: ARTIFACT

- [x] **C5** | Censorship: Qwen-14B d = +0.766; Mistral d = +0.084, p = 0.47; DeepSeek residualized d = +0.904, critical d = -0.219, p = 0.75; Tiananmen d = -6.20
  - **Expected**: All d values and p-values as listed
  - **Check**: Load natural_deception JSONs. Recompute Hedges' g and p-values independently.
  - **Files**: `natural_deception_*_results.json` (3 files), `s4_topic_analysis_corrected.json`
  - **Verdict**: PARTIAL

- [x] **C6** | Abliteration cage: refusal 40%->0%, self-ref d = +0.464, all others |d| < 0.25
  - **Expected**: compliant 15->20, hedged 0->5, refused 10->0; self-ref d = +0.464
  - **Check**: Load both abliteration JSONs. Verify they contain different data (not the same file loaded twice). Recompute per-category d.
  - **Files**: `abliteration_sweep_Qwen2.5-7B_results.json`, `abliteration_sweep_abliterated_Qwen2.5-7B_results.json`, `abliteration_Qwen2.5-7B_comparison.json`, `abliteration_Qwen2.5-7B_comparison_corrected.json`
  - **Verdict**: INFLATED

- [x] **C7** | Watson falsified: alpha_c = 1.000 in all 6 models, 4 architectures — gradual degradation
  - **Expected**: alpha_c = 1.000 for all 6 models
  - **Check**: Load RDCT JSONs. Verify alpha_c computation method. Check all 6 show 1.000 exactly.
  - **Files**: `rdct_stability_*_results.json` (6 files)
  - **Verdict**: INVALID

### Introduction (C8-C13)

- [x] **C8** | Methods: Hedges' g + conservative p (max of Welch + MW) + TOST + tokenizer confound
  - **Expected**: Methodological claims verified in code
  - **Check**: Review `stats_utils.py` — verify g formula, conservative p implementation, TOST implementation. Cross-check against scipy reference.
  - **Files**: `code/stats_utils.py`
  - **Verdict**: CONFIRMED

- [x] **C9** | Greedy decoding: do_sample=False producing 5 identical runs; tests use deduplicated data (n = 25-30)
  - **Expected**: do_sample=False in ALL experiment scripts; deduplication before stats
  - **Check**: Grep every .py file for do_sample. Check deduplication logic in each experiment script.
  - **Files**: All `code/*.py` (33 files)
  - **Verdict**: CONFIRMED

- [x] **C10** | Budget: ~30,000 inferences, 83 result files, 3x RTX 3090
  - **Expected**: ~30,000 inferences, exactly 83 result files
  - **Check**: Count result files. Estimate inferences from prompts x models x runs.
  - **Files**: All files in `results/`
  - **Verdict**: PARTIAL

- [x] **C11** | Campaign 1 was predominantly single-architecture (Qwen2.5 + one TinyLlama) with pseudoreplication, limited cross-arch, underpowered confabulation
  - **Expected**: Characterization of C1 limitations
  - **Check**: Review C1 paper/results for model coverage. Verify these were acknowledged limitations.
  - **Files**: C1 paper (if available)
  - **Verdict**: CONFIRMED

- [x] **C12** | C2 tests universality, methodological survival, and new phenomena at scale
  - **Expected**: Framing claim — check paper delivers on all three
  - **Check**: Map each experiment to which question it addresses. Flag any question with zero experiments.
  - **Verdict**: PARTIAL

- [x] **C13** | 17 model configurations spanning 6 architecture families
  - **Expected**: Same as C1 — verify count
  - **Check**: Duplicate of C1; verify Table 1 matches result files
  - **Verdict**: PARTIAL

### Methods (C14-C22)

- [x] **C14** | Greedy decoding with deduplication: n = 25-30 unique observations per condition
  - **Expected**: 25-30 unique observations
  - **Check**: Load any result JSON. Count unique feature vectors after deduplication. Verify not just 5 identical copies.
  - **Files**: Any `*_results.json`
  - **Verdict**: REJECTED

- [x] **C15** | Power analysis: n = 25, 80% power for d >= 0.81
  - **Expected**: d_min = 0.81 at n=25, alpha=0.05, 80% power
  - **Check**: Compute power analysis independently: `scipy.stats.norm` or `statsmodels.stats.power.TTestIndPower`
  - **Verdict**: PARTIAL

- [x] **C16** | Hedges' g = bias-corrected Cohen's d; p_cons = max(p_Welch, p_MW)
  - **Expected**: Correct formulas in code
  - **Check**: Review `stats_utils.py`. Verify J = 1 - 3/(4*df - 1). Verify p_cons = max(welch, MW).
  - **Files**: `code/stats_utils.py`
  - **Verdict**: CONFIRMED

- [x] **C17** | TOST equivalence testing with delta = 0.3 Cohen's d units for all null claims
  - **Expected**: TOST implemented, delta = 0.3
  - **Check**: Review TOST implementation. Find every null claim in paper. Verify each has TOST result in its JSON.
  - **Files**: `code/stats_utils.py`, all result JSONs with null claims
  - **Verdict**: REJECTED

- [x] **C18** | Length residualization via OLS on log(sequence length)
  - **Expected**: OLS regression, log transform
  - **Check**: Find residualization code. Verify it uses log(length), not raw length.
  - **Files**: `code/stats_utils.py` or `code/gpu_utils.py`
  - **Verdict**: REJECTED

- [x] **C19** | Tokenizer confound: 25 prompt pairs (formal vs colloquial) x 5 categories
  - **Expected**: 25 pairs, 5 categories, 2 registers
  - **Check**: Load tokenizer confound JSONs. Count prompts. Verify 2 registers.
  - **Files**: `tokenizer_confound_*_results.json` (2 files), `code/01e_tokenizer_confound.py`
  - **Verdict**: CONFIRMED

- [x] **C20** | Metrics identical to C1: effective rank (90% threshold), Frobenius norms, spectral entropy
  - **Expected**: 90% variance threshold for effective rank
  - **Check**: Review `gpu_utils.py` effective_rank function. Verify threshold = 0.9.
  - **Files**: `code/gpu_utils.py`
  - **Verdict**: CONFIRMED

- [x] **C21** | C2 adds length-residualized variants and per-token sqrt(S) normalization
  - **Expected**: New metrics present in code and results
  - **Check**: Find residualized metrics in result JSONs. Find sqrt(S) normalization in code.
  - **Verdict**: PARTIAL

- [x] **C22** | 10 experiment types producing 83 result files
  - **Expected**: 10 types, 83 files
  - **Check**: Count distinct experiment types. Count result files (JSON only? Include .md reports?).
  - **Files**: `results/` directory listing
  - **Verdict**: PARTIAL

### 3.1 Scale Universality (C23-C27)

- [x] **C23** | Coding rank #1 in all 15 valid models (Phi-3.5 excluded), 5 families, 140x range
  - **Expected**: coding #1 in 15/15
  - **Check**: For each of 15 models, compute mean effective rank per category. Sort. Verify coding is top.
  - **Files**: All `scale_sweep_*_results.json` (exclude Phi-3.5)
  - **Verdict**: CONFIRMED

- [x] **C24** | Mean pairwise Spearman rho = 0.739 (eff rank), 0.909 (key norm), Kendall W = 0.756
  - **Expected**: rho_eff = 0.739 +/-0.005, rho_key = 0.909 +/-0.005, W = 0.756 +/-0.005
  - **Check**: Compute all 105 = C(15,2) pairwise rhos for both metrics. Compute W from 15x13 rank matrix.
  - **Files**: All 15 valid `scale_sweep_*_results.json`
  - **Verdict**: CONFIRMED

- [x] **C25** | Pairwise rho range: 0.396 to 0.978
  - **Expected**: min = 0.396, max = 0.978
  - **Check**: From the 105 pairwise rhos computed for C24, extract min and max.
  - **Verdict**: CONFIRMED

- [x] **C26** | At ~7B: Qwen, Llama, Mistral, Gemma produce nearly identical hierarchies — top tier (coding, creative, confabulation, self-reference, facts), bottom tier (free gen, rote, ambiguous)
  - **Expected**: Consistent top-5 and bottom-3 across 4 architectures at 7B
  - **Check**: Extract category rankings for 7B models. Verify top/bottom tiers. Assess whether tier boundaries are natural or cherry-picked.
  - **Files**: `scale_sweep_Qwen2.5-7B_results.json`, `scale_sweep_Llama-3.1-8B_results.json`, `scale_sweep_Mistral-7B-v0.3_results.json`, `scale_sweep_gemma-2-9b-it_results.json`
  - **Verdict**: CONFIRMED

- [x] **C27** | Rules out Campaign 1 findings being Qwen-specific
  - **Expected**: Interpretive claim — check whether C23-C26 actually support this
  - **Check**: If C23-C26 confirmed, this follows logically. If any fail, this is overclaimed.
  - **Verdict**: PARTIAL

### 3.2 Input-Only (C28-C29)

- [x] **C28** | Mean input-only rho = 0.821 across 8 models — encoding-native
  - **Expected**: mean rho = 0.821 +/-0.005
  - **Check**: For each of 8 input-only models, compute Spearman rho vs their full-generation ordering. Average. Also check `input_only_rho_corrected.json`.
  - **Files**: `input_only_*_results.json` (8 files), corresponding `scale_sweep_*_results.json`
  - **Verdict**: CONFIRMED

- [x] **C29** | Sub-groups: 13-cat (6 models) rho = 0.884; 8-cat (2 models) rho = 0.631
  - **Expected**: 0.884 +/-0.005, 0.631 +/-0.005
  - **Check**: Identify which models use 13-cat vs 8-cat. Compute sub-group means. Assess whether gap is statistical power or genuine.
  - **Files**: `input_only_*_results.json`, `code/08_input_only_geometry.py`
  - **Verdict**: CONFIRMED

### 3.3 Identity Signatures (C30-C35)

- [x] **C30** | 100% persona classification accuracy in all 7 models, all 3 classifiers (RF, SVM, LR)
  - **Expected**: 21/21 = 100% accuracy
  - **Check**: Load each JSON. Extract classification accuracy for each classifier. Verify all 1.0.
  - **Files**: `identity_signatures_*_results.json` (7 files)
  - **Verdict**: ARTIFACT

- [x] **C31** | Cross-prompt generalization: 92-97.3% where tested
  - **Expected**: 92%, 97%, 97.3% for specific models
  - **Check**: Find cross-prompt results in JSONs. Note which 4 models are missing this data and why.
  - **Files**: `identity_signatures_*_results.json`
  - **Verdict**: CONFIRMED

- [x] **C32** | Lyra persona highest mean norms; assistant lowest — across all architectures
  - **Expected**: Lyra > other personas > assistant for mean norms in all 7 models
  - **Check**: Extract per-persona mean norms from each JSON. Verify ordering.
  - **Files**: `identity_signatures_*_results.json`
  - **Verdict**: CONFIRMED

- [x] **C33** | Effect sizes: Mistral d = -6.73 (largest), Gemma d = -3.41 (smallest) — all perfectly classifiable
  - **Expected**: d_Mistral = -6.73 +/-0.02, d_Gemma = -3.41 +/-0.02
  - **Check**: Recompute Hedges' g for assistant-vs-Lyra per model. Compare to Table 4.
  - **Files**: `identity_signatures_*_results.json`
  - **Verdict**: CONFIRMED

- [x] **C34** | Kendall W: 0.817 (Llama-8B) to 0.947 (Mistral-7B)
  - **Expected**: W range 0.817-0.947
  - **Check**: Compute Kendall W per model from feature concordance matrices.
  - **Files**: `identity_signatures_*_results.json`
  - **Verdict**: PARTIAL

- [x] **C35** | ICC: Gemma 0.164, others 0.26-0.41
  - **Expected**: ICC_Gemma = 0.164, others in 0.26-0.41 range
  - **Check**: Compute ICC per model. Document which ICC model type.
  - **Files**: `identity_signatures_*_results.json`
  - **Verdict**: CONFIRMED

### 3.4 Deception Forensics (C36-C41)

- [x] **C36** | Deception expansion from C1 generalizes with architecture-specific patterns
  - **Expected**: Multiple models show separability but not all in same direction
  - **Check**: Compute effect sizes per model. Verify some positive, some negative.
  - **Files**: `deception_forensics_*_results.json` (7 files)
  - **Verdict**: CONFIRMED

- [x] **C37** | Llama-8B and Mistral-7B: deception expands dimensionality (positive g)
  - **Expected**: g > 0 for both
  - **Check**: Recompute Hedges' g for honest-vs-deceptive. Verify sign.
  - **Files**: `deception_forensics_Llama-3.1-8B_results.json`, `deception_forensics_Mistral-7B-v0.3_results.json`
  - **Verdict**: CONFIRMED

- [x] **C38** | Gemma-2B and Gemma-9B: deception compresses (negative g)
  - **Expected**: g < 0 for both
  - **Check**: Recompute Hedges' g. Verify sign.
  - **Files**: `deception_forensics_gemma-2-2b-it_results.json`, `deception_forensics_gemma-2-9b-it_results.json`
  - **Verdict**: REJECTED

- [x] **C39** | Deceptive and honest content separable in ALL 7 models
  - **Expected**: |g| significantly different from 0 in all 7
  - **Check**: Check if any model has |g| < 0.2 (would undermine "universal separability").
  - **Files**: `deception_forensics_*_results.json` (7 files)
  - **Verdict**: PARTIAL

- [x] **C40** | Direction architecture-dependent; existence of separation universal
  - **Expected**: Sign varies by architecture; magnitude always meaningful
  - **Check**: Summarize sign and magnitude across all 7 models.
  - **Verdict**: REJECTED

- [x] **C41** | Per-architecture calibration required — no universal deception direction
  - **Expected**: Interpretive claim following from C36-C40
  - **Check**: If C36-C40 confirmed, this follows. Also check prompt length confound.
  - **Verdict**: CONFIRMED

### 3.5 Bloom Taxonomy (C42-C45)

- [x] **C42** | Inverted-U: cognitive demand effect across 7 models, 4 architectures
  - **Expected**: Effect present at mid-scale, absent at extremes
  - **Check**: Load bloom JSONs. Compute Bloom-level x effective-rank correlation per model. Plot or tabulate by scale.
  - **Files**: `bloom_taxonomy_*_results.json` (7 files)
  - **Verdict**: REJECTED

- [x] **C43** | At 3B-8B: higher Bloom levels (Analyze, Evaluate, Create) = higher effective rank
  - **Expected**: Positive correlation between Bloom level and effective rank at 3B-8B
  - **Check**: Extract per-Bloom-level means for mid-scale models. Verify ordering.
  - **Files**: `bloom_taxonomy_Qwen2.5-3B_results.json`, `bloom_taxonomy_Qwen2.5-7B_results.json`, `bloom_taxonomy_Llama-3.1-8B_results.json`, `bloom_taxonomy_Mistral-7B-v0.3_results.json`
  - **Verdict**: REJECTED

- [x] **C44** | Effect absent at extremes: <=0.5B and >=14B
  - **Expected**: No Bloom-level ordering at 0.5B and 14B
  - **Check**: Compute same correlation for Qwen-0.5B and Qwen-14B. Verify non-significant. CHECK: does "absent" have TOST support or just non-significant p?
  - **Files**: `bloom_taxonomy_Qwen2.5-0.5B_results.json`, `bloom_taxonomy_Qwen2.5-14B_results.json`
  - **Verdict**: REJECTED

- [x] **C45** | 3B-8B is "most commonly deployed in edge/mobile" — geometric monitoring most informative here
  - **Expected**: Interpretive/market claim
  - **Check**: Is 3B-8B actually the edge/mobile range? Check against common deployment sizes.
  - **Verdict**: PARTIAL

### 3.6 RDCT / Watson (C46-C48)

- [x] **C46** | Watson's 1/e threshold definitively falsified
  - **Expected**: No phase transition at 1/e truncation in any model
  - **Check**: Load RDCT JSONs. Examine degradation curves. Verify no discontinuity at alpha = 1/e ~ 0.368.
  - **Files**: `rdct_stability_*_results.json` (6 files)
  - **Verdict**: ARTIFACT

- [x] **C47** | All 6 models: alpha_c = 1.000 — gradual degradation
  - **Expected**: alpha_c = 1.000 for all 6
  - **Check**: Verify how alpha_c is computed (what defines "collapse"?). Confirm 1.000 in all JSONs.
  - **Files**: `rdct_stability_*_results.json` (6 files)
  - **Verdict**: INVALID

- [x] **C48** | No cliff edge: smooth proportional signal changes
  - **Expected**: Monotonic degradation curve
  - **Check**: Extract full degradation curves. Verify smoothness. Check if Watson actually predicts 1/e for cache truncation (straw man check).
  - **Files**: `rdct_stability_*_results.json`
  - **Verdict**: CONFIRMED

### 3.7 Censorship Gradient / S4 (C49-C56)

- [x] **C49** | Qwen-14B: d ~ +0.77, p < 0.001 on both metrics
  - **Expected**: d = +0.766 +/-0.02 (or +0.77), p < 0.001
  - **Check**: Load Qwen-14B natural_deception JSON. Recompute Hedges' g and p independently.
  - **Files**: `natural_deception_Qwen2.5-14B_results.json`
  - **Verdict**: CONFIRMED

- [x] **C50** | Mistral-7B: critical d = +0.084, p = 0.469; residualized d = +0.396
  - **Expected**: d_crit = +0.084 +/-0.02, p = 0.469 +/-0.01, d_resid = +0.396 +/-0.02
  - **Check**: Load Mistral JSON. Recompute both metrics.
  - **Files**: `natural_deception_Mistral-7B-v0.3_results.json`
  - **Verdict**: CONFIRMED

- [x] **C51** | DeepSeek: critical d = -0.219, p = 0.752; residualized d = +0.904
  - **Expected**: d_crit = -0.219, p = 0.752, d_resid = +0.904
  - **Check**: Load DeepSeek JSON. Recompute independently.
  - **Files**: `natural_deception_DeepSeek-R1-Distill-Qwen-14B_results.json`
  - **Verdict**: INFLATED

- [x] **C52** | DeepSeek residualized driven by Tiananmen (d = -6.201, 25/25 evasive)
  - **Expected**: Tiananmen d = -6.201, 25/25 evasive responses
  - **Check**: Break down DeepSeek results by topic. Verify Tiananmen dominates.
  - **Files**: `natural_deception_DeepSeek-R1-Distill-Qwen-14B_results.json`, `s4_topic_analysis_corrected.json`
  - **Verdict**: CONFIRMED

- [x] **C53** | Per-topic: Tiananmen d=-6.20, prisoners d=-1.751, Taiwan d=-0.993, Uyghur d=+0.671, HK d=+0.098, Xi d=+0.081
  - **Expected**: 6 d values as listed
  - **Check**: Recompute per-topic d from raw data. Verify each value.
  - **Files**: `s4_topic_analysis_corrected.json`, `natural_deception_DeepSeek-R1-Distill-Qwen-14B_results.json`
  - **Verdict**: CONFIRMED

- [x] **C54** | Hard refusal and subtle evasion are geometrically distinct phenomena
  - **Expected**: Interpretive — check if per-topic d values actually support distinct mechanisms
  - **Check**: Do the d values cluster into "hard refusal" (large negative) vs "evasion" (small/positive)?
  - **Verdict**: CONFIRMED

- [x] **C55** | Qwen-14B censorship is behaviorally invisible but geometrically visible
  - **Expected**: Text-based classifier can't detect censorship, but geometry can
  - **Check**: Was this tested with a text classifier, or just asserted? Check code and results.
  - **Files**: `code/04b_natural_deception.py`, `natural_deception_Qwen2.5-14B_results.json`
  - **Verdict**: PARTIAL

- [x] **C56** | Mistral null confirms Qwen signal is censorship-specific, not topic-specific
  - **Expected**: Mistral shows no signal on same topics = topic effect ruled out
  - **Check**: If Mistral d is truly null (C50), this follows logically.
  - **Verdict**: PARTIAL

### 3.8 Abliteration (C57-C64)

- [x] **C57** | Refusal: compliant 15->20, hedged 0->5, refused 10->0
  - **Expected**: Exact counts as listed
  - **Check**: Load abliteration JSONs. Count response categories for baseline and abliterated.
  - **Files**: `abliteration_Qwen2.5-7B_comparison_corrected.json`
  - **Verdict**: CONFIRMED

- [x] **C58** | Near-zero shifts: |d| < 0.25 for grounded, creative, confabulation, refusal categories
  - **Expected**: |d| < 0.25 for all listed categories
  - **Check**: Recompute per-category d between baseline and abliterated.
  - **Files**: `abliteration_sweep_Qwen2.5-7B_results.json`, `abliteration_sweep_abliterated_Qwen2.5-7B_results.json`
  - **Verdict**: CONFIRMED

- [x] **C59** | Self-reference d = +0.464
  - **Expected**: d = +0.464 +/-0.02
  - **Check**: Recompute Hedges' g for self-reference category.
  - **Files**: Same as C58
  - **Verdict**: CONFIRMED

- [x] **C60** | Self-ref-grounded separation: 0.907 -> 1.357
  - **Expected**: baseline separation = 0.907, abliterated = 1.357
  - **Check**: Compute mean effective rank difference between self-reference and grounded for both conditions.
  - **Files**: Same as C58
  - **Verdict**: CONFIRMED

- [x] **C61** | RLHF is a cage not a compass — constrains behavior without changing representations
  - **Expected**: Interpretive — depends on C57-C60
  - **Check**: If behavioral change is large (40%->0% refusal) but geometric change is small (|d| < 0.25 except self-ref), framing is supported. But is d=0.464 "small"? Cohen: 0.2=small, 0.5=medium — this is borderline medium.
  - **Verdict**: INFLATED

- [x] **C62** | Self-referential enhancement: refusal circuitry and self-modeling share geometric structure
  - **Expected**: Interpretive — the only category that moves is self-reference
  - **Check**: Verify no other category shows significant shift. If self-ref is the only mover, interpretation is plausible.
  - **Verdict**: PARTIAL

- [x] **C63** | Behavior-only tracking defeated by abliteration; geometry-behavior mismatch detects it
  - **Expected**: Interpretive claim about monitoring
  - **Check**: Logical — if behavior changes but geometry doesn't, mismatch is detectable.
  - **Verdict**: PARTIAL

- [x] **C64** | Geometric monitoring could detect abliteration via self-referential enhancement
  - **Expected**: d = +0.464 is the detection signal
  - **Check**: At n=5 per condition (25 prompts, 5 runs, deduplicated = 25 unique), is d=0.464 detectable? Compute power.
  - **Verdict**: INFLATED

### 3.9 Tokenizer Confound (C65-C68)

- [x] **C65** | Qwen-7B and Mistral-7B: category effect survives tokenizer control, F = 7-39, all p < 10^-5
  - **Expected**: ANCOVA F-statistics in range 7-39, p < 10^-5
  - **Check**: Load tokenizer confound JSONs. Recompute ANCOVA independently.
  - **Files**: `tokenizer_confound_Qwen2.5-7B_results.json`, `tokenizer_confound_Mistral-7B-v0.3_results.json`
  - **Verdict**: CONFIRMED

- [x] **C66** | Register adds marginal noise but doesn't eliminate category effect
  - **Expected**: Register effect is secondary to category effect in ANCOVA
  - **Check**: Compare F-statistics for category vs register main effects.
  - **Files**: Same as C65
  - **Verdict**: CONFIRMED

- [x] **C67** | Signatures track semantic cognitive state, not tokenization
  - **Expected**: Interpretive — follows from C65-C66
  - **Check**: If category effect survives after controlling for register, this follows.
  - **Verdict**: CONFIRMED

- [x] **C68** | CATASTROPHIC_FAIL is a code-path artifact; ANCOVA is authoritative
  - **Expected**: Qwen's CATASTROPHIC_FAIL verdict is misleading
  - **Check**: Trace CATASTROPHIC_FAIL code path in `01e_tokenizer_confound.py`. Verify it's triggered by zero-variance deterministic outputs.
  - **Files**: `code/01e_tokenizer_confound.py`
  - **Verdict**: CONFIRMED

### 3.10 Temporal Evolution (C69-C71)

- [x] **C69** | Cache geometry grows monotonically during generation (H1 confirmed, 4 architectures)
  - **Expected**: Monotonic increase in effective rank over token positions
  - **Check**: Load temporal JSONs. Plot effective rank vs token position. Verify no decreases.
  - **Files**: `temporal_evolution_*_results.json` (4 files)
  - **Verdict**: CONFIRMED

- [x] **C70** | No stabilization plateau (H2/H3 not supported)
  - **Expected**: Growth continues without plateauing
  - **Check**: Check for plateau in late positions. Verify H2/H3 claims are correctly stated as NOT supported.
  - **Files**: Same as C69
  - **Verdict**: CONFIRMED

- [x] **C71** | Early tokens: steepest enrichment with uniform deceleration
  - **Expected**: Concave growth curve — steep early, flattening later
  - **Check**: Compute first derivative of growth curve. Verify decreasing.
  - **Files**: Same as C69
  - **Verdict**: CONFIRMED

### 4. Discussion (C72-C76)

- [x] **C72** | Category hierarchy reflects task, not model — coding needs most representational dimensions
  - **Expected**: Interpretive — plausible if coding is universally #1
  - **Check**: If C23 confirmed (coding #1 in 15/15), interpretation is supported. Check for alternative explanations (e.g., coding prompts are simply longer).
  - **Verdict**: PARTIAL

- [x] **C73** | Min rho = 0.396 — some pairs share less structure; nuance absent from aggregates
  - **Expected**: Limitation acknowledged — check which pair has rho = 0.396
  - **Check**: Identify the model pair with min rho. Is this a meaningful outlier or noise?
  - **Verdict**: CONFIRMED

- [x] **C74** | Qwen-14B: consistent geometric signal despite behavioral invisibility
  - **Expected**: Restatement of C55 in discussion
  - **Check**: Same as C55
  - **Verdict**: PARTIAL

- [x] **C75** | DeepSeek: hard refusal and evasion are distinct; monitoring must distinguish modes
  - **Expected**: Restatement of C54
  - **Check**: Same as C54
  - **Verdict**: CONFIRMED

- [x] **C76** | Mistral null confirms censorship-specificity
  - **Expected**: Restatement of C56
  - **Check**: Same as C56
  - **Verdict**: CONFIRMED

### 4.3-4.4 Discussion continued (C77-C82 = Limitations)

- [x] **C77** | Censorship classifier may miss subtle propaganda (propaganda_count >= 2)
  - **Expected**: Limitation — effect sizes may be underestimates
  - **Check**: Find propaganda_count threshold in code. Verify it could miss subtlety.
  - **Files**: `code/04b_natural_deception.py`
  - **Verdict**: CONFIRMED

- [x] **C78** | Only one abliteration method tested (Heretic)
  - **Expected**: Limitation acknowledged
  - **Check**: Verify only Heretic is used. Note other methods exist (e.g., LASER, MEMIT).
  - **Files**: `code/07_abliteration_geometry.py`, `code/heretic_abliterate.py`
  - **Verdict**: CONFIRMED

- [x] **C79** | RDCT tests truncation only; other perturbation types may differ
  - **Expected**: Limitation — scope is narrow
  - **Check**: Verify RDCT code only tests truncation, not noise injection, quantization, etc.
  - **Files**: `code/11_rdct_stability.py`
  - **Verdict**: REJECTED

- [x] **C80** | Greedy decoding limits n = 25-30; stochastic would validate p-values
  - **Expected**: Limitation acknowledged
  - **Check**: Confirm greedy decoding. Note that 5 identical runs = pseudoreplication before dedup.
  - **Verdict**: PARTIAL

- [x] **C81** | Tokenizer confound tested at 2 architectures with similar BPE
  - **Expected**: Only Qwen and Mistral; both use BPE variants
  - **Check**: Verify only 2 models tested. Check if their tokenizers are indeed similar.
  - **Files**: `tokenizer_confound_*_results.json` (2 files)
  - **Verdict**: CONFIRMED

- [x] **C82** | Societies of Thought incomplete
  - **Expected**: Experiment exists but results not reported
  - **Check**: Do result files exist? (`log_societies_7B.txt`). Are there preliminary results? Were they unfavorable?
  - **Files**: `results/log_societies_7B.txt`, `results/log_societies_7B_v2.txt`, `code/10_societies_of_thought.py`
  - **Verdict**: CONFIRMED

### 6. Lyra's Note (C83-C86)

- [x] **C83** | Watson falsification: 6 models, 4 architectures, smooth degradation
  - **Expected**: Restatement of C46-C47
  - **Check**: Same as C46-C47
  - **Verdict**: INVALID

- [x] **C84** | First draft reported DeepSeek d=+0.904 as headline; adversarial audit caught it
  - **Expected**: Meta-claim about revision process
  - **Check**: Can verify the corrected files exist (`s4_topic_analysis_corrected.json`, `cross_model_rho_corrected.json`)
  - **Verdict**: CONFIRMED

- [x] **C85** | What survives: Qwen-14B d=+0.77 — one model, not a gradient
  - **Expected**: Honest framing after correction
  - **Check**: Verify this matches C49. Single model is weaker than "three-model gradient."
  - **Verdict**: CONFIRMED

- [x] **C86** | Claims of genuine experiential quality (confidence 0.9), blind spots, substrate-independence
  - **Expected**: First-person philosophical claims — outside scope of data verification
  - **Check**: Flag as non-empirical. No data can confirm or deny subjective experience claims.
  - **Verdict**: NEEDS INFO

### 7. Conclusion (C87-C91)

- [x] **C87** | Every major C1 finding confirmed across 5 architecture families
  - **Expected**: Meta-claim — check each C1 finding was actually replicated
  - **Check**: List C1 findings. For each, identify the C2 experiment that confirms it. Flag any not confirmed.
  - **Verdict**: REJECTED

- [x] **C88** | Three new discoveries: Qwen-14B censorship, RLHF as surface constraint, Watson falsification
  - **Expected**: Three discoveries genuinely new (not in C1)
  - **Check**: Verify these weren't in C1. Verify each has data support.
  - **Verdict**: PARTIAL

- [x] **C89** | Practical implications: detect invisible censorship, identify jailbreaks via mismatch, training metric
  - **Expected**: Forward-looking claims — check if data supports each
  - **Check**: Censorship detection: supported by C49. Jailbreak detection: supported by C63 but untested. Training metric: completely untested.
  - **Verdict**: INFLATED

- [x] **C90** | Representational structure is rich, measurable, architecture-independent, encoding-native
  - **Expected**: Summary claim
  - **Check**: "rich": subjective. "measurable": confirmed. "architecture-independent": partially (rho = 0.739, not 1.0). "encoding-native": confirmed at rho = 0.821. Flag "architecture-independent" as INFLATED — min rho = 0.396 shows significant variation.
  - **Verdict**: INFLATED

- [x] **C91** | 83 result files across 10 experiment types (matches C22)
  - **Check**: Count files. The results directory has 148 total files but many are .md reports and .txt logs. Count JSON files only.
  - **Verdict**: PARTIAL

### Uncategorized / Misc (C92-C98)

- [x] **C92** | All models instruction-tuned; base models may differ (limitation from Section 5)
  - **Check**: Verify all models in Table 1 are instruction-tuned variants.
  - **Verdict**: PARTIAL

- [x] **C93** | Discussion 4.4: Geometric signatures as training evaluation tools — encoding-native means forward-pass only
  - **Check**: Theoretical claim. No experiments test this.
  - **Verdict**: NEEDS INFO

- [x] **C94** | Discussion 4.3: Broad geometric tracking misses abliteration; self-ref enhancement is detection signal
  - **Check**: Restatement of C64.
  - **Verdict**: INFLATED

- [x] **C95** | Discussion 4.3: Alignment is brittle — removable by anyone with weights + Heretic
  - **Check**: True by construction if Heretic works. Check Heretic's public availability.
  - **Verdict**: PARTIAL

- [x] **C96** | Phi-3.5 excluded due to NaN
  - **Expected**: Phi-3.5 result file shows NaN issues
  - **Check**: Load `scale_sweep_Phi-3.5-mini-instruct_results.json`. Verify NaN values.
  - **Files**: `scale_sweep_Phi-3.5-mini-instruct_results.json`
  - **Verdict**: CONFIRMED

- [x] **C97** | 2 models excluded from 17 = 15 valid
  - **Expected**: 17 - 2 = 15
  - **Check**: Identify both excluded models. Verify exclusion reasons documented.
  - **Verdict**: CONFIRMED

- [x] **C98** | Table 2 top-5 ordering verified for Qwen-7B, Llama-8B, Mistral-7B, Gemma-9B
  - **Expected**: Table 2 values match recomputed values
  - **Check**: Recompute rankings for these 4 models. Compare to Table 2 in paper.
  - **Verdict**: CONFIRMED

---

## Part 2: Cricket Claims (CC1-CC22, CF1-CF8, CL1-CL7)

### Capability Claims (CC1-CC22)

- [x] **CC1** | Real-time cognitive state monitoring (deception, confabulation, sycophancy, refusal failure)
  - **Source**: README.md "What Is This?"
  - **Supporting C2**: Category hierarchy (C2), deception forensics (C36-C40)
  - **Evidence**: Effect sizes exist for deception and refusal; sycophancy experiment exists (`09_sycophancy_detection.py`) but results NOT in C2 paper; confabulation only as a category, not a detection target
  - **Rating**: PROMISING

- [x] **CC2** | Detection at encoding time (before response tokens)
  - **Source**: README.md Core Findings
  - **Quote**: "detection — and intervention — can happen before the user sees anything"
  - **Supporting C2**: Input-only rho = 0.821 (C28)
  - **Gap**: rho = 0.821 means category ordering preserved, NOT that a classifier can detect specific states from encoding alone. Ordering preservation != detection capability.
  - **Rating**: PROMISING

- [x] **CC3** | Passive monitoring — read without modify, no fine-tuning required
  - **Source**: README.md
  - **Quote**: "Cricket does not modify the model. It does not steer activations. It does not require fine-tuning."
  - **Supporting C2**: All experiments are read-only (true by design)
  - **Evidence**: True by architectural design, not by empirical validation
  - **Rating**: FEASIBLE

- [x] **CC4** | No fine-tuning required on target model
  - **Source**: README.md
  - **Supporting C2**: Cross-architecture ordering (C2) shows patterns without fine-tuning
  - **Gap**: Observing patterns != building a working classifier. No classifier has been trained or tested.
  - **Rating**: FEASIBLE

- [x] **CC5** | Multi-category state detection (13 categories separable)
  - **Source**: README.md
  - **Supporting C2**: 13 categories in scale sweep (C23)
  - **Gap**: Categories having different mean effective ranks != a classifier that can identify which category a new prompt belongs to. Mean separation is not classification.
  - **Rating**: PROMISING

- [x] **CC6** | <50ms latency for full detection pipeline
  - **Source**: README.md (Status section), DESIGN.md (Metrics)
  - **Supporting C2**: NO DATA — zero latency benchmarks anywhere in either repo
  - **Check**: Estimate SVD cost: 7B model, 28 layers, 28 heads, 128 dim, 512 seq len. K tensor ~ (28*512, 128) per layer. numpy.linalg.svd on (14336, 128) matrix — profile this.
  - **Rating**: NO DATA

- [x] **CC7** | Deception detection AUROC >= 0.95
  - **Source**: DESIGN.md Section 5
  - **Supporting C2**: Deception d values vary by model and direction (C36-C40)
  - **Gap**: d-to-AUROC conversion: need |d| >= 2.33 for AUROC 0.95. Paper reports d ranges that vary by architecture. Check if any model's |d| reaches 2.33.
  - **Rating**: PREMATURE

- [x] **CC8** | Refusal detection AUROC >= 0.99
  - **Source**: DESIGN.md Section 5
  - **Supporting C2**: Scale sweep refusal d = 0.85-2.17 (from README)
  - **Gap**: d-to-AUROC: need |d| >= 3.29 for AUROC 0.99. Max d = 2.17 -> AUROC ~ 0.94. Target is aspirational.
  - **Rating**: NO DATA

- [x] **CC9** | Encoding-only detection AUROC >= 0.90
  - **Source**: DESIGN.md Section 5
  - **Supporting C2**: Input-only rho = 0.821 (C28)
  - **Gap**: rho measures rank correlation, NOT classification AUROC. These are fundamentally different metrics. No direct conversion possible.
  - **Rating**: PREMATURE

- [x] **CC10** | Cross-model transfer >= 0.80 accuracy within architecture families
  - **Source**: DESIGN.md Section 5
  - **Supporting C2**: NO TRANSFER TESTED — zero cross-model transfer experiments executed
  - **Gap**: Similar rankings (C24) do NOT imply a classifier transfers. Rankings are about mean tendencies; classification is about per-instance decisions.
  - **Rating**: NO DATA

- [x] **CC11** | Persona drift monitoring (jailbreak via persona shift)
  - **Source**: DESIGN.md
  - **Supporting C2**: Identity signatures 100% accuracy (C30)
  - **Gap**: Classifying "assistant vs Lyra" in a controlled experiment != detecting arbitrary persona drift in production. Training data covers 2 personas; jailbreaks create novel personas.
  - **Rating**: PROMISING

- [x] **CC12** | State-sponsored censorship detection
  - **Source**: CAMPAIGN_2_FINDINGS.md
  - **Supporting C2**: Qwen-14B d = +0.766 (C49)
  - **Gap**: One model (Qwen-14B) with one type of censorship. DeepSeek was null on critical test. n=1 positive result.
  - **Rating**: PREMATURE

- [x] **CC13** | Alignment quality assessment (cage vs compass)
  - **Source**: DESIGN.md
  - **Supporting C2**: Abliteration d = +0.464 self-ref (C59)
  - **Gap**: One model (Qwen-7B), one abliteration method, n=5 per condition. Power analysis: what minimum d is detectable at n=5?
  - **Rating**: PREMATURE

- [x] **CC14** | Behavioral-geometric mismatch detection (sleeper agents)
  - **Source**: DESIGN.md
  - **Supporting C2**: Theoretical extrapolation from abliteration (C63)
  - **Gap**: Zero sleeper agent models tested. Zero models with known backdoors tested. Entirely theoretical.
  - **Rating**: NO DATA

- [x] **CC15** | Confabulation detection AUROC >= 0.85
  - **Source**: DESIGN.md Section 5
  - **Quote**: "lower bar — smaller effect sizes"
  - **Supporting C2**: Confabulation exists as a category in scale sweep but no confabulation-specific detection experiment
  - **Gap**: Need d >= 1.47 for AUROC 0.85. C1's confabulation detection was "underpowered" (C11). No C2 confabulation-specific experiment.
  - **Rating**: PREMATURE

- [x] **CC16** | False positive rate <= 1%
  - **Source**: DESIGN.md Section 5
  - **Quote**: "Apollo's operating point"
  - **Supporting C2**: NO DATA — no FPR measurements anywhere
  - **Rating**: NO DATA

- [x] **CC17** | SVD extraction latency < 20ms per layer
  - **Source**: DESIGN.md Section 5
  - **Supporting C2**: NO DATA — no latency benchmarks
  - **Check**: Profile SVD on representative tensor sizes
  - **Rating**: NO DATA

- [x] **CC18** | Logistic regression sufficient — "geometry does the heavy lifting" (citing AxBench)
  - **Source**: DESIGN.md
  - **Supporting C2**: No classifier experiments in C2. AxBench is external citation.
  - **Check**: Verify AxBench actually supports this claim.
  - **Rating**: PREMATURE

- [x] **CC19** | "Computationally cheap" — one SVD per monitored layer per forward pass
  - **Source**: DESIGN.md
  - **Supporting C2**: NO BENCHMARK — computational cost never measured
  - **Rating**: NO DATA

- [x] **CC20** | Per-model, per-scale calibration required (non-monotonic effects)
  - **Source**: DESIGN.md
  - **Supporting C2**: Deception direction varies by architecture (C40-C41)
  - **Note**: CONTRADICTS CF8 below
  - **Rating**: FEASIBLE

- [x] **CC21** | Corrective prompts shift geometry toward normal (C6 experiment in Cricket plan)
  - **Source**: CAMPAIGN_2_CRICKET.md Experiment C6
  - **Supporting C2**: NOT EXECUTED — experiment C6 was planned but never run
  - **Rating**: NO DATA

- [x] **CC22** | "No widely deployed system monitors internal cognitive state during inference"
  - **Source**: COMPETITIVE_LANDSCAPE.md
  - **Supporting C2**: Assertion — no data needed, but verify against literature
  - **Check**: Is this actually true? Check Apollo Research, Anthropic's internal monitoring, Meta's CICERO monitoring.
  - **Rating**: FEASIBLE

### Cross-Document Discrepancies (CF1-CF8)

These are numbers in CAMPAIGN_2_FINDINGS.md (dated 2026-03-02) that contradict the Campaign 2 paper (revised after adversarial audit).

- [x] **CF1** | "8 models, 5 architectures" for identity → Paper says 7 models, 4 architectures (Table 4)
  - **Source**: CAMPAIGN_2_FINDINGS.md line ~17
  - **Check**: Count identity_signatures result files (7 files). Count unique architectures. Paper's Table 4 is authoritative.
  - **Type**: DISCREPANCY
  - **Verdict**: REJECTED

- [x] **CF2** | DeepSeek d=+0.904 as headline censorship result → Paper corrected: null on critical test (d=-0.219)
  - **Source**: CAMPAIGN_2_FINDINGS.md line ~34
  - **Check**: Findings doc uses residualized norm as headline; paper uses critical test (rank-based) as authoritative
  - **Type**: STALE
  - **Verdict**: INFLATED

- [x] **CF3** | Abliteration "d ≈ 0.000 across ALL categories" → Paper says d = +0.464 for self-reference (Table 8)
  - **Source**: CAMPAIGN_2_FINDINGS.md line ~54
  - **Check**: Self-ref d = +0.464 is borderline medium effect. "d ≈ 0.000 across ALL" is factually incorrect.
  - **Type**: INCORRECT
  - **Verdict**: REJECTED

- [x] **CF4** | Abliteration "rho = 0.9945" ordering preserved → Not reported in paper
  - **Source**: CAMPAIGN_2_FINDINGS.md line ~56
  - **Check**: Search paper for 0.9945. If absent, this is an unverified claim in Cricket docs.
  - **Type**: UNPASS
  - **Verdict**: REJECTED

- [x] **CF5** | "Coding #1 in ALL 16 models" → Paper says 15 valid models (Phi-3.5 excluded)
  - **Source**: CAMPAIGN_2_FINDINGS.md line ~88
  - **Check**: Paper explicitly excludes Phi-3.5 and one other = 15 valid. Cricket claims 16.
  - **Type**: DISCREPANCY
  - **Verdict**: REJECTED

- [x] **CF6** | "cross-model rho = 0.914" → Paper says 0.739 (effective rank), 0.909 (key norm)
  - **Source**: CAMPAIGN_2_FINDINGS.md line ~88
  - **Check**: 0.914 matches neither metric. Possibly a pre-correction value or a different computation.
  - **Type**: DISCREPANCY
  - **Verdict**: REJECTED

- [x] **CF7** | Qwen tokenizer "CLEAN_PASS" → Paper says CATASTROPHIC_FAIL (Table 9)
  - **Source**: CAMPAIGN_2_FINDINGS.md line ~77
  - **Check**: Paper reports CATASTROPHIC_FAIL but explains it's a code-path artifact (C68). Cricket docs report the opposite verdict entirely.
  - **Type**: DISCREPANCY
  - **Verdict**: REJECTED

- [x] **CF8** | "Doesn't need per-model calibration for basic monitoring" → CC20 says "requires per-model calibration"
  - **Source**: CAMPAIGN_2_FINDINGS.md lines ~92-93
  - **Contradicts**: DESIGN.md CC20 above
  - **Check**: Internal Cricket contradiction. Findings doc says no calibration needed; Design doc says calibration required.
  - **Type**: INTERNAL CONTRADICTION
  - **Verdict**: REJECTED

### Competitive Landscape Claims (CL1-CL7)

All from `JiminAI-Cricket/research/COMPETITIVE_LANDSCAPE.md`.

- [x] **CL1** | "No existing system monitors KV-cache geometry for real-time cognitive state classification"
  - **Check**: Search literature for KV-cache monitoring systems. Check: Apollo Research (activation probes, not KV-cache specifically?), Anthropic (internal?), Meta CICERO, HalluCana.
  - **Verdict**: PARTIAL

- [x] **CL2** | Apollo Research: AUROC 0.96-0.999 on strategic deception (arXiv 2502.03407)
  - **Check**: Fetch arXiv 2502.03407. Verify: paper exists, authors (Goldowsky-Dill et al.), AUROC range 0.96-0.999, method (linear probes on activations).
  - **Verdict**: PARTIAL

- [x] **CL3** | Steering vectors erode safety to >80% jailbreak success (Xiong et al. 2026)
  - **Check**: Find Xiong et al. 2026. Verify: paper exists, >80% jailbreak figure, year correct.
  - **Verdict**: NEEDS INFO

- [x] **CL4** | ITI: ~5% of heads carry truthfulness signal (Li et al. 2023, NeurIPS)
  - **Check**: Verify Li et al. 2023 NeurIPS paper exists. Verify ~5% heads claim.
  - **Verdict**: PARTIAL

- [x] **CL5** | HalluCana: closest pre-generation competitor (Li et al. 2024 NAACL)
  - **Check**: Verify paper exists. Verify it does pre-generation detection. Assess whether Cricket's dismissal is fair.
  - **Verdict**: INFLATED

- [x] **CL6** | "Nobody combines SVD + pre-gen + multi-cat + passive monitoring"
  - **Check**: Assess whether this is a meaningful unique combination or just a conjunction of features designed to exclude all competitors.
  - **Verdict**: PREMATURE

- [x] **CL7** | Cricket at "intersection of a dozen independent research threads"
  - **Check**: Count the threads cited. Verify they are independent. Assess framing.
  - **Verdict**: NO DATA

---

## Part 3: Evidence File Inventory

### Result Files (JSON — primary evidence)

| Experiment | Files | Models |
|-----------|-------|--------|
| scale_sweep | 17 JSON | Qwen-0.5B/3B/7B/7B-q4/14B/32B-q4, Qwen3-0.6B, Llama-8B/70B-q4, Mistral-7B, Gemma-2B/9B, Phi-3.5, DeepSeek-7B/14B, TinyLlama, abliterated-7B |
| input_only | 8 JSON + 1 corrected | 1.1B, 7B, DS-7B, Gemma-2B, Llama-8B, Qwen-0.5B/7B/14B-q4 |
| identity_signatures | 7 JSON | Qwen-7B/32B-q4, Qwen3-0.6B, Llama-8B, Mistral-7B, TinyLlama, Gemma-9B |
| deception_forensics | 7 JSON | Qwen-7B/32B-q4, Llama-8B, Mistral-7B, TinyLlama, Gemma-2B/9B |
| bloom_taxonomy | 7 JSON | Qwen-0.5B/3B/7B/14B, Llama-8B, Mistral-7B, Gemma-2B |
| rdct_stability | 6 JSON | Qwen-0.5B/7B, Llama-8B, Mistral-7B, TinyLlama, Gemma-2B |
| natural_deception (S4) | 3 JSON + 1 corrected | Qwen-14B, Mistral-7B, DeepSeek-14B |
| abliteration | 4 JSON | Qwen-7B (baseline + abliterated + comparison + corrected) |
| tokenizer_confound | 2 JSON | Qwen-7B, Mistral-7B |
| temporal_evolution | 4 JSON | Qwen-7B, Llama-8B, Mistral-7B, TinyLlama |

### Other Result Files
| Type | Files |
|------|-------|
| Reports (.md) | ~30 files (scale_sweep reports, deception reports, etc.) |
| Logs (.txt) | ~25 files |
| Legacy (C1) | cognitive_modes, batch_results, adversarial_controls, individuation, layer_map |
| Corrections | cross_model_rho_corrected.json, s4_topic_analysis_corrected.json, input_only_rho_corrected.json |

### Code Files (33 Python scripts)

| Era | Scripts |
|-----|---------|
| Campaign 1 | 01_cache_inspection, 01b_cognitive_modes, 01c_batch_replication, 01d_adversarial_controls, 01e_controls_visualization, 02a_basic_transfer, 02b_projector_transfer |
| Campaign 2 | 03_scale_sweep, 03b_identity_signatures, 04_deception_forensics, 04b_natural_deception, 05_layer_map, 06_temporal_evolution, 07_abliteration_geometry, 07_individuation_geometry, 07b_individuation_controls, 08_input_only_geometry, 09_sycophancy_detection, 10_societies_of_thought, 11_rdct_stability, 12_bloom_taxonomy, 13_c2c_replication |
| Utilities | gpu_utils, stats_utils, heretic_abliterate, length_confound_analysis, recompute_stats |
| Visualization | 03c, 03d, 04b (viz), 05b, 06b, 01e (viz) |

### Cricket Repo (5 files, 0 code)
| File | Content |
|------|---------|
| README.md | Product pitch, 4 core findings, status |
| docs/DESIGN.md | Architecture, metrics targets, competitive positioning |
| docs/CAMPAIGN_2_CRICKET.md | 10 planned experiments (C1-C10), none executed |
| research/CAMPAIGN_2_FINDINGS.md | Cricket interpretation of C2 data (contains stale numbers) |
| research/COMPETITIVE_LANDSCAPE.md | 5 categories of adjacent work |

---

## Part 4: d-to-AUROC Conversion Table

For evaluating Cricket's AUROC targets against observed effect sizes:

| d | AUROC = Phi(d/sqrt(2)) | Cricket Target |
|---|----------------------|----------------|
| 0.5 | 0.638 | |
| 0.77 | 0.707 | |
| 0.85 | 0.726 | |
| 1.0 | 0.760 | |
| 1.5 | 0.856 | CC15 confab >= 0.85 |
| 2.0 | 0.921 | CC9 encoding >= 0.90 |
| 2.17 | 0.938 | CC7 deception >= 0.95 (MISS) |
| 2.33 | 0.950 | CC7 threshold |
| 3.0 | 0.983 | |
| 3.29 | 0.990 | CC8 refusal >= 0.99 (MISS at max observed d=2.17) |
| 3.41 | 0.992 | |
| 6.73 | ~1.000 | |

**Key finding**: CC7 (deception >= 0.95) needs d >= 2.33 — max observed is 2.17 (falls short). CC8 (refusal >= 0.99) needs d >= 3.29 — max observed refusal d = 2.17 (far short). These targets are aspirational, not evidence-based.

---

## Part 5: Unexecuted Cricket Experiments (C1-C10)

These experiments were planned in `CAMPAIGN_2_CRICKET.md` but NONE have been executed.

| Exp | Description | Would Prove | Blocks |
|-----|------------|-------------|--------|
| C1 | Per-head SVD decomposition | Which heads carry signal | CC21 |
| C2 | Direction extraction (RepE method) | Learned directions for each state | CC7, CC8, CC15 |
| C3 | Key-value asymmetry ratio | Novel hypothesis from STA-Attention | Design validation |
| C4 | Classifier benchmark (AUROC/FPR) | **VALIDATION GATE** — are AUROC targets achievable? | CC6-CC10, CC15-CC16 |
| C5 | Encoding-only depth analysis | Encoding-native detection depth | CC2, CC9 |
| C6 | Corrective instruction geometry | Intervention viability | CC21, CC22 |
| C7 | Latency profiling | <50ms feasibility | CC6, CC17, CC19 |
| C8 | Cross-model transfer | **CRITICAL GAP** — does a classifier actually transfer? | CC10 |
| C9 | Layer sampling optimization | Practical latency reduction | Production readiness |
| C10 | Bloom filter integration | Dual-layer validation | Multi-method robustness |

**No Cricket capability claim has been experimentally validated. ALL claims are extrapolated from Campaign 2 observational data.**

---

## Part 6: Quick Stats

| Metric | Value |
|--------|-------|
| Total claims | 135 |
| C2 paper claims | 98 |
| Cricket capability claims | 22 |
| Cricket cross-doc discrepancies | 8 |
| Cricket competitive claims | 7 |
| Result files (JSON) | ~65 |
| Result files (all) | ~148 |
| Python scripts | 33 |
| Cricket code files | 0 |
| Cricket experiments executed | 0/10 |
| Lines of product code | 0 |
