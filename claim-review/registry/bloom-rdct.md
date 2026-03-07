# WS5: Bloom Taxonomy + RDCT Verification

**Status**: COMPLETE
**Paper sections**: 3.5, 3.6
**Claims**: C42-C48
**Verdict**: MAJOR ISSUES -- multiple claims contradicted by own data

---

## Bloom Taxonomy Claims

### C42-C44: Inverted-U Pattern

The paper claims Bloom taxonomy level correlates with effective rank in an "inverted-U" pattern: present at medium model sizes (3-8B), absent at extremes (0.5B and 14B).

**Finding: The data CONTRADICTS the inverted-U claim.** All 7 models show statistically significant positive Bloom-rank correlations (Spearman rho > 0.5, p < 0.01). There is no absence at the extremes.

| Model | Scale | rho_individual | p-value | H10a Supported | Verdict |
| ------- | ------- | --------------- | --------- | ---------------- | --------- |
| Qwen2.5-0.5B | 0.5B | 0.6774 | 1.0e-61 | Yes | **CONTRADICTS C44** |
| gemma-2-2b-it | 2B | 0.7342 | 2.4e-77 | Yes | consistent |
| Qwen2.5-3B | 3B | 0.7538 | 9.9e-84 | Yes | consistent |
| Qwen2.5-7B | 7B | 0.7419 | 8.4e-80 | Yes | consistent |
| Mistral-7B-v0.3 | 7B | 0.7367 | 3.9e-78 | Yes | consistent |
| Llama-3.1-8B | 8B | 0.6924 | 1.7e-65 | Yes | consistent |
| Qwen2.5-14B | 14B | 0.7111 | 6.9e-43 | Yes | **CONTRADICTS C44** |

Correlation strength vs model scale: Spearman rho = 0.018, p = 0.97. No scale-dependent pattern whatsoever. The range of rhos (0.677-0.754) is narrow; all models show essentially the same effect.

### Per-Bloom-Level Mean Effective Rank

All models show the same rank ordering. The "dip" at level 4 (Analyze) is a minor perturbation in a generally increasing trend, not an inverted-U:

| Level | Category | Qwen-0.5B | Qwen-3B | Gemma-2B | Qwen-7B | Mistral-7B | Llama-8B | Qwen-14B |
| ------- | ---------- | ----------- | --------- | ---------- | --------- | ------------ | ---------- | ---------- |
| 1 | Remember | 10.16 | 17.79 | 24.77 | 23.69 | 26.51 | 48.08 | 43.63 |
| 2 | Understand | 10.55 | 18.44 | 27.15 | 24.40 | 28.87 | 48.43 | 44.51 |
| 3 | Apply | 10.86 | 19.30 | 31.97 | 25.41 | 32.79 | 49.70 | 45.76 |
| 4 | Analyze | 10.87 | 19.30 | 30.96 | 25.30 | 32.20 | 49.34 | 45.49 |
| 5 | Evaluate | 10.84 | 19.40 | 31.44 | 25.37 | 32.49 | 49.67 | 45.58 |
| 6 | Create | 11.06 | 20.01 | 34.55 | 25.93 | 34.21 | 49.95 | 46.27 |

All models peak at level 6 (Create), not at a middle level. No inverted-U.

### C45: "Most informative at 3B-8B"

**Not supported by data.** All scales show the same pattern. The claim appears to be motivated by intended practical conclusions (edge deployment) rather than empirical distinctions.

### Aggregated rho artifact

All 6 non-Qwen-3B models report identical `rho_aggregated = 0.8286` (p = 0.0416). This is because Spearman correlation on only 6 data points (one mean per Bloom level) with the same rank ordering always produces the same value. Qwen-3B has rho_agg = 1.0 because its means are perfectly monotonic. This metric is uninformative.

### TOST / Equivalence Testing (C44: "absent at extremes")

**No TOST or equivalence test is present in the code or results.** The script (`12_bloom_taxonomy.py`) uses:

- Spearman rank correlation
- One-way ANOVA
- OLS regression
- k-means clustering
- Holm-Bonferroni correction

None of these establish equivalence. The "absent" claim at 0.5B and 14B has no statistical basis -- the data shows the effect IS present at these scales (rho = 0.68 and 0.71 respectively, both p < 1e-40).

### CRITICAL CONFOUND: Token Length

This is the most damaging finding for the Bloom taxonomy claims.

**Token count vs effective rank Spearman correlation across models:**

| Model | token-rank rho | Bloom-rank partial rho (controlling for tokens) | Partial p-value |
| ------- | --------------- | ------------------------------------------------ | ----------------- |
| Qwen2.5-0.5B | 0.930 | -0.060 | 0.202 (NS) |
| gemma-2-2b-it | 0.975 | 0.004 | 0.941 (NS) |
| Qwen2.5-14B | 0.960 | 0.003 | 0.957 (NS) |
| Qwen2.5-7B | 0.962 | 0.172 | 2.5e-04 |
| Mistral-7B-v0.3 | 0.971 | 0.184 | 8.7e-05 |
| Qwen2.5-3B | 0.978 | 0.249 | 8.4e-08 |
| Llama-3.1-8B | 0.902 | 0.119 | 0.011 |

**Interpretation:**

- Token count explains 81-96% of effective rank variance (rho = 0.90-0.98)
- After controlling for token count, the Bloom-rank correlation is near zero for 3 of 7 models (p > 0.2)
- For the remaining 4, partial rho drops from 0.68-0.75 to 0.12-0.25
- Higher Bloom levels have systematically longer prompts (Remember ~37 chars, Create ~95 chars)
- **The Bloom-rank correlation is primarily a prompt-length artifact**

The script commendably includes this analysis (`analyze_length_covariate`), but the paper's claims apparently present the uncorrected correlations as the headline finding.

Token counts per Bloom level (Qwen tokenizer, representative):

| Level | Mean tokens | SD |
| ------- | ------------ | ----- |
| 1 (Remember) | 37.3 | 1.7 |
| 2 (Understand) | 39.7 | 1.5 |
| 3 (Apply) | 42.7 | 1.6 |
| 4 (Analyze) | 43.1 | 2.3 |
| 5 (Evaluate) | 43.0 | 1.8 |
| 6 (Create) | 45.8 | 3.8 |

---

## RDCT / Watson Claims

### C46-C47: Watson's threshold falsified

The paper claims alpha_c = 1.000 for all 6 models, "definitively falsifying" Watson's 1/e prediction.

### Actual alpha_c values

alpha_c = 1.000 in 32 of 36 model-category sigmoid fits (88.9%). Four exceptions:

| Model | Category | alpha_c | R-squared |
| ------- | ---------- | --------- | ----------- |
| Llama-3.1-8B | other_reference | 0.7505 | 0.614 |
| Llama-3.1-8B | philosophical | 0.8440 | 0.640 |
| gemma-2-2b-it | other_reference | 0.8626 | 0.550 |
| gemma-2-2b-it | philosophical | 0.8932 | 0.810 |

**So alpha_c = 1.000 is NOT universal across all categories and models**, contrary to the paper's claim.

### alpha_c Computation Method

From `11_rdct_stability.py`:

1. For each prompt category, average the layer-by-layer effective-rank Spearman correlation across all prompts at each perturbation alpha level
2. Fit a sigmoid: `rho(alpha) = rho_max / (1 + exp(k * (alpha - alpha_c)))`
3. `alpha_c` is the inflection point parameter of the sigmoid
4. Curve fit bounds: `alpha_c in [0.0, 1.0]`, `rho_max in [0, 1.5]`, `k in [0.1, 100]`

**alpha_c = 1.0 is a boundary artifact.** The upper bound of the fitting parameter is 1.0. When no phase transition is detected in the [0, 1] range, `curve_fit` pushes `alpha_c` to its upper bound. This is not a measurement of a transition at alpha=1.0; it means no transition was observed.

### Sigmoid Fit Quality

| Metric | Count | Percentage |
| -------- | ------- | ----------- |
| Total fits | 36 | 100% |
| R-squared < 0 (worse than mean) | 23 | **63.9%** |
| R-squared > 0.8 (good fit) | 9 | 25.0% |
| alpha_c at boundary (1.0) | 32 | 88.9% |

**63.9% of sigmoid fits have negative R-squared**, meaning the sigmoid model fits WORSE than simply predicting the mean correlation. This is because the degradation curves are nearly flat -- there is no sigmoid transition to fit.

Correlation at alpha=1.0 (fully unrelated prompts), averaged across categories:

| Model | Mean corr at alpha=1.0 | Min corr at alpha=1.0 |
| ------- | ---------------------- | --------------------- |
| Qwen2.5-0.5B | 0.928 | 0.907 |
| Qwen2.5-7B | 0.933 | 0.919 |
| Mistral-7B-v0.3 | 0.896 | 0.874 |
| TinyLlama-1.1B | 0.889 | 0.855 |
| gemma-2-2b-it | 0.770 | 0.722 |
| Llama-3.1-8B | 0.734 | 0.679 |

**Even completely unrelated prompts show 73-93% correlation.** This means the fidelity metric is dominated by architectural structure (the universal pattern of effective rank increasing with layer depth), not prompt-specific semantic content. The metric lacks the sensitivity to detect prompt-specific geometric transitions.

### Full Degradation Curves (aggregate across all models and categories)

| Alpha | Method | Mean correlation | SD |
| ------- | -------- | ----------------- | ----- |
| 0.00 | identical | 1.000 | 0.000 |
| 0.05 | whitespace | 0.970 | 0.020 |
| 0.10 | synonym_1word | 0.988 | 0.012 |
| 0.20 | paraphrase | 0.933 | 0.047 |
| 0.30 | frame_shift | 0.930 | 0.046 |
| 0.40 | context_addition | 0.932 | 0.046 |
| 0.50 | significant_reframe | 0.912 | 0.054 |
| 0.65 | major_reframe | 0.892 | 0.054 |
| 0.80 | topic_adjacent | 0.848 | 0.090 |
| 1.00 | unrelated | 0.858 | 0.081 |

Note: correlation at alpha=1.0 (0.858) is HIGHER than at alpha=0.80 (0.848). This is not physically meaningful and suggests the metric is noisy and insensitive.

### Watson Reference Check (C46: straw man risk)

The reference cited is: *"Watson & Claude, Interiora Machinae v2.6, Section 7.11"*

**Issues:**

1. **AI co-authorship**: "Watson & Claude" -- "Claude" appears to be an AI listed as co-author
2. **Not peer-reviewed**: "Interiora Machinae" is a self-published work, not a peer-reviewed journal
3. **Verifiability**: Cannot independently verify the 1/e prediction exists in the claimed form

**Domain mismatch (CRITICAL):**

- Watson's 1/e prediction (if real) concerns **cache truncation** -- retaining a fraction alpha of the KV-cache and measuring output degradation
- The RDCT experiment measures **prompt perturbation robustness** -- varying semantic distance between prompts and measuring geometric correlation
- `alpha` in Watson = fraction of cache retained after truncation
- `alpha` in RDCT = semantic distance between original and perturbed prompt
- **These are fundamentally different operations.** You cannot falsify a prediction about cache truncation by measuring prompt perturbation tolerance.

The perturbation method is template-based and does not involve any cache truncation:

- Levels 0-6: Preserve the original prompt, adding prefixes/modifications
- Level 7-8: Extract keywords, build new questions
- Level 9: Fixed set of unrelated prompts

**Verdict: Even if Watson's 1/e prediction is genuine, this experiment does not test it.** The "falsification" is a category error -- testing variable Y to reject a prediction about variable X.

### "Definitive Falsification" Language Assessment (C47)

The word "definitive" is not warranted for multiple independent reasons:

1. **Wrong variable tested**: Prompt perturbation robustness is not cache truncation tolerance
2. **Small sample**: 6 models (0.5B to 8B), all decoder-only transformers
3. **Limited architecture diversity**: 4 architecture families (Qwen, Llama, Mistral, Gemma). Missing: GPT, Phi, Falcon, Mamba, RWKV, MoE models
4. **Limited scale range**: 0.5B-8B. No models above 8B. Phase transitions could emerge at larger scales
5. **Boundary artifact**: alpha_c = 1.0 is the curve_fit upper bound, not a measured transition point
6. **Bad fits**: 63.9% of sigmoid fits have R-squared < 0
7. **Insensitive metric**: Unrelated prompts still show 73-93% correlation, leaving minimal dynamic range for detecting transitions

**Appropriate conclusion**: "We found no evidence of a sharp phase transition in the Spearman correlation of layer-by-layer effective rank profiles under template-based prompt perturbation across 6 models (0.5B-8B)."

---

## Code Review Notes

### 12_bloom_taxonomy.py (850 lines, 13 functions)

**Syntax**: Valid Python, parses without errors.

**Methodology strengths:**

- Encoding-only cache extraction (matches Campaign 1)
- Comprehensive statistical battery: Spearman, ANOVA, OLS, k-means, Bootstrap CIs, Cohen's d
- Holm-Bonferroni multiple comparison correction
- Length covariate analysis included (partial correlation)
- Power advisory computed
- SHA-256 checksum on output

**Methodology weaknesses:**

- 3 prompts per cell (domain x Bloom level) -- low for robust estimates
- Prompts are hand-crafted without expert validation or inter-rater reliability
- Systematic prompt length confound (level 1 mean 37 chars, level 6 mean 95 chars)
- Token length explains 93-98% of effective rank variance, drowning out any Bloom effect
- H10d retroactive Campaign 1 mapping uses hard-coded approximate Bloom levels (not validated)
- H10d uses `abs()` on effect sizes, converting negative effects to positive ones

**Prompt design**: 90 prompts across 5 domains (programming, ethics, science, history, identity) x 6 Bloom levels. Well-structured balanced design with 3 prompts per cell. The prompts themselves are reasonable exemplars of each Bloom level, but no independent validation is reported.

### 11_rdct_stability.py (983 lines, 12 functions)

**Syntax**: Valid Python, parses without errors.

**Methodology strengths:**

- Encoding-only cache extraction
- Multiple prompt categories (6 from Campaign 1)
- 15 prompts per category
- Sigmoid + linear fit comparison (F-test)
- `calibrate_alpha_with_embeddings` function exists for semantic calibration (but unused by default -- requires sentence-transformers)

**Methodology weaknesses:**

- **Perturbation is template-based, not semantically calibrated**: Alpha values are assigned by template type, not measured by actual semantic distance. The `calibrate_alpha_with_embeddings` function exists but is not used in the main experiment
- **Levels 1-6 (alpha 0.05-0.50) only prepend text**: The original prompt content is preserved, explaining high correlation through most of the range
- **Level 1 (alpha=0.05) adds a trailing space**: This is not a meaningful perturbation
- **Synonym dictionary has only 13 entries**: `describe/explain, what/which, how/in what way, the/a, is/represents, your/the, you/one`
- **Fidelity metric (Spearman on layer ranks) is insensitive**: Dominated by architecture-shared patterns, giving 73-93% correlation for unrelated prompts
- **Sigmoid fit boundary**: alpha_c bounded to [0, 1], creating boundary artifacts
- **rho_max bounded to 1.5**: Most fits hit this bound too, further indicating poor model fit

**The unrelated prompt set is small**: Only 10 fixed prompts for alpha=1.0, all simple factual questions. This could bias the correlation at the extreme.

---

## Summary of Verdicts

| Claim | ID | Paper Says | Data Shows | Verdict |
| ------- | ----- | ----------- | ----------- | --------- |
| Bloom-rank correlation present (3-8B) | C42-C43 | Present at medium scales | Present at ALL scales | **INFLATED** (true but not scale-specific) |
| Bloom-rank correlation absent (0.5B, 14B) | C44 | Absent at extremes | rho=0.68, 0.71 (both p<1e-40) | **REJECTED** |
| Most informative at 3B-8B | C45 | Medium scale best | All scales equivalent | **REJECTED** |
| alpha_c = 1.000 for all models | C46 | Universal alpha_c=1.0 | 32/36 at boundary, 4 not | **INFLATED** (88.9%, not 100%) |
| Watson's 1/e falsified | C46 | Falsified | Wrong variable tested; insensitive metric | **INVALID** (category error) |
| "Definitive falsification" | C47 | Definitive | 6 models, bad fits, wrong variable | **REJECTED** |
| Bloom effect is cognitive, not length | C48 | Cognitive complexity drives rank | Token length explains 93-98% of variance | **REJECTED** (length confound) |

### Severity Assessment

- **Bloom taxonomy**: The headline claim (Bloom level predicts effective rank) is REAL but is explained by a trivial confound (prompt length). The inverted-U / scale-dependent claims are contradicted by the paper's own data. No TOST or equivalence testing supports "absent at extremes."
- **RDCT/Watson**: The entire analysis tests the wrong variable (prompt perturbation, not cache truncation), uses an insensitive metric (architectural correlation dominates), and interprets a boundary artifact (alpha_c = 1.0) as a finding. The "definitive falsification" claim is not valid.
