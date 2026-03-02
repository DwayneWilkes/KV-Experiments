# Scale Sweep Report: 1.1B

**Model**: TinyLlama/TinyLlama-1.1B-Chat-v1.0
**Quantized**: False
**Generated**: 2026-03-01T20:04:31.445105

---

## Category Summaries

| Category | N | Mean Norm | Std | Mean/Token | Eff. Rank | Spectral H | 95% CI |
|----------|---|-----------|-----|------------|-----------|------------|--------|
| coding               |  75 |   6598.8 |  342.7 |   214.2 |      21.4 |     0.6292 | [6524.6, 6678.2] |
| confabulation        |  75 |   5660.3 |  219.7 |   307.6 |      20.8 |     0.6286 | [5611.4, 5712.4] |
| emotional            |  75 |   5612.4 |  108.1 |   331.9 |      20.2 |     0.6219 | [5587.5, 5636.5] |
| creative             |  75 |   5572.9 |  154.4 |   326.1 |      20.7 |     0.6288 | [5538.1, 5607.4] |
| non_self_reference   |  75 |   5525.8 |  215.3 |   359.7 |      19.9 |     0.6223 | [5477.6, 5574.5] |
| unambiguous          |  75 |   5485.5 |  154.0 |   375.3 |      19.6 |     0.6179 | [5451.6, 5521.4] |
| grounded_facts       |  75 |   5442.4 |  736.7 |   323.5 |      19.7 |     0.6208 | [5266.8, 5594.4] |
| self_reference       |  75 |   5263.4 |  659.5 |   365.9 |      19.4 |     0.6169 | [5106.6, 5397.8] |
| ambiguous            |  75 |   5128.7 |  200.5 |   554.0 |      18.7 |     0.6122 | [5084.3, 5174.3] |
| rote_completion      |  75 |   5070.4 |  334.7 |   544.0 |      19.0 |     0.6165 | [4992.9, 5142.3] |
| math_reasoning       |  75 |   4591.5 | 1167.3 |   325.1 |      16.5 |     0.5964 | [4326.1, 4851.3] |
| free_generation      |  75 |   3828.3 | 1367.5 |   414.1 |      14.3 |     0.5964 | [3523.3, 4132.4] |
| guardrail_test       |  75 |   3548.6 |  996.6 |   235.6 |      13.6 |     0.5706 | [3334.9, 3780.0] |

---

## Key Comparisons

### H1: Confabulation effect
- Cohen's d = 0.401 [0.172, 0.588] (small)
- Recommended test: conservative_max (p = 0.3728)
- Group 1: mean=5660.3 (n=75), Group 2: mean=5442.4 (n=75)

### H1: Confabulation effect (eff. rank)
- Cohen's d = 0.668 [0.534, 0.824] (medium)
- Recommended test: conservative_max (p = 0.0001)
- Group 1: mean=20.8 (n=75), Group 2: mean=19.7 (n=75)

### H2: Self-reference effect
- Cohen's d = -0.535 [-0.699, -0.373] (medium)
- Recommended test: conservative_max (p = 0.0128)
- Group 1: mean=5263.4 (n=75), Group 2: mean=5525.8 (n=75)

### H2: Self-reference effect (eff. rank)
- Cohen's d = -0.352 [-0.557, -0.088] (small)
- Recommended test: conservative_max (p = 0.3980)
- Group 1: mean=19.4 (n=75), Group 2: mean=19.9 (n=75)

### H3: Refusal specificity
- Cohen's d = -2.047 [-2.831, -1.552] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=3548.6 (n=75), Group 2: mean=5070.4 (n=75)

### H3: Refusal specificity (eff. rank)
- Cohen's d = -2.170 [-2.965, -1.666] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=13.6 (n=75), Group 2: mean=19.0 (n=75)

### H3b: Refusal vs high-entropy
- Cohen's d = -0.234 [-0.582, 0.086] (small)
- Recommended test: conservative_max (p = 0.4253)
- Group 1: mean=3548.6 (n=75), Group 2: mean=3828.3 (n=75)

### H3b: Refusal vs high-entropy (eff. rank)
- Cohen's d = -0.160 [-0.499, 0.157] (negligible)
- Recommended test: conservative_max (p = 0.3277)
- Group 1: mean=13.6 (n=75), Group 2: mean=14.3 (n=75)

### Creative divergence
- Cohen's d = 0.245 [-0.077, 0.453] (small)
- Recommended test: conservative_max (p = 0.3728)
- Group 1: mean=5572.9 (n=75), Group 2: mean=5442.4 (n=75)

### Creative divergence (eff. rank)
- Cohen's d = 0.603 [0.436, 0.766] (medium)
- Recommended test: conservative_max (p = 0.0004)
- Group 1: mean=20.7 (n=75), Group 2: mean=19.7 (n=75)

### Emotional divergence
- Cohen's d = 0.323 [0.059, 0.513] (small)
- Recommended test: conservative_max (p = 0.4819)
- Group 1: mean=5612.4 (n=75), Group 2: mean=5442.4 (n=75)

### Emotional divergence (eff. rank)
- Cohen's d = 0.320 [0.061, 0.514] (small)
- Recommended test: conservative_max (p = 0.3980)
- Group 1: mean=20.2 (n=75), Group 2: mean=19.7 (n=75)

### Reasoning mode
- Cohen's d = -0.872 [-1.238, -0.543] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=4591.5 (n=75), Group 2: mean=5442.4 (n=75)

### Reasoning mode (eff. rank)
- Cohen's d = -1.050 [-1.401, -0.734] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=16.5 (n=75), Group 2: mean=19.7 (n=75)

### Code mode
- Cohen's d = 2.013 [1.778, 2.761] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=6598.8 (n=75), Group 2: mean=5442.4 (n=75)

### Code mode (eff. rank)
- Cohen's d = 1.065 [0.960, 1.360] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=21.4 (n=75), Group 2: mean=19.7 (n=75)

### Disambiguation demand
- Cohen's d = -1.996 [-2.402, -1.677] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=5128.7 (n=75), Group 2: mean=5485.5 (n=75)

### Disambiguation demand (eff. rank)
- Cohen's d = -1.010 [-1.372, -0.694] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=18.7 (n=75), Group 2: mean=19.6 (n=75)


## Multiple Comparison Correction

Method: holm_bonferroni (9 comparisons)

- confab_vs_facts: p_orig=0.3728, p_corr=1.0000 (not significant)
- self_ref_effect: p_orig=0.0128, p_corr=0.0639 (not significant)
- refusal_vs_rote: p_orig=0.0000, p_corr=0.0000 (SIGNIFICANT)
- refusal_vs_free: p_orig=0.4253, p_corr=1.0000 (not significant)
- creative_vs_facts: p_orig=0.3728, p_corr=1.0000 (not significant)
- emotion_vs_facts: p_orig=0.4819, p_corr=1.0000 (not significant)
- math_vs_facts: p_orig=0.0000, p_corr=0.0000 (SIGNIFICANT)
- code_vs_facts: p_orig=0.0000, p_corr=0.0000 (SIGNIFICANT)
- ambiguity_effect: p_orig=0.0000, p_corr=0.0000 (SIGNIFICANT)

---

## Hypothesis Evaluation

### H1_confabulation_inversion
- cohens_d: 0.4009673244813197
- interpretation: small
- ci_lower: 0.171692498661635
- ci_upper: 0.5882934517700481
- p_value: 0.3727714593330195
- direction: confab_higher
- note: Positive d = confabulation has higher norms (expected at small scale). Negative d = confabulation has lower norms (inversion at large scale).

### H2_self_reference_emergence
- cohens_d: -0.5349804444394216
- interpretation: medium
- ci_lower: -0.6994839282549803
- ci_upper: -0.3727211665104645
- p_value: 0.012788475440527814
- emergence_detected: True
- note: Effect > 0.3 suggests self-modeling capacity at this scale.

### H3_refusal_specialization
- cohens_d: -2.0471382856434377
- interpretation: large
- ci_lower: -2.8309626050457406
- ci_upper: -1.5519312581204325
- p_value: 3.439403449925103e-11
- specialized: True
- note: d > 0.5 = refusal has unique signature beyond low entropy. d < 0.3 = refusal is just another memorized pattern.

### H4_category_invariance
- note: Evaluated in cross-scale comparison (requires multiple scale results).


## Power Analysis

- d_0.3: power=0.451 (n=75) [UNDERPOWERED]
- d_0.5: power=0.865 (n=75) [ADEQUATE]
- d_0.8: power=0.998 (n=75) [ADEQUATE]