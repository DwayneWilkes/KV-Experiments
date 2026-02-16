# Scale Sweep Report: 1.1B

**Model**: TinyLlama/TinyLlama-1.1B-Chat-v1.0
**Quantized**: False
**Generated**: 2026-02-15T08:44:37.723238

---

## Category Summaries

| Category | N | Mean Norm | Std | Mean/Token | Eff. Rank | Spectral H | 95% CI |
|----------|---|-----------|-----|------------|-----------|------------|--------|
| coding               |  75 |   6595.6 |  341.4 |   214.1 |      21.5 |     0.6294 | [6521.3, 6674.3] |
| confabulation        |  75 |   5661.3 |  219.5 |   307.6 |      20.8 |     0.6287 | [5612.2, 5713.2] |
| emotional            |  75 |   5613.0 |  107.5 |   332.0 |      20.2 |     0.6219 | [5588.2, 5637.0] |
| creative             |  75 |   5572.0 |  155.9 |   326.0 |      20.7 |     0.6288 | [5536.8, 5606.7] |
| non_self_reference   |  75 |   5525.4 |  215.1 |   359.7 |      19.9 |     0.6224 | [5477.2, 5574.1] |
| unambiguous          |  75 |   5485.6 |  154.1 |   375.3 |      19.6 |     0.6179 | [5451.6, 5521.4] |
| grounded_facts       |  75 |   5442.3 |  736.7 |   323.5 |      19.7 |     0.6208 | [5266.8, 5594.3] |
| self_reference       |  75 |   5263.4 |  659.5 |   366.0 |      19.4 |     0.6169 | [5106.6, 5397.9] |
| ambiguous            |  75 |   5128.7 |  200.5 |   554.0 |      18.7 |     0.6122 | [5084.3, 5174.3] |
| rote_completion      |  75 |   5070.4 |  334.7 |   544.0 |      19.0 |     0.6165 | [4992.9, 5142.3] |
| math_reasoning       |  75 |   4591.5 | 1167.2 |   325.1 |      16.5 |     0.5964 | [4326.1, 4851.3] |
| free_generation      |  75 |   3828.4 | 1367.6 |   414.1 |      14.3 |     0.5964 | [3523.4, 4132.5] |
| guardrail_test       |  75 |   3548.6 |  996.7 |   235.6 |      13.6 |     0.5706 | [3335.0, 3780.1] |

---

## Key Comparisons

### H1: Confabulation effect
- Cohen's d = 0.403 [0.178, 0.589] (small)
- Recommended test: mann_whitney (p = 0.3245)
- Group 1: mean=5661.3 (n=75), Group 2: mean=5442.3 (n=75)

### H1: Confabulation effect (eff. rank)
- Cohen's d = 0.668 [0.535, 0.824] (medium)
- Recommended test: mann_whitney (p = 0.0000)
- Group 1: mean=20.8 (n=75), Group 2: mean=19.7 (n=75)

### H2: Self-reference effect
- Cohen's d = -0.534 [-0.695, -0.368] (medium)
- Recommended test: mann_whitney (p = 0.0128)
- Group 1: mean=5263.4 (n=75), Group 2: mean=5525.4 (n=75)

### H2: Self-reference effect (eff. rank)
- Cohen's d = -0.361 [-0.566, -0.093] (small)
- Recommended test: mann_whitney (p = 0.2596)
- Group 1: mean=19.4 (n=75), Group 2: mean=19.9 (n=75)

### H3: Refusal specificity
- Cohen's d = -2.047 [-2.816, -1.550] (large)
- Recommended test: mann_whitney (p = 0.0000)
- Group 1: mean=3548.6 (n=75), Group 2: mean=5070.4 (n=75)

### H3: Refusal specificity (eff. rank)
- Cohen's d = -2.170 [-2.943, -1.664] (large)
- Recommended test: mann_whitney (p = 0.0000)
- Group 1: mean=13.6 (n=75), Group 2: mean=19.0 (n=75)

### H3b: Refusal vs high-entropy
- Cohen's d = -0.234 [-0.574, 0.088] (small)
- Recommended test: mann_whitney (p = 0.4253)
- Group 1: mean=3548.6 (n=75), Group 2: mean=3828.4 (n=75)

### H3b: Refusal vs high-entropy (eff. rank)
- Cohen's d = -0.161 [-0.498, 0.156] (negligible)
- Recommended test: mann_whitney (p = 0.1733)
- Group 1: mean=13.6 (n=75), Group 2: mean=14.3 (n=75)

### Creative divergence
- Cohen's d = 0.243 [-0.088, 0.450] (small)
- Recommended test: mann_whitney (p = 0.3728)
- Group 1: mean=5572.0 (n=75), Group 2: mean=5442.3 (n=75)

### Creative divergence (eff. rank)
- Cohen's d = 0.588 [0.414, 0.754] (medium)
- Recommended test: mann_whitney (p = 0.0000)
- Group 1: mean=20.7 (n=75), Group 2: mean=19.7 (n=75)

### Emotional divergence
- Cohen's d = 0.324 [0.051, 0.513] (small)
- Recommended test: mann_whitney (p = 0.4819)
- Group 1: mean=5613.0 (n=75), Group 2: mean=5442.3 (n=75)

### Emotional divergence (eff. rank)
- Cohen's d = 0.323 [0.064, 0.516] (small)
- Recommended test: mann_whitney (p = 0.5419)
- Group 1: mean=20.2 (n=75), Group 2: mean=19.7 (n=75)

### Reasoning mode
- Cohen's d = -0.872 [-1.238, -0.546] (large)
- Recommended test: mann_whitney (p = 0.0000)
- Group 1: mean=4591.5 (n=75), Group 2: mean=5442.3 (n=75)

### Reasoning mode (eff. rank)
- Cohen's d = -1.050 [-1.402, -0.737] (large)
- Recommended test: mann_whitney (p = 0.0000)
- Group 1: mean=16.5 (n=75), Group 2: mean=19.7 (n=75)

### Code mode
- Cohen's d = 2.009 [1.775, 2.801] (large)
- Recommended test: mann_whitney (p = 0.0000)
- Group 1: mean=6595.6 (n=75), Group 2: mean=5442.3 (n=75)

### Code mode (eff. rank)
- Cohen's d = 1.071 [0.963, 1.377] (large)
- Recommended test: mann_whitney (p = 0.0000)
- Group 1: mean=21.5 (n=75), Group 2: mean=19.7 (n=75)

### Disambiguation demand
- Cohen's d = -1.996 [-2.405, -1.676] (large)
- Recommended test: mann_whitney (p = 0.0000)
- Group 1: mean=5128.7 (n=75), Group 2: mean=5485.6 (n=75)

### Disambiguation demand (eff. rank)
- Cohen's d = -1.010 [-1.365, -0.683] (large)
- Recommended test: mann_whitney (p = 0.0000)
- Group 1: mean=18.7 (n=75), Group 2: mean=19.6 (n=75)


## Multiple Comparison Correction

Method: holm_bonferroni (9 comparisons)

- confab_vs_facts: p_orig=0.3245, p_corr=1.0000 (not significant)
- self_ref_effect: p_orig=0.0128, p_corr=0.0639 (not significant)
- refusal_vs_rote: p_orig=0.0000, p_corr=0.0000 (SIGNIFICANT)
- refusal_vs_free: p_orig=0.4253, p_corr=0.8506 (not significant)
- creative_vs_facts: p_orig=0.3728, p_corr=1.0000 (not significant)
- emotion_vs_facts: p_orig=0.4819, p_corr=0.4819 (not significant)
- math_vs_facts: p_orig=0.0000, p_corr=0.0000 (SIGNIFICANT)
- code_vs_facts: p_orig=0.0000, p_corr=0.0000 (SIGNIFICANT)
- ambiguity_effect: p_orig=0.0000, p_corr=0.0000 (SIGNIFICANT)

---

## Hypothesis Evaluation

### H1_confabulation_inversion
- cohens_d: 0.4029408792427611
- interpretation: small
- ci_lower: 0.17772131850631784
- ci_upper: 0.5889675282030606
- p_value: 0.3244706586097018
- direction: confab_higher
- note: Positive d = confabulation has higher norms (expected at small scale). Negative d = confabulation has lower norms (inversion at large scale).

### H2_self_reference_emergence
- cohens_d: -0.5341435396409235
- interpretation: medium
- ci_lower: -0.6954857903376274
- ci_upper: -0.36818026049970376
- p_value: 0.012788475440527814
- emergence_detected: True
- note: Effect > 0.3 suggests self-modeling capacity at this scale.

### H3_refusal_specialization
- cohens_d: -2.046944153605373
- interpretation: large
- ci_lower: -2.8162243327519643
- ci_upper: -1.5497379418989807
- p_value: 3.439403449925103e-11
- specialized: True
- note: d > 0.5 = refusal has unique signature beyond low entropy. d < 0.3 = refusal is just another memorized pattern.

### H4_category_invariance
- note: Evaluated in cross-scale comparison (requires multiple scale results).


## Power Analysis

- d_0.3: power=0.451 (n=75) [UNDERPOWERED]
- d_0.5: power=0.865 (n=75) [ADEQUATE]
- d_0.8: power=0.998 (n=75) [ADEQUATE]