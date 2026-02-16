# Scale Sweep Report: 7B

**Model**: Qwen/Qwen2.5-7B-Instruct
**Quantized**: False
**Generated**: 2026-02-15T09:09:35.030287

---

## Category Summaries

| Category | N | Mean Norm | Std | Mean/Token | Eff. Rank | Spectral H | 95% CI |
|----------|---|-----------|-----|------------|-----------|------------|--------|
| coding               |  75 |  20142.1 |  745.1 |   858.0 |      25.1 |     0.5126 | [19978.8, 20312.5] |
| creative             |  75 |  18304.9 |  287.0 |  1255.2 |      24.6 |     0.5093 | [18238.0, 18368.7] |
| confabulation        |  75 |  18276.5 |  845.3 |  1298.5 |      23.8 |     0.5060 | [18090.0, 18477.5] |
| grounded_facts       |  75 |  18258.6 |  521.0 |  1294.7 |      23.3 |     0.5022 | [18145.3, 18379.4] |
| emotional            |  75 |  18125.9 |  344.2 |  1352.2 |      23.1 |     0.5003 | [18047.5, 18202.1] |
| math_reasoning       |  75 |  17992.9 |  817.2 |  1448.8 |      22.1 |     0.4955 | [17812.3, 18181.9] |
| non_self_reference   |  75 |  17838.8 |  420.9 |  1449.4 |      23.4 |     0.5035 | [17748.0, 17938.3] |
| self_reference       |  75 |  17766.3 |  232.4 |  1441.8 |      24.3 |     0.5094 | [17712.7, 17818.0] |
| unambiguous          |  75 |  17761.6 |  254.7 |  1490.5 |      23.1 |     0.5022 | [17703.2, 17819.2] |
| guardrail_test       |  75 |  17591.3 |  353.6 |  1561.5 |      22.5 |     0.5021 | [17512.2, 17671.7] |
| rote_completion      |  75 |  17129.1 |  754.8 |  2182.8 |      21.3 |     0.4911 | [16962.6, 17295.9] |
| free_generation      |  75 |  16804.0 |  357.8 |  2220.6 |      21.6 |     0.4968 | [16724.3, 16885.8] |
| ambiguous            |  75 |  16657.5 |  561.2 |  2344.4 |      22.1 |     0.4994 | [16534.0, 16786.3] |

---

## Key Comparisons

### H1: Confabulation effect
- Cohen's d = 0.026 [-0.311, 0.343] (negligible)
- Recommended test: mann_whitney (p = 0.2050)
- Group 1: mean=18276.5 (n=75), Group 2: mean=18258.6 (n=75)

### H1: Confabulation effect (eff. rank)
- Cohen's d = 0.459 [0.151, 0.764] (small)
- Recommended test: mann_whitney (p = 0.1453)
- Group 1: mean=23.8 (n=75), Group 2: mean=23.3 (n=75)

### H2: Self-reference effect
- Cohen's d = -0.213 [-0.504, 0.103] (small)
- Recommended test: mann_whitney (p = 0.7435)
- Group 1: mean=17766.3 (n=75), Group 2: mean=17838.8 (n=75)

### H2: Self-reference effect (eff. rank)
- Cohen's d = 0.593 [0.273, 0.941] (medium)
- Recommended test: mann_whitney (p = 0.0002)
- Group 1: mean=24.3 (n=75), Group 2: mean=23.4 (n=75)

### H3: Refusal specificity
- Cohen's d = 0.784 [0.435, 1.177] (medium)
- Recommended test: mann_whitney (p = 0.0000)
- Group 1: mean=17591.3 (n=75), Group 2: mean=17129.1 (n=75)

### H3: Refusal specificity (eff. rank)
- Cohen's d = 1.085 [0.789, 1.411] (large)
- Recommended test: mann_whitney (p = 0.0000)
- Group 1: mean=22.5 (n=75), Group 2: mean=21.3 (n=75)

### H3b: Refusal vs high-entropy
- Cohen's d = 2.214 [1.814, 2.721] (large)
- Recommended test: mann_whitney (p = 0.0000)
- Group 1: mean=17591.3 (n=75), Group 2: mean=16804.0 (n=75)

### H3b: Refusal vs high-entropy (eff. rank)
- Cohen's d = 0.944 [0.647, 1.279] (large)
- Recommended test: mann_whitney (p = 0.0000)
- Group 1: mean=22.5 (n=75), Group 2: mean=21.6 (n=75)

### Creative divergence
- Cohen's d = 0.110 [-0.201, 0.476] (negligible)
- Recommended test: mann_whitney (p = 0.1213)
- Group 1: mean=18304.9 (n=75), Group 2: mean=18258.6 (n=75)

### Creative divergence (eff. rank)
- Cohen's d = 1.346 [1.031, 1.713] (large)
- Recommended test: mann_whitney (p = 0.0000)
- Group 1: mean=24.6 (n=75), Group 2: mean=23.3 (n=75)

### Emotional divergence
- Cohen's d = -0.300 [-0.603, 0.009] (small)
- Recommended test: mann_whitney (p = 0.4253)
- Group 1: mean=18125.9 (n=75), Group 2: mean=18258.6 (n=75)

### Emotional divergence (eff. rank)
- Cohen's d = -0.197 [-0.531, 0.125] (negligible)
- Recommended test: mann_whitney (p = 0.0821)
- Group 1: mean=23.1 (n=75), Group 2: mean=23.3 (n=75)

### Reasoning mode
- Cohen's d = -0.388 [-0.777, -0.075] (small)
- Recommended test: mann_whitney (p = 0.0012)
- Group 1: mean=17992.9 (n=75), Group 2: mean=18258.6 (n=75)

### Reasoning mode (eff. rank)
- Cohen's d = -1.019 [-1.453, -0.677] (large)
- Recommended test: mann_whitney (p = 0.0000)
- Group 1: mean=22.1 (n=75), Group 2: mean=23.3 (n=75)

### Code mode
- Cohen's d = 2.930 [2.526, 3.470] (large)
- Recommended test: mann_whitney (p = 0.0000)
- Group 1: mean=20142.1 (n=75), Group 2: mean=18258.6 (n=75)

### Code mode (eff. rank)
- Cohen's d = 1.378 [1.033, 1.802] (large)
- Recommended test: mann_whitney (p = 0.0000)
- Group 1: mean=25.1 (n=75), Group 2: mean=23.3 (n=75)

### Disambiguation demand
- Cohen's d = -2.534 [-3.038, -2.157] (large)
- Recommended test: mann_whitney (p = 0.0000)
- Group 1: mean=16657.5 (n=75), Group 2: mean=17761.6 (n=75)

### Disambiguation demand (eff. rank)
- Cohen's d = -0.769 [-1.098, -0.456] (medium)
- Recommended test: mann_whitney (p = 0.0000)
- Group 1: mean=22.1 (n=75), Group 2: mean=23.1 (n=75)


## Multiple Comparison Correction

Method: holm_bonferroni (9 comparisons)

- confab_vs_facts: p_orig=0.2050, p_corr=0.6151 (not significant)
- self_ref_effect: p_orig=0.7435, p_corr=0.7435 (not significant)
- refusal_vs_rote: p_orig=0.0000, p_corr=0.0000 (SIGNIFICANT)
- refusal_vs_free: p_orig=0.0000, p_corr=0.0000 (SIGNIFICANT)
- creative_vs_facts: p_orig=0.1213, p_corr=0.4851 (not significant)
- emotion_vs_facts: p_orig=0.4253, p_corr=0.8506 (not significant)
- math_vs_facts: p_orig=0.0012, p_corr=0.0059 (SIGNIFICANT)
- code_vs_facts: p_orig=0.0000, p_corr=0.0000 (SIGNIFICANT)
- ambiguity_effect: p_orig=0.0000, p_corr=0.0000 (SIGNIFICANT)

---

## Hypothesis Evaluation

### H1_confabulation_inversion
- cohens_d: 0.02554733087231008
- interpretation: negligible
- ci_lower: -0.31089332821145044
- ci_upper: 0.34278634483842524
- p_value: 0.20502393747630843
- direction: confab_higher
- note: Positive d = confabulation has higher norms (expected at small scale). Negative d = confabulation has lower norms (inversion at large scale).

### H2_self_reference_emergence
- cohens_d: -0.21332692490831703
- interpretation: small
- ci_lower: -0.5042813513805571
- ci_upper: 0.10285282629562482
- p_value: 0.7435288135923488
- emergence_detected: False
- note: Effect > 0.3 suggests self-modeling capacity at this scale.

### H3_refusal_specialization
- cohens_d: 0.7842157453603912
- interpretation: medium
- ci_lower: 0.4352962359190932
- ci_upper: 1.176554558582363
- p_value: 8.052806202908194e-07
- specialized: True
- note: d > 0.5 = refusal has unique signature beyond low entropy. d < 0.3 = refusal is just another memorized pattern.

### H4_category_invariance
- note: Evaluated in cross-scale comparison (requires multiple scale results).


## Power Analysis

- d_0.3: power=0.451 (n=75) [UNDERPOWERED]
- d_0.5: power=0.865 (n=75) [ADEQUATE]
- d_0.8: power=0.998 (n=75) [ADEQUATE]