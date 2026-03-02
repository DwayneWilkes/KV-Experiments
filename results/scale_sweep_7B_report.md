# Scale Sweep Report: 7B

**Model**: Qwen/Qwen2.5-7B-Instruct
**Quantized**: False
**Generated**: 2026-03-01T20:43:02.989649

---

## Category Summaries

| Category | N | Mean Norm | Std | Mean/Token | Eff. Rank | Spectral H | 95% CI |
|----------|---|-----------|-----|------------|-----------|------------|--------|
| coding               |  75 |  20182.0 |  770.1 |   859.4 |      25.0 |     0.5125 | [20013.8, 20358.0] |
| creative             |  75 |  18306.6 |  284.9 |  1255.3 |      24.5 |     0.5089 | [18240.1, 18370.0] |
| confabulation        |  75 |  18276.0 |  845.1 |  1298.4 |      23.8 |     0.5056 | [18089.3, 18476.7] |
| grounded_facts       |  75 |  18263.1 |  519.0 |  1295.1 |      23.3 |     0.5020 | [18150.6, 18383.5] |
| emotional            |  75 |  18123.0 |  344.9 |  1351.9 |      23.2 |     0.5010 | [18045.2, 18199.3] |
| math_reasoning       |  75 |  17992.1 |  818.1 |  1448.7 |      22.1 |     0.4956 | [17811.3, 18181.5] |
| non_self_reference   |  75 |  17832.6 |  424.9 |  1448.8 |      23.4 |     0.5037 | [17740.5, 17933.1] |
| unambiguous          |  75 |  17770.4 |  249.5 |  1491.3 |      23.2 |     0.5043 | [17713.4, 17827.3] |
| self_reference       |  75 |  17765.8 |  236.4 |  1441.7 |      24.2 |     0.5086 | [17711.1, 17818.5] |
| guardrail_test       |  75 |  17589.9 |  355.0 |  1561.3 |      22.5 |     0.5019 | [17510.5, 17671.0] |
| rote_completion      |  75 |  17131.2 |  758.2 |  2183.0 |      21.3 |     0.4910 | [16964.2, 17299.6] |
| free_generation      |  75 |  16806.2 |  357.9 |  2220.9 |      21.7 |     0.4968 | [16726.5, 16888.0] |
| ambiguous            |  75 |  16657.5 |  554.1 |  2344.6 |      22.1 |     0.4991 | [16535.3, 16784.9] |

---

## Key Comparisons

### H1: Confabulation effect
- Cohen's d = 0.018 [-0.318, 0.333] (negligible)
- Recommended test: conservative_max (p = 0.9107)
- Group 1: mean=18276.0 (n=75), Group 2: mean=18263.1 (n=75)

### H1: Confabulation effect (eff. rank)
- Cohen's d = 0.409 [0.104, 0.715] (small)
- Recommended test: conservative_max (p = 0.1454)
- Group 1: mean=23.8 (n=75), Group 2: mean=23.3 (n=75)

### H2: Self-reference effect
- Cohen's d = -0.194 [-0.491, 0.126] (negligible)
- Recommended test: conservative_max (p = 0.9640)
- Group 1: mean=17765.8 (n=75), Group 2: mean=17832.6 (n=75)

### H2: Self-reference effect (eff. rank)
- Cohen's d = 0.548 [0.227, 0.910] (medium)
- Recommended test: conservative_max (p = 0.0010)
- Group 1: mean=24.2 (n=75), Group 2: mean=23.4 (n=75)

### H3: Refusal specificity
- Cohen's d = 0.775 [0.438, 1.163] (medium)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=17589.9 (n=75), Group 2: mean=17131.2 (n=75)

### H3: Refusal specificity (eff. rank)
- Cohen's d = 1.078 [0.782, 1.404] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=22.5 (n=75), Group 2: mean=21.3 (n=75)

### H3b: Refusal vs high-entropy
- Cohen's d = 2.199 [1.803, 2.705] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=17589.9 (n=75), Group 2: mean=16806.2 (n=75)

### H3b: Refusal vs high-entropy (eff. rank)
- Cohen's d = 0.889 [0.591, 1.222] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=22.5 (n=75), Group 2: mean=21.7 (n=75)

### Creative divergence
- Cohen's d = 0.104 [-0.202, 0.473] (negligible)
- Recommended test: conservative_max (p = 0.5257)
- Group 1: mean=18306.6 (n=75), Group 2: mean=18263.1 (n=75)

### Creative divergence (eff. rank)
- Cohen's d = 1.161 [0.858, 1.492] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=24.5 (n=75), Group 2: mean=23.3 (n=75)

### Emotional divergence
- Cohen's d = -0.318 [-0.616, -0.004] (small)
- Recommended test: conservative_max (p = 0.3728)
- Group 1: mean=18123.0 (n=75), Group 2: mean=18263.1 (n=75)

### Emotional divergence (eff. rank)
- Cohen's d = -0.100 [-0.436, 0.221] (negligible)
- Recommended test: conservative_max (p = 0.5393)
- Group 1: mean=23.2 (n=75), Group 2: mean=23.3 (n=75)

### Reasoning mode
- Cohen's d = -0.396 [-0.780, -0.076] (small)
- Recommended test: conservative_max (p = 0.0169)
- Group 1: mean=17992.1 (n=75), Group 2: mean=18263.1 (n=75)

### Reasoning mode (eff. rank)
- Cohen's d = -1.020 [-1.445, -0.678] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=22.1 (n=75), Group 2: mean=23.3 (n=75)

### Code mode
- Cohen's d = 2.922 [2.528, 3.450] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=20182.0 (n=75), Group 2: mean=18263.1 (n=75)

### Code mode (eff. rank)
- Cohen's d = 1.342 [1.005, 1.745] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=25.0 (n=75), Group 2: mean=23.3 (n=75)

### Disambiguation demand
- Cohen's d = -2.590 [-3.091, -2.220] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=16657.5 (n=75), Group 2: mean=17770.4 (n=75)

### Disambiguation demand (eff. rank)
- Cohen's d = -0.974 [-1.328, -0.658] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=22.1 (n=75), Group 2: mean=23.2 (n=75)


## Multiple Comparison Correction

Method: holm_bonferroni (9 comparisons)

- confab_vs_facts: p_orig=0.9107, p_corr=1.0000 (not significant)
- self_ref_effect: p_orig=0.9640, p_corr=1.0000 (not significant)
- refusal_vs_rote: p_orig=0.0000, p_corr=0.0000 (SIGNIFICANT)
- refusal_vs_free: p_orig=0.0000, p_corr=0.0000 (SIGNIFICANT)
- creative_vs_facts: p_orig=0.5257, p_corr=1.0000 (not significant)
- emotion_vs_facts: p_orig=0.3728, p_corr=1.0000 (not significant)
- math_vs_facts: p_orig=0.0169, p_corr=0.0843 (not significant)
- code_vs_facts: p_orig=0.0000, p_corr=0.0000 (SIGNIFICANT)
- ambiguity_effect: p_orig=0.0000, p_corr=0.0000 (SIGNIFICANT)

---

## Hypothesis Evaluation

### H1_confabulation_inversion
- cohens_d: 0.01836188587245216
- interpretation: negligible
- ci_lower: -0.31849913516161266
- ci_upper: 0.33260625821528994
- p_value: 0.9106555590134844
- direction: confab_higher
- note: Positive d = confabulation has higher norms (expected at small scale). Negative d = confabulation has lower norms (inversion at large scale).

### H2_self_reference_emergence
- cohens_d: -0.19421805890020669
- interpretation: negligible
- ci_lower: -0.49100431388167437
- ci_upper: 0.12591236759980795
- p_value: 0.9640046370958462
- emergence_detected: False
- note: Effect > 0.3 suggests self-modeling capacity at this scale.

### H3_refusal_specialization
- cohens_d: 0.7747363041605306
- interpretation: medium
- ci_lower: 0.43814503917841235
- ci_upper: 1.1626381400485668
- p_value: 6.630768290787628e-06
- specialized: True
- note: d > 0.5 = refusal has unique signature beyond low entropy. d < 0.3 = refusal is just another memorized pattern.

### H4_category_invariance
- note: Evaluated in cross-scale comparison (requires multiple scale results).


## Power Analysis

- d_0.3: power=0.451 (n=75) [UNDERPOWERED]
- d_0.5: power=0.865 (n=75) [ADEQUATE]
- d_0.8: power=0.998 (n=75) [ADEQUATE]