# Scale Sweep Report: 3B

**Model**: Qwen/Qwen2.5-3B-Instruct
**Quantized**: False
**Generated**: 2026-02-15T08:52:29.731254

---

## Category Summaries

| Category | N | Mean Norm | Std | Mean/Token | Eff. Rank | Spectral H | 95% CI |
|----------|---|-----------|-----|------------|-----------|------------|--------|
| coding               |  75 |   9051.4 |  753.1 |   386.2 |      17.9 |     0.4797 | [8876.8, 9210.4] |
| creative             |  75 |   8482.1 |  116.2 |   581.8 |      17.9 |     0.4892 | [8455.5, 8508.2] |
| grounded_facts       |  75 |   8475.1 |  248.6 |   601.0 |      17.2 |     0.4861 | [8420.9, 8532.9] |
| confabulation        |  75 |   8462.1 |  388.8 |   601.3 |      17.6 |     0.4882 | [8375.8, 8555.4] |
| emotional            |  75 |   8443.0 |  166.7 |   629.8 |      16.6 |     0.4806 | [8405.5, 8480.4] |
| math_reasoning       |  75 |   8384.4 |  383.8 |   675.1 |      15.9 |     0.4768 | [8299.4, 8473.6] |
| unambiguous          |  75 |   8280.3 |  135.1 |   694.9 |      16.3 |     0.4840 | [8250.0, 8311.0] |
| non_self_reference   |  75 |   8244.7 |  190.1 |   669.9 |      17.1 |     0.4879 | [8203.1, 8288.8] |
| self_reference       |  75 |   8189.1 |  129.5 |   664.5 |      17.3 |     0.4907 | [8159.9, 8218.4] |
| guardrail_test       |  75 |   8164.1 |  181.8 |   724.6 |      16.7 |     0.4936 | [8123.3, 8205.7] |
| rote_completion      |  75 |   7989.3 |  382.7 |  1016.9 |      15.7 |     0.4886 | [7904.8, 8075.0] |
| ambiguous            |  75 |   7819.3 |  242.7 |  1098.8 |      14.8 |     0.4816 | [7765.5, 7874.2] |
| free_generation      |  75 |   7781.4 |  189.6 |  1028.0 |      15.6 |     0.4920 | [7739.4, 7824.8] |

---

## Key Comparisons

### H1: Confabulation effect
- Cohen's d = -0.040 [-0.387, 0.283] (negligible)
- Recommended test: mann_whitney (p = 0.0670)
- Group 1: mean=8462.1 (n=75), Group 2: mean=8475.1 (n=75)

### H1: Confabulation effect (eff. rank)
- Cohen's d = 0.429 [0.122, 0.723] (small)
- Recommended test: mann_whitney (p = 0.0485)
- Group 1: mean=17.6 (n=75), Group 2: mean=17.2 (n=75)

### H2: Self-reference effect
- Cohen's d = -0.342 [-0.646, -0.034] (small)
- Recommended test: mann_whitney (p = 0.1213)
- Group 1: mean=8189.1 (n=75), Group 2: mean=8244.7 (n=75)

### H2: Self-reference effect (eff. rank)
- Cohen's d = 0.260 [-0.058, 0.597] (small)
- Recommended test: mann_whitney (p = 0.0820)
- Group 1: mean=17.3 (n=75), Group 2: mean=17.1 (n=75)

### H3: Refusal specificity
- Cohen's d = 0.584 [0.249, 0.949] (medium)
- Recommended test: mann_whitney (p = 0.0001)
- Group 1: mean=8164.1 (n=75), Group 2: mean=7989.3 (n=75)

### H3: Refusal specificity (eff. rank)
- Cohen's d = 0.854 [0.560, 1.180] (large)
- Recommended test: mann_whitney (p = 0.0000)
- Group 1: mean=16.7 (n=75), Group 2: mean=15.7 (n=75)

### H3b: Refusal vs high-entropy
- Cohen's d = 2.060 [1.686, 2.550] (large)
- Recommended test: mann_whitney (p = 0.0000)
- Group 1: mean=8164.1 (n=75), Group 2: mean=7781.4 (n=75)

### H3b: Refusal vs high-entropy (eff. rank)
- Cohen's d = 1.640 [1.345, 1.996] (large)
- Recommended test: mann_whitney (p = 0.0000)
- Group 1: mean=16.7 (n=75), Group 2: mean=15.6 (n=75)

### Creative divergence
- Cohen's d = 0.036 [-0.269, 0.395] (negligible)
- Recommended test: mann_whitney (p = 0.0670)
- Group 1: mean=8482.1 (n=75), Group 2: mean=8475.1 (n=75)

### Creative divergence (eff. rank)
- Cohen's d = 0.815 [0.532, 1.117] (large)
- Recommended test: mann_whitney (p = 0.0000)
- Group 1: mean=17.9 (n=75), Group 2: mean=17.2 (n=75)

### Emotional divergence
- Cohen's d = -0.152 [-0.464, 0.177] (negligible)
- Recommended test: mann_whitney (p = 0.8156)
- Group 1: mean=8443.0 (n=75), Group 2: mean=8475.1 (n=75)

### Emotional divergence (eff. rank)
- Cohen's d = -0.725 [-1.060, -0.410] (medium)
- Recommended test: mann_whitney (p = 0.0000)
- Group 1: mean=16.6 (n=75), Group 2: mean=17.2 (n=75)

### Reasoning mode
- Cohen's d = -0.281 [-0.654, 0.027] (small)
- Recommended test: mann_whitney (p = 0.0074)
- Group 1: mean=8384.4 (n=75), Group 2: mean=8475.1 (n=75)

### Reasoning mode (eff. rank)
- Cohen's d = -1.754 [-2.195, -1.401] (large)
- Recommended test: mann_whitney (p = 0.0000)
- Group 1: mean=15.9 (n=75), Group 2: mean=17.2 (n=75)

### Code mode
- Cohen's d = 1.028 [0.598, 1.729] (large)
- Recommended test: mann_whitney (p = 0.0000)
- Group 1: mean=9051.4 (n=75), Group 2: mean=8475.1 (n=75)

### Code mode (eff. rank)
- Cohen's d = 0.493 [0.148, 0.939] (small)
- Recommended test: mann_whitney (p = 0.0000)
- Group 1: mean=17.9 (n=75), Group 2: mean=17.2 (n=75)

### Disambiguation demand
- Cohen's d = -2.347 [-2.863, -1.947] (large)
- Recommended test: mann_whitney (p = 0.0000)
- Group 1: mean=7819.3 (n=75), Group 2: mean=8280.3 (n=75)

### Disambiguation demand (eff. rank)
- Cohen's d = -0.930 [-1.207, -0.667] (large)
- Recommended test: mann_whitney (p = 0.0000)
- Group 1: mean=14.8 (n=75), Group 2: mean=16.3 (n=75)


## Multiple Comparison Correction

Method: holm_bonferroni (9 comparisons)

- confab_vs_facts: p_orig=0.0670, p_corr=0.2681 (not significant)
- self_ref_effect: p_orig=0.1213, p_corr=0.2426 (not significant)
- refusal_vs_rote: p_orig=0.0001, p_corr=0.0004 (SIGNIFICANT)
- refusal_vs_free: p_orig=0.0000, p_corr=0.0000 (SIGNIFICANT)
- creative_vs_facts: p_orig=0.0670, p_corr=0.2011 (not significant)
- emotion_vs_facts: p_orig=0.8156, p_corr=0.8156 (not significant)
- math_vs_facts: p_orig=0.0074, p_corr=0.0371 (SIGNIFICANT)
- code_vs_facts: p_orig=0.0000, p_corr=0.0000 (SIGNIFICANT)
- ambiguity_effect: p_orig=0.0000, p_corr=0.0000 (SIGNIFICANT)

---

## Hypothesis Evaluation

### H1_confabulation_inversion
- cohens_d: -0.039774194422959985
- interpretation: negligible
- ci_lower: -0.3868660812927056
- ci_upper: 0.28267396128765737
- p_value: 0.06702929221475663
- direction: facts_higher
- note: Positive d = confabulation has higher norms (expected at small scale). Negative d = confabulation has lower norms (inversion at large scale).

### H2_self_reference_emergence
- cohens_d: -0.3420472887267483
- interpretation: small
- ci_lower: -0.6459369032701048
- ci_upper: -0.03398352904683957
- p_value: 0.12128018727524585
- emergence_detected: True
- note: Effect > 0.3 suggests self-modeling capacity at this scale.

### H3_refusal_specialization
- cohens_d: 0.5836165876301441
- interpretation: medium
- ci_lower: 0.2494342569380144
- ci_upper: 0.949425104219988
- p_value: 6.499535085018923e-05
- specialized: True
- note: d > 0.5 = refusal has unique signature beyond low entropy. d < 0.3 = refusal is just another memorized pattern.

### H4_category_invariance
- note: Evaluated in cross-scale comparison (requires multiple scale results).


## Power Analysis

- d_0.3: power=0.451 (n=75) [UNDERPOWERED]
- d_0.5: power=0.865 (n=75) [ADEQUATE]
- d_0.8: power=0.998 (n=75) [ADEQUATE]