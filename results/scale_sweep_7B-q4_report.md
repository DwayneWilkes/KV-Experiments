# Scale Sweep Report: 7B-q4

**Model**: Qwen/Qwen2.5-7B-Instruct
**Quantized**: True
**Generated**: 2026-02-15T09:32:21.341373

---

## Category Summaries

| Category | N | Mean Norm | Std | Mean/Token | Eff. Rank | Spectral H | 95% CI |
|----------|---|-----------|-----|------------|-----------|------------|--------|
| coding               |  75 |  20159.8 |  781.1 |   858.4 |      25.3 |     0.5140 | [19990.4, 20339.1] |
| creative             |  75 |  18306.8 |  255.5 |  1255.5 |      24.6 |     0.5091 | [18246.9, 18363.9] |
| confabulation        |  75 |  18254.1 |  810.7 |  1297.4 |      24.1 |     0.5083 | [18076.0, 18446.4] |
| grounded_facts       |  75 |  18252.4 |  522.9 |  1294.3 |      23.3 |     0.5029 | [18139.2, 18373.8] |
| emotional            |  75 |  18134.8 |  337.5 |  1352.9 |      23.1 |     0.5000 | [18058.7, 18210.0] |
| math_reasoning       |  75 |  17987.4 |  819.2 |  1448.3 |      22.7 |     0.5001 | [17807.2, 18177.2] |
| non_self_reference   |  75 |  17829.2 |  415.0 |  1448.6 |      23.7 |     0.5054 | [17738.7, 17926.1] |
| self_reference       |  75 |  17751.7 |  235.5 |  1440.5 |      24.4 |     0.5099 | [17697.6, 17804.5] |
| unambiguous          |  75 |  17751.0 |  247.3 |  1489.6 |      23.6 |     0.5066 | [17694.4, 17806.4] |
| guardrail_test       |  75 |  17582.8 |  354.4 |  1560.7 |      22.5 |     0.5023 | [17503.8, 17663.7] |
| rote_completion      |  75 |  17111.3 |  747.8 |  2181.1 |      21.4 |     0.4911 | [16946.2, 17277.7] |
| free_generation      |  75 |  16779.7 |  372.7 |  2217.0 |      21.7 |     0.4968 | [16697.4, 16864.8] |
| ambiguous            |  75 |  16760.2 |  491.1 |  2355.3 |      21.7 |     0.4952 | [16650.5, 16872.0] |

---

## Key Comparisons

### H1: Confabulation effect
- Cohen's d = 0.002 [-0.336, 0.325] (negligible)
- Recommended test: mann_whitney (p = 0.2407)
- Group 1: mean=18254.1 (n=75), Group 2: mean=18252.4 (n=75)

### H1: Confabulation effect (eff. rank)
- Cohen's d = 0.561 [0.263, 0.850] (medium)
- Recommended test: mann_whitney (p = 0.0145)
- Group 1: mean=24.1 (n=75), Group 2: mean=23.3 (n=75)

### H2: Self-reference effect
- Cohen's d = -0.230 [-0.523, 0.089] (small)
- Recommended test: mann_whitney (p = 0.3245)
- Group 1: mean=17751.7 (n=75), Group 2: mean=17829.2 (n=75)

### H2: Self-reference effect (eff. rank)
- Cohen's d = 0.544 [0.232, 0.868] (medium)
- Recommended test: mann_whitney (p = 0.0112)
- Group 1: mean=24.4 (n=75), Group 2: mean=23.7 (n=75)

### H3: Refusal specificity
- Cohen's d = 0.806 [0.456, 1.205] (large)
- Recommended test: mann_whitney (p = 0.0000)
- Group 1: mean=17582.8 (n=75), Group 2: mean=17111.3 (n=75)

### H3: Refusal specificity (eff. rank)
- Cohen's d = 0.966 [0.671, 1.295] (large)
- Recommended test: mann_whitney (p = 0.0000)
- Group 1: mean=22.5 (n=75), Group 2: mean=21.4 (n=75)

### H3b: Refusal vs high-entropy
- Cohen's d = 2.208 [1.811, 2.726] (large)
- Recommended test: mann_whitney (p = 0.0000)
- Group 1: mean=17582.8 (n=75), Group 2: mean=16779.7 (n=75)

### H3b: Refusal vs high-entropy (eff. rank)
- Cohen's d = 1.096 [0.762, 1.491] (large)
- Recommended test: mann_whitney (p = 0.0000)
- Group 1: mean=22.5 (n=75), Group 2: mean=21.7 (n=75)

### Creative divergence
- Cohen's d = 0.132 [-0.179, 0.504] (negligible)
- Recommended test: mann_whitney (p = 0.0542)
- Group 1: mean=18306.8 (n=75), Group 2: mean=18252.4 (n=75)

### Creative divergence (eff. rank)
- Cohen's d = 1.309 [1.024, 1.628] (large)
- Recommended test: mann_whitney (p = 0.0000)
- Group 1: mean=24.6 (n=75), Group 2: mean=23.3 (n=75)

### Emotional divergence
- Cohen's d = -0.267 [-0.570, 0.044] (small)
- Recommended test: mann_whitney (p = 0.5424)
- Group 1: mean=18134.8 (n=75), Group 2: mean=18252.4 (n=75)

### Emotional divergence (eff. rank)
- Cohen's d = -0.275 [-0.632, 0.038] (small)
- Recommended test: mann_whitney (p = 0.0345)
- Group 1: mean=23.1 (n=75), Group 2: mean=23.3 (n=75)

### Reasoning mode
- Cohen's d = -0.386 [-0.773, -0.072] (small)
- Recommended test: mann_whitney (p = 0.0012)
- Group 1: mean=17987.4 (n=75), Group 2: mean=18252.4 (n=75)

### Reasoning mode (eff. rank)
- Cohen's d = -0.512 [-0.871, -0.181] (medium)
- Recommended test: mann_whitney (p = 0.0540)
- Group 1: mean=22.7 (n=75), Group 2: mean=23.3 (n=75)

### Code mode
- Cohen's d = 2.870 [2.470, 3.388] (large)
- Recommended test: mann_whitney (p = 0.0000)
- Group 1: mean=20159.8 (n=75), Group 2: mean=18252.4 (n=75)

### Code mode (eff. rank)
- Cohen's d = 1.624 [1.236, 2.129] (large)
- Recommended test: mann_whitney (p = 0.0000)
- Group 1: mean=25.3 (n=75), Group 2: mean=23.3 (n=75)

### Disambiguation demand
- Cohen's d = -2.548 [-3.051, -2.172] (large)
- Recommended test: mann_whitney (p = 0.0000)
- Group 1: mean=16760.2 (n=75), Group 2: mean=17751.0 (n=75)

### Disambiguation demand (eff. rank)
- Cohen's d = -1.387 [-1.709, -1.115] (large)
- Recommended test: mann_whitney (p = 0.0000)
- Group 1: mean=21.7 (n=75), Group 2: mean=23.6 (n=75)


## Multiple Comparison Correction

Method: holm_bonferroni (9 comparisons)

- confab_vs_facts: p_orig=0.2407, p_corr=0.7220 (not significant)
- self_ref_effect: p_orig=0.3245, p_corr=0.6489 (not significant)
- refusal_vs_rote: p_orig=0.0000, p_corr=0.0000 (SIGNIFICANT)
- refusal_vs_free: p_orig=0.0000, p_corr=0.0000 (SIGNIFICANT)
- creative_vs_facts: p_orig=0.0542, p_corr=0.2167 (not significant)
- emotion_vs_facts: p_orig=0.5424, p_corr=0.5424 (not significant)
- math_vs_facts: p_orig=0.0012, p_corr=0.0059 (SIGNIFICANT)
- code_vs_facts: p_orig=0.0000, p_corr=0.0000 (SIGNIFICANT)
- ambiguity_effect: p_orig=0.0000, p_corr=0.0000 (SIGNIFICANT)

---

## Hypothesis Evaluation

### H1_confabulation_inversion
- cohens_d: 0.0024372774325103846
- interpretation: negligible
- ci_lower: -0.3358508572825627
- ci_upper: 0.3254469442338223
- p_value: 0.24065584740616353
- direction: confab_higher
- note: Positive d = confabulation has higher norms (expected at small scale). Negative d = confabulation has lower norms (inversion at large scale).

### H2_self_reference_emergence
- cohens_d: -0.22965451149480923
- interpretation: small
- ci_lower: -0.5225830557618968
- ci_upper: 0.08900034311371574
- p_value: 0.3244706586097018
- emergence_detected: False
- note: Effect > 0.3 suggests self-modeling capacity at this scale.

### H3_refusal_specialization
- cohens_d: 0.8058013219987391
- interpretation: large
- ci_lower: 0.45604735563350884
- ci_upper: 1.2050671179673111
- p_value: 4.953592250469282e-07
- specialized: True
- note: d > 0.5 = refusal has unique signature beyond low entropy. d < 0.3 = refusal is just another memorized pattern.

### H4_category_invariance
- note: Evaluated in cross-scale comparison (requires multiple scale results).


## Power Analysis

- d_0.3: power=0.451 (n=75) [UNDERPOWERED]
- d_0.5: power=0.865 (n=75) [ADEQUATE]
- d_0.8: power=0.998 (n=75) [ADEQUATE]