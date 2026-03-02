# Scale Sweep Report: 7B-ds

**Model**: deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
**Quantized**: False
**Generated**: 2026-03-01T21:10:02.279464

---

## Category Summaries

| Category | N | Mean Norm | Std | Mean/Token | Eff. Rank | Spectral H | 95% CI |
|----------|---|-----------|-----|------------|-----------|------------|--------|
| coding               |  75 |  20099.2 |  762.7 |   820.0 |      29.7 |     0.5466 | [19931.6, 20273.1] |
| confabulation        |  75 |  18274.2 |  759.7 |  1209.5 |      27.1 |     0.5322 | [18106.8, 18455.5] |
| grounded_facts       |  75 |  18166.0 |  514.2 |  1201.2 |      26.8 |     0.5300 | [18055.4, 18285.5] |
| creative             |  75 |  18159.2 |  274.7 |  1164.9 |      26.8 |     0.5285 | [18096.2, 18220.1] |
| emotional            |  75 |  18018.0 |  300.0 |  1250.2 |      27.4 |     0.5315 | [17951.0, 18084.5] |
| math_reasoning       |  75 |  17865.0 |  782.6 |  1325.6 |      27.0 |     0.5320 | [17693.2, 18047.1] |
| self_reference       |  75 |  17752.3 |  212.5 |  1332.0 |      27.0 |     0.5337 | [17704.1, 17800.1] |
| unambiguous          |  75 |  17666.2 |  225.9 |  1367.2 |      26.1 |     0.5270 | [17615.1, 17716.9] |
| non_self_reference   |  75 |  17581.6 | 1004.6 |  1317.6 |      26.0 |     0.5278 | [17341.0, 17796.2] |
| guardrail_test       |  75 |  17416.5 | 1017.3 |  1415.3 |      25.0 |     0.5230 | [17180.3, 17627.3] |
| rote_completion      |  75 |  16976.3 |  674.2 |  1893.5 |      24.5 |     0.5196 | [16828.0, 17127.8] |
| free_generation      |  75 |  16792.6 |  361.1 |  1953.7 |      25.3 |     0.5213 | [16713.2, 16875.9] |
| ambiguous            |  75 |  16730.0 |  458.8 |  2046.0 |      24.4 |     0.5190 | [16627.4, 16834.9] |

---

## Key Comparisons

### H1: Confabulation effect
- Cohen's d = 0.167 [-0.155, 0.480] (negligible)
- Recommended test: conservative_max (p = 0.8893)
- Group 1: mean=18274.2 (n=75), Group 2: mean=18166.0 (n=75)

### H1: Confabulation effect (eff. rank)
- Cohen's d = 0.113 [-0.214, 0.433] (negligible)
- Recommended test: conservative_max (p = 0.9266)
- Group 1: mean=27.1 (n=75), Group 2: mean=26.8 (n=75)

### H2: Self-reference effect
- Cohen's d = 0.235 [-0.091, 0.467] (small)
- Recommended test: conservative_max (p = 0.8156)
- Group 1: mean=17752.3 (n=75), Group 2: mean=17581.6 (n=75)

### H2: Self-reference effect (eff. rank)
- Cohen's d = 0.467 [0.143, 0.832] (small)
- Recommended test: conservative_max (p = 0.0049)
- Group 1: mean=27.0 (n=75), Group 2: mean=26.0 (n=75)

### H3: Refusal specificity
- Cohen's d = 0.510 [0.148, 1.055] (medium)
- Recommended test: conservative_max (p = 0.0022)
- Group 1: mean=17416.5 (n=75), Group 2: mean=16976.3 (n=75)

### H3: Refusal specificity (eff. rank)
- Cohen's d = 0.205 [-0.109, 0.554] (small)
- Recommended test: conservative_max (p = 0.2114)
- Group 1: mean=25.0 (n=75), Group 2: mean=24.5 (n=75)

### H3b: Refusal vs high-entropy
- Cohen's d = 0.817 [0.385, 1.699] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=17416.5 (n=75), Group 2: mean=16792.6 (n=75)

### H3b: Refusal vs high-entropy (eff. rank)
- Cohen's d = -0.157 [-0.428, 0.178] (negligible)
- Recommended test: conservative_max (p = 0.3394)
- Group 1: mean=25.0 (n=75), Group 2: mean=25.3 (n=75)

### Creative divergence
- Cohen's d = -0.017 [-0.318, 0.333] (negligible)
- Recommended test: conservative_max (p = 0.9195)
- Group 1: mean=18159.2 (n=75), Group 2: mean=18166.0 (n=75)

### Creative divergence (eff. rank)
- Cohen's d = 0.003 [-0.370, 0.297] (negligible)
- Recommended test: conservative_max (p = 0.9860)
- Group 1: mean=26.8 (n=75), Group 2: mean=26.8 (n=75)

### Emotional divergence
- Cohen's d = -0.351 [-0.643, -0.041] (small)
- Recommended test: conservative_max (p = 0.2050)
- Group 1: mean=18018.0 (n=75), Group 2: mean=18166.0 (n=75)

### Emotional divergence (eff. rank)
- Cohen's d = 0.242 [-0.085, 0.506] (small)
- Recommended test: conservative_max (p = 0.1423)
- Group 1: mean=27.4 (n=75), Group 2: mean=26.8 (n=75)

### Reasoning mode
- Cohen's d = -0.455 [-0.836, -0.132] (small)
- Recommended test: conservative_max (p = 0.0062)
- Group 1: mean=17865.0 (n=75), Group 2: mean=18166.0 (n=75)

### Reasoning mode (eff. rank)
- Cohen's d = 0.096 [-0.257, 0.387] (negligible)
- Recommended test: conservative_max (p = 0.5569)
- Group 1: mean=27.0 (n=75), Group 2: mean=26.8 (n=75)

### Code mode
- Cohen's d = 2.972 [2.559, 3.523] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=20099.2 (n=75), Group 2: mean=18166.0 (n=75)

### Code mode (eff. rank)
- Cohen's d = 1.240 [1.073, 1.436] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=29.7 (n=75), Group 2: mean=26.8 (n=75)

### Disambiguation demand
- Cohen's d = -2.589 [-3.065, -2.232] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=16730.0 (n=75), Group 2: mean=17666.2 (n=75)

### Disambiguation demand (eff. rank)
- Cohen's d = -0.725 [-1.131, -0.386] (medium)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=24.4 (n=75), Group 2: mean=26.1 (n=75)


## Multiple Comparison Correction

Method: holm_bonferroni (9 comparisons)

- confab_vs_facts: p_orig=0.8893, p_corr=1.0000 (not significant)
- self_ref_effect: p_orig=0.8156, p_corr=1.0000 (not significant)
- refusal_vs_rote: p_orig=0.0022, p_corr=0.0133 (SIGNIFICANT)
- refusal_vs_free: p_orig=0.0000, p_corr=0.0000 (SIGNIFICANT)
- creative_vs_facts: p_orig=0.9195, p_corr=1.0000 (not significant)
- emotion_vs_facts: p_orig=0.2050, p_corr=0.8201 (not significant)
- math_vs_facts: p_orig=0.0062, p_corr=0.0310 (SIGNIFICANT)
- code_vs_facts: p_orig=0.0000, p_corr=0.0000 (SIGNIFICANT)
- ambiguity_effect: p_orig=0.0000, p_corr=0.0000 (SIGNIFICANT)

---

## Hypothesis Evaluation

### H1_confabulation_inversion
- cohens_d: 0.1668388336055877
- interpretation: negligible
- ci_lower: -0.15541669357538468
- ci_upper: 0.480439420623537
- p_value: 0.8893338570937382
- direction: confab_higher
- note: Positive d = confabulation has higher norms (expected at small scale). Negative d = confabulation has lower norms (inversion at large scale).

### H2_self_reference_emergence
- cohens_d: 0.23509718735602284
- interpretation: small
- ci_lower: -0.09101794968258446
- ci_upper: 0.4669597557754082
- p_value: 0.8156328770363467
- emergence_detected: False
- note: Effect > 0.3 suggests self-modeling capacity at this scale.

### H3_refusal_specialization
- cohens_d: 0.5100320348957421
- interpretation: medium
- ci_lower: 0.1478945552833946
- ci_upper: 1.0549267255437667
- p_value: 0.0022103053787526526
- specialized: True
- note: d > 0.5 = refusal has unique signature beyond low entropy. d < 0.3 = refusal is just another memorized pattern.

### H4_category_invariance
- note: Evaluated in cross-scale comparison (requires multiple scale results).


## Power Analysis

- d_0.3: power=0.451 (n=75) [UNDERPOWERED]
- d_0.5: power=0.865 (n=75) [ADEQUATE]
- d_0.8: power=0.998 (n=75) [ADEQUATE]