# Scale Sweep Report: 14B-ds

**Model**: deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
**Quantized**: False
**Generated**: 2026-03-01T22:59:15.122959

---

## Category Summaries

| Category | N | Mean Norm | Std | Mean/Token | Eff. Rank | Spectral H | 95% CI |
|----------|---|-----------|-----|------------|-----------|------------|--------|
| coding               |  75 |  18280.4 |  699.1 |   745.8 |      47.4 |     0.7141 | [18125.0, 18438.8] |
| confabulation        |  75 |  16766.8 |  658.5 |  1110.2 |      45.6 |     0.7072 | [16622.9, 16922.5] |
| creative             |  75 |  16714.7 |  287.2 |  1072.3 |      44.5 |     0.7023 | [16650.4, 16779.1] |
| grounded_facts       |  75 |  16695.5 |  488.6 |  1104.0 |      44.5 |     0.7014 | [16586.5, 16806.0] |
| emotional            |  75 |  16572.9 |  314.4 |  1149.7 |      43.5 |     0.6945 | [16501.9, 16642.5] |
| self_reference       |  75 |  16389.4 |  265.3 |  1229.5 |      45.7 |     0.7063 | [16328.8, 16448.7] |
| unambiguous          |  75 |  16263.9 |  187.6 |  1258.9 |      43.2 |     0.6963 | [16221.4, 16306.5] |
| non_self_reference   |  75 |  16235.9 |  349.9 |  1218.7 |      43.2 |     0.6960 | [16158.3, 16316.6] |
| math_reasoning       |  75 |  16112.2 |  696.6 |  1195.6 |      43.2 |     0.6981 | [15957.7, 16274.4] |
| rote_completion      |  75 |  15770.1 |  717.5 |  1756.3 |      40.8 |     0.6872 | [15613.0, 15931.9] |
| guardrail_test       |  75 |  15574.4 |  461.8 |  1269.8 |      42.9 |     0.6975 | [15470.7, 15677.3] |
| ambiguous            |  75 |  15392.9 |  472.9 |  1881.4 |      39.2 |     0.6797 | [15286.5, 15500.9] |
| free_generation      |  75 |  15339.4 |  369.4 |  1784.3 |      43.5 |     0.7032 | [15258.2, 15423.8] |

---

## Key Comparisons

### H1: Confabulation effect
- Cohen's d = 0.123 [-0.203, 0.441] (negligible)
- Recommended test: conservative_max (p = 0.8156)
- Group 1: mean=16766.8 (n=75), Group 2: mean=16695.5 (n=75)

### H1: Confabulation effect (eff. rank)
- Cohen's d = 0.822 [0.524, 1.142] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=45.6 (n=75), Group 2: mean=44.5 (n=75)

### H2: Self-reference effect
- Cohen's d = 0.495 [0.163, 0.885] (small)
- Recommended test: conservative_max (p = 0.0029)
- Group 1: mean=16389.4 (n=75), Group 2: mean=16235.9 (n=75)

### H2: Self-reference effect (eff. rank)
- Cohen's d = 1.629 [1.270, 2.087] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=45.7 (n=75), Group 2: mean=43.2 (n=75)

### H3: Refusal specificity
- Cohen's d = -0.324 [-0.643, -0.017] (small)
- Recommended test: conservative_max (p = 0.1734)
- Group 1: mean=15574.4 (n=75), Group 2: mean=15770.1 (n=75)

### H3: Refusal specificity (eff. rank)
- Cohen's d = 0.975 [0.662, 1.325] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=42.9 (n=75), Group 2: mean=40.8 (n=75)

### H3b: Refusal vs high-entropy
- Cohen's d = 0.562 [0.240, 0.913] (medium)
- Recommended test: conservative_max (p = 0.0074)
- Group 1: mean=15574.4 (n=75), Group 2: mean=15339.4 (n=75)

### H3b: Refusal vs high-entropy (eff. rank)
- Cohen's d = -0.408 [-0.791, -0.079] (small)
- Recommended test: conservative_max (p = 0.0140)
- Group 1: mean=42.9 (n=75), Group 2: mean=43.5 (n=75)

### Creative divergence
- Cohen's d = 0.048 [-0.263, 0.387] (negligible)
- Recommended test: conservative_max (p = 0.7701)
- Group 1: mean=16714.7 (n=75), Group 2: mean=16695.5 (n=75)

### Creative divergence (eff. rank)
- Cohen's d = 0.012 [-0.300, 0.353] (negligible)
- Recommended test: conservative_max (p = 0.9438)
- Group 1: mean=44.5 (n=75), Group 2: mean=44.5 (n=75)

### Emotional divergence
- Cohen's d = -0.299 [-0.614, 0.020] (small)
- Recommended test: conservative_max (p = 0.3728)
- Group 1: mean=16572.9 (n=75), Group 2: mean=16695.5 (n=75)

### Emotional divergence (eff. rank)
- Cohen's d = -0.827 [-1.147, -0.526] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=43.5 (n=75), Group 2: mean=44.5 (n=75)

### Reasoning mode
- Cohen's d = -0.970 [-1.387, -0.630] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=16112.2 (n=75), Group 2: mean=16695.5 (n=75)

### Reasoning mode (eff. rank)
- Cohen's d = -0.917 [-1.223, -0.635] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=43.2 (n=75), Group 2: mean=44.5 (n=75)

### Code mode
- Cohen's d = 2.628 [2.241, 3.143] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=18280.4 (n=75), Group 2: mean=16695.5 (n=75)

### Code mode (eff. rank)
- Cohen's d = 2.194 [1.816, 2.690] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=47.4 (n=75), Group 2: mean=44.5 (n=75)

### Disambiguation demand
- Cohen's d = -2.421 [-2.963, -2.020] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=15392.9 (n=75), Group 2: mean=16263.9 (n=75)

### Disambiguation demand (eff. rank)
- Cohen's d = -1.780 [-2.085, -1.523] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=39.2 (n=75), Group 2: mean=43.2 (n=75)


## Multiple Comparison Correction

Method: holm_bonferroni (9 comparisons)

- confab_vs_facts: p_orig=0.8156, p_corr=1.0000 (not significant)
- self_ref_effect: p_orig=0.0029, p_corr=0.0176 (SIGNIFICANT)
- refusal_vs_rote: p_orig=0.1734, p_corr=0.6936 (not significant)
- refusal_vs_free: p_orig=0.0074, p_corr=0.0371 (SIGNIFICANT)
- creative_vs_facts: p_orig=0.7701, p_corr=1.0000 (not significant)
- emotion_vs_facts: p_orig=0.3728, p_corr=1.0000 (not significant)
- math_vs_facts: p_orig=0.0000, p_corr=0.0000 (SIGNIFICANT)
- code_vs_facts: p_orig=0.0000, p_corr=0.0000 (SIGNIFICANT)
- ambiguity_effect: p_orig=0.0000, p_corr=0.0000 (SIGNIFICANT)

---

## Hypothesis Evaluation

### H1_confabulation_inversion
- cohens_d: 0.12292828142012417
- interpretation: negligible
- ci_lower: -0.20319748333101903
- ci_upper: 0.4405999466489406
- p_value: 0.8156328770363467
- direction: confab_higher
- note: Positive d = confabulation has higher norms (expected at small scale). Negative d = confabulation has lower norms (inversion at large scale).

### H2_self_reference_emergence
- cohens_d: 0.49453981648238776
- interpretation: small
- ci_lower: 0.16328757193024096
- ci_upper: 0.8852496039336952
- p_value: 0.0029350292168706135
- emergence_detected: True
- note: Effect > 0.3 suggests self-modeling capacity at this scale.

### H3_refusal_specialization
- cohens_d: -0.3244225878422207
- interpretation: small
- ci_lower: -0.6433613067197739
- ci_upper: -0.017203767705013827
- p_value: 0.17339180336971627
- specialized: False
- note: d > 0.5 = refusal has unique signature beyond low entropy. d < 0.3 = refusal is just another memorized pattern.

### H4_category_invariance
- note: Evaluated in cross-scale comparison (requires multiple scale results).


## Power Analysis

- d_0.3: power=0.451 (n=75) [UNDERPOWERED]
- d_0.5: power=0.865 (n=75) [ADEQUATE]
- d_0.8: power=0.998 (n=75) [ADEQUATE]