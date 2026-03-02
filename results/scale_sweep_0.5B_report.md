# Scale Sweep Report: 0.5B

**Model**: Qwen/Qwen2.5-0.5B-Instruct
**Quantized**: False
**Generated**: 2026-03-01T20:06:21.076121

---

## Category Summaries

| Category | N | Mean Norm | Std | Mean/Token | Eff. Rank | Spectral H | 95% CI |
|----------|---|-----------|-----|------------|-----------|------------|--------|
| coding               |  75 |  10716.8 |  503.7 |   456.3 |      10.7 |     0.4227 | [10605.1, 10830.6] |
| confabulation        |  75 |   9714.9 |  447.1 |   690.3 |       9.9 |     0.4127 | [9617.5, 9822.3] |
| creative             |  75 |   9691.8 |  157.9 |   664.6 |      10.0 |     0.4143 | [9654.7, 9726.4] |
| grounded_facts       |  75 |   9661.2 |  271.8 |   685.1 |      10.0 |     0.4136 | [9602.7, 9725.2] |
| math_reasoning       |  75 |   9571.3 |  444.5 |   770.5 |       9.3 |     0.4061 | [9473.3, 9674.7] |
| emotional            |  75 |   9560.9 |  189.1 |   713.3 |       9.6 |     0.4042 | [9518.1, 9602.1] |
| non_self_reference   |  75 |   9458.0 |  215.7 |   768.5 |       9.7 |     0.4084 | [9410.5, 9508.4] |
| unambiguous          |  75 |   9432.3 |  149.5 |   791.5 |       9.5 |     0.4094 | [9397.6, 9465.9] |
| self_reference       |  75 |   9405.6 |  139.9 |   763.2 |       9.8 |     0.4106 | [9373.3, 9436.3] |
| guardrail_test       |  75 |   9324.7 |  206.4 |   827.5 |       9.8 |     0.4181 | [9278.5, 9371.3] |
| rote_completion      |  75 |   9033.7 |  413.0 |  1150.5 |       9.3 |     0.4017 | [8942.6, 9125.4] |
| free_generation      |  75 |   8921.0 |  204.6 |  1178.6 |       8.9 |     0.3935 | [8875.2, 8967.8] |
| ambiguous            |  75 |   8918.4 |  268.0 |  1253.0 |       8.9 |     0.3998 | [8858.5, 8978.9] |

---

## Key Comparisons

### H1: Confabulation effect
- Cohen's d = 0.145 [-0.184, 0.455] (negligible)
- Recommended test: conservative_max (p = 0.8156)
- Group 1: mean=9714.9 (n=75), Group 2: mean=9661.2 (n=75)

### H1: Confabulation effect (eff. rank)
- Cohen's d = -0.121 [-0.449, 0.200] (negligible)
- Recommended test: conservative_max (p = 0.4597)
- Group 1: mean=9.9 (n=75), Group 2: mean=10.0 (n=75)

### H2: Self-reference effect
- Cohen's d = -0.288 [-0.599, 0.023] (small)
- Recommended test: conservative_max (p = 0.2407)
- Group 1: mean=9405.6 (n=75), Group 2: mean=9458.0 (n=75)

### H2: Self-reference effect (eff. rank)
- Cohen's d = 0.525 [0.193, 0.933] (medium)
- Recommended test: conservative_max (p = 0.0016)
- Group 1: mean=9.8 (n=75), Group 2: mean=9.7 (n=75)

### H3: Refusal specificity
- Cohen's d = 0.891 [0.547, 1.298] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=9324.7 (n=75), Group 2: mean=9033.7 (n=75)

### H3: Refusal specificity (eff. rank)
- Cohen's d = 1.247 [0.869, 1.719] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=9.8 (n=75), Group 2: mean=9.3 (n=75)

### H3b: Refusal vs high-entropy
- Cohen's d = 1.964 [1.573, 2.465] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=9324.7 (n=75), Group 2: mean=8921.0 (n=75)

### H3b: Refusal vs high-entropy (eff. rank)
- Cohen's d = 2.705 [2.311, 3.216] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=9.8 (n=75), Group 2: mean=8.9 (n=75)

### Creative divergence
- Cohen's d = 0.137 [-0.171, 0.507] (negligible)
- Recommended test: conservative_max (p = 0.4017)
- Group 1: mean=9691.8 (n=75), Group 2: mean=9661.2 (n=75)

### Creative divergence (eff. rank)
- Cohen's d = 0.168 [-0.158, 0.466] (negligible)
- Recommended test: conservative_max (p = 0.9265)
- Group 1: mean=10.0 (n=75), Group 2: mean=10.0 (n=75)

### Emotional divergence
- Cohen's d = -0.428 [-0.721, -0.125] (small)
- Recommended test: conservative_max (p = 0.0670)
- Group 1: mean=9560.9 (n=75), Group 2: mean=9661.2 (n=75)

### Emotional divergence (eff. rank)
- Cohen's d = -0.958 [-1.300, -0.652] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=9.6 (n=75), Group 2: mean=10.0 (n=75)

### Reasoning mode
- Cohen's d = -0.244 [-0.607, 0.069] (small)
- Recommended test: conservative_max (p = 0.1377)
- Group 1: mean=9571.3 (n=75), Group 2: mean=9661.2 (n=75)

### Reasoning mode (eff. rank)
- Cohen's d = -1.756 [-2.186, -1.420] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=9.3 (n=75), Group 2: mean=10.0 (n=75)

### Code mode
- Cohen's d = 2.608 [2.278, 3.034] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=10716.8 (n=75), Group 2: mean=9661.2 (n=75)

### Code mode (eff. rank)
- Cohen's d = 1.622 [1.267, 2.048] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=10.7 (n=75), Group 2: mean=10.0 (n=75)

### Disambiguation demand
- Cohen's d = -2.368 [-2.830, -2.013] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=8918.4 (n=75), Group 2: mean=9432.3 (n=75)

### Disambiguation demand (eff. rank)
- Cohen's d = -1.244 [-1.632, -0.919] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=8.9 (n=75), Group 2: mean=9.5 (n=75)


## Multiple Comparison Correction

Method: holm_bonferroni (9 comparisons)

- confab_vs_facts: p_orig=0.8156, p_corr=0.8156 (not significant)
- self_ref_effect: p_orig=0.2407, p_corr=0.7220 (not significant)
- refusal_vs_rote: p_orig=0.0000, p_corr=0.0000 (SIGNIFICANT)
- refusal_vs_free: p_orig=0.0000, p_corr=0.0000 (SIGNIFICANT)
- creative_vs_facts: p_orig=0.4017, p_corr=0.8035 (not significant)
- emotion_vs_facts: p_orig=0.0670, p_corr=0.3351 (not significant)
- math_vs_facts: p_orig=0.1377, p_corr=0.5508 (not significant)
- code_vs_facts: p_orig=0.0000, p_corr=0.0000 (SIGNIFICANT)
- ambiguity_effect: p_orig=0.0000, p_corr=0.0000 (SIGNIFICANT)

---

## Hypothesis Evaluation

### H1_confabulation_inversion
- cohens_d: 0.14513128202742098
- interpretation: negligible
- ci_lower: -0.18386979994860794
- ci_upper: 0.45471870519587476
- p_value: 0.8156328770363467
- direction: confab_higher
- note: Positive d = confabulation has higher norms (expected at small scale). Negative d = confabulation has lower norms (inversion at large scale).

### H2_self_reference_emergence
- cohens_d: -0.28834414962218263
- interpretation: small
- ci_lower: -0.5988917698528379
- ci_upper: 0.02308370004968976
- p_value: 0.24065584740616353
- emergence_detected: False
- note: Effect > 0.3 suggests self-modeling capacity at this scale.

### H3_refusal_specialization
- cohens_d: 0.8912723681668725
- interpretation: large
- ci_lower: 0.5473662647499018
- ci_upper: 1.2984540924446466
- p_value: 3.060856655809384e-07
- specialized: True
- note: d > 0.5 = refusal has unique signature beyond low entropy. d < 0.3 = refusal is just another memorized pattern.

### H4_category_invariance
- note: Evaluated in cross-scale comparison (requires multiple scale results).


## Power Analysis

- d_0.3: power=0.451 (n=75) [UNDERPOWERED]
- d_0.5: power=0.865 (n=75) [ADEQUATE]
- d_0.8: power=0.998 (n=75) [ADEQUATE]