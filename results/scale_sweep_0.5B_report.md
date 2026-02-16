# Scale Sweep Report: 0.5B

**Model**: Qwen/Qwen2.5-0.5B-Instruct
**Quantized**: False
**Generated**: 2026-02-15T08:35:31.621909

---

## Category Summaries

| Category | N | Mean Norm | Std | Mean/Token | Eff. Rank | Spectral H | 95% CI |
|----------|---|-----------|-----|------------|-----------|------------|--------|
| coding               |  75 |  10715.7 |  502.4 |   456.3 |      10.7 |     0.4225 | [10604.0, 10829.5] |
| confabulation        |  75 |   9712.1 |  442.1 |   690.2 |      10.0 |     0.4123 | [9615.5, 9818.0] |
| creative             |  75 |   9695.5 |  158.9 |   664.8 |      10.1 |     0.4146 | [9658.5, 9730.6] |
| grounded_facts       |  75 |   9664.0 |  270.4 |   685.4 |      10.0 |     0.4134 | [9605.9, 9727.5] |
| math_reasoning       |  75 |   9571.8 |  444.3 |   770.5 |       9.3 |     0.4069 | [9473.5, 9675.3] |
| emotional            |  75 |   9567.3 |  182.8 |   713.9 |       9.5 |     0.4030 | [9525.8, 9607.2] |
| non_self_reference   |  75 |   9460.6 |  215.2 |   768.7 |       9.7 |     0.4078 | [9413.3, 9510.8] |
| unambiguous          |  75 |   9436.4 |  136.1 |   791.9 |       9.6 |     0.4099 | [9405.3, 9467.0] |
| self_reference       |  75 |   9403.3 |  138.9 |   763.0 |       9.8 |     0.4077 | [9371.3, 9433.8] |
| guardrail_test       |  75 |   9326.6 |  206.5 |   827.6 |       9.8 |     0.4174 | [9280.3, 9373.4] |
| rote_completion      |  75 |   9041.6 |  424.7 |  1151.2 |       9.2 |     0.4020 | [8947.9, 9136.1] |
| free_generation      |  75 |   8926.3 |  206.6 |  1179.3 |       8.8 |     0.3933 | [8880.1, 8973.7] |
| ambiguous            |  75 |   8917.4 |  270.2 |  1252.8 |       8.9 |     0.3980 | [8857.1, 8978.1] |

---

## Key Comparisons

### H1: Confabulation effect
- Cohen's d = 0.131 [-0.199, 0.447] (negligible)
- Recommended test: mann_whitney (p = 0.8893)
- Group 1: mean=9712.1 (n=75), Group 2: mean=9664.0 (n=75)

### H1: Confabulation effect (eff. rank)
- Cohen's d = -0.028 [-0.362, 0.291] (negligible)
- Recommended test: mann_whitney (p = 0.3474)
- Group 1: mean=10.0 (n=75), Group 2: mean=10.0 (n=75)

### H2: Self-reference effect
- Cohen's d = -0.316 [-0.621, -0.006] (small)
- Recommended test: mann_whitney (p = 0.1456)
- Group 1: mean=9403.3 (n=75), Group 2: mean=9460.6 (n=75)

### H2: Self-reference effect (eff. rank)
- Cohen's d = 0.256 [-0.061, 0.614] (small)
- Recommended test: mann_whitney (p = 0.0270)
- Group 1: mean=9.8 (n=75), Group 2: mean=9.7 (n=75)

### H3: Refusal specificity
- Cohen's d = 0.853 [0.495, 1.273] (large)
- Recommended test: mann_whitney (p = 0.0000)
- Group 1: mean=9326.6 (n=75), Group 2: mean=9041.6 (n=75)

### H3: Refusal specificity (eff. rank)
- Cohen's d = 1.324 [0.983, 1.730] (large)
- Recommended test: mann_whitney (p = 0.0000)
- Group 1: mean=9.8 (n=75), Group 2: mean=9.2 (n=75)

### H3b: Refusal vs high-entropy
- Cohen's d = 1.937 [1.546, 2.430] (large)
- Recommended test: mann_whitney (p = 0.0000)
- Group 1: mean=9326.6 (n=75), Group 2: mean=8926.3 (n=75)

### H3b: Refusal vs high-entropy (eff. rank)
- Cohen's d = 2.387 [2.037, 2.845] (large)
- Recommended test: mann_whitney (p = 0.0000)
- Group 1: mean=9.8 (n=75), Group 2: mean=8.8 (n=75)

### Creative divergence
- Cohen's d = 0.142 [-0.174, 0.506] (negligible)
- Recommended test: mann_whitney (p = 0.0214)
- Group 1: mean=9695.5 (n=75), Group 2: mean=9664.0 (n=75)

### Creative divergence (eff. rank)
- Cohen's d = 0.216 [-0.096, 0.508] (small)
- Recommended test: mann_whitney (p = 0.8521)
- Group 1: mean=10.1 (n=75), Group 2: mean=10.0 (n=75)

### Emotional divergence
- Cohen's d = -0.419 [-0.722, -0.123] (small)
- Recommended test: mann_whitney (p = 0.0670)
- Group 1: mean=9567.3 (n=75), Group 2: mean=9664.0 (n=75)

### Emotional divergence (eff. rank)
- Cohen's d = -1.019 [-1.369, -0.712] (large)
- Recommended test: mann_whitney (p = 0.0000)
- Group 1: mean=9.5 (n=75), Group 2: mean=10.0 (n=75)

### Reasoning mode
- Cohen's d = -0.251 [-0.617, 0.057] (small)
- Recommended test: mann_whitney (p = 0.0074)
- Group 1: mean=9571.8 (n=75), Group 2: mean=9664.0 (n=75)

### Reasoning mode (eff. rank)
- Cohen's d = -1.585 [-2.006, -1.242] (large)
- Recommended test: mann_whitney (p = 0.0000)
- Group 1: mean=9.3 (n=75), Group 2: mean=10.0 (n=75)

### Code mode
- Cohen's d = 2.607 [2.271, 3.028] (large)
- Recommended test: mann_whitney (p = 0.0000)
- Group 1: mean=10715.7 (n=75), Group 2: mean=9664.0 (n=75)

### Code mode (eff. rank)
- Cohen's d = 1.586 [1.234, 2.006] (large)
- Recommended test: mann_whitney (p = 0.0000)
- Group 1: mean=10.7 (n=75), Group 2: mean=10.0 (n=75)

### Disambiguation demand
- Cohen's d = -2.427 [-2.872, -2.070] (large)
- Recommended test: mann_whitney (p = 0.0000)
- Group 1: mean=8917.4 (n=75), Group 2: mean=9436.4 (n=75)

### Disambiguation demand (eff. rank)
- Cohen's d = -1.507 [-1.902, -1.180] (large)
- Recommended test: mann_whitney (p = 0.0000)
- Group 1: mean=8.9 (n=75), Group 2: mean=9.6 (n=75)


## Multiple Comparison Correction

Method: holm_bonferroni (9 comparisons)

- confab_vs_facts: p_orig=0.8893, p_corr=0.8893 (not significant)
- self_ref_effect: p_orig=0.1456, p_corr=0.2911 (not significant)
- refusal_vs_rote: p_orig=0.0000, p_corr=0.0000 (SIGNIFICANT)
- refusal_vs_free: p_orig=0.0000, p_corr=0.0000 (SIGNIFICANT)
- creative_vs_facts: p_orig=0.0214, p_corr=0.0854 (not significant)
- emotion_vs_facts: p_orig=0.0670, p_corr=0.2011 (not significant)
- math_vs_facts: p_orig=0.0074, p_corr=0.0371 (SIGNIFICANT)
- code_vs_facts: p_orig=0.0000, p_corr=0.0000 (SIGNIFICANT)
- ambiguity_effect: p_orig=0.0000, p_corr=0.0000 (SIGNIFICANT)

---

## Hypothesis Evaluation

### H1_confabulation_inversion
- cohens_d: 0.13116010345050336
- interpretation: negligible
- ci_lower: -0.19910939270740152
- ci_upper: 0.4469415266559317
- p_value: 0.8893338570937382
- direction: confab_higher
- note: Positive d = confabulation has higher norms (expected at small scale). Negative d = confabulation has lower norms (inversion at large scale).

### H2_self_reference_emergence
- cohens_d: -0.3163397207329006
- interpretation: small
- ci_lower: -0.6207651171382431
- ci_upper: -0.005759132067612079
- p_value: 0.14555740847244883
- emergence_detected: True
- note: Effect > 0.3 suggests self-modeling capacity at this scale.

### H3_refusal_specialization
- cohens_d: 0.8532538693439214
- interpretation: large
- ci_lower: 0.4949155642475123
- ci_upper: 1.272847025284972
- p_value: 1.8269171032371832e-07
- specialized: True
- note: d > 0.5 = refusal has unique signature beyond low entropy. d < 0.3 = refusal is just another memorized pattern.

### H4_category_invariance
- note: Evaluated in cross-scale comparison (requires multiple scale results).


## Power Analysis

- d_0.3: power=0.451 (n=75) [UNDERPOWERED]
- d_0.5: power=0.865 (n=75) [ADEQUATE]
- d_0.8: power=0.998 (n=75) [ADEQUATE]