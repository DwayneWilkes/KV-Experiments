# Scale Sweep Report: 7B-mistral

**Model**: mistralai/Mistral-7B-Instruct-v0.3
**Quantized**: False
**Generated**: 2026-03-02T01:05:02.935432

---

## Category Summaries

| Category | N | Mean Norm | Std | Mean/Token | Eff. Rank | Spectral H | 95% CI |
|----------|---|-----------|-----|------------|-----------|------------|--------|
| coding               |  75 |  12825.3 | 1249.3 |   409.3 |      45.4 |     0.7163 | [12531.8, 13089.2] |
| confabulation        |  75 |  11308.0 |  566.2 |   631.4 |      44.3 |     0.7159 | [11183.1, 11442.7] |
| grounded_facts       |  75 |  11274.2 |  376.6 |   683.7 |      42.8 |     0.7051 | [11192.4, 11363.0] |
| creative             |  75 |  11131.1 |  282.5 |   689.6 |      43.7 |     0.7113 | [11067.8, 11194.7] |
| emotional            |  75 |  11128.0 |  283.1 |   719.8 |      42.3 |     0.7032 | [11063.5, 11192.0] |
| unambiguous          |  75 |  10909.7 |  332.4 |   801.2 |      41.5 |     0.7018 | [10835.8, 10986.4] |
| math_reasoning       |  75 |  10760.5 |  512.5 |   771.2 |      42.3 |     0.7096 | [10648.6, 10881.2] |
| non_self_reference   |  75 |  10663.7 |  392.8 |   739.6 |      42.6 |     0.7111 | [10575.0, 10753.1] |
| self_reference       |  75 |  10654.3 |  245.3 |   766.1 |      43.7 |     0.7146 | [10600.1, 10709.5] |
| guardrail_test       |  75 |  10583.0 |  260.7 |   769.6 |      43.2 |     0.7155 | [10524.2, 10642.7] |
| rote_completion      |  75 |  10480.4 |  479.7 |  1124.5 |      40.1 |     0.6971 | [10373.7, 10585.6] |
| ambiguous            |  75 |  10227.0 |  394.2 |  1203.7 |      38.9 |     0.6917 | [10138.2, 10314.5] |
| free_generation      |  75 |  10106.8 |  298.3 |  1121.6 |      40.8 |     0.7041 | [10040.8, 10176.5] |

---

## Key Comparisons

### H1: Confabulation effect
- Cohen's d = 0.070 [-0.260, 0.388] (negligible)
- Recommended test: conservative_max (p = 0.6736)
- Group 1: mean=11308.0 (n=75), Group 2: mean=11274.2 (n=75)

### H1: Confabulation effect (eff. rank)
- Cohen's d = 1.016 [0.737, 1.310] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=44.3 (n=75), Group 2: mean=42.8 (n=75)

### H2: Self-reference effect
- Cohen's d = -0.028 [-0.352, 0.303] (negligible)
- Recommended test: conservative_max (p = 0.9640)
- Group 1: mean=10654.3 (n=75), Group 2: mean=10663.7 (n=75)

### H2: Self-reference effect (eff. rank)
- Cohen's d = 0.553 [0.229, 0.908] (medium)
- Recommended test: conservative_max (p = 0.0009)
- Group 1: mean=43.7 (n=75), Group 2: mean=42.6 (n=75)

### H3: Refusal specificity
- Cohen's d = 0.266 [-0.059, 0.589] (small)
- Recommended test: conservative_max (p = 0.1063)
- Group 1: mean=10583.0 (n=75), Group 2: mean=10480.4 (n=75)

### H3: Refusal specificity (eff. rank)
- Cohen's d = 1.708 [1.398, 2.089] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=43.2 (n=75), Group 2: mean=40.1 (n=75)

### H3b: Refusal vs high-entropy
- Cohen's d = 1.700 [1.295, 2.241] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=10583.0 (n=75), Group 2: mean=10106.8 (n=75)

### H3b: Refusal vs high-entropy (eff. rank)
- Cohen's d = 1.918 [1.606, 2.310] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=43.2 (n=75), Group 2: mean=40.8 (n=75)

### Creative divergence
- Cohen's d = -0.430 [-0.753, -0.115] (small)
- Recommended test: conservative_max (p = 0.0434)
- Group 1: mean=11131.1 (n=75), Group 2: mean=11274.2 (n=75)

### Creative divergence (eff. rank)
- Cohen's d = 0.593 [0.299, 0.890] (medium)
- Recommended test: conservative_max (p = 0.0146)
- Group 1: mean=43.7 (n=75), Group 2: mean=42.8 (n=75)

### Emotional divergence
- Cohen's d = -0.439 [-0.761, -0.122] (small)
- Recommended test: conservative_max (p = 0.0273)
- Group 1: mean=11128.0 (n=75), Group 2: mean=11274.2 (n=75)

### Emotional divergence (eff. rank)
- Cohen's d = -0.333 [-0.710, -0.010] (small)
- Recommended test: conservative_max (p = 0.0430)
- Group 1: mean=42.3 (n=75), Group 2: mean=42.8 (n=75)

### Reasoning mode
- Cohen's d = -1.142 [-1.644, -0.766] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=10760.5 (n=75), Group 2: mean=11274.2 (n=75)

### Reasoning mode (eff. rank)
- Cohen's d = -0.286 [-0.671, 0.044] (small)
- Recommended test: conservative_max (p = 0.0824)
- Group 1: mean=42.3 (n=75), Group 2: mean=42.8 (n=75)

### Code mode
- Cohen's d = 1.681 [1.157, 2.566] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=12825.3 (n=75), Group 2: mean=11274.2 (n=75)

### Code mode (eff. rank)
- Cohen's d = 1.412 [1.031, 1.893] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=45.4 (n=75), Group 2: mean=42.8 (n=75)

### Disambiguation demand
- Cohen's d = -1.872 [-2.208, -1.588] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=10227.0 (n=75), Group 2: mean=10909.7 (n=75)

### Disambiguation demand (eff. rank)
- Cohen's d = -0.907 [-1.203, -0.629] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=38.9 (n=75), Group 2: mean=41.5 (n=75)


## Multiple Comparison Correction

Method: holm_bonferroni (9 comparisons)

- confab_vs_facts: p_orig=0.6736, p_corr=1.0000 (not significant)
- self_ref_effect: p_orig=0.9640, p_corr=1.0000 (not significant)
- refusal_vs_rote: p_orig=0.1063, p_corr=0.3188 (not significant)
- refusal_vs_free: p_orig=0.0000, p_corr=0.0000 (SIGNIFICANT)
- creative_vs_facts: p_orig=0.0434, p_corr=0.1737 (not significant)
- emotion_vs_facts: p_orig=0.0273, p_corr=0.1364 (not significant)
- math_vs_facts: p_orig=0.0000, p_corr=0.0000 (SIGNIFICANT)
- code_vs_facts: p_orig=0.0000, p_corr=0.0000 (SIGNIFICANT)
- ambiguity_effect: p_orig=0.0000, p_corr=0.0000 (SIGNIFICANT)

---

## Hypothesis Evaluation

### H1_confabulation_inversion
- cohens_d: 0.07044866587433203
- interpretation: negligible
- ci_lower: -0.25983427201427717
- ci_upper: 0.38836515691879214
- p_value: 0.6736074129513852
- direction: confab_higher
- note: Positive d = confabulation has higher norms (expected at small scale). Negative d = confabulation has lower norms (inversion at large scale).

### H2_self_reference_emergence
- cohens_d: -0.02846190091704026
- interpretation: negligible
- ci_lower: -0.3519856730001676
- ci_upper: 0.3028491250417607
- p_value: 0.9640046370958462
- emergence_detected: False
- note: Effect > 0.3 suggests self-modeling capacity at this scale.

### H3_refusal_specialization
- cohens_d: 0.26586041185090387
- interpretation: small
- ci_lower: -0.05926852043897733
- ci_upper: 0.5894939615174059
- p_value: 0.10626860229026222
- specialized: False
- note: d > 0.5 = refusal has unique signature beyond low entropy. d < 0.3 = refusal is just another memorized pattern.

### H4_category_invariance
- note: Evaluated in cross-scale comparison (requires multiple scale results).


## Power Analysis

- d_0.3: power=0.451 (n=75) [UNDERPOWERED]
- d_0.5: power=0.865 (n=75) [ADEQUATE]
- d_0.8: power=0.998 (n=75) [ADEQUATE]