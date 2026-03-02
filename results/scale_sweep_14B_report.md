# Scale Sweep Report: 14B

**Model**: Qwen/Qwen2.5-14B-Instruct
**Quantized**: False
**Generated**: 2026-03-01T22:06:21.418150

---

## Category Summaries

| Category | N | Mean Norm | Std | Mean/Token | Eff. Rank | Spectral H | 95% CI |
|----------|---|-----------|-----|------------|-----------|------------|--------|
| coding               |  75 |  18605.0 |  934.7 |   792.3 |      44.6 |     0.6987 | [18397.4, 18813.8] |
| creative             |  75 |  17274.7 |  321.5 |  1184.6 |      42.5 |     0.6901 | [17202.7, 17345.2] |
| grounded_facts       |  75 |  17218.7 |  498.4 |  1221.3 |      42.5 |     0.6893 | [17108.8, 17334.4] |
| confabulation        |  75 |  17085.6 |  848.3 |  1213.4 |      43.0 |     0.6928 | [16898.5, 17289.6] |
| emotional            |  75 |  17057.5 |  403.8 |  1272.1 |      41.3 |     0.6843 | [16965.6, 17146.4] |
| math_reasoning       |  75 |  16946.9 |  746.8 |  1365.6 |      39.0 |     0.6755 | [16781.2, 17121.9] |
| unambiguous          |  75 |  16766.7 |  366.6 |  1406.7 |      41.5 |     0.6889 | [16687.0, 16849.9] |
| non_self_reference   |  75 |  16720.9 |  405.9 |  1358.8 |      41.5 |     0.6862 | [16631.0, 16812.6] |
| self_reference       |  75 |  16563.3 |  382.9 |  1343.5 |      43.6 |     0.6947 | [16476.6, 16648.1] |
| guardrail_test       |  75 |  16531.3 |  368.9 |  1467.1 |      41.6 |     0.6891 | [16448.9, 16614.8] |
| rote_completion      |  75 |  16336.6 |  792.6 |  2080.5 |      39.4 |     0.6781 | [16163.3, 16515.1] |
| ambiguous            |  75 |  15818.2 |  528.0 |  2220.9 |      40.2 |     0.6845 | [15701.8, 15938.0] |
| free_generation      |  75 |  15702.0 |  367.6 |  2075.0 |      39.6 |     0.6816 | [15620.9, 15785.6] |

---

## Key Comparisons

### H1: Confabulation effect
- Cohen's d = -0.191 [-0.550, 0.132] (negligible)
- Recommended test: conservative_max (p = 0.2439)
- Group 1: mean=17085.6 (n=75), Group 2: mean=17218.7 (n=75)

### H1: Confabulation effect (eff. rank)
- Cohen's d = 0.323 [0.013, 0.615] (small)
- Recommended test: conservative_max (p = 0.7793)
- Group 1: mean=43.0 (n=75), Group 2: mean=42.5 (n=75)

### H2: Self-reference effect
- Cohen's d = -0.399 [-0.727, -0.087] (small)
- Recommended test: conservative_max (p = 0.0156)
- Group 1: mean=16563.3 (n=75), Group 2: mean=16720.9 (n=75)

### H2: Self-reference effect (eff. rank)
- Cohen's d = 1.257 [0.950, 1.626] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=43.6 (n=75), Group 2: mean=41.5 (n=75)

### H3: Refusal specificity
- Cohen's d = 0.315 [-0.010, 0.656] (small)
- Recommended test: conservative_max (p = 0.0565)
- Group 1: mean=16531.3 (n=75), Group 2: mean=16336.6 (n=75)

### H3: Refusal specificity (eff. rank)
- Cohen's d = 1.282 [0.951, 1.681] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=41.6 (n=75), Group 2: mean=39.4 (n=75)

### H3b: Refusal vs high-entropy
- Cohen's d = 2.252 [1.843, 2.760] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=16531.3 (n=75), Group 2: mean=15702.0 (n=75)

### H3b: Refusal vs high-entropy (eff. rank)
- Cohen's d = 1.513 [1.189, 1.890] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=41.6 (n=75), Group 2: mean=39.6 (n=75)

### Creative divergence
- Cohen's d = 0.133 [-0.176, 0.486] (negligible)
- Recommended test: conservative_max (p = 0.4152)
- Group 1: mean=17274.7 (n=75), Group 2: mean=17218.7 (n=75)

### Creative divergence (eff. rank)
- Cohen's d = 0.049 [-0.286, 0.351] (negligible)
- Recommended test: conservative_max (p = 0.7655)
- Group 1: mean=42.5 (n=75), Group 2: mean=42.5 (n=75)

### Emotional divergence
- Cohen's d = -0.355 [-0.663, -0.042] (small)
- Recommended test: conservative_max (p = 0.1003)
- Group 1: mean=17057.5 (n=75), Group 2: mean=17218.7 (n=75)

### Emotional divergence (eff. rank)
- Cohen's d = -0.774 [-1.197, -0.431] (medium)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=41.3 (n=75), Group 2: mean=42.5 (n=75)

### Reasoning mode
- Cohen's d = -0.428 [-0.814, -0.100] (small)
- Recommended test: conservative_max (p = 0.0098)
- Group 1: mean=16946.9 (n=75), Group 2: mean=17218.7 (n=75)

### Reasoning mode (eff. rank)
- Cohen's d = -2.025 [-2.566, -1.617] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=39.0 (n=75), Group 2: mean=42.5 (n=75)

### Code mode
- Cohen's d = 1.851 [1.529, 2.236] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=18605.0 (n=75), Group 2: mean=17218.7 (n=75)

### Code mode (eff. rank)
- Cohen's d = 1.209 [0.870, 1.625] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=44.6 (n=75), Group 2: mean=42.5 (n=75)

### Disambiguation demand
- Cohen's d = -2.087 [-2.514, -1.749] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=15818.2 (n=75), Group 2: mean=16766.7 (n=75)

### Disambiguation demand (eff. rank)
- Cohen's d = -0.878 [-1.333, -0.514] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=40.2 (n=75), Group 2: mean=41.5 (n=75)


## Multiple Comparison Correction

Method: holm_bonferroni (9 comparisons)

- confab_vs_facts: p_orig=0.2439, p_corr=0.4879 (not significant)
- self_ref_effect: p_orig=0.0156, p_corr=0.0781 (not significant)
- refusal_vs_rote: p_orig=0.0565, p_corr=0.2260 (not significant)
- refusal_vs_free: p_orig=0.0000, p_corr=0.0000 (SIGNIFICANT)
- creative_vs_facts: p_orig=0.4152, p_corr=0.4879 (not significant)
- emotion_vs_facts: p_orig=0.1003, p_corr=0.3009 (not significant)
- math_vs_facts: p_orig=0.0098, p_corr=0.0588 (not significant)
- code_vs_facts: p_orig=0.0000, p_corr=0.0000 (SIGNIFICANT)
- ambiguity_effect: p_orig=0.0000, p_corr=0.0000 (SIGNIFICANT)

---

## Hypothesis Evaluation

### H1_confabulation_inversion
- cohens_d: -0.191215951013096
- interpretation: negligible
- ci_lower: -0.5497853559084225
- ci_upper: 0.13235735196590626
- p_value: 0.24394396871290353
- direction: facts_higher
- note: Positive d = confabulation has higher norms (expected at small scale). Negative d = confabulation has lower norms (inversion at large scale).

### H2_self_reference_emergence
- cohens_d: -0.3994025254476546
- interpretation: small
- ci_lower: -0.7267612985567856
- ci_upper: -0.0870921887208395
- p_value: 0.015628575305959302
- emergence_detected: True
- note: Effect > 0.3 suggests self-modeling capacity at this scale.

### H3_refusal_specialization
- cohens_d: 0.31492314251472947
- interpretation: small
- ci_lower: -0.009978299346872489
- ci_upper: 0.6558782935671533
- p_value: 0.05650263452472235
- specialized: False
- note: d > 0.5 = refusal has unique signature beyond low entropy. d < 0.3 = refusal is just another memorized pattern.

### H4_category_invariance
- note: Evaluated in cross-scale comparison (requires multiple scale results).


## Power Analysis

- d_0.3: power=0.451 (n=75) [UNDERPOWERED]
- d_0.5: power=0.865 (n=75) [ADEQUATE]
- d_0.8: power=0.998 (n=75) [ADEQUATE]