# Scale Sweep Report: 14B

**Model**: Qwen/Qwen2.5-14B-Instruct
**Quantized**: False
**Generated**: 2026-02-15T21:59:43.921688

---

## Category Summaries

| Category | N | Mean Norm | Std | Mean/Token | Eff. Rank | Spectral H | 95% CI |
|----------|---|-----------|-----|------------|-----------|------------|--------|
| coding               |  75 |  18606.6 |  932.9 |   792.4 |      44.6 |     0.6988 | [18399.4, 18814.6] |
| creative             |  75 |  17229.7 |  336.8 |  1181.4 |      42.5 |     0.6895 | [17152.6, 17305.3] |
| grounded_facts       |  75 |  17218.0 |  498.5 |  1221.3 |      42.5 |     0.6893 | [17108.0, 17333.7] |
| confabulation        |  75 |  17069.5 |  812.7 |  1212.8 |      42.9 |     0.6921 | [16888.9, 17262.3] |
| emotional            |  75 |  17048.3 |  408.6 |  1271.4 |      41.3 |     0.6843 | [16955.5, 17138.8] |
| math_reasoning       |  75 |  16945.9 |  747.8 |  1365.5 |      38.9 |     0.6755 | [16780.2, 17121.2] |
| unambiguous          |  75 |  16783.4 |  393.6 |  1408.1 |      41.4 |     0.6875 | [16697.1, 16872.4] |
| non_self_reference   |  75 |  16722.7 |  407.8 |  1359.0 |      41.6 |     0.6863 | [16632.3, 16814.7] |
| self_reference       |  75 |  16562.5 |  371.2 |  1343.6 |      43.7 |     0.6946 | [16478.2, 16645.8] |
| guardrail_test       |  75 |  16533.7 |  369.3 |  1467.3 |      41.6 |     0.6892 | [16450.8, 16616.9] |
| rote_completion      |  75 |  16342.4 |  799.9 |  2080.8 |      39.4 |     0.6782 | [16167.3, 16522.7] |
| ambiguous            |  75 |  15835.8 |  510.7 |  2223.8 |      40.4 |     0.6851 | [15723.2, 15951.8] |
| free_generation      |  75 |  15702.6 |  363.7 |  2075.2 |      39.4 |     0.6811 | [15622.8, 15785.5] |

---

## Key Comparisons

### H1: Confabulation effect
- Cohen's d = -0.220 [-0.580, 0.111] (small)
- Recommended test: mann_whitney (p = 0.0056)
- Group 1: mean=17069.5 (n=75), Group 2: mean=17218.0 (n=75)

### H1: Confabulation effect (eff. rank)
- Cohen's d = 0.258 [-0.065, 0.558] (small)
- Recommended test: mann_whitney (p = 0.9640)
- Group 1: mean=42.9 (n=75), Group 2: mean=42.5 (n=75)

### H2: Self-reference effect
- Cohen's d = -0.411 [-0.737, -0.095] (small)
- Recommended test: mann_whitney (p = 0.0166)
- Group 1: mean=16562.5 (n=75), Group 2: mean=16722.7 (n=75)

### H2: Self-reference effect (eff. rank)
- Cohen's d = 1.218 [0.929, 1.566] (large)
- Recommended test: mann_whitney (p = 0.0000)
- Group 1: mean=43.7 (n=75), Group 2: mean=41.6 (n=75)

### H3: Refusal specificity
- Cohen's d = 0.307 [-0.025, 0.648] (small)
- Recommended test: mann_whitney (p = 0.0074)
- Group 1: mean=16533.7 (n=75), Group 2: mean=16342.4 (n=75)

### H3: Refusal specificity (eff. rank)
- Cohen's d = 1.285 [0.953, 1.691] (large)
- Recommended test: mann_whitney (p = 0.0000)
- Group 1: mean=41.6 (n=75), Group 2: mean=39.4 (n=75)

### H3b: Refusal vs high-entropy
- Cohen's d = 2.268 [1.859, 2.786] (large)
- Recommended test: mann_whitney (p = 0.0000)
- Group 1: mean=16533.7 (n=75), Group 2: mean=15702.6 (n=75)

### H3b: Refusal vs high-entropy (eff. rank)
- Cohen's d = 1.569 [1.247, 1.937] (large)
- Recommended test: mann_whitney (p = 0.0000)
- Group 1: mean=41.6 (n=75), Group 2: mean=39.4 (n=75)

### Creative divergence
- Cohen's d = 0.027 [-0.281, 0.368] (negligible)
- Recommended test: mann_whitney (p = 0.6064)
- Group 1: mean=17229.7 (n=75), Group 2: mean=17218.0 (n=75)

### Creative divergence (eff. rank)
- Cohen's d = -0.017 [-0.357, 0.287] (negligible)
- Recommended test: mann_whitney (p = 0.3982)
- Group 1: mean=42.5 (n=75), Group 2: mean=42.5 (n=75)

### Emotional divergence
- Cohen's d = -0.372 [-0.677, -0.059] (small)
- Recommended test: mann_whitney (p = 0.0542)
- Group 1: mean=17048.3 (n=75), Group 2: mean=17218.0 (n=75)

### Emotional divergence (eff. rank)
- Cohen's d = -0.787 [-1.194, -0.457] (medium)
- Recommended test: mann_whitney (p = 0.0000)
- Group 1: mean=41.3 (n=75), Group 2: mean=42.5 (n=75)

### Reasoning mode
- Cohen's d = -0.428 [-0.820, -0.103] (small)
- Recommended test: mann_whitney (p = 0.0002)
- Group 1: mean=16945.9 (n=75), Group 2: mean=17218.0 (n=75)

### Reasoning mode (eff. rank)
- Cohen's d = -2.047 [-2.607, -1.639] (large)
- Recommended test: mann_whitney (p = 0.0000)
- Group 1: mean=38.9 (n=75), Group 2: mean=42.5 (n=75)

### Code mode
- Cohen's d = 1.857 [1.534, 2.234] (large)
- Recommended test: mann_whitney (p = 0.0000)
- Group 1: mean=18606.6 (n=75), Group 2: mean=17218.0 (n=75)

### Code mode (eff. rank)
- Cohen's d = 1.232 [0.890, 1.648] (large)
- Recommended test: mann_whitney (p = 0.0000)
- Group 1: mean=44.6 (n=75), Group 2: mean=42.5 (n=75)

### Disambiguation demand
- Cohen's d = -2.078 [-2.493, -1.749] (large)
- Recommended test: mann_whitney (p = 0.0000)
- Group 1: mean=15835.8 (n=75), Group 2: mean=16783.4 (n=75)

### Disambiguation demand (eff. rank)
- Cohen's d = -0.636 [-1.066, -0.282] (medium)
- Recommended test: mann_whitney (p = 0.0000)
- Group 1: mean=40.4 (n=75), Group 2: mean=41.4 (n=75)


## Multiple Comparison Correction

Method: holm_bonferroni (9 comparisons)

- confab_vs_facts: p_orig=0.0056, p_corr=0.0279 (SIGNIFICANT)
- self_ref_effect: p_orig=0.0166, p_corr=0.0498 (SIGNIFICANT)
- refusal_vs_rote: p_orig=0.0074, p_corr=0.0297 (SIGNIFICANT)
- refusal_vs_free: p_orig=0.0000, p_corr=0.0000 (SIGNIFICANT)
- creative_vs_facts: p_orig=0.6064, p_corr=0.6064 (not significant)
- emotion_vs_facts: p_orig=0.0542, p_corr=0.1083 (not significant)
- math_vs_facts: p_orig=0.0002, p_corr=0.0012 (SIGNIFICANT)
- code_vs_facts: p_orig=0.0000, p_corr=0.0000 (SIGNIFICANT)
- ambiguity_effect: p_orig=0.0000, p_corr=0.0000 (SIGNIFICANT)

---

## Hypothesis Evaluation

### H1_confabulation_inversion
- cohens_d: -0.22031282016728032
- interpretation: small
- ci_lower: -0.5803674072328603
- ci_upper: 0.11084464920355712
- p_value: 0.0055771062147464655
- direction: facts_higher
- note: Positive d = confabulation has higher norms (expected at small scale). Negative d = confabulation has lower norms (inversion at large scale).

### H2_self_reference_emergence
- cohens_d: -0.4107477482959369
- interpretation: small
- ci_lower: -0.7365147849489625
- ci_upper: -0.09465415997257304
- p_value: 0.016593743690830443
- emergence_detected: True
- note: Effect > 0.3 suggests self-modeling capacity at this scale.

### H3_refusal_specialization
- cohens_d: 0.30712915905900384
- interpretation: small
- ci_lower: -0.024831124138373257
- ci_upper: 0.6484849647827098
- p_value: 0.007414224574308374
- specialized: False
- note: d > 0.5 = refusal has unique signature beyond low entropy. d < 0.3 = refusal is just another memorized pattern.

### H4_category_invariance
- note: Evaluated in cross-scale comparison (requires multiple scale results).


## Power Analysis

- d_0.3: power=0.451 (n=75) [UNDERPOWERED]
- d_0.5: power=0.865 (n=75) [ADEQUATE]
- d_0.8: power=0.998 (n=75) [ADEQUATE]