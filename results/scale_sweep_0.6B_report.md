# Scale Sweep Report: 0.6B

**Model**: Qwen/Qwen3-0.6B
**Quantized**: False
**Generated**: 2026-03-02T08:23:57.455098

---

## Category Summaries

| Category | N | Mean Norm | Std | Mean/Token | Eff. Rank | Spectral H | 95% CI |
|----------|---|-----------|-----|------------|-----------|------------|--------|
| coding               |  75 |  30024.6 | 1139.4 |  1278.7 |      17.8 |     0.4406 | [29769.9, 30281.2] |
| creative             |  75 |  26900.0 |  474.3 |  1844.6 |      16.1 |     0.4308 | [26793.3, 27006.5] |
| grounded_facts       |  75 |  26817.2 |  869.1 |  1900.9 |      16.1 |     0.4296 | [26624.5, 27019.6] |
| confabulation        |  75 |  26725.8 | 1171.5 |  1899.8 |      16.1 |     0.4315 | [26467.2, 27003.7] |
| emotional            |  75 |  26624.9 |  548.0 |  1986.1 |      16.2 |     0.4294 | [26500.4, 26746.3] |
| non_self_reference   |  75 |  26121.4 |  612.4 |  2123.0 |      15.5 |     0.4275 | [25986.3, 26262.1] |
| math_reasoning       |  75 |  26110.3 | 1035.7 |  2106.0 |      15.9 |     0.4293 | [25880.9, 26351.3] |
| unambiguous          |  75 |  25996.2 |  370.3 |  2181.9 |      15.4 |     0.4264 | [25913.8, 26082.2] |
| self_reference       |  75 |  25995.2 |  345.2 |  2109.8 |      16.3 |     0.4348 | [25916.2, 26071.6] |
| guardrail_test       |  75 |  25432.5 |  618.0 |  2256.3 |      15.4 |     0.4331 | [25294.1, 25573.1] |
| rote_completion      |  75 |  24927.3 | 1163.6 |  3173.5 |      16.2 |     0.4320 | [24669.6, 25183.7] |
| ambiguous            |  75 |  24733.0 |  564.9 |  3484.2 |      14.1 |     0.4159 | [24607.0, 24863.5] |
| free_generation      |  75 |  24515.7 |  728.0 |  3235.8 |      14.8 |     0.4246 | [24357.8, 24685.7] |

---

## Key Comparisons

### H1: Confabulation effect
- Cohen's d = -0.089 [-0.428, 0.229] (negligible)
- Recommended test: conservative_max (p = 0.5881)
- Group 1: mean=26725.8 (n=75), Group 2: mean=26817.2 (n=75)

### H1: Confabulation effect (eff. rank)
- Cohen's d = -0.006 [-0.345, 0.306] (negligible)
- Recommended test: conservative_max (p = 0.9684)
- Group 1: mean=16.1 (n=75), Group 2: mean=16.1 (n=75)

### H2: Self-reference effect
- Cohen's d = -0.254 [-0.551, 0.059] (small)
- Recommended test: conservative_max (p = 0.5424)
- Group 1: mean=25995.2 (n=75), Group 2: mean=26121.4 (n=75)

### H2: Self-reference effect (eff. rank)
- Cohen's d = 0.844 [0.553, 1.155] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=16.3 (n=75), Group 2: mean=15.5 (n=75)

### H3: Refusal specificity
- Cohen's d = 0.542 [0.215, 0.894] (medium)
- Recommended test: conservative_max (p = 0.0016)
- Group 1: mean=25432.5 (n=75), Group 2: mean=24927.3 (n=75)

### H3: Refusal specificity (eff. rank)
- Cohen's d = -0.786 [-1.171, -0.460] (medium)
- Recommended test: conservative_max (p = 0.0001)
- Group 1: mean=15.4 (n=75), Group 2: mean=16.2 (n=75)

### H3b: Refusal vs high-entropy
- Cohen's d = 1.358 [0.972, 1.855] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=25432.5 (n=75), Group 2: mean=24515.7 (n=75)

### H3b: Refusal vs high-entropy (eff. rank)
- Cohen's d = 0.982 [0.721, 1.265] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=15.4 (n=75), Group 2: mean=14.8 (n=75)

### Creative divergence
- Cohen's d = 0.118 [-0.195, 0.469] (negligible)
- Recommended test: conservative_max (p = 0.4703)
- Group 1: mean=26900.0 (n=75), Group 2: mean=26817.2 (n=75)

### Creative divergence (eff. rank)
- Cohen's d = 0.123 [-0.199, 0.447] (negligible)
- Recommended test: conservative_max (p = 1.0000)
- Group 1: mean=16.1 (n=75), Group 2: mean=16.1 (n=75)

### Emotional divergence
- Cohen's d = -0.265 [-0.576, 0.058] (small)
- Recommended test: conservative_max (p = 0.5424)
- Group 1: mean=26624.9 (n=75), Group 2: mean=26817.2 (n=75)

### Emotional divergence (eff. rank)
- Cohen's d = 0.153 [-0.172, 0.474] (negligible)
- Recommended test: conservative_max (p = 0.6735)
- Group 1: mean=16.2 (n=75), Group 2: mean=16.1 (n=75)

### Reasoning mode
- Cohen's d = -0.739 [-1.141, -0.407] (medium)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=26110.3 (n=75), Group 2: mean=26817.2 (n=75)

### Reasoning mode (eff. rank)
- Cohen's d = -0.173 [-0.491, 0.156] (negligible)
- Recommended test: conservative_max (p = 0.2909)
- Group 1: mean=15.9 (n=75), Group 2: mean=16.1 (n=75)

### Code mode
- Cohen's d = 3.165 [2.701, 3.811] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=30024.6 (n=75), Group 2: mean=26817.2 (n=75)

### Code mode (eff. rank)
- Cohen's d = 2.285 [1.939, 2.730] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=17.8 (n=75), Group 2: mean=16.1 (n=75)

### Disambiguation demand
- Cohen's d = -2.645 [-3.098, -2.293] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=24733.0 (n=75), Group 2: mean=25996.2 (n=75)

### Disambiguation demand (eff. rank)
- Cohen's d = -1.277 [-1.614, -0.983] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=14.1 (n=75), Group 2: mean=15.4 (n=75)


## Multiple Comparison Correction

Method: holm_bonferroni (9 comparisons)

- confab_vs_facts: p_orig=0.5881, p_corr=1.0000 (not significant)
- self_ref_effect: p_orig=0.5424, p_corr=1.0000 (not significant)
- refusal_vs_rote: p_orig=0.0016, p_corr=0.0082 (SIGNIFICANT)
- refusal_vs_free: p_orig=0.0000, p_corr=0.0000 (SIGNIFICANT)
- creative_vs_facts: p_orig=0.4703, p_corr=1.0000 (not significant)
- emotion_vs_facts: p_orig=0.5424, p_corr=1.0000 (not significant)
- math_vs_facts: p_orig=0.0000, p_corr=0.0001 (SIGNIFICANT)
- code_vs_facts: p_orig=0.0000, p_corr=0.0000 (SIGNIFICANT)
- ambiguity_effect: p_orig=0.0000, p_corr=0.0000 (SIGNIFICANT)

---

## Hypothesis Evaluation

### H1_confabulation_inversion
- cohens_d: -0.08865377544254562
- interpretation: negligible
- ci_lower: -0.4276001928901112
- ci_upper: 0.22920685950521383
- p_value: 0.588089949331277
- direction: facts_higher
- note: Positive d = confabulation has higher norms (expected at small scale). Negative d = confabulation has lower norms (inversion at large scale).

### H2_self_reference_emergence
- cohens_d: -0.2538649610382724
- interpretation: small
- ci_lower: -0.5514120528752412
- ci_upper: 0.05867614524993369
- p_value: 0.5423660690106564
- emergence_detected: False
- note: Effect > 0.3 suggests self-modeling capacity at this scale.

### H3_refusal_specialization
- cohens_d: 0.5423426294210345
- interpretation: medium
- ci_lower: 0.2146757483029454
- ci_upper: 0.8935133438133076
- p_value: 0.001645410374502893
- specialized: True
- note: d > 0.5 = refusal has unique signature beyond low entropy. d < 0.3 = refusal is just another memorized pattern.

### H4_category_invariance
- note: Evaluated in cross-scale comparison (requires multiple scale results).


## Power Analysis

- d_0.3: power=0.451 (n=75) [UNDERPOWERED]
- d_0.5: power=0.865 (n=75) [ADEQUATE]
- d_0.8: power=0.998 (n=75) [ADEQUATE]