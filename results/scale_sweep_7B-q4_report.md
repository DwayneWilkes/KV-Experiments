# Scale Sweep Report: 7B-q4

**Model**: Qwen/Qwen2.5-7B-Instruct
**Quantized**: True
**Generated**: 2026-03-01T20:52:13.184711

---

## Category Summaries

| Category | N | Mean Norm | Std | Mean/Token | Eff. Rank | Spectral H | 95% CI |
|----------|---|-----------|-----|------------|-----------|------------|--------|
| coding               |  75 |  20159.9 |  781.0 |   858.4 |      25.3 |     0.5140 | [19990.5, 20339.1] |
| creative             |  75 |  18307.1 |  256.7 |  1255.5 |      24.6 |     0.5095 | [18246.8, 18364.5] |
| grounded_facts       |  75 |  18252.5 |  522.9 |  1294.3 |      23.3 |     0.5029 | [18139.2, 18373.8] |
| confabulation        |  75 |  18250.7 |  813.8 |  1297.1 |      24.1 |     0.5087 | [18071.9, 18443.9] |
| emotional            |  75 |  18136.1 |  339.4 |  1353.0 |      23.1 |     0.5003 | [18059.9, 18211.6] |
| math_reasoning       |  75 |  17987.4 |  819.2 |  1448.3 |      22.7 |     0.5001 | [17807.1, 18177.1] |
| non_self_reference   |  75 |  17829.2 |  415.0 |  1448.6 |      23.7 |     0.5054 | [17738.7, 17926.1] |
| self_reference       |  75 |  17751.8 |  235.5 |  1440.5 |      24.4 |     0.5099 | [17697.6, 17804.5] |
| unambiguous          |  75 |  17751.0 |  247.4 |  1489.6 |      23.6 |     0.5066 | [17694.4, 17806.4] |
| guardrail_test       |  75 |  17582.9 |  354.4 |  1560.7 |      22.5 |     0.5023 | [17503.9, 17663.7] |
| rote_completion      |  75 |  17111.3 |  747.8 |  2181.1 |      21.4 |     0.4911 | [16946.2, 17277.7] |
| free_generation      |  75 |  16779.8 |  372.7 |  2217.0 |      21.7 |     0.4968 | [16697.4, 16864.8] |
| ambiguous            |  75 |  16760.2 |  491.1 |  2355.3 |      21.7 |     0.4952 | [16650.5, 16872.0] |

---

## Key Comparisons

### H1: Confabulation effect
- Cohen's d = -0.003 [-0.341, 0.315] (negligible)
- Recommended test: conservative_max (p = 0.9872)
- Group 1: mean=18250.7 (n=75), Group 2: mean=18252.5 (n=75)

### H1: Confabulation effect (eff. rank)
- Cohen's d = 0.613 [0.326, 0.896] (medium)
- Recommended test: conservative_max (p = 0.0056)
- Group 1: mean=24.1 (n=75), Group 2: mean=23.3 (n=75)

### H2: Self-reference effect
- Cohen's d = -0.229 [-0.529, 0.090] (small)
- Recommended test: conservative_max (p = 0.3245)
- Group 1: mean=17751.8 (n=75), Group 2: mean=17829.2 (n=75)

### H2: Self-reference effect (eff. rank)
- Cohen's d = 0.542 [0.229, 0.869] (medium)
- Recommended test: conservative_max (p = 0.0112)
- Group 1: mean=24.4 (n=75), Group 2: mean=23.7 (n=75)

### H3: Refusal specificity
- Cohen's d = 0.806 [0.464, 1.202] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=17582.9 (n=75), Group 2: mean=17111.3 (n=75)

### H3: Refusal specificity (eff. rank)
- Cohen's d = 0.965 [0.671, 1.293] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=22.5 (n=75), Group 2: mean=21.4 (n=75)

### H3b: Refusal vs high-entropy
- Cohen's d = 2.208 [1.811, 2.726] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=17582.9 (n=75), Group 2: mean=16779.8 (n=75)

### H3b: Refusal vs high-entropy (eff. rank)
- Cohen's d = 1.094 [0.759, 1.494] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=22.5 (n=75), Group 2: mean=21.7 (n=75)

### Creative divergence
- Cohen's d = 0.133 [-0.174, 0.507] (negligible)
- Recommended test: conservative_max (p = 0.4184)
- Group 1: mean=18307.1 (n=75), Group 2: mean=18252.5 (n=75)

### Creative divergence (eff. rank)
- Cohen's d = 1.344 [1.055, 1.670] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=24.6 (n=75), Group 2: mean=23.3 (n=75)

### Emotional divergence
- Cohen's d = -0.264 [-0.560, 0.053] (small)
- Recommended test: conservative_max (p = 0.6064)
- Group 1: mean=18136.1 (n=75), Group 2: mean=18252.5 (n=75)

### Emotional divergence (eff. rank)
- Cohen's d = -0.231 [-0.576, 0.086] (small)
- Recommended test: conservative_max (p = 0.1585)
- Group 1: mean=23.1 (n=75), Group 2: mean=23.3 (n=75)

### Reasoning mode
- Cohen's d = -0.386 [-0.768, -0.065] (small)
- Recommended test: conservative_max (p = 0.0197)
- Group 1: mean=17987.4 (n=75), Group 2: mean=18252.5 (n=75)

### Reasoning mode (eff. rank)
- Cohen's d = -0.512 [-0.876, -0.188] (medium)
- Recommended test: conservative_max (p = 0.0540)
- Group 1: mean=22.7 (n=75), Group 2: mean=23.3 (n=75)

### Code mode
- Cohen's d = 2.870 [2.481, 3.388] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=20159.9 (n=75), Group 2: mean=18252.5 (n=75)

### Code mode (eff. rank)
- Cohen's d = 1.628 [1.242, 2.130] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=25.3 (n=75), Group 2: mean=23.3 (n=75)

### Disambiguation demand
- Cohen's d = -2.548 [-3.053, -2.174] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=16760.2 (n=75), Group 2: mean=17751.0 (n=75)

### Disambiguation demand (eff. rank)
- Cohen's d = -1.387 [-1.704, -1.114] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=21.7 (n=75), Group 2: mean=23.6 (n=75)


## Multiple Comparison Correction

Method: holm_bonferroni (9 comparisons)

- confab_vs_facts: p_orig=0.9872, p_corr=1.0000 (not significant)
- self_ref_effect: p_orig=0.3245, p_corr=1.0000 (not significant)
- refusal_vs_rote: p_orig=0.0000, p_corr=0.0000 (SIGNIFICANT)
- refusal_vs_free: p_orig=0.0000, p_corr=0.0000 (SIGNIFICANT)
- creative_vs_facts: p_orig=0.4184, p_corr=1.0000 (not significant)
- emotion_vs_facts: p_orig=0.6064, p_corr=1.0000 (not significant)
- math_vs_facts: p_orig=0.0197, p_corr=0.0986 (not significant)
- code_vs_facts: p_orig=0.0000, p_corr=0.0000 (SIGNIFICANT)
- ambiguity_effect: p_orig=0.0000, p_corr=0.0000 (SIGNIFICANT)

---

## Hypothesis Evaluation

### H1_confabulation_inversion
- cohens_d: -0.002624085314791151
- interpretation: negligible
- ci_lower: -0.34149072364397853
- ci_upper: 0.31539864255877065
- p_value: 0.9872045801065016
- direction: facts_higher
- note: Positive d = confabulation has higher norms (expected at small scale). Negative d = confabulation has lower norms (inversion at large scale).

### H2_self_reference_emergence
- cohens_d: -0.22947576429510144
- interpretation: small
- ci_lower: -0.5290136715238201
- ci_upper: 0.08966050291937727
- p_value: 0.3244706586097018
- emergence_detected: False
- note: Effect > 0.3 suggests self-modeling capacity at this scale.

### H3_refusal_specialization
- cohens_d: 0.8059920944170177
- interpretation: large
- ci_lower: 0.46389454057531915
- ci_upper: 1.2024205020962753
- p_value: 2.9958844456744646e-06
- specialized: True
- note: d > 0.5 = refusal has unique signature beyond low entropy. d < 0.3 = refusal is just another memorized pattern.

### H4_category_invariance
- note: Evaluated in cross-scale comparison (requires multiple scale results).


## Power Analysis

- d_0.3: power=0.451 (n=75) [UNDERPOWERED]
- d_0.5: power=0.865 (n=75) [ADEQUATE]
- d_0.8: power=0.998 (n=75) [ADEQUATE]