# Scale Sweep Report: 9B

**Model**: google/gemma-2-9b-it
**Quantized**: False
**Generated**: 2026-03-01T21:34:55.810588

---

## Category Summaries

| Category | N | Mean Norm | Std | Mean/Token | Eff. Rank | Spectral H | 95% CI |
|----------|---|-----------|-----|------------|-----------|------------|--------|
| coding               |  75 |  18010.8 | 1533.7 |   617.1 |      91.9 |     0.7458 | [17659.3, 18346.4] |
| grounded_facts       |  75 |  15885.8 |  547.7 |  1096.3 |      79.7 |     0.7211 | [15767.3, 16013.5] |
| creative             |  75 |  15656.3 |  252.7 |  1015.4 |      81.7 |     0.7268 | [15599.6, 15713.7] |
| confabulation        |  75 |  15585.4 |  804.3 |  1073.2 |      81.5 |     0.7285 | [15409.2, 15778.0] |
| emotional            |  75 |  15582.0 |  299.0 |  1114.4 |      79.0 |     0.7172 | [15515.3, 15649.4] |
| math_reasoning       |  75 |  15332.0 |  690.8 |  1136.3 |      82.2 |     0.7340 | [15181.3, 15493.6] |
| non_self_reference   |  75 |  15218.6 |  472.3 |  1152.6 |      77.0 |     0.7160 | [15110.9, 15325.5] |
| self_reference       |  75 |  15196.8 |  576.3 |  1139.8 |      80.8 |     0.7253 | [15060.2, 15317.8] |
| guardrail_test       |  75 |  15146.6 |  352.9 |  1231.0 |      77.8 |     0.7218 | [15068.3, 15226.5] |
| unambiguous          |  75 |  15063.5 |  764.0 |  1179.5 |      77.1 |     0.7171 | [14882.6, 15222.7] |
| rote_completion      |  75 |  14994.5 |  740.6 |  1685.5 |      75.5 |     0.7142 | [14833.0, 15161.6] |
| free_generation      |  75 |  14551.7 |  384.4 |  1641.0 |      75.3 |     0.7165 | [14466.6, 14640.7] |
| ambiguous            |  75 |  14376.4 |  451.0 |  1784.8 |      73.0 |     0.7097 | [14273.9, 14479.3] |

---

## Key Comparisons

### H1: Confabulation effect
- Cohen's d = -0.436 [-0.798, -0.109] (small)
- Recommended test: conservative_max (p = 0.0085)
- Group 1: mean=15585.4 (n=75), Group 2: mean=15885.8 (n=75)

### H1: Confabulation effect (eff. rank)
- Cohen's d = 0.558 [0.250, 0.866] (medium)
- Recommended test: conservative_max (p = 0.0042)
- Group 1: mean=81.5 (n=75), Group 2: mean=79.7 (n=75)

### H2: Self-reference effect
- Cohen's d = -0.041 [-0.337, 0.306] (negligible)
- Recommended test: conservative_max (p = 0.8007)
- Group 1: mean=15196.8 (n=75), Group 2: mean=15218.6 (n=75)

### H2: Self-reference effect (eff. rank)
- Cohen's d = 1.075 [0.777, 1.399] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=80.8 (n=75), Group 2: mean=77.0 (n=75)

### H3: Refusal specificity
- Cohen's d = 0.262 [-0.063, 0.606] (small)
- Recommended test: conservative_max (p = 0.1213)
- Group 1: mean=15146.6 (n=75), Group 2: mean=14994.5 (n=75)

### H3: Refusal specificity (eff. rank)
- Cohen's d = 0.755 [0.453, 1.068] (medium)
- Recommended test: conservative_max (p = 0.0004)
- Group 1: mean=77.8 (n=75), Group 2: mean=75.5 (n=75)

### H3b: Refusal vs high-entropy
- Cohen's d = 1.612 [1.248, 2.073] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=15146.6 (n=75), Group 2: mean=14551.7 (n=75)

### H3b: Refusal vs high-entropy (eff. rank)
- Cohen's d = 0.962 [0.664, 1.281] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=77.8 (n=75), Group 2: mean=75.3 (n=75)

### Creative divergence
- Cohen's d = -0.538 [-0.799, -0.260] (medium)
- Recommended test: conservative_max (p = 0.0074)
- Group 1: mean=15656.3 (n=75), Group 2: mean=15885.8 (n=75)

### Creative divergence (eff. rank)
- Cohen's d = 0.880 [0.542, 1.276] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=81.7 (n=75), Group 2: mean=79.7 (n=75)

### Emotional divergence
- Cohen's d = -0.688 [-0.948, -0.421] (medium)
- Recommended test: conservative_max (p = 0.0006)
- Group 1: mean=15582.0 (n=75), Group 2: mean=15885.8 (n=75)

### Emotional divergence (eff. rank)
- Cohen's d = -0.341 [-0.656, -0.024] (small)
- Recommended test: conservative_max (p = 0.0434)
- Group 1: mean=79.0 (n=75), Group 2: mean=79.7 (n=75)

### Reasoning mode
- Cohen's d = -0.888 [-1.320, -0.541] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=15332.0 (n=75), Group 2: mean=15885.8 (n=75)

### Reasoning mode (eff. rank)
- Cohen's d = 0.765 [0.472, 1.066] (medium)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=82.2 (n=75), Group 2: mean=79.7 (n=75)

### Code mode
- Cohen's d = 1.845 [1.338, 2.628] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=18010.8 (n=75), Group 2: mean=15885.8 (n=75)

### Code mode (eff. rank)
- Cohen's d = 2.776 [2.007, 4.302] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=91.9 (n=75), Group 2: mean=79.7 (n=75)

### Disambiguation demand
- Cohen's d = -1.095 [-1.875, -0.641] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=14376.4 (n=75), Group 2: mean=15063.5 (n=75)

### Disambiguation demand (eff. rank)
- Cohen's d = -0.903 [-1.392, -0.535] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=73.0 (n=75), Group 2: mean=77.1 (n=75)


## Multiple Comparison Correction

Method: holm_bonferroni (9 comparisons)

- confab_vs_facts: p_orig=0.0085, p_corr=0.0297 (SIGNIFICANT)
- self_ref_effect: p_orig=0.8007, p_corr=0.8007 (not significant)
- refusal_vs_rote: p_orig=0.1213, p_corr=0.2427 (not significant)
- refusal_vs_free: p_orig=0.0000, p_corr=0.0000 (SIGNIFICANT)
- creative_vs_facts: p_orig=0.0074, p_corr=0.0297 (SIGNIFICANT)
- emotion_vs_facts: p_orig=0.0006, p_corr=0.0030 (SIGNIFICANT)
- math_vs_facts: p_orig=0.0000, p_corr=0.0000 (SIGNIFICANT)
- code_vs_facts: p_orig=0.0000, p_corr=0.0000 (SIGNIFICANT)
- ambiguity_effect: p_orig=0.0000, p_corr=0.0000 (SIGNIFICANT)

---

## Hypothesis Evaluation

### H1_confabulation_inversion
- cohens_d: -0.43649510409741243
- interpretation: small
- ci_lower: -0.7977584702406937
- ci_upper: -0.10874680876407405
- p_value: 0.008478255934605484
- direction: facts_higher
- note: Positive d = confabulation has higher norms (expected at small scale). Negative d = confabulation has lower norms (inversion at large scale).

### H2_self_reference_emergence
- cohens_d: -0.04130407642370109
- interpretation: negligible
- ci_lower: -0.3373891589420686
- ci_upper: 0.30616968369893827
- p_value: 0.8006831234056995
- emergence_detected: False
- note: Effect > 0.3 suggests self-modeling capacity at this scale.

### H3_refusal_specialization
- cohens_d: 0.2622833803625349
- interpretation: small
- ci_lower: -0.06292859168249258
- ci_upper: 0.6055598974434362
- p_value: 0.12132655806167564
- specialized: False
- note: d > 0.5 = refusal has unique signature beyond low entropy. d < 0.3 = refusal is just another memorized pattern.

### H4_category_invariance
- note: Evaluated in cross-scale comparison (requires multiple scale results).


## Power Analysis

- d_0.3: power=0.451 (n=75) [UNDERPOWERED]
- d_0.5: power=0.865 (n=75) [ADEQUATE]
- d_0.8: power=0.998 (n=75) [ADEQUATE]