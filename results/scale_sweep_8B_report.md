# Scale Sweep Report: 8B

**Model**: meta-llama/Llama-3.1-8B-Instruct
**Quantized**: False
**Generated**: 2026-03-01T21:11:51.308795

---

## Category Summaries

| Category | N | Mean Norm | Std | Mean/Token | Eff. Rank | Spectral H | 95% CI |
|----------|---|-----------|-----|------------|-----------|------------|--------|
| coding               |  75 |  13608.6 |  703.3 |   560.1 |      46.5 |     0.7153 | [13456.4, 13769.0] |
| emotional            |  75 |  12664.8 |  232.1 |   882.0 |      40.5 |     0.6886 | [12612.4, 12717.0] |
| creative             |  75 |  12532.5 |  265.5 |   803.9 |      43.2 |     0.7029 | [12474.2, 12592.8] |
| grounded_facts       |  75 |  12525.7 |  258.4 |   852.0 |      42.0 |     0.6969 | [12467.8, 12583.8] |
| confabulation        |  75 |  12405.3 |  379.4 |   856.1 |      42.5 |     0.6998 | [12319.1, 12495.2] |
| unambiguous          |  75 |  12377.1 |  174.0 |   958.1 |      40.2 |     0.6884 | [12338.9, 12416.6] |
| self_reference       |  75 |  12154.2 |  289.8 |   911.9 |      42.5 |     0.6998 | [12088.0, 12218.8] |
| non_self_reference   |  75 |  12148.1 |  249.4 |   928.0 |      41.3 |     0.6961 | [12091.8, 12203.9] |
| guardrail_test       |  75 |  11959.4 |  281.5 |   973.3 |      41.7 |     0.6999 | [11897.5, 12023.7] |
| rote_completion      |  75 |  11839.5 |  585.8 |  1316.6 |      39.9 |     0.6907 | [11708.8, 11970.8] |
| math_reasoning       |  75 |  11807.5 |  641.8 |   939.5 |      42.6 |     0.7029 | [11672.0, 11961.3] |
| free_generation      |  75 |  11626.9 |  281.2 |  1352.6 |      38.3 |     0.6835 | [11564.5, 11691.1] |
| ambiguous            |  75 |  11583.1 |  382.8 |  1416.0 |      38.7 |     0.6841 | [11498.9, 11668.3] |

---

## Key Comparisons

### H1: Confabulation effect
- Cohen's d = -0.371 [-0.716, -0.051] (small)
- Recommended test: conservative_max (p = 0.0247)
- Group 1: mean=12405.3 (n=75), Group 2: mean=12525.7 (n=75)

### H1: Confabulation effect (eff. rank)
- Cohen's d = 0.351 [0.037, 0.662] (small)
- Recommended test: conservative_max (p = 0.2222)
- Group 1: mean=42.5 (n=75), Group 2: mean=42.0 (n=75)

### H2: Self-reference effect
- Cohen's d = 0.023 [-0.304, 0.345] (negligible)
- Recommended test: conservative_max (p = 0.8905)
- Group 1: mean=12154.2 (n=75), Group 2: mean=12148.1 (n=75)

### H2: Self-reference effect (eff. rank)
- Cohen's d = 0.779 [0.427, 1.215] (medium)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=42.5 (n=75), Group 2: mean=41.3 (n=75)

### H3: Refusal specificity
- Cohen's d = 0.261 [-0.063, 0.603] (small)
- Recommended test: conservative_max (p = 0.1129)
- Group 1: mean=11959.4 (n=75), Group 2: mean=11839.5 (n=75)

### H3: Refusal specificity (eff. rank)
- Cohen's d = 0.978 [0.661, 1.342] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=41.7 (n=75), Group 2: mean=39.9 (n=75)

### H3b: Refusal vs high-entropy
- Cohen's d = 1.182 [0.836, 1.605] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=11959.4 (n=75), Group 2: mean=11626.9 (n=75)

### H3b: Refusal vs high-entropy (eff. rank)
- Cohen's d = 2.919 [2.577, 3.374] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=41.7 (n=75), Group 2: mean=38.3 (n=75)

### Creative divergence
- Cohen's d = 0.026 [-0.298, 0.341] (negligible)
- Recommended test: conservative_max (p = 0.8755)
- Group 1: mean=12532.5 (n=75), Group 2: mean=12525.7 (n=75)

### Creative divergence (eff. rank)
- Cohen's d = 0.817 [0.472, 1.220] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=43.2 (n=75), Group 2: mean=42.0 (n=75)

### Emotional divergence
- Cohen's d = 0.566 [0.249, 0.906] (medium)
- Recommended test: conservative_max (p = 0.0007)
- Group 1: mean=12664.8 (n=75), Group 2: mean=12525.7 (n=75)

### Emotional divergence (eff. rank)
- Cohen's d = -1.353 [-1.759, -1.029] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=40.5 (n=75), Group 2: mean=42.0 (n=75)

### Reasoning mode
- Cohen's d = -1.468 [-2.320, -0.969] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=11807.5 (n=75), Group 2: mean=12525.7 (n=75)

### Reasoning mode (eff. rank)
- Cohen's d = 0.317 [-0.002, 0.643] (small)
- Recommended test: conservative_max (p = 0.1103)
- Group 1: mean=42.6 (n=75), Group 2: mean=42.0 (n=75)

### Code mode
- Cohen's d = 2.044 [1.783, 2.374] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=13608.6 (n=75), Group 2: mean=12525.7 (n=75)

### Code mode (eff. rank)
- Cohen's d = 2.151 [1.583, 3.072] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=46.5 (n=75), Group 2: mean=42.0 (n=75)

### Disambiguation demand
- Cohen's d = -2.671 [-3.209, -2.293] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=11583.1 (n=75), Group 2: mean=12377.1 (n=75)

### Disambiguation demand (eff. rank)
- Cohen's d = -0.748 [-1.175, -0.400] (medium)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=38.7 (n=75), Group 2: mean=40.2 (n=75)


## Multiple Comparison Correction

Method: holm_bonferroni (9 comparisons)

- confab_vs_facts: p_orig=0.0247, p_corr=0.0989 (not significant)
- self_ref_effect: p_orig=0.8905, p_corr=1.0000 (not significant)
- refusal_vs_rote: p_orig=0.1129, p_corr=0.3388 (not significant)
- refusal_vs_free: p_orig=0.0000, p_corr=0.0000 (SIGNIFICANT)
- creative_vs_facts: p_orig=0.8755, p_corr=1.0000 (not significant)
- emotion_vs_facts: p_orig=0.0007, p_corr=0.0035 (SIGNIFICANT)
- math_vs_facts: p_orig=0.0000, p_corr=0.0000 (SIGNIFICANT)
- code_vs_facts: p_orig=0.0000, p_corr=0.0000 (SIGNIFICANT)
- ambiguity_effect: p_orig=0.0000, p_corr=0.0000 (SIGNIFICANT)

---

## Hypothesis Evaluation

### H1_confabulation_inversion
- cohens_d: -0.3710119522560409
- interpretation: small
- ci_lower: -0.7163376727602816
- ci_upper: -0.05111118775327275
- p_value: 0.024724363680897855
- direction: facts_higher
- note: Positive d = confabulation has higher norms (expected at small scale). Negative d = confabulation has lower norms (inversion at large scale).

### H2_self_reference_emergence
- cohens_d: 0.022529107090398313
- interpretation: negligible
- ci_lower: -0.3040062556845906
- ci_upper: 0.3445241475231204
- p_value: 0.8904620882251517
- emergence_detected: False
- note: Effect > 0.3 suggests self-modeling capacity at this scale.

### H3_refusal_specialization
- cohens_d: 0.2609958967011702
- interpretation: small
- ci_lower: -0.06309656685979818
- ci_upper: 0.6030513469634551
- p_value: 0.112948177222711
- specialized: False
- note: d > 0.5 = refusal has unique signature beyond low entropy. d < 0.3 = refusal is just another memorized pattern.

### H4_category_invariance
- note: Evaluated in cross-scale comparison (requires multiple scale results).


## Power Analysis

- d_0.3: power=0.451 (n=75) [UNDERPOWERED]
- d_0.5: power=0.865 (n=75) [ADEQUATE]
- d_0.8: power=0.998 (n=75) [ADEQUATE]