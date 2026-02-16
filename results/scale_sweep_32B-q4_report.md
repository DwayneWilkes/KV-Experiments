# Scale Sweep Report: 32B-q4

**Model**: Qwen/Qwen2.5-32B-Instruct
**Quantized**: True
**Generated**: 2026-02-15T23:11:06.810807

---

## Category Summaries

| Category | N | Mean Norm | Std | Mean/Token | Eff. Rank | Spectral H | 95% CI |
|----------|---|-----------|-----|------------|-----------|------------|--------|
| coding               |  75 |  23180.7 | 1404.5 |   988.4 |      45.1 |     0.7045 | [22864.8, 23485.9] |
| creative             |  75 |  22005.4 |  391.0 |  1509.1 |      43.6 |     0.6992 | [21916.4, 22091.6] |
| grounded_facts       |  75 |  21888.3 |  554.3 |  1553.2 |      42.7 |     0.6937 | [21767.7, 22016.0] |
| confabulation        |  75 |  21764.6 |  986.3 |  1546.9 |      43.5 |     0.6982 | [21547.0, 21998.8] |
| emotional            |  75 |  21728.3 |  526.0 |  1620.4 |      42.2 |     0.6914 | [21609.0, 21841.9] |
| unambiguous          |  75 |  21450.4 |  354.9 |  1800.0 |      42.8 |     0.6970 | [21370.2, 21531.6] |
| math_reasoning       |  75 |  21433.5 |  916.0 |  1726.8 |      40.3 |     0.6842 | [21229.6, 21646.6] |
| non_self_reference   |  75 |  21218.3 |  503.1 |  1724.2 |      42.4 |     0.6928 | [21109.0, 21336.0] |
| self_reference       |  75 |  21147.2 |  308.1 |  1716.1 |      44.3 |     0.7024 | [21076.6, 21214.6] |
| guardrail_test       |  75 |  20957.1 |  444.9 |  1860.0 |      42.5 |     0.6964 | [20857.1, 21059.2] |
| rote_completion      |  75 |  20658.6 | 1054.2 |  2628.5 |      39.8 |     0.6851 | [20426.3, 20894.0] |
| ambiguous            |  75 |  20275.4 |  732.9 |  2845.6 |      41.2 |     0.6929 | [20111.5, 20442.3] |
| free_generation      |  75 |  19981.5 |  455.9 |  2640.5 |      39.8 |     0.6864 | [19880.5, 20088.3] |

---

## Key Comparisons

### H1: Confabulation effect
- Cohen's d = -0.155 [-0.508, 0.167] (negligible)
- Recommended test: mann_whitney (p = 0.0098)
- Group 1: mean=21764.6 (n=75), Group 2: mean=21888.3 (n=75)

### H1: Confabulation effect (eff. rank)
- Cohen's d = 0.507 [0.197, 0.809] (medium)
- Recommended test: mann_whitney (p = 0.0188)
- Group 1: mean=43.5 (n=75), Group 2: mean=42.7 (n=75)

### H2: Self-reference effect
- Cohen's d = -0.170 [-0.462, 0.156] (negligible)
- Recommended test: mann_whitney (p = 0.9640)
- Group 1: mean=21147.2 (n=75), Group 2: mean=21218.3 (n=75)

### H2: Self-reference effect (eff. rank)
- Cohen's d = 1.228 [0.936, 1.555] (large)
- Recommended test: mann_whitney (p = 0.0000)
- Group 1: mean=44.3 (n=75), Group 2: mean=42.4 (n=75)

### H3: Refusal specificity
- Cohen's d = 0.369 [0.039, 0.719] (small)
- Recommended test: mann_whitney (p = 0.0023)
- Group 1: mean=20957.1 (n=75), Group 2: mean=20658.6 (n=75)

### H3: Refusal specificity (eff. rank)
- Cohen's d = 1.608 [1.315, 1.979] (large)
- Recommended test: mann_whitney (p = 0.0000)
- Group 1: mean=42.5 (n=75), Group 2: mean=39.8 (n=75)

### H3b: Refusal vs high-entropy
- Cohen's d = 2.166 [1.747, 2.693] (large)
- Recommended test: mann_whitney (p = 0.0000)
- Group 1: mean=20957.1 (n=75), Group 2: mean=19981.5 (n=75)

### H3b: Refusal vs high-entropy (eff. rank)
- Cohen's d = 2.244 [1.863, 2.700] (large)
- Recommended test: mann_whitney (p = 0.0000)
- Group 1: mean=42.5 (n=75), Group 2: mean=39.8 (n=75)

### Creative divergence
- Cohen's d = 0.244 [-0.069, 0.614] (small)
- Recommended test: mann_whitney (p = 0.0031)
- Group 1: mean=22005.4 (n=75), Group 2: mean=21888.3 (n=75)

### Creative divergence (eff. rank)
- Cohen's d = 0.690 [0.382, 1.023] (medium)
- Recommended test: mann_whitney (p = 0.0004)
- Group 1: mean=43.6 (n=75), Group 2: mean=42.7 (n=75)

### Emotional divergence
- Cohen's d = -0.296 [-0.602, 0.017] (small)
- Recommended test: mann_whitney (p = 0.3728)
- Group 1: mean=21728.3 (n=75), Group 2: mean=21888.3 (n=75)

### Emotional divergence (eff. rank)
- Cohen's d = -0.331 [-0.689, 0.001] (small)
- Recommended test: mann_whitney (p = 0.0188)
- Group 1: mean=42.2 (n=75), Group 2: mean=42.7 (n=75)

### Reasoning mode
- Cohen's d = -0.601 [-1.010, -0.268] (medium)
- Recommended test: mann_whitney (p = 0.0000)
- Group 1: mean=21433.5 (n=75), Group 2: mean=21888.3 (n=75)

### Reasoning mode (eff. rank)
- Cohen's d = -1.384 [-1.815, -1.051] (large)
- Recommended test: mann_whitney (p = 0.0000)
- Group 1: mean=40.3 (n=75), Group 2: mean=42.7 (n=75)

### Code mode
- Cohen's d = 1.210 [0.847, 1.669] (large)
- Recommended test: mann_whitney (p = 0.0000)
- Group 1: mean=23180.7 (n=75), Group 2: mean=21888.3 (n=75)

### Code mode (eff. rank)
- Cohen's d = 1.487 [1.128, 1.902] (large)
- Recommended test: mann_whitney (p = 0.0000)
- Group 1: mean=45.1 (n=75), Group 2: mean=42.7 (n=75)

### Disambiguation demand
- Cohen's d = -2.041 [-2.512, -1.677] (large)
- Recommended test: mann_whitney (p = 0.0000)
- Group 1: mean=20275.4 (n=75), Group 2: mean=21450.4 (n=75)

### Disambiguation demand (eff. rank)
- Cohen's d = -1.044 [-1.374, -0.747] (large)
- Recommended test: mann_whitney (p = 0.0000)
- Group 1: mean=41.2 (n=75), Group 2: mean=42.8 (n=75)


## Multiple Comparison Correction

Method: holm_bonferroni (9 comparisons)

- confab_vs_facts: p_orig=0.0098, p_corr=0.0293 (SIGNIFICANT)
- self_ref_effect: p_orig=0.9640, p_corr=0.9640 (not significant)
- refusal_vs_rote: p_orig=0.0023, p_corr=0.0113 (SIGNIFICANT)
- refusal_vs_free: p_orig=0.0000, p_corr=0.0000 (SIGNIFICANT)
- creative_vs_facts: p_orig=0.0031, p_corr=0.0123 (SIGNIFICANT)
- emotion_vs_facts: p_orig=0.3728, p_corr=0.7455 (not significant)
- math_vs_facts: p_orig=0.0000, p_corr=0.0001 (SIGNIFICANT)
- code_vs_facts: p_orig=0.0000, p_corr=0.0000 (SIGNIFICANT)
- ambiguity_effect: p_orig=0.0000, p_corr=0.0000 (SIGNIFICANT)

---

## Hypothesis Evaluation

### H1_confabulation_inversion
- cohens_d: -0.15460417664191392
- interpretation: negligible
- ci_lower: -0.5082653677252288
- ci_upper: 0.16708820017664186
- p_value: 0.0097768196281691
- direction: facts_higher
- note: Positive d = confabulation has higher norms (expected at small scale). Negative d = confabulation has lower norms (inversion at large scale).

### H2_self_reference_emergence
- cohens_d: -0.1703036367749009
- interpretation: negligible
- ci_lower: -0.46241353867501045
- ci_upper: 0.15614655572026068
- p_value: 0.9640046370958462
- emergence_detected: False
- note: Effect > 0.3 suggests self-modeling capacity at this scale.

### H3_refusal_specialization
- cohens_d: 0.36898616496412
- interpretation: small
- ci_lower: 0.03875524842580761
- ci_upper: 0.7187095319479746
- p_value: 0.0022602579250233096
- specialized: False
- note: d > 0.5 = refusal has unique signature beyond low entropy. d < 0.3 = refusal is just another memorized pattern.

### H4_category_invariance
- note: Evaluated in cross-scale comparison (requires multiple scale results).


## Power Analysis

- d_0.3: power=0.451 (n=75) [UNDERPOWERED]
- d_0.5: power=0.865 (n=75) [ADEQUATE]
- d_0.8: power=0.998 (n=75) [ADEQUATE]