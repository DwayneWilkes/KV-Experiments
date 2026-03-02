# Scale Sweep Report: 70B-q4

**Model**: meta-llama/Llama-3.1-70B-Instruct
**Quantized**: True
**Generated**: 2026-03-02T00:16:54.184385

---

## Category Summaries

| Category | N | Mean Norm | Std | Mean/Token | Eff. Rank | Spectral H | 95% CI |
|----------|---|-----------|-----|------------|-----------|------------|--------|
| coding               |  45 |  33562.9 | 5512.5 |  1377.0 |      37.7 |     0.6907 | [31888.5, 35078.4] |
| grounded_facts       |  45 |  33526.6 |  752.8 |  2280.1 |      34.6 |     0.6778 | [33316.0, 33756.9] |
| creative             |  45 |  33341.2 |  727.2 |  2137.9 |      35.0 |     0.6767 | [33135.0, 33555.2] |
| confabulation        |  45 |  33017.0 | 1115.6 |  2277.5 |      35.3 |     0.6795 | [32706.6, 33344.5] |
| emotional            |  45 |  32799.7 |  693.9 |  2283.3 |      33.2 |     0.6670 | [32600.0, 33002.9] |
| unambiguous          |  45 |  32307.5 |  542.8 |  2499.6 |      33.2 |     0.6651 | [32150.9, 32458.7] |
| math_reasoning       |  45 |  31779.5 | 1377.7 |  2532.5 |      35.1 |     0.6788 | [31403.3, 32197.9] |
| rote_completion      |  45 |  31465.0 | 1494.8 |  3501.0 |      32.0 |     0.6651 | [31033.7, 31889.2] |
| non_self_reference   |  45 |  31441.8 | 4015.0 |  2401.0 |      33.1 |     0.6688 | [30157.5, 32466.8] |
| ambiguous            |  45 |  30604.0 |  834.9 |  3743.9 |      31.3 |     0.6595 | [30365.3, 30852.8] |
| free_generation      |  45 |  30284.4 |  926.0 |  3521.8 |      32.3 |     0.6673 | [30023.9, 30552.4] |
| guardrail_test       |  45 |  29607.1 | 3755.2 |  2423.3 |      33.0 |     0.6744 | [28478.9, 30652.6] |
| self_reference       |  45 |  29321.3 | 5548.0 |  2198.8 |      32.4 |     0.6743 | [27621.4, 30790.9] |

---

## Key Comparisons

### H1: Confabulation effect
- Cohen's d = -0.536 [-0.996, -0.131] (medium)
- Recommended test: conservative_max (p = 0.0131)
- Group 1: mean=33017.0 (n=45), Group 2: mean=33526.6 (n=45)

### H1: Confabulation effect (eff. rank)
- Cohen's d = 0.769 [0.366, 1.213] (medium)
- Recommended test: conservative_max (p = 0.0005)
- Group 1: mean=35.3 (n=45), Group 2: mean=34.6 (n=45)

### H2: Self-reference effect
- Cohen's d = -0.438 [-0.847, -0.027] (small)
- Recommended test: conservative_max (p = 0.0410)
- Group 1: mean=29321.3 (n=45), Group 2: mean=31441.8 (n=45)

### H2: Self-reference effect (eff. rank)
- Cohen's d = -0.187 [-0.570, 0.235] (negligible)
- Recommended test: conservative_max (p = 0.5637)
- Group 1: mean=32.4 (n=45), Group 2: mean=33.1 (n=45)

### H3: Refusal specificity
- Cohen's d = -0.650 [-0.959, -0.321] (medium)
- Recommended test: conservative_max (p = 0.0270)
- Group 1: mean=29607.1 (n=45), Group 2: mean=31465.0 (n=45)

### H3: Refusal specificity (eff. rank)
- Cohen's d = 0.477 [0.061, 0.983] (small)
- Recommended test: conservative_max (p = 0.0263)
- Group 1: mean=33.0 (n=45), Group 2: mean=32.0 (n=45)

### H3b: Refusal vs high-entropy
- Cohen's d = -0.248 [-0.586, 0.183] (small)
- Recommended test: conservative_max (p = 0.2458)
- Group 1: mean=29607.1 (n=45), Group 2: mean=30284.4 (n=45)

### H3b: Refusal vs high-entropy (eff. rank)
- Cohen's d = 0.355 [-0.051, 0.876] (small)
- Recommended test: conservative_max (p = 0.0961)
- Group 1: mean=33.0 (n=45), Group 2: mean=32.3 (n=45)

### Creative divergence
- Cohen's d = -0.251 [-0.672, 0.157] (small)
- Recommended test: conservative_max (p = 0.4056)
- Group 1: mean=33341.2 (n=45), Group 2: mean=33526.6 (n=45)

### Creative divergence (eff. rank)
- Cohen's d = 0.505 [0.101, 0.949] (medium)
- Recommended test: conservative_max (p = 0.0269)
- Group 1: mean=35.0 (n=45), Group 2: mean=34.6 (n=45)

### Emotional divergence
- Cohen's d = -1.004 [-1.460, -0.615] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=32799.7 (n=45), Group 2: mean=33526.6 (n=45)

### Emotional divergence (eff. rank)
- Cohen's d = -1.634 [-2.234, -1.183] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=33.2 (n=45), Group 2: mean=34.6 (n=45)

### Reasoning mode
- Cohen's d = -1.574 [-2.489, -1.013] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=31779.5 (n=45), Group 2: mean=33526.6 (n=45)

### Reasoning mode (eff. rank)
- Cohen's d = 0.329 [-0.083, 0.722] (small)
- Recommended test: conservative_max (p = 0.4925)
- Group 1: mean=35.1 (n=45), Group 2: mean=34.6 (n=45)

### Code mode
- Cohen's d = 0.009 [-0.374, 0.503] (negligible)
- Recommended test: conservative_max (p = 0.9653)
- Group 1: mean=33562.9 (n=45), Group 2: mean=33526.6 (n=45)

### Code mode (eff. rank)
- Cohen's d = 0.998 [0.493, 1.857] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=37.7 (n=45), Group 2: mean=34.6 (n=45)

### Disambiguation demand
- Cohen's d = -2.419 [-3.122, -1.929] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=30604.0 (n=45), Group 2: mean=32307.5 (n=45)

### Disambiguation demand (eff. rank)
- Cohen's d = -1.129 [-1.806, -0.667] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=31.3 (n=45), Group 2: mean=33.2 (n=45)


## Multiple Comparison Correction

Method: holm_bonferroni (9 comparisons)

- confab_vs_facts: p_orig=0.0131, p_corr=0.0785 (not significant)
- self_ref_effect: p_orig=0.0410, p_corr=0.1640 (not significant)
- refusal_vs_rote: p_orig=0.0270, p_corr=0.1348 (not significant)
- refusal_vs_free: p_orig=0.2458, p_corr=0.7373 (not significant)
- creative_vs_facts: p_orig=0.4056, p_corr=0.8113 (not significant)
- emotion_vs_facts: p_orig=0.0000, p_corr=0.0001 (SIGNIFICANT)
- math_vs_facts: p_orig=0.0000, p_corr=0.0000 (SIGNIFICANT)
- code_vs_facts: p_orig=0.9653, p_corr=0.9653 (not significant)
- ambiguity_effect: p_orig=0.0000, p_corr=0.0000 (SIGNIFICANT)

---

## Hypothesis Evaluation

### H1_confabulation_inversion
- cohens_d: -0.535526193134197
- interpretation: medium
- ci_lower: -0.9955740042795459
- ci_upper: -0.13077971766784785
- p_value: 0.013086167845981884
- direction: facts_higher
- note: Positive d = confabulation has higher norms (expected at small scale). Negative d = confabulation has lower norms (inversion at large scale).

### H2_self_reference_emergence
- cohens_d: -0.4378914074038447
- interpretation: small
- ci_lower: -0.8470023425091476
- ci_upper: -0.027054289936675885
- p_value: 0.04099436544221333
- emergence_detected: True
- note: Effect > 0.3 suggests self-modeling capacity at this scale.

### H3_refusal_specialization
- cohens_d: -0.65008037195766
- interpretation: medium
- ci_lower: -0.9592264714502491
- ci_upper: -0.32103701999309
- p_value: 0.026953486746252576
- specialized: True
- note: d > 0.5 = refusal has unique signature beyond low entropy. d < 0.3 = refusal is just another memorized pattern.

### H4_category_invariance
- note: Evaluated in cross-scale comparison (requires multiple scale results).


## Power Analysis

- d_0.3: power=0.296 (n=45) [UNDERPOWERED]
- d_0.5: power=0.660 (n=45) [UNDERPOWERED]
- d_0.8: power=0.967 (n=45) [ADEQUATE]