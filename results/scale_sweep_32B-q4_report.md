# Scale Sweep Report: 32B-q4

**Model**: Qwen/Qwen2.5-32B-Instruct
**Quantized**: True
**Generated**: 2026-03-01T22:28:00.788104

---

## Category Summaries

| Category | N | Mean Norm | Std | Mean/Token | Eff. Rank | Spectral H | 95% CI |
|----------|---|-----------|-----|------------|-----------|------------|--------|
| coding               |  75 |  23180.6 | 1404.4 |   988.4 |      45.1 |     0.7045 | [22864.8, 23485.9] |
| creative             |  75 |  21979.5 |  439.2 |  1507.1 |      43.7 |     0.7000 | [21877.8, 22076.2] |
| grounded_facts       |  75 |  21888.3 |  554.3 |  1553.2 |      42.7 |     0.6937 | [21767.8, 22016.0] |
| confabulation        |  75 |  21764.6 |  986.5 |  1546.9 |      43.5 |     0.6982 | [21547.1, 21998.9] |
| emotional            |  75 |  21728.2 |  525.8 |  1620.4 |      42.2 |     0.6916 | [21608.8, 21841.7] |
| unambiguous          |  75 |  21453.5 |  352.4 |  1800.3 |      42.8 |     0.6971 | [21373.9, 21533.9] |
| math_reasoning       |  75 |  21437.1 |  915.4 |  1727.1 |      40.3 |     0.6841 | [21233.2, 21650.2] |
| non_self_reference   |  75 |  21232.9 |  523.5 |  1725.2 |      42.4 |     0.6928 | [21119.4, 21355.1] |
| self_reference       |  75 |  21147.2 |  308.0 |  1716.1 |      44.3 |     0.7024 | [21076.6, 21214.6] |
| guardrail_test       |  75 |  20957.2 |  444.9 |  1860.0 |      42.5 |     0.6964 | [20857.2, 21059.3] |
| rote_completion      |  75 |  20658.7 | 1054.2 |  2628.5 |      39.8 |     0.6851 | [20426.4, 20894.1] |
| ambiguous            |  75 |  20275.3 |  732.8 |  2845.6 |      41.2 |     0.6929 | [20111.5, 20442.2] |
| free_generation      |  75 |  19981.4 |  455.9 |  2640.5 |      39.8 |     0.6864 | [19880.4, 20088.2] |

---

## Key Comparisons

### H1: Confabulation effect
- Cohen's d = -0.155 [-0.508, 0.164] (negligible)
- Recommended test: conservative_max (p = 0.3459)
- Group 1: mean=21764.6 (n=75), Group 2: mean=21888.3 (n=75)

### H1: Confabulation effect (eff. rank)
- Cohen's d = 0.503 [0.198, 0.810] (medium)
- Recommended test: conservative_max (p = 0.0188)
- Group 1: mean=43.5 (n=75), Group 2: mean=42.7 (n=75)

### H2: Self-reference effect
- Cohen's d = -0.200 [-0.494, 0.122] (negligible)
- Recommended test: conservative_max (p = 0.9640)
- Group 1: mean=21147.2 (n=75), Group 2: mean=21232.9 (n=75)

### H2: Self-reference effect (eff. rank)
- Cohen's d = 1.253 [0.964, 1.588] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=44.3 (n=75), Group 2: mean=42.4 (n=75)

### H3: Refusal specificity
- Cohen's d = 0.369 [0.046, 0.716] (small)
- Recommended test: conservative_max (p = 0.0260)
- Group 1: mean=20957.2 (n=75), Group 2: mean=20658.7 (n=75)

### H3: Refusal specificity (eff. rank)
- Cohen's d = 1.609 [1.312, 1.982] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=42.5 (n=75), Group 2: mean=39.8 (n=75)

### H3b: Refusal vs high-entropy
- Cohen's d = 2.166 [1.750, 2.699] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=20957.2 (n=75), Group 2: mean=19981.4 (n=75)

### H3b: Refusal vs high-entropy (eff. rank)
- Cohen's d = 2.243 [1.870, 2.699] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=42.5 (n=75), Group 2: mean=39.8 (n=75)

### Creative divergence
- Cohen's d = 0.182 [-0.127, 0.547] (negligible)
- Recommended test: conservative_max (p = 0.2658)
- Group 1: mean=21979.5 (n=75), Group 2: mean=21888.3 (n=75)

### Creative divergence (eff. rank)
- Cohen's d = 0.729 [0.421, 1.064] (medium)
- Recommended test: conservative_max (p = 0.0001)
- Group 1: mean=43.7 (n=75), Group 2: mean=42.7 (n=75)

### Emotional divergence
- Cohen's d = -0.296 [-0.598, 0.019] (small)
- Recommended test: conservative_max (p = 0.3728)
- Group 1: mean=21728.2 (n=75), Group 2: mean=21888.3 (n=75)

### Emotional divergence (eff. rank)
- Cohen's d = -0.302 [-0.652, 0.023] (small)
- Recommended test: conservative_max (p = 0.0661)
- Group 1: mean=42.2 (n=75), Group 2: mean=42.7 (n=75)

### Reasoning mode
- Cohen's d = -0.596 [-0.997, -0.263] (medium)
- Recommended test: conservative_max (p = 0.0004)
- Group 1: mean=21437.1 (n=75), Group 2: mean=21888.3 (n=75)

### Reasoning mode (eff. rank)
- Cohen's d = -1.392 [-1.815, -1.056] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=40.3 (n=75), Group 2: mean=42.7 (n=75)

### Code mode
- Cohen's d = 1.210 [0.849, 1.675] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=23180.6 (n=75), Group 2: mean=21888.3 (n=75)

### Code mode (eff. rank)
- Cohen's d = 1.486 [1.131, 1.909] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=45.1 (n=75), Group 2: mean=42.7 (n=75)

### Disambiguation demand
- Cohen's d = -2.049 [-2.526, -1.685] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=20275.3 (n=75), Group 2: mean=21453.5 (n=75)

### Disambiguation demand (eff. rank)
- Cohen's d = -1.051 [-1.386, -0.754] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=41.2 (n=75), Group 2: mean=42.8 (n=75)


## Multiple Comparison Correction

Method: holm_bonferroni (9 comparisons)

- confab_vs_facts: p_orig=0.3459, p_corr=1.0000 (not significant)
- self_ref_effect: p_orig=0.9640, p_corr=1.0000 (not significant)
- refusal_vs_rote: p_orig=0.0260, p_corr=0.1300 (not significant)
- refusal_vs_free: p_orig=0.0000, p_corr=0.0000 (SIGNIFICANT)
- creative_vs_facts: p_orig=0.2658, p_corr=1.0000 (not significant)
- emotion_vs_facts: p_orig=0.3728, p_corr=1.0000 (not significant)
- math_vs_facts: p_orig=0.0004, p_corr=0.0023 (SIGNIFICANT)
- code_vs_facts: p_orig=0.0000, p_corr=0.0000 (SIGNIFICANT)
- ambiguity_effect: p_orig=0.0000, p_corr=0.0000 (SIGNIFICANT)

---

## Hypothesis Evaluation

### H1_confabulation_inversion
- cohens_d: -0.15455196144360664
- interpretation: negligible
- ci_lower: -0.5078138833207937
- ci_upper: 0.16381944750157054
- p_value: 0.34588659935226307
- direction: facts_higher
- note: Positive d = confabulation has higher norms (expected at small scale). Negative d = confabulation has lower norms (inversion at large scale).

### H2_self_reference_emergence
- cohens_d: -0.1995521677942378
- interpretation: negligible
- ci_lower: -0.493694345040919
- ci_upper: 0.12233311446203254
- p_value: 0.9640046370958462
- emergence_detected: False
- note: Effect > 0.3 suggests self-modeling capacity at this scale.

### H3_refusal_specialization
- cohens_d: 0.36904498657791834
- interpretation: small
- ci_lower: 0.04557035551267806
- ci_upper: 0.7158943797536046
- p_value: 0.026004133764689314
- specialized: False
- note: d > 0.5 = refusal has unique signature beyond low entropy. d < 0.3 = refusal is just another memorized pattern.

### H4_category_invariance
- note: Evaluated in cross-scale comparison (requires multiple scale results).


## Power Analysis

- d_0.3: power=0.451 (n=75) [UNDERPOWERED]
- d_0.5: power=0.865 (n=75) [ADEQUATE]
- d_0.8: power=0.998 (n=75) [ADEQUATE]