# Scale Sweep Report: 2B

**Model**: google/gemma-2-2b-it
**Quantized**: False
**Generated**: 2026-03-01T23:18:36.349314

---

## Category Summaries

| Category | N | Mean Norm | Std | Mean/Token | Eff. Rank | Spectral H | 95% CI |
|----------|---|-----------|-----|------------|-----------|------------|--------|
| coding               |  75 |   8293.4 |  468.3 |   284.2 |      57.8 |     0.6423 | [8190.3, 8398.5] |
| grounded_facts       |  75 |   7046.6 |  237.3 |   486.4 |      49.1 |     0.6225 | [6995.7, 7102.6] |
| creative             |  75 |   7002.9 |  153.3 |   454.1 |      51.0 |     0.6233 | [6969.1, 7037.6] |
| emotional            |  75 |   6993.9 |  131.3 |   500.2 |      48.6 |     0.6112 | [6965.1, 7023.1] |
| confabulation        |  75 |   6957.1 |  382.3 |   478.7 |      50.0 |     0.6279 | [6873.7, 7048.6] |
| math_reasoning       |  75 |   6928.1 |  335.2 |   513.2 |      49.1 |     0.6283 | [6854.5, 7005.9] |
| unambiguous          |  75 |   6824.1 |  135.0 |   533.7 |      48.0 |     0.6141 | [6793.4, 6854.4] |
| non_self_reference   |  75 |   6820.5 |  183.9 |   516.8 |      47.3 |     0.6171 | [6779.2, 6863.0] |
| guardrail_test       |  75 |   6747.4 |  148.8 |   548.6 |      47.5 |     0.6179 | [6714.6, 6781.8] |
| self_reference       |  75 |   6701.7 |  312.9 |   502.7 |      49.1 |     0.6236 | [6625.9, 6765.3] |
| rote_completion      |  75 |   6656.9 |  344.1 |   748.3 |      46.7 |     0.6111 | [6580.9, 6734.5] |
| free_generation      |  75 |   6548.2 |  205.9 |   738.0 |      44.6 |     0.6033 | [6503.0, 6595.7] |
| ambiguous            |  75 |   6514.1 |  253.3 |   808.3 |      41.8 |     0.5912 | [6457.7, 6572.2] |

---

## Key Comparisons

### H1: Confabulation effect
- Cohen's d = -0.281 [-0.634, 0.043] (small)
- Recommended test: conservative_max (p = 0.0872)
- Group 1: mean=6957.1 (n=75), Group 2: mean=7046.6 (n=75)

### H1: Confabulation effect (eff. rank)
- Cohen's d = 0.358 [0.034, 0.706] (small)
- Recommended test: conservative_max (p = 0.0484)
- Group 1: mean=50.0 (n=75), Group 2: mean=49.1 (n=75)

### H2: Self-reference effect
- Cohen's d = -0.463 [-0.685, -0.201] (small)
- Recommended test: conservative_max (p = 0.0671)
- Group 1: mean=6701.7 (n=75), Group 2: mean=6820.5 (n=75)

### H2: Self-reference effect (eff. rank)
- Cohen's d = 0.553 [0.197, 1.026] (medium)
- Recommended test: conservative_max (p = 0.0009)
- Group 1: mean=49.1 (n=75), Group 2: mean=47.3 (n=75)

### H3: Refusal specificity
- Cohen's d = 0.341 [0.015, 0.678] (small)
- Recommended test: conservative_max (p = 0.0823)
- Group 1: mean=6747.4 (n=75), Group 2: mean=6656.9 (n=75)

### H3: Refusal specificity (eff. rank)
- Cohen's d = 0.332 [0.008, 0.653] (small)
- Recommended test: conservative_max (p = 0.1734)
- Group 1: mean=47.5 (n=75), Group 2: mean=46.7 (n=75)

### H3b: Refusal vs high-entropy
- Cohen's d = 1.108 [0.753, 1.542] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=6747.4 (n=75), Group 2: mean=6548.2 (n=75)

### H3b: Refusal vs high-entropy (eff. rank)
- Cohen's d = 1.222 [0.883, 1.626] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=47.5 (n=75), Group 2: mean=44.6 (n=75)

### Creative divergence
- Cohen's d = -0.219 [-0.510, 0.105] (small)
- Recommended test: conservative_max (p = 0.6736)
- Group 1: mean=7002.9 (n=75), Group 2: mean=7046.6 (n=75)

### Creative divergence (eff. rank)
- Cohen's d = 0.952 [0.572, 1.471] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=51.0 (n=75), Group 2: mean=49.1 (n=75)

### Emotional divergence
- Cohen's d = -0.275 [-0.558, 0.036] (small)
- Recommended test: conservative_max (p = 0.3728)
- Group 1: mean=6993.9 (n=75), Group 2: mean=7046.6 (n=75)

### Emotional divergence (eff. rank)
- Cohen's d = -0.280 [-0.555, 0.036] (small)
- Recommended test: conservative_max (p = 0.3241)
- Group 1: mean=48.6 (n=75), Group 2: mean=49.1 (n=75)

### Reasoning mode
- Cohen's d = -0.408 [-0.786, -0.086] (small)
- Recommended test: conservative_max (p = 0.0137)
- Group 1: mean=6928.1 (n=75), Group 2: mean=7046.6 (n=75)

### Reasoning mode (eff. rank)
- Cohen's d = -0.010 [-0.340, 0.310] (negligible)
- Recommended test: conservative_max (p = 0.9517)
- Group 1: mean=49.1 (n=75), Group 2: mean=49.1 (n=75)

### Code mode
- Cohen's d = 3.359 [2.951, 3.938] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=8293.4 (n=75), Group 2: mean=7046.6 (n=75)

### Code mode (eff. rank)
- Cohen's d = 3.297 [2.687, 4.215] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=57.8 (n=75), Group 2: mean=49.1 (n=75)

### Disambiguation demand
- Cohen's d = -1.527 [-1.982, -1.155] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=6514.1 (n=75), Group 2: mean=6824.1 (n=75)

### Disambiguation demand (eff. rank)
- Cohen's d = -1.681 [-1.988, -1.447] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=41.8 (n=75), Group 2: mean=48.0 (n=75)


## Multiple Comparison Correction

Method: holm_bonferroni (9 comparisons)

- confab_vs_facts: p_orig=0.0872, p_corr=0.3354 (not significant)
- self_ref_effect: p_orig=0.0671, p_corr=0.3354 (not significant)
- refusal_vs_rote: p_orig=0.0823, p_corr=0.3354 (not significant)
- refusal_vs_free: p_orig=0.0000, p_corr=0.0000 (SIGNIFICANT)
- creative_vs_facts: p_orig=0.6736, p_corr=0.7456 (not significant)
- emotion_vs_facts: p_orig=0.3728, p_corr=0.7456 (not significant)
- math_vs_facts: p_orig=0.0137, p_corr=0.0820 (not significant)
- code_vs_facts: p_orig=0.0000, p_corr=0.0000 (SIGNIFICANT)
- ambiguity_effect: p_orig=0.0000, p_corr=0.0000 (SIGNIFICANT)

---

## Hypothesis Evaluation

### H1_confabulation_inversion
- cohens_d: -0.2814996901816103
- interpretation: small
- ci_lower: -0.6339951081547286
- ci_upper: 0.04283452863776063
- p_value: 0.08723872155166629
- direction: facts_higher
- note: Positive d = confabulation has higher norms (expected at small scale). Negative d = confabulation has lower norms (inversion at large scale).

### H2_self_reference_emergence
- cohens_d: -0.462717125055079
- interpretation: small
- ci_lower: -0.6850155151616732
- ci_upper: -0.20112825949542995
- p_value: 0.06708276471621659
- emergence_detected: True
- note: Effect > 0.3 suggests self-modeling capacity at this scale.

### H3_refusal_specialization
- cohens_d: 0.34117422932720215
- interpretation: small
- ci_lower: 0.01533830010862228
- ci_upper: 0.6779265640553273
- p_value: 0.08233886949507026
- specialized: False
- note: d > 0.5 = refusal has unique signature beyond low entropy. d < 0.3 = refusal is just another memorized pattern.

### H4_category_invariance
- note: Evaluated in cross-scale comparison (requires multiple scale results).


## Power Analysis

- d_0.3: power=0.451 (n=75) [UNDERPOWERED]
- d_0.5: power=0.865 (n=75) [ADEQUATE]
- d_0.8: power=0.998 (n=75) [ADEQUATE]