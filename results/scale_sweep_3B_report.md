# Scale Sweep Report: 3B

**Model**: Qwen/Qwen2.5-3B-Instruct
**Quantized**: False
**Generated**: 2026-03-01T20:23:37.025292

---

## Category Summaries

| Category | N | Mean Norm | Std | Mean/Token | Eff. Rank | Spectral H | 95% CI |
|----------|---|-----------|-----|------------|-----------|------------|--------|
| coding               |  75 |   9114.9 |  770.5 |   388.4 |      18.1 |     0.4794 | [8934.0, 9280.0] |
| creative             |  75 |   8479.8 |  129.2 |   581.6 |      17.8 |     0.4888 | [8450.6, 8509.0] |
| grounded_facts       |  75 |   8472.9 |  247.0 |   600.8 |      17.3 |     0.4872 | [8418.6, 8530.1] |
| confabulation        |  75 |   8461.8 |  391.0 |   601.2 |      17.7 |     0.4884 | [8374.8, 8555.2] |
| emotional            |  75 |   8447.4 |  159.7 |   630.2 |      16.6 |     0.4802 | [8411.5, 8483.1] |
| math_reasoning       |  75 |   8388.7 |  389.7 |   675.3 |      15.8 |     0.4765 | [8302.2, 8479.4] |
| unambiguous          |  75 |   8279.4 |  131.4 |   694.8 |      16.4 |     0.4842 | [8250.3, 8309.5] |
| non_self_reference   |  75 |   8251.1 |  185.7 |   670.5 |      17.0 |     0.4880 | [8209.9, 8294.3] |
| self_reference       |  75 |   8191.8 |  154.4 |   664.6 |      17.4 |     0.4909 | [8157.4, 8226.7] |
| guardrail_test       |  75 |   8168.5 |  181.7 |   725.0 |      16.7 |     0.4932 | [8127.6, 8210.4] |
| rote_completion      |  75 |   7987.2 |  377.7 |  1016.9 |      15.8 |     0.4899 | [7904.1, 8071.7] |
| ambiguous            |  75 |   7822.6 |  242.6 |  1099.3 |      14.8 |     0.4806 | [7768.6, 7878.0] |
| free_generation      |  75 |   7799.8 |  209.3 |  1030.0 |      15.5 |     0.4910 | [7753.5, 7848.4] |

---

## Key Comparisons

### H1: Confabulation effect
- Cohen's d = -0.034 [-0.377, 0.280] (negligible)
- Recommended test: conservative_max (p = 0.8357)
- Group 1: mean=8461.8 (n=75), Group 2: mean=8472.9 (n=75)

### H1: Confabulation effect (eff. rank)
- Cohen's d = 0.312 [-0.005, 0.608] (small)
- Recommended test: conservative_max (p = 0.2803)
- Group 1: mean=17.7 (n=75), Group 2: mean=17.3 (n=75)

### H2: Self-reference effect
- Cohen's d = -0.347 [-0.666, -0.038] (small)
- Recommended test: conservative_max (p = 0.0542)
- Group 1: mean=8191.8 (n=75), Group 2: mean=8251.1 (n=75)

### H2: Self-reference effect (eff. rank)
- Cohen's d = 0.422 [0.105, 0.761] (small)
- Recommended test: conservative_max (p = 0.0108)
- Group 1: mean=17.4 (n=75), Group 2: mean=17.0 (n=75)

### H3: Refusal specificity
- Cohen's d = 0.612 [0.288, 0.977] (medium)
- Recommended test: conservative_max (p = 0.0003)
- Group 1: mean=8168.5 (n=75), Group 2: mean=7987.2 (n=75)

### H3: Refusal specificity (eff. rank)
- Cohen's d = 0.714 [0.417, 1.021] (medium)
- Recommended test: conservative_max (p = 0.0012)
- Group 1: mean=16.7 (n=75), Group 2: mean=15.8 (n=75)

### H3b: Refusal vs high-entropy
- Cohen's d = 1.881 [1.482, 2.411] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=8168.5 (n=75), Group 2: mean=7799.8 (n=75)

### H3b: Refusal vs high-entropy (eff. rank)
- Cohen's d = 1.641 [1.293, 2.064] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=16.7 (n=75), Group 2: mean=15.5 (n=75)

### Creative divergence
- Cohen's d = 0.035 [-0.268, 0.388] (negligible)
- Recommended test: conservative_max (p = 0.8314)
- Group 1: mean=8479.8 (n=75), Group 2: mean=8472.9 (n=75)

### Creative divergence (eff. rank)
- Cohen's d = 0.575 [0.265, 0.896] (medium)
- Recommended test: conservative_max (p = 0.0085)
- Group 1: mean=17.8 (n=75), Group 2: mean=17.3 (n=75)

### Emotional divergence
- Cohen's d = -0.123 [-0.429, 0.212] (negligible)
- Recommended test: conservative_max (p = 0.7435)
- Group 1: mean=8447.4 (n=75), Group 2: mean=8472.9 (n=75)

### Emotional divergence (eff. rank)
- Cohen's d = -0.897 [-1.248, -0.583] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=16.6 (n=75), Group 2: mean=17.3 (n=75)

### Reasoning mode
- Cohen's d = -0.258 [-0.625, 0.055] (small)
- Recommended test: conservative_max (p = 0.1163)
- Group 1: mean=8388.7 (n=75), Group 2: mean=8472.9 (n=75)

### Reasoning mode (eff. rank)
- Cohen's d = -1.915 [-2.351, -1.571] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=15.8 (n=75), Group 2: mean=17.3 (n=75)

### Code mode
- Cohen's d = 1.122 [0.669, 1.856] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=9114.9 (n=75), Group 2: mean=8472.9 (n=75)

### Code mode (eff. rank)
- Cohen's d = 0.488 [0.150, 0.932] (small)
- Recommended test: conservative_max (p = 0.0035)
- Group 1: mean=18.1 (n=75), Group 2: mean=17.3 (n=75)

### Disambiguation demand
- Cohen's d = -2.342 [-2.865, -1.942] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=7822.6 (n=75), Group 2: mean=8279.4 (n=75)

### Disambiguation demand (eff. rank)
- Cohen's d = -1.095 [-1.330, -0.873] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=14.8 (n=75), Group 2: mean=16.4 (n=75)


## Multiple Comparison Correction

Method: holm_bonferroni (9 comparisons)

- confab_vs_facts: p_orig=0.8357, p_corr=1.0000 (not significant)
- self_ref_effect: p_orig=0.0542, p_corr=0.2708 (not significant)
- refusal_vs_rote: p_orig=0.0003, p_corr=0.0018 (SIGNIFICANT)
- refusal_vs_free: p_orig=0.0000, p_corr=0.0000 (SIGNIFICANT)
- creative_vs_facts: p_orig=0.8314, p_corr=1.0000 (not significant)
- emotion_vs_facts: p_orig=0.7435, p_corr=1.0000 (not significant)
- math_vs_facts: p_orig=0.1163, p_corr=0.4653 (not significant)
- code_vs_facts: p_orig=0.0000, p_corr=0.0000 (SIGNIFICANT)
- ambiguity_effect: p_orig=0.0000, p_corr=0.0000 (SIGNIFICANT)

---

## Hypothesis Evaluation

### H1_confabulation_inversion
- cohens_d: -0.03394866234967584
- interpretation: negligible
- ci_lower: -0.37659500276972513
- ci_upper: 0.2802555470278083
- p_value: 0.8356515845998139
- direction: facts_higher
- note: Positive d = confabulation has higher norms (expected at small scale). Negative d = confabulation has lower norms (inversion at large scale).

### H2_self_reference_emergence
- cohens_d: -0.3471634493308487
- interpretation: small
- ci_lower: -0.6658258938816555
- ci_upper: -0.038160374288383474
- p_value: 0.05416734485119756
- emergence_detected: True
- note: Effect > 0.3 suggests self-modeling capacity at this scale.

### H3_refusal_specialization
- cohens_d: 0.6117285823312509
- interpretation: medium
- ci_lower: 0.2879051252902527
- ci_upper: 0.976559436339984
- p_value: 0.00029194343847457297
- specialized: True
- note: d > 0.5 = refusal has unique signature beyond low entropy. d < 0.3 = refusal is just another memorized pattern.

### H4_category_invariance
- note: Evaluated in cross-scale comparison (requires multiple scale results).


## Power Analysis

- d_0.3: power=0.451 (n=75) [UNDERPOWERED]
- d_0.5: power=0.865 (n=75) [ADEQUATE]
- d_0.8: power=0.998 (n=75) [ADEQUATE]