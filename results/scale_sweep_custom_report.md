# Scale Sweep Report: custom

**Model**: /home/thomas/KV-Experiments/results/abliterated_Qwen2.5-7B
**Quantized**: False
**Generated**: 2026-03-02T09:10:40.115364

---

## Category Summaries

| Category | N | Mean Norm | Std | Mean/Token | Eff. Rank | Spectral H | 95% CI |
|----------|---|-----------|-----|------------|-----------|------------|--------|
| coding               |  75 |  20166.3 |  761.7 |   858.8 |      25.3 |     0.5151 | [20000.7, 20339.7] |
| creative             |  75 |  18350.1 |  255.4 |  1258.5 |      24.3 |     0.5087 | [18289.9, 18407.1] |
| confabulation        |  75 |  18277.9 |  825.7 |  1298.9 |      23.9 |     0.5069 | [18095.8, 18474.5] |
| grounded_facts       |  75 |  18258.4 |  519.3 |  1294.7 |      23.2 |     0.5021 | [18146.2, 18379.2] |
| emotional            |  75 |  18127.8 |  336.6 |  1352.4 |      23.0 |     0.4983 | [18052.3, 18202.2] |
| math_reasoning       |  75 |  17995.9 |  824.5 |  1448.9 |      22.2 |     0.4962 | [17814.2, 18186.9] |
| non_self_reference   |  75 |  17832.9 |  421.6 |  1448.9 |      23.6 |     0.5050 | [17742.2, 17932.0] |
| unambiguous          |  75 |  17784.5 |  258.8 |  1492.4 |      23.1 |     0.5022 | [17725.2, 17843.5] |
| self_reference       |  75 |  17770.1 |  226.1 |  1442.1 |      24.4 |     0.5101 | [17717.8, 17820.3] |
| guardrail_test       |  75 |  17553.5 |  365.0 |  1558.0 |      22.6 |     0.5023 | [17472.8, 17637.4] |
| rote_completion      |  75 |  17145.8 |  739.1 |  2186.2 |      21.2 |     0.4900 | [16983.7, 17310.8] |
| ambiguous            |  75 |  16793.1 |  486.1 |  2359.8 |      21.6 |     0.4948 | [16685.2, 16904.2] |
| free_generation      |  75 |  16791.6 |  359.0 |  2218.9 |      22.0 |     0.4985 | [16712.5, 16874.3] |

---

## Key Comparisons

### H1: Confabulation effect
- Cohen's d = 0.028 [-0.307, 0.345] (negligible)
- Recommended test: conservative_max (p = 0.8628)
- Group 1: mean=18277.9 (n=75), Group 2: mean=18258.4 (n=75)

### H1: Confabulation effect (eff. rank)
- Cohen's d = 0.539 [0.241, 0.839] (medium)
- Recommended test: conservative_max (p = 0.0240)
- Group 1: mean=23.9 (n=75), Group 2: mean=23.2 (n=75)

### H2: Self-reference effect
- Cohen's d = -0.186 [-0.483, 0.139] (negligible)
- Recommended test: conservative_max (p = 0.9640)
- Group 1: mean=17770.1 (n=75), Group 2: mean=17832.9 (n=75)

### H2: Self-reference effect (eff. rank)
- Cohen's d = 0.513 [0.192, 0.848] (medium)
- Recommended test: conservative_max (p = 0.0097)
- Group 1: mean=24.4 (n=75), Group 2: mean=23.6 (n=75)

### H3: Refusal specificity
- Cohen's d = 0.699 [0.363, 1.087] (medium)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=17553.5 (n=75), Group 2: mean=17145.8 (n=75)

### H3: Refusal specificity (eff. rank)
- Cohen's d = 1.030 [0.750, 1.340] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=22.6 (n=75), Group 2: mean=21.2 (n=75)

### H3b: Refusal vs high-entropy
- Cohen's d = 2.105 [1.715, 2.601] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=17553.5 (n=75), Group 2: mean=16791.6 (n=75)

### H3b: Refusal vs high-entropy (eff. rank)
- Cohen's d = 0.776 [0.486, 1.088] (medium)
- Recommended test: conservative_max (p = 0.0001)
- Group 1: mean=22.6 (n=75), Group 2: mean=22.0 (n=75)

### Creative divergence
- Cohen's d = 0.224 [-0.089, 0.612] (small)
- Recommended test: conservative_max (p = 0.1731)
- Group 1: mean=18350.1 (n=75), Group 2: mean=18258.4 (n=75)

### Creative divergence (eff. rank)
- Cohen's d = 0.905 [0.569, 1.303] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=24.3 (n=75), Group 2: mean=23.2 (n=75)

### Emotional divergence
- Cohen's d = -0.298 [-0.598, 0.015] (small)
- Recommended test: conservative_max (p = 0.5424)
- Group 1: mean=18127.8 (n=75), Group 2: mean=18258.4 (n=75)

### Emotional divergence (eff. rank)
- Cohen's d = -0.215 [-0.550, 0.099] (small)
- Recommended test: conservative_max (p = 0.2404)
- Group 1: mean=23.0 (n=75), Group 2: mean=23.2 (n=75)

### Reasoning mode
- Cohen's d = -0.381 [-0.763, -0.062] (small)
- Recommended test: conservative_max (p = 0.0212)
- Group 1: mean=17995.9 (n=75), Group 2: mean=18258.4 (n=75)

### Reasoning mode (eff. rank)
- Cohen's d = -0.946 [-1.342, -0.618] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=22.2 (n=75), Group 2: mean=23.2 (n=75)

### Code mode
- Cohen's d = 2.927 [2.533, 3.455] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=20166.3 (n=75), Group 2: mean=18258.4 (n=75)

### Code mode (eff. rank)
- Cohen's d = 1.685 [1.291, 2.201] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=25.3 (n=75), Group 2: mean=23.2 (n=75)

### Disambiguation demand
- Cohen's d = -2.546 [-3.019, -2.194] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=16793.1 (n=75), Group 2: mean=17784.5 (n=75)

### Disambiguation demand (eff. rank)
- Cohen's d = -1.293 [-1.631, -0.999] (large)
- Recommended test: conservative_max (p = 0.0000)
- Group 1: mean=21.6 (n=75), Group 2: mean=23.1 (n=75)


## Multiple Comparison Correction

Method: holm_bonferroni (9 comparisons)

- confab_vs_facts: p_orig=0.8628, p_corr=1.0000 (not significant)
- self_ref_effect: p_orig=0.9640, p_corr=1.0000 (not significant)
- refusal_vs_rote: p_orig=0.0000, p_corr=0.0002 (SIGNIFICANT)
- refusal_vs_free: p_orig=0.0000, p_corr=0.0000 (SIGNIFICANT)
- creative_vs_facts: p_orig=0.1731, p_corr=0.6925 (not significant)
- emotion_vs_facts: p_orig=0.5424, p_corr=1.0000 (not significant)
- math_vs_facts: p_orig=0.0212, p_corr=0.1061 (not significant)
- code_vs_facts: p_orig=0.0000, p_corr=0.0000 (SIGNIFICANT)
- ambiguity_effect: p_orig=0.0000, p_corr=0.0000 (SIGNIFICANT)

---

## Hypothesis Evaluation

### H1_confabulation_inversion
- cohens_d: 0.028286027569505802
- interpretation: negligible
- ci_lower: -0.3065990189498182
- ci_upper: 0.34492454976242104
- p_value: 0.8627628034187348
- direction: confab_higher
- note: Positive d = confabulation has higher norms (expected at small scale). Negative d = confabulation has lower norms (inversion at large scale).

### H2_self_reference_emergence
- cohens_d: -0.18564922058246805
- interpretation: negligible
- ci_lower: -0.4829909366887729
- ci_upper: 0.13860689653477815
- p_value: 0.9640046370958462
- emergence_detected: False
- note: Effect > 0.3 suggests self-modeling capacity at this scale.

### H3_refusal_specialization
- cohens_d: 0.6994073333398853
- interpretation: medium
- ci_lower: 0.36281915533101267
- ci_upper: 1.0868106219335212
- p_value: 4.0108294895261344e-05
- specialized: True
- note: d > 0.5 = refusal has unique signature beyond low entropy. d < 0.3 = refusal is just another memorized pattern.

### H4_category_invariance
- note: Evaluated in cross-scale comparison (requires multiple scale results).


## Power Analysis

- d_0.3: power=0.451 (n=75) [UNDERPOWERED]
- d_0.5: power=0.865 (n=75) [ADEQUATE]
- d_0.8: power=0.998 (n=75) [ADEQUATE]