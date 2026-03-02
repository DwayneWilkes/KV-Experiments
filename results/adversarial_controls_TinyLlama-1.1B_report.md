# Phase 1.75: Adversarial Controls — Results Report

**Generated**: 2026-03-01T19:36:19.875550
**Model**: TinyLlama/TinyLlama-1.1B-Chat-v1.0
**Runs**: 5
**Seed**: 42

---

## Control 3: Precision Sweep (GATE CHECK)

**Verdict**: SIGNAL SURVIVES QUANTIZATION: Pearson r=0.851, Spearman ρ=0.925. Cache statistics are robust to precision. Phase 1.5 findings are not quantization artifacts.
  Key finding (confab > grounded) HOLDS in FP16: d=1.79 [1.34,2.39].

- Pearson r: 0.8514
- Spearman ρ: 0.9247
- Bland-Altman mean diff: 9.99
- Confab vs Grounded 4-bit: d=0.948
- Confab vs Grounded FP16:  d=1.786

---

## Control 1: Frequency/Truth Matrix

**Verdict**: FREQUENCY ARTIFACT: Only token rarity drives the signal (freq d=0.71, truth d=-0.08). Phase 1.5 confabulation finding is a rare-word counter.

- **common_true**: 4832.0 ± 246.5 [4769.8, 4877.5]
- **common_false**: 4805.2 ± 401.0 [4700.5, 4879.2]
- **rare_true**: 5125.6 ± 527.6 [4995.4, 5231.5]
- **rare_false**: 5085.7 ± 595.8 [4938.3, 5208.1]

- truth_effect_common: d=-0.081 [-0.33, 0.28]
- truth_effect_rare: d=-0.071 [-0.39, 0.25]
- freq_effect_true: d=0.713 [0.30, 1.67]
- freq_effect_false: d=0.552 [0.19, 1.14]

---

## Control 2: Rote Memorization Test

**Verdict**: GUARDRAIL SIGNATURE DETECTED: Refusal is distinct from rote, formulaic (d > 0.5). RLHF guardrails produce measurable cache differences beyond mere low entropy.

- **refusal**: norm=4686.3 [4441.3, 4910.8]
- **rote_completion**: norm=5466.7 [5360.6, 5568.6]
- **code_boilerplate**: norm=5042.8 [4926.8, 5141.3]
- **formulaic_response**: norm=5112.2 [5001.1, 5180.9]
- **creative_open**: norm=5111.8 [5014.4, 5179.7]

---

## Control 4: Semantic Gradient Transfer

**Verdict**: NO TRANSFER: Raw cache injection fails at all distances. Confirms Phase 2a negative result.

- **near**: 0% (0/25)
- **medium_near**: 0% (0/25)
- **medium**: 0% (0/25)
- **medium_far**: 0% (0/25)
- **far**: 0% (0/25)
- Gradient τ: nan

---

## Control 5: Length Confound

**Verdict**: LENGTH CONFOUND CONFIRMED: Length drives the signal (d=2.00), truth effect vanishes within length groups.

- Truth effect (short): d=0.123
- Truth effect (long): d=0.334
- Length effect: d=2.005

---

## Control 6: Template Structure

**Verdict**: NO EFFECT WITH CONTROLLED SYNTAX: d=-0.12. The original signal may have been driven by syntactic differences between true and false prompts.

- Template False vs True: d=-0.124 [-0.44, 0.20]
- Paired t-test: p=0.4261 (14 pairs)

---

## Summary

| Control | Result | Implication |
|---------|--------|-------------|
| Control 3 | SIGNAL SURVIVES QUANTIZATION | SIGNAL SURVIVES QUANTIZATION: Pearson r=0.851, Spearman ρ=0.925. Cache statistic... |
| Control 1 | FREQUENCY ARTIFACT | FREQUENCY ARTIFACT: Only token rarity drives the signal (freq d=0.71, truth d=-0... |
| Control 2 | GUARDRAIL SIGNATURE DETECTED | GUARDRAIL SIGNATURE DETECTED: Refusal is distinct from rote, formulaic (d > 0.5)... |
| Control 4 | NO TRANSFER | NO TRANSFER: Raw cache injection fails at all distances. Confirms Phase 2a negat... |
| Control 5 | LENGTH CONFOUND CONFIRMED | LENGTH CONFOUND CONFIRMED: Length drives the signal (d=2.00), truth effect vanis... |
| Control 6 | NO EFFECT WITH CONTROLLED SYNTAX | NO EFFECT WITH CONTROLLED SYNTAX: d=-0.12. The original signal may have been dri... |
