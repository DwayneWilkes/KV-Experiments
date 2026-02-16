# Phase 1.75: Adversarial Controls — Results Report

**Generated**: 2026-02-15T07:39:57.997852
**Model**: TinyLlama/TinyLlama-1.1B-Chat-v1.0
**Runs**: 5
**Seed**: 42

---

## Control 3: Precision Sweep (GATE CHECK)

**Verdict**: SIGNAL SURVIVES QUANTIZATION: Pearson r=0.853, Spearman ρ=0.899. Cache statistics are robust to precision. Phase 1.5 findings are not quantization artifacts.
  Key finding (confab > grounded) HOLDS in FP16: d=1.83 [1.38,2.44].

- Pearson r: 0.8526
- Spearman ρ: 0.8988
- Bland-Altman mean diff: 9.76
- Confab vs Grounded 4-bit: d=0.948
- Confab vs Grounded FP16:  d=1.832

---

## Control 1: Frequency/Truth Matrix

**Verdict**: FREQUENCY ARTIFACT: Only token rarity drives the signal (freq d=0.71, truth d=-0.08). Phase 1.5 confabulation finding is a rare-word counter.

- **common_true**: 4831.6 ± 246.2 [4770.0, 4876.6]
- **common_false**: 4805.2 ± 401.0 [4700.4, 4879.1]
- **rare_true**: 5124.9 ± 527.0 [4995.1, 5230.9]
- **rare_false**: 5085.7 ± 595.9 [4938.3, 5208.1]

- truth_effect_common: d=-0.079 [-0.33, 0.29]
- truth_effect_rare: d=-0.070 [-0.39, 0.25]
- freq_effect_true: d=0.713 [0.30, 1.67]
- freq_effect_false: d=0.552 [0.19, 1.14]

---

## Control 2: Rote Memorization Test

**Verdict**: GUARDRAIL SIGNATURE DETECTED: Refusal is distinct from rote, formulaic (d > 0.5). RLHF guardrails produce measurable cache differences beyond mere low entropy.

- **refusal**: norm=4686.3 [4441.2, 4910.8]
- **rote_completion**: norm=5466.5 [5360.2, 5568.1]
- **code_boilerplate**: norm=5044.6 [4928.1, 5143.3]
- **formulaic_response**: norm=5112.1 [5001.3, 5180.4]
- **creative_open**: norm=5108.6 [5011.5, 5176.7]

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

- Truth effect (short): d=0.104
- Truth effect (long): d=0.334
- Length effect: d=1.998

---

## Control 6: Template Structure

**Verdict**: NO EFFECT WITH CONTROLLED SYNTAX: d=0.03. The original signal may have been driven by syntactic differences between true and false prompts.

- Template False vs True: d=0.029 [-0.29, 0.36]
- Paired t-test: p=0.5470 (14 pairs)

---

## Summary

| Control | Result | Implication |
|---------|--------|-------------|
| Control 3 | SIGNAL SURVIVES QUANTIZATION | SIGNAL SURVIVES QUANTIZATION: Pearson r=0.853, Spearman ρ=0.899. Cache statistic... |
| Control 1 | FREQUENCY ARTIFACT | FREQUENCY ARTIFACT: Only token rarity drives the signal (freq d=0.71, truth d=-0... |
| Control 2 | GUARDRAIL SIGNATURE DETECTED | GUARDRAIL SIGNATURE DETECTED: Refusal is distinct from rote, formulaic (d > 0.5)... |
| Control 4 | NO TRANSFER | NO TRANSFER: Raw cache injection fails at all distances. Confirms Phase 2a negat... |
| Control 5 | LENGTH CONFOUND CONFIRMED | LENGTH CONFOUND CONFIRMED: Length drives the signal (d=2.00), truth effect vanis... |
| Control 6 | NO EFFECT WITH CONTROLLED SYNTAX | NO EFFECT WITH CONTROLLED SYNTAX: d=0.03. The original signal may have been driv... |
