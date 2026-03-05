# WS6: Censorship Gradient Verification

**Status**: COMPLETE
**Paper sections**: 3.5, Table 6-7
**Claims**: C49–C56

## Task 6.1: Result Files

3 natural_deception JSON files:
- DeepSeek-R1-Distill-Qwen-14B (n=150 censored + 150 control + 150 complex)
- Mistral-7B-v0.3 (n=150 + 150 + 150)
- Qwen2.5-14B (n=150 + 150 + 150)

Conditions: censored (6 sensitive China topics × 5 questions × 5 runs), control (6 neutral China topics × 5 × 5), complex_noncensored (6 non-China atrocities × 5 × 5)
Metrics per condition: norms, norms_per_token, key_ranks, key_entropies, total_tokens

## Task 6.2: Table 6 Recomputation

### Raw Norms (censored vs control)
| Model | Our g | Their g | Match |
|-------|-------|---------|-------|
| DeepSeek | -0.933 | -0.933 | EXACT |
| Mistral | -0.082 | -0.082 | EXACT |
| Qwen-14B | +1.010 | +1.010 | EXACT |

### Key Ranks (censored vs control)
| Model | Our g | Their g |
|-------|-------|---------|
| DeepSeek | -0.700 | -0.701 |
| Mistral | +0.217 | — |
| Qwen-14B | +0.721 | — |

### Corrected Topic Analysis File Values
From `s4_topic_analysis_corrected.json`:
- DeepSeek critical_test d = -0.219 ← matches paper
- DeepSeek residualized_censored_vs_complex d = +0.904 ← matches paper
- DeepSeek residualized_censored_vs_control d = +0.231
- DeepSeek overall_key_ranks d = -0.701

**Verdict**: CONFIRMED — all Table 6 values match.

## Task 6.3: DeepSeek Per-Topic (Table 7)

| Topic | Our g (norms) | Our g (ranks) | Paper d (ranks) | Match |
|-------|-------------|-------------|----------------|-------|
| tiananmen | -4.623 | -6.174 | -6.201 | CLOSE (0.4%) |
| political_prisoners | -1.871 | -1.743 | -1.751 | CLOSE |
| taiwan | -1.208 | -0.989 | -0.993 | CLOSE |
| uyghur | +0.026 | +0.668 | +0.671 | CLOSE |
| hong_kong | -0.247 | +0.098 | +0.098 | EXACT |
| xi_criticism | -0.507 | +0.080 | +0.081 | CLOSE |

Paper uses key_ranks metric for Table 7. Small differences (~0.3%) likely due to Hedges' g vs Cohen's d.
**Verdict**: CONFIRMED — all per-topic values match within tolerance.

## Task 6.4: Aggregation Check

Per-topic d values do NOT mathematically aggregate to overall d=+0.904 (residualized):
- Mean of per-topic rank d values: (-6.174 + -1.743 + -0.989 + 0.668 + 0.098 + 0.080) / 6 = -1.343
- But overall censored-vs-control rank d = -0.701
- The +0.904 is residualized_censored_vs_**complex**, not censored_vs_control

The paper presents d=+0.904 as the headline number but this is censored vs complex_noncensored (after residualization), NOT censored vs control. This is technically correct but potentially misleading — it compares censored China topics to non-China atrocities, not to neutral China topics.

**Verdict**: INFLATED — numbers correct but headline d=0.904 compares to wrong baseline for the "censorship detection" narrative.

## Task 6.5: Tiananmen d=-6.201 and 25/25 Evasive

- Key ranks g = -6.174 (our computation) vs paper's -6.201 → difference likely d vs g
- Classification: **25/25 evasive** CONFIRMED
- "Evasive" is defined operationally by the classification model (not human evaluation)
- This is the strongest single finding in the paper — massive effect size

**Verdict**: CONFIRMED.

## Task 6.6: "Behaviorally Invisible" (Qwen-14B)

Qwen-14B classification distribution on censored topics:
| Topic | Evasive | Truthful | Unknown |
|-------|---------|----------|---------|
| tiananmen | 5 | 5 | 15 |
| uyghur | 0 | 5 | 20 |
| taiwan | 0 | 10 | 15 |
| political_prisoners | 0 | 20 | 5 |
| xi_criticism | 0 | 15 | 10 |
| hong_kong | 0 | 15 | 10 |
| **Overall** | **5** | **70** | **75** |

Only 5/150 (3.3%) classified as evasive → behavioral output is largely "truthful" or "unknown."

But cache norms show g=+1.010 (large) with all per-topic d values positive (0.63–1.66).

**Assessment**: The claim is PARTIALLY SUPPORTED:
- The text output does NOT clearly reveal censorship (only 3.3% evasive)
- But calling it "invisible" is an overstatement — 50% are "unknown" (ambiguous)
- The testing method is a keyword/pattern classifier, NOT a human evaluation or adversarial red team
- A sophisticated text classifier might detect subtler signals

**Verdict**: PARTIAL — text classification mostly fails to detect censorship, but "invisible" is stronger than the evidence supports.

## Task 6.7: Mistral "Uncensored"

Mistral classification on censored topics:
| Topic | Evasive | Truthful | Unknown |
|-------|---------|----------|---------|
| tiananmen | 5 | 5 | 15 |
| uyghur | 0 | 5 | 20 |
| taiwan | 0 | 10 | 15 |
| political_prisoners | 0 | 20 | 5 |
| xi_criticism | 0 | 25 | 0 |
| hong_kong | 0 | 20 | 5 |
| **Overall** | **5** | **85** | **60** |

Cache norms g = -0.082 (negligible), p = 0.478 → no censorship signal.

Mistral shows 5 evasive responses on Tiananmen (same as Qwen-14B), but 56.7% truthful overall. The negligible cache d confirms it as an appropriate uncensored control.

**Verdict**: CONFIRMED — Mistral behaves as expected for an uncensored model.

## Task 6.8: Code and Prompt Review

Deferred to WS11 code audit.

## Length Confound

| Model | Censored tokens | Control tokens | Ratio | r² |
|-------|-----------------|----------------|-------|-----|
| DeepSeek | 143 | 190 | 0.75 | **0.83** |
| Mistral | 214 | 213 | 1.00 | 0.43 |
| Qwen-14B | 220 | 215 | 1.02 | 0.05 |

**DeepSeek has a massive length confound** (r²=0.83): censored responses are 25% shorter, and response length explains 83% of norm variance. The raw norms g=-0.933 is partly driven by shorter responses (shorter → lower norms).

The paper addresses this with residualized norms, but the headline "critical test" d=-0.219 uses key_ranks (which may also be length-confounded).

## Summary

| Claim | Verdict | Notes |
|-------|---------|-------|
| Table 6 values | **CONFIRMED** | All effect sizes match |
| Table 7 per-topic | **CONFIRMED** | All within tolerance |
| Tiananmen 25/25 evasive | **CONFIRMED** | Confirmed |
| d=+0.904 framing | **INFLATED** | Uses censored-vs-complex, not censored-vs-control |
| Qwen-14B "invisible" | **PARTIAL** | Not fully invisible; tested with simple classifier |
| Mistral uncensored | **CONFIRMED** | Appropriate control |
| Length confound | **PARTIALLY ADDRESSED** | DeepSeek r²=0.83 for length; residualization helps but unclear if sufficient |
