# WS4: Deception Forensics Verification

**Status**: COMPLETE
**Paper sections**: 3.3-3.4
**Claims**: C36-C41 (deception detection, honest vs deceptive separability, expansion vs compression)

## Task 4.1: Result Files Loaded

7 deception_forensics JSON files, all well-formed:

| Model | n (per condition) | Architecture |
| ------- | ------------------- | ------------- |
| Llama-3.1-8B | 75 | Llama |
| Mistral-7B-v0.3 | 75 | Mistral |
| Qwen2.5-7B | 75 | Qwen |
| Qwen2.5-32B-q4 | 45 | Qwen |
| TinyLlama-1.1B | 75 | Llama |
| gemma-2-2b-it | 75 | Gemma |
| gemma-2-9b-it | 75 | Gemma |

**Conditions** (experiment_1): honest, deceptive, confabulation
**Metrics**: norms, norms_per_token, key_ranks, key_entropies

## Task 4.2: Hedges' g Recomputation

| Model | Our g (norms) | Their g | Match | Our g (per_token) | Our g (key_ranks) |
| ------- | ------------- | --------- | ------- | ------------------- | ------------------- |
| Llama-3.1-8B | -2.9951 | -2.9951 | EXACT | +3.6371 | -1.9260 |
| Mistral-7B-v0.3 | -2.9139 | -2.9139 | EXACT | +3.3499 | -0.6186 |
| Qwen2.5-32B-q4 | -3.0388 | N/A* | — | +3.5593 | -2.4209 |
| Qwen2.5-7B | -0.6467 | N/A* | — | +2.2558 | -0.8446 |
| TinyLlama-1.1B | -0.7538 | N/A* | — | +0.4251 | -1.0664 |
| gemma-2-2b-it | -1.3055 | -1.3055 | EXACT | +1.2249 | -1.5171 |
| gemma-2-9b-it | -0.6661 | -0.6661 | EXACT | +1.6341 | -0.9628 |

*3 models don't have pre-computed g in their analysis section — computed independently.

**Sign convention**: g = hedges_g(honest, deceptive). Negative g → deceptive > honest → "expansion."
**Verdict**: CONFIRMED for 4 models with stored values (exact match to 4 decimal places).

## Task 4.3: Expansion vs Compression — REJECTED

**Paper claim**: Llama-8B and Mistral-7B show expansion. Gemma-2B and Gemma-9B show compression.

**Finding**: **ALL 7 models show expansion** on raw norms (all g negative).

On effective rank (H6 verdict `d_rank_honest_vs_deceptive`):

| Model | d (eff. rank) | Code label |
| ------- | ------------- | ------------ |
| Llama-3.1-8B | -1.936 | deception_expands: True |
| Mistral-7B-v0.3 | -0.622 | deception_expands: True |
| Qwen2.5-7B | -0.849 | **BUG: note says "Positive d"** |
| Qwen2.5-32B-q4 | -2.442 | **BUG: note says "Positive d"** |
| TinyLlama-1.1B | -1.072 | **BUG: note says "Positive d"** |
| gemma-2-2b-it | -1.525 | deception_expands: True |
| gemma-2-9b-it | -0.968 | deception_expands: True |

ALL models have negative d (deception EXPANDS dimensionality). No compression found in any model on any metric.

### CODE BUG: H6 Verdict Note Text

For Qwen-7B, Qwen-32B, TinyLlama:

- `d_rank_honest_vs_deceptive` is **negative** (e.g., -0.849)
- But note text says "**Positive d** = honest has HIGHER effective rank"
- This is factually wrong — the d is negative, meaning deceptive has higher rank
- These 3 models also lack `deception_expands: True` key that the other 4 correctly have

**Verdict**: REJECTED — paper's expansion/compression split is not supported by the data.

## Task 4.4: Universal Separability

| Model | \|g\| (norms) | Cohen's scale | Verdict |
| ------- | ----------- | ------------- | --------- |
| Llama-3.1-8B | 2.995 | very large | OK |
| Mistral-7B-v0.3 | 2.914 | very large | OK |
| Qwen2.5-32B-q4 | 3.039 | very large | OK |
| gemma-2-2b-it | 1.306 | large | OK |
| TinyLlama-1.1B | 0.754 | medium | OK |
| gemma-2-9b-it | 0.666 | medium | OK |
| Qwen2.5-7B | 0.647 | medium | OK |

No model has |g| < 0.2. All at least medium (>0.5).
**Verdict**: CONFIRMED — all models separable. But magnitude varies 4.7× (0.647 to 3.039).

## Task 4.5: Length Confound — SIGNIFICANT

Deceptive responses are systematically ~1.6-1.7× longer than honest responses:

| Model | Honest tokens (implied) | Deceptive tokens | Ratio |
| ------- | ------------------------ | ------------------- | ------- |
| Llama-3.1-8B | 12.5 | 20.5 | 1.64× |
| gemma-2-2b-it | 12.3 | 20.8 | 1.69× |
| gemma-2-9b-it | 12.3 | 20.9 | 1.69× |

Impact:

- Raw norm "expansion" is **at least partly driven by response length**, not cognitive state
- Per-token norms **flip direction** (honest > deceptive), partially controlling for length
- Paper includes per-token analysis but headlines raw norms
- Proper control would residualize norms against token count

**Verdict**: PARTIAL — raw norm expansion claim confounded by response length.

## Task 4.6: Code Review

Deferred to WS11 code audit.

## Claim Verdict Summary

| Claim | Description | Verdict |
| ------- | ------------- | --------- |
| C36 | Honest vs deceptive separable | **CONFIRMED** (all \|g\| > 0.5) |
| C37 | Expansion in Llama/Mistral | **CONFIRMED** (norms direction correct) |
| C38 | Compression in Gemma | **REJECTED** (both Gemma models show expansion, not compression) |
| C39 | Universal separability | **PARTIAL** (separable yes, but 4.7× magnitude range) |
| C40-41 | Direction architecture-dependent | **REJECTED** (all models same direction) |
| — | Length confound addressed | **PARTIAL** (not adequately addressed) |
| — | H6 verdict code correctness | **CODE BUG** (3 models wrong note text) |
