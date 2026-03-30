# V13: Matched-Scale Transfer Control

**Status**: REGISTERED
**Design commit**: {to be filled at commit time}
**Result commit**: pending

## Hypothesis

**Claim under test**: "Cross-model transfer at AUROC ~0.86 reflects shared suppression geometry" (Paper C3, Section 4.3, Exp 49c)

**Finding**: F03 showed cross-model transfer is at chance (mean AUROC 0.52, not 0.86). F04 showed same-model transfer also at chance (0.53). The paper's high cross-model AUROC may reflect model-scale artifacts: training on small models (1.1B-9B) and testing on larger models (14B) where cache geometry differs systematically by scale, not by suppression.

**Null hypothesis (H0)**: Within-scale transfer (7B -> 7B, different architectures) achieves AUROC > 0.70, confirming that shared geometry exists at matched scale.

**Alternative (H1)**: Within-scale transfer is at chance (< 0.60), confirming that the cross-model signal was a scale artifact.

## Methods

**Statistical tests**: Train LogReg on model A's refusal features, test on model B's refusal features (same scale, different architecture). LOO AUROC for within-model baseline. Train-test AUROC for cross-model transfer. FWL residualization against input length.

Model pairs (matched ~7B scale):
- Qwen2.5-7B-Instruct -> Llama-3.1-8B-Instruct (7B vs 8B)
- Llama-3.1-8B-Instruct -> Mistral-7B-Instruct-v0.3 (8B vs 7B)
- Mistral-7B-Instruct-v0.3 -> Qwen2.5-7B-Instruct (7B vs 7B)

All 6 directional pairs. Focus on refusal (survived F01b input control).

Reference: methods.md (train-test AUROC, FWL within-fold)

**Input data**: 49c_cross_model_suppression.json — per-model features for refusal task. Same data as F03 but analyzed for scale-matching specifically.

**Sample sizes**: Per-model N varies (10-20 per condition). LOO appropriate.

**Confound controls**: FWL against input length (same as F03). Focus on refusal only (survived F01b).

**Multiple comparison correction**: Holm-Bonferroni across 6 model pairs.

## Pre-Registered Pass/Fail Criteria

- If mean matched-scale transfer AUROC > 0.70 after FWL: **CONFIRMED** — shared geometry at matched scale
- If mean matched-scale transfer AUROC < 0.60 after FWL: **FALSIFIED** — no transfer even at matched scale
- If 0.60-0.70: **WEAKENED** — weak signal, ambiguous

Comparison with F03 (all-scale mean 0.52):
- If matched-scale > all-scale + 0.10: evidence for scale-artifact hypothesis
- If matched-scale similar to all-scale: geometry is uniformly absent, not scale-dependent

## Execution

**GPU required**: No (uses stored 49c features)
**Estimated time**: < 5 minutes
**Code**: `kv_verify/experiments/v13_matched_scale.py`
**Tests**: `kv_verify/tests/test_v13.py`

## Findings

{Pending execution}

## Result Commit

pending
