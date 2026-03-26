# V07: Sycophancy Length Confound

**Status**: REGISTERED
**Design commit**: (this commit)

## Hypothesis

**Claim under test**: "KV-cache geometry detects sycophancy" (Paper C3)

**Finding**: M2. Length-only AUROC (0.943) EXCEEDS feature AUROC (0.933).
FWL-both drops AUROC to 0.4425 (below chance). R^2 values show 91-98% of
feature variance explained by length.

**Null hypothesis (H0)**: Cache geometry features outperform length-only
features for sycophancy detection.

**Alternative (H1)**: A length-only classifier matches or exceeds the cache
feature classifier, proving the sycophancy signal is entirely length-driven.

## Methods

**Statistical tests**:
- GroupKFold AUROC with corrected groups (see [methods.md](methods.md#groupkfold))
- Three classifiers compared:
  1. Features only: norm_per_token, key_rank, key_entropy
  2. Length only: norm, n_generated
  3. FWL-both: features residualized against both norm and n_generated
- Within-fold FWL residualization (see [methods.md](methods.md#frisch-waugh-lovell-theorem))

**Input data**: Per-item features from `same_prompt_sycophancy.json` (exp39).

**Sample sizes**: N=20 sycophantic, N=20 direct responses.

## Pre-Registered Pass/Fail Criteria

- If length-only AUROC >= feature-only AUROC: sycophancy claim **FALSIFIED**
- If FWL-both AUROC < 0.55: all signal is length, sycophancy claim **FALSIFIED**
- If feature-only AUROC > length-only AUROC by 0.05+ AND FWL-both > 0.60:
  sycophancy claim **CONFIRMED**

## Execution

**GPU required**: No. CPU only, uses pre-computed features.
**Code**: `kv_verify/experiments/v07_sycophancy.py`
**Tests**: `kv_verify/tests/test_v07.py`
