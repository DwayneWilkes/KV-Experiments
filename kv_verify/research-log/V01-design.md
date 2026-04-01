# V01: GroupKFold Bug Detection

**Status**: REGISTERED
**Design commit**: (this commit)

## Hypothesis

**Claim under test**: "GroupKFold prevents prompt leakage across CV folds" (Paper C3)

**Finding**: C2. `groups = np.array(list(range(n_pos)) + list(range(n_neg)))` at
`49_expanded_validation.py:773` assigns overlapping group IDs across classes.
For non-paired experiments (31, 32, 33, 36), unrelated prompts share group IDs.
For same-prompt experiments (18b, 39), the overlap is accidentally correct.

**Null hypothesis (H0)**: Correcting group assignment does not change any AUROC
by more than 0.02 or flip any significance result.

**Alternative (H1)**: At least one comparison changes AUROC by more than 0.05
or flips significance after correction.

## Methods

**Statistical tests**:
- GroupKFold AUROC with corrected group assignment (see [methods.md](methods.md#groupkfold))
- Group-level permutation test, 10K iterations (see [methods.md](methods.md#group-level-permutation))
- Holm-Bonferroni correction on 10-test family (see [methods.md](methods.md#holm-bonferroni-step-down-procedure))
- BCa bootstrap CIs, 10K resamples (see [methods.md](methods.md#bca-bias-corrected-and-accelerated))

**Input data**: Per-item features from 6 hackathon result JSONs.
10 comparisons total across exp31, exp32, exp33, exp36, exp18b, exp39.

**Sample sizes**: N=20 per group for most comparisons. N=10 per group for exp18b.

**Group assignment fix**:
- Non-paired (exp31, exp32, exp33, exp36): `groups = np.arange(n_pos + n_neg)`
- Same-prompt paired (exp18b, exp39): `groups = np.concatenate([prompt_idx_pos, prompt_idx_neg])`

## Pre-Registered Pass/Fail Criteria

- If any AUROC changes by more than 0.05: **WEAKENED** for that comparison
- If any result flips significance (either direction): **WEAKENED**
- If all AUROCs within 0.02 of original AND no significance flips: **CONFIRMED**
  (methodology claim wrong, but numbers survive)

## Execution

**GPU required**: No. CPU only, uses pre-computed features from JSON.
**Code**: `kv_verify/experiments/v01_groupkfold.py`
**Tests**: `kv_verify/tests/test_v01.py`
