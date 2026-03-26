# V04: Multiple Comparison Correction (Holm-Bonferroni)

**Status**: COMPLETE
**Registration**: Retroactive. This design was specified in the pipeline
spec before implementation but was not committed as a separate design
document before execution. All future experiments will be pre-registered.

## Hypothesis

**Claim under test**: "9/10 comparisons significant in Exp 47" (Paper C3, Section 4.1)

**Finding**: C5. The `holm_bonferroni()` function exists in `stats_utils.py:210-232`
but was never called in Exp 47 or Exp 48. Ten binary comparisons are reported
without family-wise correction.

**Null hypothesis (H0)**: All 9 significant results survive Holm-Bonferroni correction.

**Alternative (H1)**: At least one significant result loses significance after correction.

## Methods

**Statistical test**: Holm-Bonferroni step-down procedure (Holm, 1979).
See [methods.md](methods.md#holm-bonferroni-step-down-procedure).

**Input data**: 10 permutation p-values from `corrected_evaluation.json`,
extracted into `kv_verify/fixtures.py::EXP47_COMPARISONS`.

**Family definition**: The 10 comparisons constitute one family because
they are reported together as a unified evaluation in Paper C3 Section 4.1.

**Alpha**: 0.05 (two-sided, as in the original experiments).

## Pre-Registered Pass/Fail Criteria

- If any result flips from significant to non-significant: **WEAKENED**
- If no results lose significance: **CONFIRMED**
- If multiple results lose significance: **WEAKENED** (with count)

## Execution

**GPU required**: No. CPU-only, pure computation.
**Code**: `kv_verify/experiments/v04_holm_bonferroni.py`
**Tests**: `kv_verify/tests/test_v04.py` (8 tests)

## Findings

**Verdict: WEAKENED**

9/10 significant before correction, 8/10 after Holm-Bonferroni.

`exp36_impossible_vs_harmful` lost significance:
- Raw p = 0.0366
- Corrected p = 0.0732 (rank 9 of 10, multiplier = 2)
- This was the "linchpin" comparison distinguishing output suppression
  from safety refusal. Its loss of significance weakens the mechanistic
  interpretation.

`exp32_jailbreak_vs_refusal` was already non-significant (raw p = 0.0516,
corrected p = 0.0732). No change in status.

All other 8 comparisons remain significant with corrected p <= 0.001.

**Deviations from plan**: None.
**Post-hoc analyses**: None.

## Result Commit

`e59139a` — feat(kv-verify): V4 Holm-Bonferroni — first verdict WEAKENED (chunk 4)
