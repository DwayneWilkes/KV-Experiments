# V10: Power Analysis

**Status**: REGISTERED
**Design commit**: (this commit)

## Hypothesis

**Claim under test**: Various. All AUROC claims assume adequate statistical
power. The experiments use N=10-20 per group.

**Finding**: M7. N=10-20 limits power. Non-significant results cannot be
interpreted as absence of effect. The "linchpin" comparison
(exp36_impossible_vs_harmful, AUROC=0.65) may be underpowered.

**Null hypothesis (H0)**: All significant results have achieved power > 0.80.

**Alternative (H1)**: At least one significant result has achieved power < 0.50,
meaning it was a lucky draw rather than reliable detection.

## Methods

**Statistical tests**:
- Simulation-based power analysis (see [methods.md](methods.md#simulation-based-power))
- For every comparison from V01/V03 corrected results:
  1. Achieved power at observed N and AUROC
  2. Minimum detectable AUROC at N and 80% power
  3. Required N for 80% power at observed AUROC
- Hanley-McNeil SE for each AUROC estimate

**AUROC-to-d conversion**: d = sqrt(2) * Phi^{-1}(AUROC).
Assumes equal-variance normal distributions (approximation).

**Input data**: Corrected AUROCs and sample sizes from V01 and V03 results.

**Sample sizes**: As reported in each comparison (N=10-20 per group).

## Pre-Registered Pass/Fail Criteria

- If power > 0.80 for all significant results: **CONFIRMED** (adequate power)
- If power < 0.50 for any significant result: **WEAKENED** (underpowered)
- If linchpin comparison (exp36_impossible_vs_harmful, AUROC=0.65) has
  power < 0.30: **LINCHPIN UNDERPOWERED** (specific verdict annotation)

## Execution

**GPU required**: No. CPU only, pure computation.
**Depends on**: V01 and V03 corrected results.
**Code**: `kv_verify/experiments/v10_power_analysis.py`
**Tests**: `kv_verify/tests/test_v10.py`
