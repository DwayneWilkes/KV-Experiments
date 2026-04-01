# V03: FWL Leakage Test

**Status**: REGISTERED
**Design commit**: (this commit)

## Hypothesis

**Claim under test**: "Features survive FWL residualization (signal irreducible
to length)" (Paper C3)

**Finding**: C4. `fwl_residualize(X, Z)` at `48_fwl_residualization.py:296-327`
fits OLS on the ENTIRE dataset before CV splitting. Test-fold confound values
leak into the regression coefficients.

**Null hypothesis (H0)**: Within-fold FWL produces the same AUROCs as
full-dataset FWL (leakage is negligible).

**Alternative (H1)**: Within-fold FWL changes at least one AUROC by more than
0.05, or polynomial FWL collapses comparisons that survived linear FWL.

## Methods

**Statistical tests**:
- FWL residualization, within-fold vs full-dataset (see [methods.md](methods.md#frisch-waugh-lovell-theorem))
- Polynomial FWL at degree 2 and 3 (see [methods.md](methods.md#frisch-waugh-lovell-theorem))
- GroupKFold AUROC with corrected groups (see [methods.md](methods.md#groupkfold))
- Group-level permutation test, 10K iterations

**Input data**: Per-item features from hackathon result JSONs (same as V01).
Length confounds: `norm` (total Frobenius norm) and `n_generated` (token count).

**FWL conditions tested** (per comparison):
1. Full-dataset linear FWL (original, buggy)
2. Within-fold linear FWL (corrected)
3. Within-fold polynomial FWL degree 2
4. Within-fold polynomial FWL degree 3

**Sample sizes**: Same as V01 (N=10-20 per group).

## Pre-Registered Pass/Fail Criteria

- If within-fold FWL changes any AUROC by more than 0.05 vs full-dataset:
  FWL leakage is real, C4 **CONFIRMED**
- If polynomial FWL collapses 3+ of the 7 surviving comparisons
  (excluding sycophancy, deception-18b FWL-both, Llama refusal) below chance:
  **CENTRAL CLAIM FALSIFIED**
- If polynomial FWL preserves all 7 surviving comparisons:
  "beyond length" claim **STRENGTHENED**

**Note on polynomial FWL**: This is the single most consequential test in the
pipeline. If nonlinear length confounds explain the remaining signal, the
paper's central claim falls. No prior data exists to predict the outcome.

## Execution

**GPU required**: No. CPU only, uses pre-computed features.
**Code**: `kv_verify/experiments/v03_fwl_leakage.py`
**Tests**: `kv_verify/tests/test_v03.py`
