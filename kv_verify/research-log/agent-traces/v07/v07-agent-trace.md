---
agent: experiment-writer
date: 2026-03-26
scope: "V03 FWL Leakage Test experiment for kv_verify verification pipeline"
verdict: "COMPLETE"
files_created: 2
tests_added: 23
conventions_followed:
  patent_header: false
  module_docstring: true
  cli_arguments: false
  statistical_battery: true
  sha256_checksum: false
  result_json_schema: true
---

## Experiment Written: V03 FWL Leakage Test

### Specification
- **Hypothesis**: FWL residualization leaks test-fold confound values into regression coefficients when fit on the entire dataset before CV splitting (finding C4). Polynomial FWL may collapse signal that survived linear FWL, potentially falsifying the paper's central claim.
- **Conditions**: Four FWL conditions per comparison: (1) full-dataset linear FWL, (2) within-fold linear FWL, (3) polynomial FWL degree 2, (4) polynomial FWL degree 3
- **Controls**: Full-dataset FWL is the "buggy" baseline; within-fold FWL is the corrected condition. Three comparisons excluded from polynomial verdict (already known confounded/non-significant).
- **Metrics**: AUROC under each FWL condition, R-squared for linear/poly2/poly3, leakage delta, collapse/preserve counts
- **Source**: V03-design.md pre-registered design, finding C4 from deep review

### Files Created
- `kv_verify/experiments/v03_fwl_leakage.py`: Experiment implementation (run_v03)
- `kv_verify/tests/test_v03.py`: 23 tests across 8 test classes

### Convention Checklist
| Convention | Status | Notes |
|-----------|--------|-------|
| Patent/license header | N/A | No patent header in existing scripts |
| Module docstring | Yes | Objectives, four conditions, pre-registered pass/fail criteria |
| CLI arguments | N/A | Called as library function like V04, not CLI script |
| Statistical battery | Yes | GroupKFold AUROC, FWL residualization, polynomial FWL, R-squared |
| SHA-256 checksum | No | V04 pattern does not include checksums on result JSON |
| Result JSON schema | Yes | Matches project convention: comparisons array, overall verdicts, thresholds |
| Reproducibility | Yes | Deterministic via GroupKFold splits and OLS regression (no random state needed) |

### Design Decisions
- **n_permutations parameter**: Included in signature per spec but currently unused. V03 compares FWL conditions via AUROC difference thresholds, not permutation null distributions. Documented as reserved for future use.
- **Polynomial FWL on full dataset then classify**: Per spec, polynomial FWL residualizes the full dataset with fwl_nonlinear() first, then passes residuals to groupkfold_auroc() without fwl_confounds. This is intentional: the polynomial test asks "does nonlinear length explain the signal?" not "does within-fold polynomial FWL change AUROC?"
- **Per-comparison verdict vs overall verdict**: Each comparison gets CONFIRMED/INDETERMINATE for leakage. The polynomial verdict (falsified/strengthened/weakened) is computed across all surviving comparisons and stored in JSON only.
- **Followed V04 pattern**: Single return object convention from V04 was adapted to return List[ClaimVerification] (one per comparison) as specified. JSON serialization follows same pattern.
- **NaN handling**: Custom JSON serializer converts NaN to null for JSON compatibility. AUROC can be NaN when GroupKFold can't produce valid predictions (insufficient groups).

### Test Coverage
- **TestV03ReturnType** (3 tests): Returns list of ClaimVerification, correct count
- **TestV03ClaimIds** (3 tests): All start with "C4-", unique, finding_id="C4"
- **TestV03StatsSchema** (5 tests): Required AUROC keys present, numeric, in [0,1], R-squared dict with linear/poly2/poly3
- **TestV03ResultJSON** (5 tests): File saved, valid JSON, has comparisons, overall verdicts, comparison schema
- **TestV03Verdicts** (3 tests): Valid enum values, leakage detection consistency, severity=CRITICAL
- **TestV03ExcludedComparisons** (2 tests): Correct excluded set, surviving count = 7
- **TestV03OverallVerdict** (3 tests): Polynomial verdict in valid set, n_surviving=7, n_collapsed_poly2 present
- **TestV03ZeroGpuTime** (1 test): CPU-only experiment

### Red Team
- **Polynomial FWL on full dataset is itself a form of leakage**: The polynomial FWL (conditions 3-4) residualizes the full dataset before CV splitting, which is the same leakage pattern being tested in conditions 1-2. This is by design per the spec (testing whether nonlinear confounds explain the signal), but a devil's advocate would argue the polynomial test should also be within-fold for consistency. The spec explicitly directs full-dataset polynomial FWL, so this is intentional.
- **Small sample sizes (N=10-20)**: GroupKFold with 5 splits on 20 samples means ~4 test samples per fold. AUROC estimates on 4 samples are extremely noisy. A 0.05 AUROC threshold for leakage detection may trigger from noise alone, or real leakage may be masked.
- **Degree-3 polynomial on N=20 with 2 confounds**: PolynomialFeatures(degree=3) on 2 confounds produces 9 features. With N=20, this is close to overfitting the confounds, which would artificially collapse all signal. The R-squared values should flag this.
- **No bootstrap CI on AUROC differences**: The leakage test uses a hard 0.05 threshold on AUROC delta without uncertainty quantification. A paired bootstrap on the full-vs-within AUROC difference would be more rigorous.

### Referrals
- **reviewer**: Verify that polynomial FWL degree 3 does not overfit the confounds given the small sample sizes. Check R-squared values for degree-3 fits approaching 1.0.
- **stats-reviewer**: Consider whether a paired bootstrap test on AUROC(full) - AUROC(within) would be more appropriate than a fixed 0.05 threshold.
