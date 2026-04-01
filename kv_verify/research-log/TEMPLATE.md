# V{XX}: {Title}

**Status**: REGISTERED | IN PROGRESS | COMPLETE
**Design commit**: {hash} (must be committed BEFORE implementation)
**Result commit**: {hash}

## Hypothesis

**Claim under test**: "{exact claim from paper}" (Paper C3, Section X.Y)

**Finding**: {C1-C6, M1-M10 from synthesis.md}

**Null hypothesis (H0)**: {what would need to be true for the claim to stand}

**Alternative (H1)**: {what the review predicts}

## Methods

**Statistical tests**: {name each test, cite from methods.md}

**Input data**: {source files, how features are extracted}

**Sample sizes**: {N per group, effective N if pseudoreplication}

**Confound controls**: {FWL, length matching, format matching, etc.}

**Multiple comparison correction**: {if applicable, which method, family size}

## Pre-Registered Pass/Fail Criteria

{Concrete thresholds. These CANNOT be changed after seeing results.}

- If {condition}: **{VERDICT}**
- If {condition}: **{VERDICT}**

## Execution

**GPU required**: Yes/No
**Estimated time**: {hours}
**Code**: `kv_verify/experiments/v{xx}_{name}.py`
**Tests**: `kv_verify/tests/test_v{xx}.py`

## Findings

{Filled in AFTER execution. Must reference pre-registered criteria.}

**Verdict: {VERDICT}**

{Evidence summary}

**Deviations from plan**: {Any changes to methodology after seeing data. MUST be documented.}
**Post-hoc analyses**: {Analyses not in the pre-registered plan. Clearly labeled as exploratory.}

## Result Commit

`{hash}` — {commit message}
