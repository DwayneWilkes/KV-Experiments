# Research Log

Pre-registered experimental designs and findings for the KV-Cache
verification pipeline. Version-controlled for reproducibility.

## Protocol

1. **REGISTER**: Before implementing any experiment, commit a design
   document specifying the hypothesis, methodology, pass/fail criteria,
   and statistical tests. This commit establishes temporal proof that
   the design preceded the results.

2. **IMPLEMENT**: Write test code and experiment code. Commit.

3. **EXECUTE**: Run the experiment. Commit results.

4. **RECORD**: Update the design document with findings, noting any
   deviations from the pre-registered plan. Commit.

Any deviation from the pre-registered design must be documented and
justified. Post-hoc analyses are permitted but must be clearly labeled.

## Log Index

| ID | Experiment | Design Commit | Result Commit | Verdict |
|----|-----------|---------------|---------------|---------|
| V01 | [GroupKFold Bug](V01-design.md) | (this commit) | pending | pending |
| V03 | [FWL Leakage](V03-design.md) | (this commit) | pending | pending |
| V04 | [Holm-Bonferroni](V04-design.md) | retroactive (see note) | `e59139a` | WEAKENED |
| V07 | [Sycophancy Length](V07-design.md) | (this commit) | pending | pending |
| V10 | [Power Analysis](V10-design.md) | (this commit) | pending | pending |

## Methodological References

Cited methods are documented in [methods.md](methods.md).
