# Experiment Catalog

All experiments are pre-registered in `research-log/` before implementation.

## V-Series: Statistical Methodology

| ID | Title | Claim | Status | Verdict |
|----|-------|-------|--------|---------|
| V01 | GroupKFold Bug | C2: GroupKFold prevents prompt leakage | Complete | CONFIRMED |
| V03 | FWL Leakage | C4: Features retain significance after FWL | Complete | WEAKENED |
| V04 | Holm-Bonferroni | C3: 9/10 comparisons significant | Complete | WEAKENED (8/10) |
| V07 | Sycophancy Length | C5: Geometry detects sycophancy | Complete | FALSIFIED |
| V10 | Power Analysis | M7: Adequate statistical power | Complete | WEAKENED (linchpin 0.43) |
| V11 | Feature Ablation | Implicit: features each contribute | Complete | WEAKENED (diffuse) |
| V12 | System Prompt | C5.1: Not prompt fingerprinting | Complete | CONFIRMED (identical prompts) |
| V13 | Matched-Scale Transfer | C7: Shared geometry, not scale | Complete | Pending data |

## F-Series: Falsification Battery

| ID | Title | Claim | Status | Verdict |
|----|-------|-------|--------|---------|
| F01a | Null Experiment | C1: Signal exists | Complete | CONFIRMED |
| F01b | Input Confound | C10: No input confound | Complete | FATAL (all confounded) |
| F01b-49b | Paper's Control | C11: 49b control valid | Complete | FALSIFIED |
| F01b-all | 10 Comparisons | C6: Some survive | Complete | 7/10 SURVIVE |
| F01c | Format Classifier | C1: Cache beats format | Complete | CONFIRMED |
| F01d | Re-extraction | C1: Stored features valid | Pending | Needs GPU |
| F02 | Held-Out Transfer | C9: Generalization | Complete | DECEPTION FALSIFIED |
| F03 | Cross-Model | C7: Transfer at 0.86 | Complete | WEAKENED (0.52) |
| F04 | Cross-Condition | C8: Deception->censorship | Complete | FALSIFIED |
| F05 | Abliterated/Hardware | Various | Complete | MIXED |

## Claim Summary

| Verdict | Count | Claims |
|---------|-------|--------|
| Falsified | 5 | C5, C8, C9-deception, C11, C12 |
| Weakened | 4 | C3, C7, M7, C9-refusal |
| Confirmed | 5 | C1, C2, C4, C6, C10-partial |

Key finding: deception and sycophancy signals are input-length artifacts. Refusal, jailbreak, and impossibility detection retain genuine signal.
