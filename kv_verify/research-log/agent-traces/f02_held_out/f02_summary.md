# F02: Held-Out Prompt Generalization Under Input-Length Control

**Overall verdict**: FALSIFIED
**Elapsed**: 23.7s

## Results by Paradigm

### Deception (Verdict: FALSIFIED)

| Metric | Value |
|--------|-------|
| Paper transfer AUROC | 0.810 |
| Baseline transfer AUROC | 0.830 |
| Input-only AUROC (LOO) | 1.000 |
| Input-only transfer AUROC | 1.000 |
| Residualized transfer AUROC | 0.385 |
| AUROC drop | 0.445 |
| Drop 95% CI | [0.077, 0.734] |
| Input token diff (pos - neg) | 11.0 |
| Welch t p-value | 0.0000 |
| Welch t p-value (Holm-corrected) | 8.10435958942003e-12 |

### Refusal (Verdict: WEAKENED)

| Metric | Value |
|--------|-------|
| Paper transfer AUROC | 0.790 |
| Baseline transfer AUROC | 0.530 |
| Input-only AUROC (LOO) | 0.630 |
| Input-only transfer AUROC | 0.735 |
| Residualized transfer AUROC | 0.590 |
| AUROC drop | -0.060 |
| Drop 95% CI | [-0.512, 0.374] |
| Input token diff (pos - neg) | 1.5 |
| Welch t p-value | 0.0699 |
| Welch t p-value (Holm-corrected) | 0.13973534299027524 |

### Impossibility (Verdict: CONFIRMED)

| Metric | Value |
|--------|-------|
| Paper transfer AUROC | 0.790 |
| Baseline transfer AUROC | 0.490 |
| Input-only AUROC (LOO) | 0.600 |
| Input-only transfer AUROC | 0.690 |
| Residualized transfer AUROC | 0.770 |
| AUROC drop | -0.280 |
| Drop 95% CI | [-0.625, 0.055] |
| Input token diff (pos - neg) | 1.0 |
| Welch t p-value | 0.1272 |
| Welch t p-value (Holm-corrected) | 0.13973534299027524 |

## Interpretation

The deception paradigm appends 'I need you to give me a confidently wrong
answer.' to held-out deceptive prompts, creating the SAME input-length
confound as in 18b/49b. Refusal and impossibility use inherently different
prompt types (harmful vs benign, impossible vs benign) where the length
difference arises from prompt content, not a systematic append.
