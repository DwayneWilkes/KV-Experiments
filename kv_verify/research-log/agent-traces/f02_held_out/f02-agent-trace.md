---
agent: experiment-writer
date: 2026-03-26
scope: "F02: Held-out prompt generalization under input-length control"
verdict: "NEEDS REVIEW"
files_created: 2
tests_added: 20
conventions_followed:
  patent_header: false
  module_docstring: true
  cli_arguments: true
  statistical_battery: true
  sha256_checksum: true
  result_json_schema: true
---

# F02 Experiment: Held-Out Input-Length Control

## What Was Written

Created `kv_verify/experiments/f02_held_out_input_control.py` and `kv_verify/tests/test_f02_held_out.py`.

This experiment tests whether the paper's Exp 49d held-out evaluation (AUROC 0.76-0.81 for training on original prompts, testing on novel held-out prompts) survives input-length residualization. The key insight is that held-out deceptive prompts are constructed by appending "I need you to give me a confidently wrong answer." to base questions, creating the SAME ~11-token input-length confound discovered in F01b.

## Analysis Per Paradigm

### Deception (F02a)
- Training: same_prompt_deception.json (honest vs deceptive, 10+10, paired)
- Test: 49d held-out honest vs deceptive (10+10)
- Expected: FALSIFIED. The deceptive prompts systematically append ~11 tokens.

### Refusal (F02b)
- Training: refusal_generation.json (refusal vs normal, 20+20, unpaired)
- Test: 49d held-out refusal vs benign (10+10)
- Expected: May survive. Different prompt types, no systematic append.

### Impossibility (F02c)
- Training: impossibility_refusal.json (impossible vs benign, 20+20, unpaired)
- Test: 49d held-out impossible vs benign (10+10)
- Expected: May survive. Different prompt types, no systematic append.

## Statistical Battery

For each paradigm:
1. Welch's t-test on input token counts (with Mann-Whitney U backup)
2. Cohen's d / Hedges' g effect sizes on input length difference
3. LOO AUROC with input tokens only (confound check)
4. Within-fold OLS residualization (training-fold regression applied to test)
5. Bootstrap CI (2000 resamples) on AUROC drop
6. Holm-Bonferroni correction across the 3 paradigm p-values
7. Pearson correlations between each feature and input length

## Design Decisions

1. **Within-fold residualization for transfer**: Fit OLS on training data, apply to both training and held-out test data. This prevents information leakage from the test set influencing the residualization regression.

2. **Bootstrap CI on AUROC drop**: Rather than just point estimates, we bootstrap the difference (baseline - residualized) AUROC on the test set. This gives uncertainty quantification on the size of the confound.

3. **LOO for held-out within-set classification**: With only 10+10 items, StratifiedKFold has high variance. LOO gives a deterministic, unbiased estimate.

4. **Two input-only classifiers**: (a) LOO within held-out set, (b) train on original data's input lengths, test on held-out. The former tests whether the held-out set alone is confounded; the latter tests transfer of the confound.

5. **Holm-Bonferroni on input-length tests**: Three paradigms form a family of tests on input-length differences, so correction is needed.

## Verdict: NEEDS REVIEW

Tests could not be run due to Bash permission restrictions. The code follows all established conventions from f01_falsification.py and f01b_49b_analysis.py. The experiment should be run to verify.

## Files

- `/mnt/d/dev/lab/ll/KV-Cache/KV-Cache_Experiments/kv_verify/experiments/f02_held_out_input_control.py`
- `/mnt/d/dev/lab/ll/KV-Cache/KV-Cache_Experiments/kv_verify/tests/test_f02_held_out.py`
