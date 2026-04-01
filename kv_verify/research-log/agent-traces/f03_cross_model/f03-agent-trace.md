---
agent: experiment-writer
date: 2026-03-26
scope: "F03 cross-model transfer with input-length control"
verdict: "NEEDS REVIEW"
files_created: 2
tests_added: 14
conventions_followed:
  patent_header: false
  module_docstring: true
  cli_arguments: false
  statistical_battery: true
  sha256_checksum: true
  result_json_schema: true
---

## Experiment Written: F03 Cross-Model Input Control

### Specification
- **Hypothesis**: Cross-model suppression transfer (Exp 49c, AUROC ~0.86 claimed) is confounded by input-length differences between refusal and benign prompts
- **Conditions**: 6 cross-model pairs (train on A, test on B) plus 3 within-model tests, all on refusal task
- **Controls**: Input-only classifier (just token count), input-length residualization
- **Metrics**: Raw AUROC, input-only AUROC, residualized AUROC, AUROC drop
- **Source**: F01b analysis showed refusal survived input control within-model; this tests whether cross-model transfer also survives

### Files Created
- `kv_verify/experiments/f03_cross_model_input_control.py`: Main experiment script
- `kv_verify/tests/test_f03_cross_model.py`: 14 tests covering data loading, statistical helpers, integration

### Convention Checklist
| Convention | Status | Notes |
|-----------|--------|-------|
| Patent/license header | N/A | No headers in existing scripts |
| Module docstring | Yes | Objectives, sub-experiments, hypothesis, expected outcomes |
| CLI arguments | No | Follows existing convention: function-based API with Path arg, no argparse |
| Statistical battery | Yes | Mann-Whitney U for asymmetry, Pearson r for correlations, LOO CV, cross-model train/test |
| SHA-256 checksum | Yes | Computed on pre-checksum JSON, stored in result |
| Result JSON schema | Yes | Matches project convention (claim_id, verdict, stats, checksum) |
| Reproducibility | Yes | All sklearn uses deterministic settings (max_iter=5000, random_state=42 in LOO) |

### Design Decisions

1. **Refusal only, not impossible**: The spec focuses on refusal since it survived F01b input control. Impossible task is excluded from this experiment to keep scope focused. The data also shows cross-model impossible transfer is mostly at chance (0.5) already.

2. **Cross-model residualization procedure**: The residualization regression is fit on TRAINING model data only, then applied to both train and test data. This is the correct procedure for cross-model evaluation -- it mirrors how a real deployment would work and avoids test-set leakage in the residualization step.

3. **LOO for within-model, train/test split for cross-model**: Within-model uses LOO (same as F01b) because n=30 is small. Cross-model uses full train/test since models A and B are independent datasets. This is more powerful than LOO for the cross-model case.

4. **No argparse**: Existing experiments (f01b, v07, v03) all use function-based APIs with Path arguments, not argparse CLI. Followed this convention.

5. **Paper AUROC lookup uses substring matching**: The paper's cross_model_transfer uses full model paths (e.g., "Qwen/Qwen2.5-7B-Instruct") while per_model_data uses short names. Used `in` operator for matching.

### Test Coverage
- `TestDataLoading` (4 tests): Verifies 49c data structure, feature extraction shape, input token computation
- `TestStatisticalHelpers` (7 tests): Safe AUROC edge cases, LOO on separable/random data, cross-model AUROC, residualization shape/decorrelation
- `TestRunF03` (5 tests): End-to-end run, JSON schema, verdict type, no self-pairs, SHA-256 integrity

### Red Team

1. **Small sample size**: 15 items per condition per model = 30 total per model pair. Cross-model: train on 30, test on 30. This is very small for logistic regression. The experiment may not have enough power to detect a genuine signal after residualization. The LOO within-model analysis helps, but cross-model AUROC variance is high at n=30.

2. **Linear residualization assumes linear confound**: If the relationship between input length and features is nonlinear, linear residualization will undercontrol. However, the F01b precedent uses linear residualization and it was sufficient there. Polynomial residualization (as in V03) could be added as a robustness check.

3. **Paper's reported AUROC of ~0.86 vs actual data**: The raw cross_model_transfer data in 49c shows most refusal pairs at AUROC=0.5 (chance!), with only Mistral->Qwen at 0.942. The mean is nowhere near 0.86. The paper's claim may aggregate differently (including impossible task, or using a different evaluation method). This pre-existing discrepancy weakens the experiment since there may be nothing to residualize.

4. **Same prompts across models**: All models use the same 15 harmful + 15 benign prompts. This means input-length structure is identical across models (same text, just different tokenization). If tokenization is similar, the input-only classifier will transfer trivially. This is actually evidence FOR the input-length confound hypothesis.

5. **n_generated varies**: Response lengths are not controlled (range 9-50 tokens). The `n_tokens - n_generated` computation correctly isolates input tokens, but the varying response length means `norm_per_token` is influenced by both input and output token counts. The residualization on input tokens only partially controls for this.

### Referrals
- PI: Review whether the paper's AUROC ~0.86 claim is for refusal specifically or averaged across tasks. The raw 49c data shows most refusal cross-model pairs at chance (0.5). If the claim is only for impossible task, the experiment scope may need adjustment.
- PI: Confirm whether running tests requires Bash permission (currently denied in this session). Tests could not be verified as passing.
