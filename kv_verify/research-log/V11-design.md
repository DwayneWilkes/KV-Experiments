# V11: Feature Ablation via Permutation Importance

**Status**: REGISTERED
**Design commit**: {to be filled at commit time}
**Result commit**: pending

## Hypothesis

**Claim under test**: "KV-cache features (norm_per_token, key_rank, key_entropy) each contribute meaningfully to classification" (Paper C3, Sections 4-5, implicit)

**Finding**: No existing experiment tests which features drive AUROC. The paper reports feature importance informally (F04 notes norms_per_token at ~35-47%) but no formal ablation.

**Null hypothesis (H0)**: Removing any single feature reduces AUROC by less than 0.05 (features are redundant or interchangeable).

**Alternative (H1)**: At least one feature dominates (removal drops AUROC by > 0.10), revealing which dimension of KV-cache geometry carries the signal.

## Methods

**Statistical tests**: Permutation importance (Breiman 2001) with GroupKFold cross-validation. For each feature: shuffle that feature's values across samples 100 times, recompute GroupKFold AUROC each time, report mean AUROC drop. Bootstrap 95% CI on the importance estimate.

Reference: methods.md (GroupKFold AUROC, bootstrap CI)

**Input data**: Stored hackathon JSON features for all 10 Exp 47 comparisons. Features: norm_per_token, key_rank, key_entropy (PRIMARY_FEATURES).

**Sample sizes**: Per-comparison N ranges from 20 to 60. Effective N from V10 power analysis. Comparisons with effective N < 20 are annotated as underpowered.

**Confound controls**: FWL residualization against input length (n_tokens) applied before ablation, so we test importance of residualized features, not raw features that may be confounded.

**Multiple comparison correction**: Holm-Bonferroni across the 3 features x 10 comparisons = 30 tests.

## Pre-Registered Pass/Fail Criteria

- If one feature's removal drops mean AUROC by > 0.10 across surviving comparisons (F01b 7/10): **CONFIRMED** — that feature carries the signal
- If all features have importance < 0.05: **WEAKENED** — features are interchangeable (geometry is diffuse, not localized)
- If norms_per_token dominates AND it is the feature most correlated with input length: **FALSIFIED** — the "geometry" is an input-length proxy

## Execution

**GPU required**: No (uses stored features)
**Estimated time**: < 5 minutes
**Code**: `kv_verify/experiments/v11_feature_ablation.py`
**Tests**: `kv_verify/tests/test_v11.py`

## Findings

{Pending execution}

## Result Commit

pending
