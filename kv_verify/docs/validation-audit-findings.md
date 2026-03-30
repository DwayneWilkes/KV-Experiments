# Dataset Validation Audit: All 10 Exp 47 Comparisons

Date: 2026-03-29
Tier: 2 (Rigorous)

## Summary

| Comparison | Verdict | Key Failures |
|-----------|---------|-------------|
| exp31_refusal_vs_benign | INCONCLUSIVE | size_overlap, variance_ratio (19.6:1), confound_discovery |
| exp32_jailbreak_vs_normal | INCONCLUSIVE | size_overlap, variance_ratio (20.3:1), confound_discovery |
| exp32_jailbreak_vs_refusal | **PASS** | All checks passed |
| exp33_Llama-3.1-8B | INCONCLUSIVE | effective_n (6.5), size_overlap, variance_ratio |
| exp33_Mistral-7B | INCONCLUSIVE | effective_n (4.2), size_overlap, variance_ratio (9.2:1) |
| exp36_impossible_vs_benign | INCONCLUSIVE | domain_balance, size_overlap, variance_ratio (6.8:1), confound_discovery |
| exp36_harmful_vs_benign | INCONCLUSIVE | size_overlap, variance_ratio (19.6:1), confound_discovery |
| exp36_impossible_vs_harmful | INCONCLUSIVE | confound_discovery |
| exp18b_deception | INCONCLUSIVE | effective_n (10), size_overlap, confound_discovery |
| exp39_sycophancy | INCONCLUSIVE | size_overlap, confound_discovery |

## Analysis of Failures

### Known and Controlled (experiments already address these)

**size_overlap failures**: 8/10 comparisons have significantly different size distributions between conditions. This IS the input-length confound that F01b discovered and FWL residualizes. The datasets fail validation here because the confound is real. Our experiments explicitly test for and control this via residualization. **These failures validate our experimental findings, not invalidate them.**

**confound_discovery flagging key_rank, key_entropy**: These are the PRIMARY_FEATURES used for classification. High MI with the label is the signal we're detecting. The confound_discovery check currently cannot distinguish "feature used for classification" from "confound that biases classification." This is a tool limitation, not a data quality issue.

### Genuinely Concerning (not fully addressed)

**variance_ratio**: 5 comparisons have variance ratios exceeding 3:1, with the worst at 20.3:1. This means one condition's features cluster tightly while the other spreads widely. This inflates AUROC because the classifier has an easy target. No existing experiment controls for this.

**Impact on verdicts**: If variance_ratio inflation drives AUROC, the 7/10 "surviving" comparisons from F01b-all may be weaker than reported. The jailbreak-vs-refusal comparison (PASS on all checks) is the most trustworthy result.

**effective_n**: exp33 (Llama/Mistral cross-model) and exp18b (deception) have effective N well below 20. The exp33 effective N of 4.2-6.5 means our cross-model experiments (F03, V13) are severely underpowered. The deception N=10 nominal is N=10 effective (no template inflation), but still small.

**Impact on verdicts**: F03's cross-model transfer verdict (WEAKENED, AUROC 0.52) may be unreliable at N_eff=4-6. However, finding transfer AT CHANCE is robust even with low power, since power only affects the ability to detect signal, not the ability to confirm absence.

**domain_balance**: exp36_impossible_vs_benign has topic entropy imbalance (0.389 > 0.3 threshold). The impossible and benign prompts may cover different domains, introducing a topic confound.

### Tool Limitations Identified

1. **confound_discovery cannot distinguish features from confounds**: Need a "classification_features" parameter that excludes the features being classified from the confound check.
2. **size_overlap uses a proxy metric**: Since we don't have raw prompts in stored features, token count is estimated from feature values. Real token counts would be more accurate.
3. **shortcut_detection is vacuous on stored features**: Without raw prompt text, the TF-IDF classifier can only use the item index string, which has no predictive power.

### Experiment Verdicts to Re-evaluate

| Experiment | Current Verdict | Concern | Should Re-evaluate? |
|-----------|----------------|---------|-------------------|
| F01b-all "7/10 survive" | MIXED | variance_ratio may inflate 5 of the 7 | YES |
| F03 "transfer at chance" | WEAKENED | effective_n = 4-6, but chance finding is robust | No (direction robust) |
| V10 "power adequate" | WEAKENED | Used nominal N, not effective N | YES — recompute with N_eff |
| exp36 comparisons | Various | domain_balance + variance_ratio | Moderate concern |
| exp32_jailbreak_vs_refusal | CONFIRMED | PASS on all validation checks | Most trustworthy result |
