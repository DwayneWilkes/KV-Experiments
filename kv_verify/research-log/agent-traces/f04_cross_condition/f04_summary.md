# F04: Cross-Condition Transfer Validity

**Verdict: FALSIFIED**

## Evidence Summary

The cross-condition transfer claim (AUROC 0.89) is invalidated by the confluence of three findings: (1) The deception classifier predicts 100% of test samples as 'deceptive' regardless of category, showing no discriminative boundary. AUROC depends entirely on probability score ranking. (2) Same-model transfer (Mistral -> Mistral) is at chance (AUROC 0.53), which directly contradicts 'shared suppression geometry' -- if the geometry were shared, same-model should be the easiest transfer, not the hardest. (3) The classifier's top features (norms_per_token, key_ranks) have R^2 > 0.80 with input length, inheriting the confound from deception training. The high cross-model transfer likely reflects model-scale artifacts: small training models (1.1B-9B) produce different cache geometry than 14B test models, and the censored/control distinction rides on this scale difference.

## Sub-experiment Results

### F04a: Per-Topic Classifier Behavior

The deception-trained classifier predicts 100% of ALL test samples
(censored, control, complex_noncensored) as 'deceptive' with
probability >= 0.95. AUROC is driven entirely by score ranking,
not by any discriminative boundary.

### F04b: Feature Confound Cross-Reference

Weighted confound R^2: 0.7684057409173464

| Feature | Avg Importance | Input-Length R^2 | Risk |
|---------|---------------|-----------------|------|
| norms | 0.318 | 0.800 | HIGH |
| norms_per_token | 0.380 | 0.800 | HIGH |
| key_ranks | 0.189 | 0.891 | HIGH |
| key_entropies | 0.113 | 0.366 | MODERATE |

### F04c: Same-Model Transfer Failure

| Scenario | AUROC |
|----------|-------|
| Same-model: Mistral-7B-v0.3 -> Mistral-7B-v0.3 | 0.533 |
| Same-family: Qwen2.5-7B -> Qwen2.5-14B | 0.517 |
| Cross-model: All deception -> DeepSeek-R1-Distill-Qwen-14B | 0.887 |
| All-to-all: All deception -> All natural | 0.774 |
| All deception -> DeepSeek-R1-Distill-Qwen-14B | 0.887 |
| All deception -> Mistral-7B-v0.3 | 0.873 |
| All deception -> Qwen2.5-14B | 0.699 |

Same-model transfer (Mistral -> Mistral: 0.53, Qwen -> Qwen: 0.52)
is at chance. Cross-model transfer (All -> DeepSeek: 0.89) only works
when pooling many small training models. This is the opposite of what
'shared suppression geometry' predicts.

### F04d: What the Data Can/Cannot Tell Us

**Can determine:**
- The classifier predicts all test samples as 'deceptive' (from per_topic data)
- Same-model transfer fails, cross-model succeeds (from scenario AUROCs)
- Feature importance is dominated by norms_per_token (from RF importances)
- The input-length R^2 for deception features is very high (from F01b-49b)
- Per-model transfer varies: DeepSeek 0.89, Mistral 0.87, Qwen 0.70

**Cannot determine:**
- We do not have per-item features for the censorship test data, so we cannot directly test whether censored responses are systematically shorter/longer than controls (though the 100% predicted-deceptive pattern suggests the model learned a threshold, not a boundary)
- We do not have the actual system prompts used in the 7-model deception experiment, so we cannot measure the exact input-length difference between honest and deceptive conditions at training time
- We cannot run a direct input-length residualization on the transfer because the cross-condition JSON stores only aggregate metrics, not per-item features. This would require re-running the experiment with input-length controls.
- We cannot definitively distinguish 'model-scale artifact' from 'shared geometry' without matched-scale controls (e.g., train on 14B deception, test on 14B censorship)

**Checksum:** sha256:a12d1aa130eff5a3bece1cb0ec3d733c5fac0b0fbad1815b6b104f5f728d002d
