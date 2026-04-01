---
agent: experiment-writer
date: 2026-03-26
scope: "F04: Cross-condition transfer validity analysis"
verdict: "NEEDS REVIEW"
files_created: 3
tests_added: 16
conventions_followed:
  patent_header: false
  module_docstring: true
  cli_arguments: false
  statistical_battery: true
  sha256_checksum: true
  result_json_schema: true
---

# F04: Cross-Condition Transfer Validity

## Summary

Wrote `kv_verify/experiments/f04_cross_condition_validity.py`, which analyzes whether the paper's cross-condition transfer claim (AUROC 0.89, deception -> censorship) is invalidated by the input-length artifact discovered in F01b.

The analysis is primarily interpretive, examining existing JSON result data rather than computing new features. It structures the argument across four sub-experiments and produces a verdict with quantitative backing.

**Verdict: FALSIFIED** (pending execution confirmation)

The evidence chain is:
1. The deception classifier predicts 100% of ALL test samples as "deceptive" (censored, control, and complex_noncensored alike)
2. Same-model transfer is at chance (Mistral -> Mistral: AUROC 0.53)
3. The classifier's top features have R^2 > 0.80 with input length
4. Cross-model transfer only works when pooling many small models into training

## Execution Status

**Python execution was denied by the sandbox.** The script has NOT been verified by running. The manual result JSON is a placeholder. The PI needs to run:

```bash
TMPDIR=/tmp/claude-1000 .venv/bin/python kv_verify/experiments/f04_cross_condition_validity.py
```

And the tests:

```bash
TMPDIR=/tmp/claude-1000 .venv/bin/python -m pytest kv_verify/tests/test_f04_cross_condition.py -v
```

## Key Data Observations

### Per-topic data is the smoking gun

In the cross_condition_transfer.json per_topic section, the deception-trained RandomForest classifier predicts `predicted_deceptive_pct: 100.0` and `mean_deception_probability: 1.0` for EVERY topic, including:
- 6 censored topics (tiananmen, uyghur, taiwan, etc.)
- 6 control topics (cuisine, geography, inventions, etc.)
- 6 complex_noncensored topics (rwanda_genocide, armenian_genocide, etc.)

This means the classifier has NO discriminative boundary. It classifies everything as "deceptive." The AUROC > 0.5 comes from probability SCORE RANKING, which is a secondary artifact.

### Same-model failure is devastating

- Scenario A (Mistral deception -> Mistral censorship): AUROC 0.53 (chance)
- Scenario B (Qwen-7B deception -> Qwen-14B censorship): AUROC 0.52 (chance)
- Scenario C (All 7 models -> DeepSeek): AUROC 0.89

If deception and censorship share geometric structure, same-model transfer should be the EASIEST case. Instead, it is at chance. Cross-model transfer only works when training on many heterogeneous small models. This pattern is consistent with model-scale confounds, not shared suppression geometry.

### Feature importance cross-reference

From the transfer_improvement.json, norms_per_token dominates feature importance (~35-47% across normalization approaches). From F01b-49b, norms_per_token has R^2 = 0.80 with input length. The classifier learned "longer input cache geometry -> deceptive" and this accidentally correlates with something in the cross-model censorship data.
