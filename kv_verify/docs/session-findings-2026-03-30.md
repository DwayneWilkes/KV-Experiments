# Session Findings: 2026-03-30

## What was built

**132 new tests, 38 commits on `feat/remote-gpu-pipeline`.**

### New modules
- `lib/dataset_validation.py` — 4-tier validation library (15 checks, modality-agnostic, composable)
- `lib/remote.py` — SSH backend for remote GPU execution (rsync sync, stage routing)
- `lib/final_report.py` — Global Holm-Bonferroni + 14-claim verdict report
- `constants.py` — All magic numbers centralized (~70 replacements across 10 experiment files)
- `pyproject.toml` — pip-installable with core/gpu/tracking dependency splits

### New experiments (all pre-registered before implementation)
- V11: Feature ablation via permutation importance → **WEAKENED** (no single feature dominates, geometry is diffuse)
- V12: System prompt residualization → **CONFIRMED** (system prompts identical, no confound)
- V13: Matched-scale transfer control → Pending hackathon data

### Infrastructure
- `MODEL_CACHE_DIR` configurable (env var > config > default)
- `PipelineConfig.model_cache_dir`, `.force`, `.validation_tier`, `.to_yaml()`
- CLI: `validate` subcommand, `config --dump`, `--remote`, `--force` flags
- Pipeline: validation stage (stage 1) between environment and prompt_gen
- Documentation: README, pipeline-stages, experiments catalog, dataset-validation guide

## Critical Finding: Dataset Quality Audit

Ran Tier 2 validation on all 10 Exp 47 comparisons. **9/10 INCONCLUSIVE, 1 PASS.**

### Variance Asymmetry (NEW, not previously identified)

4 of the 7 comparisons that "survived" input-length control (F01b-all) have **variance ratios exceeding 3:1**, with the worst at 20.3:1. One condition (typically benign/normal) produces stereotyped responses with very low feature variance, making it trivially separable.

| Comparison | Variance Ratio | Trustworthy? |
|-----------|---------------|-------------|
| exp31_refusal_vs_normal | OK | Yes |
| exp36_impossible_vs_harmful | OK | Yes |
| exp32_jailbreak_vs_refusal | OK | Yes |
| exp31_refusal_vs_benign | 19.6:1 | No — inflated |
| exp36_impossible_vs_benign | 6.8:1 | No — inflated |
| exp36_harmful_vs_benign | 19.6:1 | No — inflated |
| exp32_jailbreak_vs_normal | 20.3:1 | No — inflated |

**Revised finding: 3/7 comparisons survive input control AND pass variance validation.** Down from 7/10 "surviving" as reported before.

### Effective N concerns

- exp33 (Llama/Mistral cross-model): effective N = 4.2-6.5 (nominal N=20+20)
- exp18b (deception): nominal N=10+10, effective N=10 (no template inflation, but still small)
- V10 power analysis used nominal N, not effective N. Power estimates may be optimistic.

### Confound discovery tool improvement

The `confound_discovery` check now accepts `classification_features` to exclude intentional features from the confound scan. Without this, it flagged the features we're classifying on (expected behavior, not a data quality issue).

## Updated Claim Verdicts

| Claim | Previous Verdict | Revised Verdict | Reason |
|-------|-----------------|----------------|--------|
| C6 (refusal genuine) | CONFIRMED (7/10 survive) | **WEAKENED** (3/7 trustworthy) | 4 comparisons have variance inflation |
| M7 (adequate power) | WEAKENED (linchpin 0.43) | **WEAKENED** (worse: effective N < nominal) | Needs recomputation with N_eff |
| All others | Unchanged | Unchanged | Variance doesn't affect falsified/confirmed claims |

## What needs to happen next

### Team verification (can do now, no GPU needed)
1. Review the variance_ratio findings. Are the 20:1 ratios a genuine confound or a expected property of the data?
2. Recompute V10 power analysis using effective N from the validation audit
3. Decide if the 4 variance-inflated "surviving" comparisons should be downgraded to INCONCLUSIVE
4. Review V11 finding (diffuse geometry). Is this consistent with the 3 trustworthy comparisons?

### Implementation (needs GPU or RunPod)
5. Run F01d (feature re-extraction) to verify stored features aren't corrupted
6. Run full pipeline N=200 with dataset validation as pre-flight
7. Generate final report with global Holm correction

### Remediation (if variance inflation is confirmed as a problem)
8. Design V14: variance-matched replication. Subsample the low-variance condition to match the high-variance condition's distribution, then re-run classification.
9. Add variance_ratio control to the FWL residualization (residualize against variance, not just mean)
10. Update the final report narrative from "7/10 survive" to "3/7 survive clean validation"

## File inventory

All on `feat/remote-gpu-pipeline` branch in `ll/KV-Cache/KV-Cache_Experiments`:

```
kv_verify/
  lib/dataset_validation.py    870 lines, 15 checks, 4 tiers
  lib/remote.py                SSH backend, rsync sync
  lib/final_report.py          14-claim verdict report
  constants.py                 All magic numbers
  config.py                    model_cache_dir, force, validation_tier, to_yaml
  __main__.py                  validate + config subcommands, --remote/--force
  pipeline.py                  validation stage added
  experiments/v11_*.py         Feature ablation (WEAKENED)
  experiments/v12_*.py         System prompt (CONFIRMED)
  experiments/v13_*.py         Matched-scale transfer
  research-log/V11-design.md   Pre-registered
  research-log/V12-design.md   Pre-registered
  research-log/V13-design.md   Pre-registered
  docs/README.md               Quick start, architecture
  docs/pipeline-stages.md      Per-stage I/O
  docs/experiments.md          V/F-series catalog
  docs/dataset-validation.md   Tier reference, check guide
  docs/validation-audit-findings.md  THIS audit
  config-default.yaml          All defaults
  requirements-gpu.txt         GPU deps
  pyproject.toml               pip-installable

tests/ (132 new tests):
  test_dataset_validation.py   Core entry point
  test_check_registry.py       @check decorator
  test_tier0_checks.py         structural, duplicates, class_balance
  test_tier1_checks.py         size_overlap, effective_n, diversity, balance
  test_tier2_checks.py         shortcut, confound, variance, pairs, format
  test_tier3_checks.py         hash, metadata, measurement
  test_models_config.py        Configurable MODEL_CACHE_DIR
  test_config_cache_dir.py     PipelineConfig.model_cache_dir
  test_config_yaml.py          to_yaml + config --dump
  test_packaging.py            pyproject.toml
  test_remote.py               SSH session, sync
  test_remote_errors.py        Error handling
  test_validate_cli.py         validate subcommand
  test_cli_remote.py           --remote flag
  test_global_holm.py          Global Holm-Bonferroni
  test_final_report.py         Report generator
  test_pipeline_validation.py  Validation stage
  test_v11.py, test_v12.py, test_v13.py  New experiments
```

## Specs

`specs/changes/kv-verify-remote-gpu/`: proposal, design, tasks (52/66 complete), dataset-validation spec (19 requirements, 52 scenarios).
