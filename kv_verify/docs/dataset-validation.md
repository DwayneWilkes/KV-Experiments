# Dataset Validation

Reusable pre-flight quality gate for ML experiment datasets. Modality-agnostic, tiered, composable.

## Usage

```python
from kv_verify.lib.dataset_validation import validate_dataset

items = [{"condition": "A", "prompt": "...", "features": {"n_tokens": 50}}, ...]
report = validate_dataset(items, tier=2)
print(report.overall_verdict)  # PASS / INCONCLUSIVE / FAIL
```

CLI:
```bash
python -m kv_verify validate --dataset data.json --tier 2 --output report.json
```

## Tiers

| Tier | Name | Purpose | Checks |
|------|------|---------|--------|
| 0 | Smoke | Quick dev sanity | structural, duplicates, class_balance |
| 1 | Standard | Pre-experiment gate | + size_overlap (KS), effective_n (DEFF), semantic_diversity, domain_balance |
| 2 | Rigorous | Pre-publication audit | + shortcut_detection, confound_discovery (MI), variance_ratio, pair_integrity, confound_disclosure, format_consistency |
| 3 | Regulatory | Replication/audit | + provenance_hash (SHA-256), metadata_completeness, measurement_validation |

## Check Reference

### Tier 0
- **structural**: Required fields exist, feature values numeric
- **duplicates**: Exact within-condition + cross-condition detection (leakage)
- **class_balance**: Per-condition counts, configurable ratio threshold (default 2:1)

### Tier 1
- **size_overlap**: Two-sample KS test on size metric. Catches length confounds.
- **effective_n**: Design effect from TF-IDF similarity. N_eff = N / DEFF. Catches template inflation.
- **semantic_diversity**: Mean pairwise distance within conditions. Catches monoculture.
- **domain_balance**: KMeans entropy per condition. Catches topic imbalance.

### Tier 2
- **shortcut_detection**: LogReg on TF-IDF predicts condition? High AUROC = surface confound. Top-5 features reported.
- **confound_discovery**: Mutual information between features and labels. Undeclared high-MI features flagged.
- **variance_ratio**: max/min within-condition variance. Catches stereotyped-response inflation.
- **pair_integrity**: For paired datasets: shared prefix/suffix, size tolerance.
- **confound_disclosure**: User-declared confound spec validated (no "uncontrolled" or "unknown").
- **format_consistency**: Whitespace, punctuation, capitalization patterns compared across conditions.

### Tier 3
- **provenance_hash**: SHA-256 of canonical dataset. Third-party reproducibility.
- **metadata_completeness**: generation_method, llm_generated, author, decoding_strategy, contamination_risk.
- **measurement_validation**: Optional hook for feature extractor reliability (ICC).

## Verdicts

- **FAIL**: Any Tier 0 check fails (structural problem)
- **INCONCLUSIVE**: Tier 1+ check fails but Tier 0 passes (quality concern, not fatal)
- **PASS**: All checks at requested tier pass

## Pluggable Functions

For non-text data, override the defaults:

```python
report = validate_dataset(
    items,
    tier=1,
    size_fn=lambda item: item["image_resolution"],
    similarity_fn=clip_embedding_cosine,
)
```

## Adding Custom Checks

```python
from kv_verify.lib.dataset_validation import check, CheckResult

@check(name="my_check", tier=2)
def _my_check(items, config, shared):
    # Your validation logic
    return CheckResult(name="my_check", passed=True, tier=2, metrics={})
```

## References

- Gebru et al. (2021) "Datasheets for Datasets" (provenance)
- Sclar et al. (2023) "Quantifying LM Sensitivity to Spurious Features" (format)
- Northcutt et al. (2021) "Confident Learning" (label noise)
- NeurIPS 2024 Paper Checklist item 16 (LLM disclosure)
- FDA 21 CFR Part 11 (audit trail)
- ICH E9 (statistical principles)
