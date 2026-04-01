---
agent: code-reviewer
date: 2026-03-27
scope: "kv_verify/ full package: lib/, experiments/, pipeline.py, config.py, shims, types.py, fixtures.py, data_loader.py"
verdict: "MINOR ISSUES"
issues:
  critical: 0
  major: 0
  minor: 8
---

# Code Reuse Review: kv_verify package

## Summary

The `kv_verify` package has a clean `lib/` vs `experiments/` vs `shims` architecture. The lib layer (`lib/stats.py`, `lib/tracking.py`, `lib/models.py`, etc.) contains reusable utilities. The experiment files (`f01_*.py`, `f02_*.py`, etc.) consume those utilities but also contain duplicated helper functions. The root-level shim files are all clean 2-line re-exports. No critical duplication that changes results, but there are several DRY violations where experiment files reimplement functionality already available in the library layer.

## Shim Files: CLEAN

All 9 root-level shim files are exactly 2-line re-exports with no stale code:

| Shim File | Target | Status |
|-----------|--------|--------|
| `stats.py` | `lib.stats` | Clean re-export |
| `tracking.py` | `lib.tracking` | Clean re-export |
| `models.py` | `lib.models` | Clean re-export |
| `feature_extractor.py` | `lib.feature_extractor` | Clean re-export |
| `scorers.py` | `lib.scorers` | Clean re-export |
| `prompt_gen.py` | `lib.prompts.gen` | Clean re-export |
| `prompt_analyzer.py` | `lib.prompts.analyzer` | Clean re-export |
| `prompt_gap_filler.py` | `lib.prompts.gap_filler` | Clean re-export |
| `prompt_generator.py` | `lib.prompts.generator` | Clean re-export |

The `lib/prompts/__init__.py` also does explicit named re-exports of all public symbols from `gen.py`, `analyzer.py`, `gap_filler.py`, and `generator.py`. This is good for IDE support but creates a third import path (e.g., `from kv_verify.lib.prompts import MinimalPair`). Not a bug, just something to be aware of.

## Issues

### 1. [MINOR] [CONFIDENCE: HIGH] [read] Duplicated `_extract_features()` across 3 files

**Locations:**
- `data_loader.py:34` -- takes `items` and `feature_names`, returns `np.ndarray`
- `experiments/f02_held_out_input_control.py:114` -- same logic, hardcoded to `PRIMARY_FEATURES`
- `experiments/f03_cross_model_input_control.py:61` -- same logic, takes `feature_names` param

All three do the exact same thing: extract a feature matrix from a list of dicts with `item["features"][f]`.

**Recommendation:** `f02` and `f03` should import `_extract_features` from `data_loader.py` (or make it public as `extract_features`). The `f02` version is a strict subset (hardcodes feature names); `f03` matches the `data_loader` signature exactly.

### 2. [MINOR] [CONFIDENCE: HIGH] [read] Duplicated `_load_json()` / `HACKATHON_DIR` across 4 files

**Locations:**
- `data_loader.py:19,27` -- canonical definition
- `experiments/f02_held_out_input_control.py:54` -- redefined
- `experiments/f04_cross_condition_validity.py:60,63` -- redefined (identical function)
- `experiments/f01d_feature_reextraction.py:46` -- redefined (constant only)
- `experiments/f01b_49b_analysis.py:58` -- inline path construction

The `data_loader.py` already exports `_load_json` (used by `f01_falsification.py`). The other experiment files redefine both `HACKATHON_DIR` and `_load_json` instead of importing from `data_loader`.

**Recommendation:** Import `_load_json` from `data_loader`. If the leading underscore is a concern, rename to `load_json` or add it to `data_loader`'s public API.

### 3. [MINOR] [CONFIDENCE: HIGH] [read] Duplicated `_loo_auroc()` in 2 files

**Locations:**
- `experiments/f02_held_out_input_control.py:208` -- LOO cross-validated AUROC
- `experiments/f03_cross_model_input_control.py:86` -- identical logic

Both create `make_pipeline(StandardScaler(), LogisticRegression(...))`, loop over LOO splits, and compute AUROC. The only difference is `f03` uses `max_iter=5000` without `solver="lbfgs"` (sklearn defaults to lbfgs anyway in recent versions).

**Recommendation:** Extract to `lib/stats.py` as `loo_auroc()` alongside the existing `groupkfold_auroc()`.

### 4. [MINOR] [CONFIDENCE: HIGH] [read] Duplicated `_train_test_auroc()` pattern

**Location:** `experiments/f02_held_out_input_control.py:226`

This is a simple "fit on train, predict on test, return AUROC" function. `f03` does the same thing inline in multiple places (lines 207+, 270+). This is a natural companion to `_loo_auroc` and `groupkfold_auroc` in `lib/stats.py`.

**Recommendation:** Extract to `lib/stats.py` as `train_test_auroc()`.

### 5. [MINOR] [CONFIDENCE: HIGH] [read] Duplicated `_residualize_train_test()` in f02/f03

**Locations:**
- `experiments/f02_held_out_input_control.py:247-270`
- `experiments/f03_cross_model_input_control.py:115` (`_residualize_cross_model()`)

These are train/test split versions of FWL residualization. `lib/stats.py` already has `fwl_residualize()` with `within_fold=True, train_idx=..., test_idx=...` parameters that could serve the same purpose. However, the experiment versions return separate `X_train_resid` and `X_test_resid` arrays (rather than a single array with indices), which is a slightly different API shape.

**Recommendation:** Either refactor `fwl_residualize` to support returning separate train/test arrays, or extract a `fwl_residualize_split(X_train, Z_train, X_test, Z_test)` to `lib/stats.py`.

### 6. [MINOR] [CONFIDENCE: HIGH] [read] Duplicated input-token extraction across 4 files

**Locations:**
- `experiments/f02_held_out_input_control.py:97` -- `_get_input_tokens()` (per-item) + `_extract_input_lengths()` (vectorized)
- `experiments/f03_cross_model_input_control.py:69` -- `_extract_input_tokens()` (same logic, different name)
- `data_loader.py:49` -- `_extract_confounds()` (extracts `[norm, n_generated]`, overlapping logic)
- `pipeline.py:383-387` -- inline `get_input_tokens()` local function

All do variations of `features["n_tokens"] - features["n_generated"]` or `features["n_input_tokens"]`. The `f02` version has a fallback to `n_input_tokens` if present (the most robust).

**Recommendation:** Add a `get_input_token_count(item)` utility to `data_loader.py` and use it everywhere.

### 7. [MINOR] [CONFIDENCE: HIGH] [read] `_stratified_auroc()` in f01 has no lib equivalent

**Location:** `experiments/f01_falsification.py:35-48`

This function does StratifiedKFold AUROC. `lib/stats.py` has `groupkfold_auroc()` which does GroupKFold. They are different enough in purpose (StratifiedKFold has no group structure), but the boilerplate is nearly identical.

**Recommendation:** Consider adding `stratified_auroc()` to `lib/stats.py` alongside `groupkfold_auroc()`. Low priority since it is only used in one place.

### 8. [MINOR] [CONFIDENCE: MEDIUM] [read] Double TF-IDF+KMeans computation in gap_filler.py

**Locations:**
- `lib/prompts/gap_filler.py:251-254` (inside `_compute_gaps()`)
- `lib/prompts/gap_filler.py:320-323` (inside `analyze_gaps()`)
- `lib/prompts/analyzer.py:201-204` (inside `analyze_semantic_clusters()`)

The `analyze_gaps()` function (line 285) calls `_compute_gaps()` on line 334. Both independently compute `TfidfVectorizer(max_features=500, stop_words="english")` + `KMeans(n_clusters=n_clusters, random_state=42, n_init=10)` on the same prompt texts. The TF-IDF + KMeans is computed twice in a single call to `analyze_gaps()`.

The `analyzer.py` version uses the same parameters but is in a different module (legitimate separate use case).

**Recommendation:** Extract a shared `_cluster_prompts(texts, n_clusters)` helper that returns `(labels, cluster_centers, feature_names)`. Call it once in `analyze_gaps()` and pass results to `_compute_gaps()`.

## Duplicated Patterns (Not Individual Functions)

### make_pipeline(StandardScaler(), LogisticRegression(...))

This pattern appears 11 times across the codebase:
- `lib/stats.py:310` (1 time)
- `experiments/f01_falsification.py:37` (1 time)
- `experiments/f01b_49b_analysis.py:86` (1 time)
- `experiments/f02_held_out_input_control.py:211,233,366,404,410` (5 times)
- `experiments/f03_cross_model_input_control.py:89,109,207` (3 times)

Most use `max_iter=5000, solver="lbfgs"`, but `f03` omits `solver="lbfgs"` (no practical difference in sklearn 1.0+). One instance (`f01b`) uses `max_iter=5000` without `solver`.

**Recommendation:** Define a factory function in `lib/stats.py`:
```python
def make_classifier(max_iter=5000):
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=max_iter, solver="lbfgs"))
```

### Duplicated factual question lists

**Locations:**
- `fixtures.py:37-58` -- 20 questions in `FACTUAL_QUESTIONS`
- `pipeline.py:500-686` -- 200 questions in `_FACTUAL_QUESTIONS`

The `fixtures.py` list is a subset of the `pipeline.py` list (the first ~20 overlap but are not in the same order). These serve different purposes: `fixtures.py` is for test fixtures, `pipeline.py` is for production prompt generation.

**Recommendation:** Low priority. The lists serve different purposes and are intentionally different sizes.

## Verified Clean (No Duplication)

- **lib/tracking.py**: Entirely self-contained. The `ExperimentTracker`, `@tracked`, `@stage`, and `@validated` decorators are used correctly from the shim by all callers. Note: the `_sanitize_key` method (line 164-167) does sanitize keys -- the path traversal concern from the earlier security review has been addressed.
- **lib/models.py**: Self-contained model management. The `generate_text()` function is imported by `lib/prompts/generator.py` properly. No duplication.
- **lib/feature_extractor.py**: Self-contained feature extraction. `get_cache_accessor()` and `extract_from_cache()` have no duplicates. Imports `MODEL_CACHE_DIR` from `models.py` (single source).
- **lib/scorers.py**: Self-contained scoring functions. Used correctly by `pipeline.py:421`. No duplication.
- **lib/prompts/gen.py**: Clean data model (MinimalPair, PairSet) and generation functions. Used by both `generator.py` and `pipeline.py` via imports.
- **config.py**: Clean dataclass with YAML loading. No duplication.
- **types.py**: Clean type definitions used across the codebase. Single source of truth for `FeatureVector`, `ClassificationResult`, `ClaimVerification`, etc.
- **data_loader.py**: Clean loader for hackathon JSON data. Its `_extract_features` and `_load_json` are the canonical versions that should be imported by experiment files.
- **fixtures.py**: Test fixtures and reference data. Clean separation from production code.

## Summary Table

| Duplicated Function | Files | Severity |
|---------------------|-------|----------|
| `_extract_features()` | data_loader, f02, f03 | MINOR |
| `_load_json()` / `HACKATHON_DIR` | data_loader, f02, f04, f01d, f01b | MINOR |
| `_loo_auroc()` | f02, f03 | MINOR |
| `_train_test_auroc()` | f02 (f03 inline) | MINOR |
| `_residualize_train_test()` | f02, f03 (different shape) | MINOR |
| `_extract_input_tokens()` | f02, f03, data_loader, pipeline | MINOR |
| `_stratified_auroc()` | f01 (no equivalent in lib) | MINOR |
| TF-IDF+KMeans double compute | gap_filler (2x in one call) | MINOR |
| `make_pipeline(...)` pattern | 11 occurrences across 5 files | MINOR (pattern) |

## Verdict

**MINOR ISSUES.** The codebase has a well-structured lib layer with clean shim re-exports. The duplication is entirely in the experiment files, which reimplement data-loading and classification helpers that already exist in `data_loader.py` and `lib/stats.py`. None of the duplications change computed results -- they are code quality issues only. The most actionable cleanups would be:

1. Extract `_loo_auroc`, `_train_test_auroc`, `_stratified_auroc`, and `make_classifier` into `lib/stats.py`
2. Have experiment files import `_load_json` and `_extract_features` from `data_loader.py`
3. Add a `get_input_token_count(item)` utility to `data_loader.py`
4. Fix the double-computation of TF-IDF+KMeans in `gap_filler.py:analyze_gaps()`

## Referrals

- **impl-auditor**: The experiment files (f02, f03) each have their own residualization implementations (`_residualize_train_test`, `_residualize_cross_model`) that differ slightly from `lib/stats.py:fwl_residualize()`. Verify that the train/test split residualization in experiment files produces identical results to what `fwl_residualize(within_fold=True, train_idx=..., test_idx=...)` would produce.
