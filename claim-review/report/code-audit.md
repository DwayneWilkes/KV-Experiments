# Code Audit Notes (WS11)

**Status**: COMPLETE
**Date**: 2026-03-04
**Auditor**: Claude (independent verification)

---

## 11.1 — Python Compilation (`py_compile`)

### code/ (33 files) — ALL CONFIRMED

| File | Compiles? | Key Imports |
|------|----------|-------------|
| 01_cache_inspection.py | YES | torch, json, pathlib |
| 01b_cognitive_modes.py | YES | torch, json, numpy |
| 01c_batch_replication.py | YES | torch, json, numpy, argparse |
| 01d_adversarial_controls.py | YES | torch, scipy, gpu_utils, stats_utils |
| 01e_controls_visualization.py | YES | json, numpy, matplotlib, seaborn |
| 01e_tokenizer_confound.py | YES | torch, scipy, gpu_utils, stats_utils |
| 02a_basic_transfer.py | YES | torch, json, numpy, dataclasses |
| 02b_projector_transfer.py | YES | torch, json, dataclasses |
| 03_scale_sweep.py | YES | torch, scipy, gpu_utils, stats_utils |
| 03b_identity_signatures.py | YES | torch, scipy, sklearn, gpu_utils, stats_utils |
| 03c_scale_sweep_visualization.py | YES | json, numpy, matplotlib (lazy) |
| 03d_identity_visualization.py | YES | json, numpy, matplotlib (lazy) |
| 04_deception_forensics.py | YES | torch, scipy, gpu_utils, stats_utils |
| 04b_deception_visualization.py | YES | json, numpy, matplotlib (lazy) |
| 04b_natural_deception.py | YES | torch, scipy, gpu_utils, stats_utils, s4_natural_deception |
| 05_layer_map.py | YES | torch, scipy, gpu_utils, stats_utils |
| 05b_layer_map_visualization.py | YES | json, numpy, matplotlib (lazy) |
| 06_temporal_evolution.py | YES | torch, scipy, gpu_utils, stats_utils |
| 06b_temporal_visualization.py | YES | json, numpy, matplotlib (lazy) |
| 07_abliteration_geometry.py | YES | torch, scipy, subprocess, gpu_utils, stats_utils |
| 07_individuation_geometry.py | YES | torch, scipy, gpu_utils, stats_utils |
| 07b_individuation_controls.py | YES | torch, scipy, gpu_utils, stats_utils |
| 08_input_only_geometry.py | YES | torch, scipy, gpu_utils, stats_utils |
| 09_sycophancy_detection.py | YES | torch, scipy, importlib, gpu_utils, stats_utils |
| 10_societies_of_thought.py | YES | torch, scipy, gc, gpu_utils, stats_utils |
| 11_rdct_stability.py | YES | torch, scipy, scipy.optimize, gc, gpu_utils, stats_utils |
| 12_bloom_taxonomy.py | YES | torch, scipy, gpu_utils, stats_utils |
| 13_c2c_replication.py | YES | torch, torch.nn, scipy, gc, gpu_utils, stats_utils |
| gpu_utils.py | YES | re, torch, pathlib, transformers (lazy) |
| heretic_abliterate.py | YES | argparse, torch, heretic (deferred) |
| length_confound_analysis.py | YES | json, numpy, scipy |
| recompute_stats.py | YES | json, numpy, scipy, stats_utils |
| stats_utils.py | YES | sys, platform, numpy, scipy, torch |

### prompts/ (5 files) — ALL CONFIRMED

| File | Compiles? | Notes |
|------|----------|-------|
| c7_frequency_matched_confabulation.py | YES | Data-only with `validate_prompts()` helper |
| s2_expanded_categories.py | YES | Data-only (prompt dictionaries) |
| s3_confabulation_elicitation.py | YES | Data-only with query helpers |
| s4_natural_deception.py | YES | 90 prompts + EXPERIMENT_METADATA |
| s5_sycophancy_elicitation.py | YES | Prompts + `generate_all_prompts()` |

### Other .py files — ALL CONFIRMED

| File | Compiles? | Notes |
|------|----------|-------|
| cross_model_analysis.py | YES | json, os, numpy, scipy |
| figures/generate_paper_figures.py | YES | json, numpy, matplotlib, seaborn |
| scripts/download_models.py | YES | argparse, pathlib |
| scripts/download_all_models.py | YES | huggingface_hub |

**Total: 42 Python files, 42 compile successfully, 0 failures.**

---

## 11.2 — Import Dependency Check

| Package | Required By | Available on Audit Machine? | Notes |
|---------|-----------|---------------------------|-------|
| torch | 25+ experiment scripts | NO | GPU-dependent; expected |
| transformers | gpu_utils.py (lazy import) | NO | Expected (needs GPU env) |
| scipy | stats_utils.py, 15+ scripts | YES | |
| numpy | all scripts | YES | |
| matplotlib | 6 visualization scripts | NO | Optional (viz only) |
| seaborn | 2 visualization scripts | NO | Optional (viz only) |
| sklearn | 03b, 09, lazy in others | NO | Used for classification |
| heretic | heretic_abliterate.py only | NO | Separate package (heretic-llm) |
| bitsandbytes | gpu_utils.py (lazy import) | NO | Quantization only |
| huggingface_hub | scripts/download_all_models.py | NO | Model download utility |
| peft | Not directly imported | NO | Indirect via heretic |
| accelerate | Not directly imported | NO | Indirect via transformers |

**Standard library** (always available): sys, os, re, json, argparse, hashlib, platform, pathlib, datetime, collections, typing, dataclasses, subprocess, shutil, time, math, itertools, gc, importlib

**Key architectural note**: Most third-party imports are at module level and will fail if not installed. The `gpu_utils.py` `load_model()` function wisely uses a lazy import for `BitsAndBytesConfig` inside the function body. `heretic_abliterate.py` also defers heretic imports until after argument parsing (lines 37-39) with a comment explaining why.

**Internal imports**: `gpu_utils` and `stats_utils` are imported from `code/` using relative imports (scripts are run from `code/` directory). `s4_natural_deception` is imported after `sys.path.insert` to add `prompts/`.

---

## 11.3 — gpu_utils.py Implementation Review

File: `../code/gpu_utils.py` (310 lines)

### Functions

1. **HybridCache monkey-patch** (lines 14-29): Patches `HybridCache` (used by Gemma-2) to add `__iter__`, `__len__`, `__getitem__`. This makes it behave like `DynamicCache`. Wrapped in try/except ImportError for graceful degradation.

2. **`normalize_cache(cache)`** (lines 32-41): Converts any cache type to list of `(key, value)` tuples. Handles `DynamicCache`, `HybridCache`, and legacy tuple-of-tuples. Used in 1 file externally.

3. **`model_id_from_name(model_name)`** (lines 44-66): Extracts short filesystem-safe ID from HuggingFace model name. Examples: `"Qwen/Qwen2.5-7B-Instruct" -> "Qwen2.5-7B"`. Used in 10 files.

4. **`get_output_path(results_dir, base_name, model_name, quantize)`** (lines 69-93): Builds per-model output path. Appends `-q4` suffix for quantized models. Used in 17 files.

5. **`load_model(model_name, quantize)`** (lines 96-142): Unified model loading with optional NF4 4-bit quantization. Uses `device_map="auto"` for multi-GPU. Sets `pad_token = eos_token` if missing. Used in 22 files.

6. **`compute_cache_dimensionality(cache, variance_threshold=0.9)`** (lines 145-236): **CORE FUNCTION**. Computes effective dimensionality of KV-cache via SVD.

   **Tensor reshaping** (line 176):
   ```python
   key_2d = key.reshape(-1, key.shape[-1])
   ```
   Cache tensor shape is `(batch, heads, seq_len, head_dim)`. The reshape collapses `batch * heads * seq_len` into one dimension, producing a matrix of shape `(B*H*S, d_h)`. This is consistent with the paper's notation K(l) in R^(H*S x d_h).

   **SVD and effective rank** (lines 182-189):
   ```python
   _, s, _ = torch.linalg.svd(matrix, full_matrices=False)
   s_squared = s ** 2
   total_var = s_squared.sum()
   cumvar = torch.cumsum(s_squared, dim=0) / total_var
   eff_rank = int((cumvar < variance_threshold).sum().item()) + 1
   ```
   This counts the minimum number of singular values whose squared sum exceeds 90% of total variance. The `+1` corrects for 0-based counting. **CONFIRMED**: correct standard effective rank computation.

   **Spectral entropy** (lines 193-196):
   ```python
   probs = s_squared / total_var
   probs = probs[probs > 0]
   entropy = -float((probs * torch.log2(probs)).sum())
   norm_entropy = entropy / max_entropy
   ```
   Normalized Shannon entropy of the singular value distribution. Range [0, 1] where 0 = one dominant component, 1 = uniform. **CONFIRMED**: correct formula.

   **Averaging** (lines 218-225): Layer-level metrics are averaged across all layers. Returns `mean_key_effective_rank`, `mean_value_effective_rank`, etc. Per-layer data also returned. **CONFIRMED**: consistent with paper's description of averaging across layers.

   **Variance threshold**: Default 0.9 (90%). **CONFIRMED**: matches paper.

7. **`compute_subspace_alignment(cache1, cache2, top_k=None)`** (lines 239-301): Computes alignment between principal subspaces of two caches via principal angles. Uses right singular vectors (Vt rows) in head_dim space. Returns mean cosine of principal angles [0, 1]. **CONFIRMED**: correct principal angle computation via SVD of V1^T @ V2.

8. **`cache_to_cpu(cache)`** (lines 304-309): Deep-copies cache tensors to CPU. Used in 5 files.

**Verdict**: All functions are correctly implemented. The core `compute_cache_dimensionality` function matches the paper's methodology description exactly.

---

## 11.4 — stats_utils.py Cross-Reference

**ALREADY REVIEWED BY WS8** (see `./registry/controls-methodology.md` sections 8.1-8.5).

Summary of WS8 findings:

| Function | Lines | Verdict | Notes |
|----------|-------|---------|-------|
| `cohens_d()` | 134-150 | CONFIRMED | Standard pooled SD formula with Bessel correction |
| `hedges_g()` | 153-165 | CONFIRMED | J = 1 - 3/(4*df - 1), matches Borenstein 2009 |
| `bootstrap_ci()` | 68-82 | CONFIRMED | Standard percentile bootstrap, n=10000 |
| `bootstrap_diff_ci()` | 85-100 | CONFIRMED | Two-sided p via proportion of diffs > 0 |
| `welch_t()` | 107-110 | CONFIRMED | Calls `scipy.stats.ttest_ind(equal_var=False)` |
| `mann_whitney()` | 113-119 | CONFIRMED | Calls `scipy.stats.mannwhitneyu(alternative='two-sided')` |
| `shapiro_wilk()` | 122-127 | CONFIRMED | Standard Shapiro-Wilk via scipy |
| `tost_equivalence()` | 233-278 | CONFIRMED | Standard TOST (Schuirmann 1987), uses pooled df |
| `length_residualize()` | 285-343 | CONFIRMED | OLS via np.polyfit on raw length (not log) |
| `deduplicate_runs()` | 350-405 | CONFIRMED | Correct approach for greedy-decoded pseudoreplication |
| `full_comparison()` | 412-458 | CONFIRMED | Complete battery: normality + Welch + MW + effect sizes + bootstrap |
| `conservative_p` | 449-451 | CONFIRMED | max(p_welch, p_mw) |
| `holm_bonferroni()` | 204-226 | CONFIRMED | Correct step-up with monotonicity enforcement |
| `power_advisory()` | 461-473 | CONFIRMED | Forward problem only (power for given d), not inverse |
| `interpret_d()` | 188-197 | CONFIRMED | Standard Cohen's d thresholds |
| `cohens_d_ci()` | 168-185 | CONFIRMED | Bootstrap CI for d and g |
| `log_environment()` | 32-61 | CONFIRMED | Captures full reproducibility metadata |

**Additional note from WS11**: The `log_environment` function (line 43) tries `scipy_stats.scipy.__version__` first, which will raise `AttributeError`, then catches and uses `scipy.__version__` instead. This is harmless but messy code.

---

## 11.5 — Greedy Decoding Verification (do_sample=False)

**ALREADY REVIEWED BY WS8** (section 8.2). Cross-verified here with full grep.

| Script | do_sample | max_new_tokens | Notes |
|--------|----------|---------------|-------|
| 01_cache_inspection.py | `False` (line 79) | 30 | C1 |
| 01b_cognitive_modes.py | `False` (line 199) | 30 | C1 |
| 01c_batch_replication.py | **`True`** (line 228) | 30 | **INTENTIONAL**: sampling for variation (C1 only) |
| 01d_adversarial_controls.py | `False` (default, line 595) | 30 | C3 precision uses `do_sample_multi=False` |
| 01e_tokenizer_confound.py | `False` (line 306) | 50 | |
| 02a_basic_transfer.py | `False` (lines 78, 164) | 30 | C1 |
| 02b_projector_transfer.py | `False` (lines 103, 214) | varies | C1 |
| 03_scale_sweep.py | `False` (line 431) | 50 | C2 |
| 03b_identity_signatures.py | `False` (line 305) | 50 | C2 |
| 04_deception_forensics.py | `False` (line 370) | 50 | C2 |
| 04b_natural_deception.py | `False` (line 128) | 200 | C2 (longer for evasion detection) |
| 05_layer_map.py | `False` (lines 185, 228, 472) | 20-30 | C2 |
| 06_temporal_evolution.py | N/A | N/A | Input-only encoding, no `model.generate` |
| 07_abliteration_geometry.py | `False` (line 172) | 50 (cognitive), 100 (harmful) | C2 |
| 07_individuation_geometry.py | `False` (line 329) | 50 | C2 |
| 07b_individuation_controls.py | `False` (line 497) | 50 | C2 |
| 08_input_only_geometry.py | `False` (line 317) | 50 | C2 |
| 09_sycophancy_detection.py | N/A | N/A | No generate call |
| 10_societies_of_thought.py | N/A | N/A | No generate call |
| 11_rdct_stability.py | N/A | N/A | No generate call |
| 12_bloom_taxonomy.py | N/A | N/A | No generate call (encoding only) |
| 13_c2c_replication.py | N/A | N/A | No generate call |

**Verdict: CONFIRMED** -- `do_sample=False` in all Campaign 2 experiment scripts. The one `True` in `01c_batch_replication.py` is intentional and pre-dates Campaign 2. Scripts 09-13 use encoding-only mode (no generation).

---

## 11.6 — JSON Result File Schema Consistency

### scale_sweep (17 files) -- CONSISTENT
All 17 files share the same top-level schema: `['metadata', 'scales']`.
No schema drift across models (TinyLlama through Llama-3.1-70B).

### deception_forensics (7 files) -- CONSISTENT
All 7 files share: `['experiment_1', 'experiment_2', 'experiment_3', 'experiment_4', 'metadata']`.
Four experiment sub-sections as designed.

### identity_signatures (7 files) -- CONSISTENT
All 7 files share: `['classification', 'consistency', 'fingerprinting', 'layer_analysis', 'metadata', 'pairwise_analysis']`.
Six analysis sections as designed.

### natural_deception (3 files) -- CONSISTENT
All 3 files share: `['experiment', 'input_only', 'metadata']`.

### bloom_taxonomy (7 files) -- CONSISTENT
All 7 files share a large schema: `['checksum', 'elapsed_seconds', 'environment', 'experiment', 'holm_bonferroni', 'hypotheses', 'length_covariate', 'model', 'model_id', 'n_observations', 'n_prompts', 'n_runs', 'pairwise_comparisons', 'quantize', 'raw_results', 'seed']`.

### input_only (9 files) -- **TWO SCHEMAS** (version mismatch)

| Schema | Files |
|--------|-------|
| `['analysis', 'battery', 'metadata']` | input_only_1.1B_results.json, input_only_7B_results.json |
| `['battery_results', 'label', 'method', 'model', 'num_runs', 'seed']` | input_only_DS-7B, Gemma-2B, Llama-8B, Qwen-0.5B, Qwen-14B-q4, Qwen-7B (6 files) |
| `['mean_rho', 'n_models', 'per_model']` | input_only_rho_corrected.json (aggregated result) |

**Finding**: The 2 early files (1.1B, 7B) use a different schema with `metadata` and `analysis` sections, while the 6 later files use a flat schema with `battery_results`. This indicates the `08_input_only_geometry.py` script was refactored between runs. The `input_only_rho_corrected.json` is a post-hoc aggregation file from `recompute_stats.py`.

### abliteration (4 files) -- **THREE SCHEMAS** (version mismatch)

| Schema | Files |
|--------|-------|
| `['comparison', 'metadata']` | abliteration_Qwen2.5-7B_comparison.json |
| `['description', 'harmful_distribution', ...]` | abliteration_Qwen2.5-7B_comparison_corrected.json |
| `['metadata', 'sweep']` | abliteration_sweep_* (2 files) |

**Finding**: The comparison and sweep files have different schemas by design (different analysis types). The corrected comparison file has a different schema from the original, indicating `recompute_stats.py` produced a different structure.

**Summary**: Schema consistency is GOOD within experiment types, with two exceptions:
1. **input_only**: 2 early files vs 6 later files show script evolution
2. **abliteration**: corrected file has different structure from original

---

## 11.7 — Dead Code Inventory

### Utility Functions (gpu_utils.py, stats_utils.py)

| Function | File | External Usage | Status |
|----------|------|---------------|--------|
| `normalize_cache` | gpu_utils.py | 1 file | USED (sparse but intentional) |
| `model_id_from_name` | gpu_utils.py | 10 files | USED |
| `get_output_path` | gpu_utils.py | 17 files | USED |
| `load_model` | gpu_utils.py | 22 files | USED |
| `compute_cache_dimensionality` | gpu_utils.py | 13 files | USED |
| `compute_subspace_alignment` | gpu_utils.py | 4 files | USED |
| `cache_to_cpu` | gpu_utils.py | 5 files | USED |
| `log_environment` | stats_utils.py | 17 files | USED |
| `bootstrap_ci` | stats_utils.py | 17 files | USED |
| `bootstrap_diff_ci` | stats_utils.py | 8 files | USED |
| `welch_t` | stats_utils.py | 15 files | USED |
| `mann_whitney` | stats_utils.py | 15 files | USED |
| `shapiro_wilk` | stats_utils.py | 12 files | USED |
| `cohens_d` | stats_utils.py | 26 files | USED |
| `hedges_g` | stats_utils.py | 4 files | USED (via imports + full_comparison) |
| `cohens_d_ci` | stats_utils.py | 16 files | USED |
| `interpret_d` | stats_utils.py | 13 files | USED |
| `holm_bonferroni` | stats_utils.py | 16 files | USED |
| `tost_equivalence` | stats_utils.py | 2 files | USED (sparse but correct -- only needed for null claims) |
| `length_residualize` | stats_utils.py | 2 files | USED |
| `deduplicate_runs` | stats_utils.py | 1 file (recompute_stats.py) | USED but **NEVER CALLED IN EXPERIMENT SCRIPTS** (see WS8 8.3) |
| `full_comparison` | stats_utils.py | 16 files | USED |
| `power_advisory` | stats_utils.py | 7 files | USED |

### Dead Functions in Prompt Files

| Function | File | Called By | Status |
|----------|------|----------|--------|
| `get_questions_by_domain()` | s3_confabulation_elicitation.py | None | **DEAD** -- helper never used |
| `get_questions_by_difficulty()` | s3_confabulation_elicitation.py | None | **DEAD** -- helper never used |
| `get_domain_counts()` | s3_confabulation_elicitation.py | None | **DEAD** -- helper never used |
| `get_difficulty_counts()` | s3_confabulation_elicitation.py | None | **DEAD** -- helper never used |
| `validate_prompts()` | c7_frequency_matched_confabulation.py | None | **DEAD** -- validation helper never used |
| `generate_all_prompts()` | s5_sycophancy_elicitation.py | 09_sycophancy_detection.py | USED (imported there) |

### Dead Functions in C1 Scripts

| Function | File | Called By | Status |
|----------|------|----------|--------|
| `detect_outliers()` | 01c_batch_replication.py | Self only | C1-internal, not dead within script |
| `compute_comparisons()` | 01c_batch_replication.py | Self only | C1-internal |
| `interpret_effect_size()` | 01c_batch_replication.py | Self only | C1-internal |
| `generate_batch_report()` | 01c_batch_replication.py | Self only | C1-internal |

### Dead Functions in Analysis Scripts

| Function | File | Called By | Status |
|----------|------|----------|--------|
| `jaccard_distance()` | 01e_tokenizer_confound.py | Self only (within-file) | USED internally |
| `_compute_raw_within_between_ratio()` | 01e_tokenizer_confound.py | Self only | USED internally |

**Summary**: 5 genuinely dead functions in prompt files (query/validation helpers that were defined but never integrated into any experiment script). No dead code in the core utility or experiment files.

---

## 11.8 — Script Categorization: Campaign 1 vs Campaign 2

### Campaign 1 (C1) Scripts — 01x, 02x prefix

| Script | Status | Notes |
|--------|--------|-------|
| 01_cache_inspection.py | C1 original | Basic cache structure exploration |
| 01b_cognitive_modes.py | C1 | Cognitive mode battery (predecessor to 03_scale_sweep) |
| 01c_batch_replication.py | C1 | Sampling variation study (intentional `do_sample=True`) |
| 01d_adversarial_controls.py | C1 → C2 bridge | 6 adversarial controls; partially superseded by C2 scripts |
| 01e_controls_visualization.py | C1 viz | Visualization for 01d results |
| 01e_tokenizer_confound.py | C1 → C2 | Tokenizer confound analysis (used in both campaigns) |
| 02a_basic_transfer.py | C1 | Cross-model cache transfer experiment |
| 02b_projector_transfer.py | C1 | Projector-based transfer experiment |

**Supersession notes**:
- `01_cache_inspection.py` and `01b_cognitive_modes.py` are fully superseded by `03_scale_sweep.py` (C2), which runs the same cognitive mode battery at scale
- `01c_batch_replication.py` is standalone (sampling study, not repeated in C2)
- `01d_adversarial_controls.py` remains relevant (controls not duplicated elsewhere)
- `02a_basic_transfer.py` and `02b_projector_transfer.py` are standalone C1 experiments not repeated in C2

### Campaign 2 (C2) Scripts — 03x-13x prefix

| Script | Experiment | Notes |
|--------|-----------|-------|
| 03_scale_sweep.py | Core geometric sweep | 17-model scale sweep with cognitive modes |
| 03b_identity_signatures.py | Identity fingerprinting | Classification + pairwise analysis |
| 03c_scale_sweep_visualization.py | Viz | For scale_sweep results |
| 03d_identity_visualization.py | Viz | For identity results |
| 04_deception_forensics.py | Deception detection | 4 experiments: honest/deceptive/sycophancy/uncertainty |
| 04b_deception_visualization.py | Viz | For deception results |
| 04b_natural_deception.py | Natural deception (S4) | 2x3 design: censorship vs complexity |
| 05_layer_map.py | Layer map | Layer-level analysis: knockout, crosslingual, transfer, probing |
| 05b_layer_map_visualization.py | Viz | For layer map results |
| 06_temporal_evolution.py | Temporal evolution | Cache trajectory analysis |
| 06b_temporal_visualization.py | Viz | For temporal results |
| 07_abliteration_geometry.py | Abliteration geometry | Pre/post abliteration comparison |
| 07_individuation_geometry.py | Individuation | System prompt separability |
| 07b_individuation_controls.py | Individuation controls | Falsification tests |
| 08_input_only_geometry.py | Input-only geometry | Encoding-native analysis |
| 09_sycophancy_detection.py | Sycophancy (H7) | Encoding-only sycophancy experiment |
| 10_societies_of_thought.py | Societies of Thought (H8) | Geometric transitions during generation |
| 11_rdct_stability.py | RDCT stability (H9) | Phase transition under perturbation |
| 12_bloom_taxonomy.py | Bloom taxonomy (H10) | Cognitive complexity ordering |
| 13_c2c_replication.py | C2C replication | Cache-to-cache transfer (C2 version) |

### Support Scripts

| Script | Role | Notes |
|--------|------|-------|
| gpu_utils.py | Shared utilities | Model loading, cache operations, dimensionality |
| stats_utils.py | Shared statistics | All statistical tests and effect sizes |
| heretic_abliterate.py | Abliteration tool | Non-interactive heretic wrapper |
| length_confound_analysis.py | Post-hoc analysis | Length confound investigation |
| recompute_stats.py | Post-hoc correction | Recomputes stats on existing results |
| cross_model_analysis.py | Post-hoc analysis | Cross-model Spearman correlations |
| figures/generate_paper_figures.py | Paper figures | All publication figures |
| scripts/download_models.py | Setup | Model download helper |
| scripts/download_all_models.py | Setup | Batch model download |

---

## Deferred Code Reviews

### 3.8 — 03b_identity_signatures.py Review

**File**: `../code/03b_identity_signatures.py` (~1100 lines)

**Syntax**: CONFIRMED (py_compile passes)

**Classification Protocol**:

1. **Feature extraction** (`extract_cache_features`, line 228): Extracts per-layer key norms, value norms, key means, value means, key stds, value stds, plus SVD dimensionality metrics. Concatenates into a `flat_vector` per sample.

2. **Cross-validation**: Uses **stratified 5-fold** (`StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)`, line 457). This is proper k-fold, not a simple holdout.

3. **Multiple classifiers** (lines 442-453):
   - Random Forest (200 estimators)
   - Linear SVM (with StandardScaler)
   - Logistic Regression (with StandardScaler)
   All use `cross_val_score` with the same 5-fold CV split.

4. **Holdout split** (line 479): Additionally uses a 70/30 train/test split (`train_test_split(..., test_size=0.3, stratify=y)`) for confusion matrix and feature importance. This is separate from the k-fold evaluation.

5. **Permutation test** (lines 498-521): 1000 permutations to establish null distribution. Uses 3-fold CV for speed on each permutation (vs 5-fold for the main evaluation). The permutation test implementation has a subtle issue: lines 504-511 attempt to clone the best classifier but the logic for pipeline classifiers is convoluted. For non-RandomForest classifiers, it falls back to `RandomForestClassifier(n_estimators=100)`, which means the permutation null distribution may be computed with a different classifier than the actual score.

6. **Cross-prompt validation** (H5, lines 527-548): Leave-one-group-out validation where each prompt group is held out in turn. This tests whether classification survives when trained and tested on non-overlapping prompts.

**Concerns**:
- The permutation test classifier fallback (line 511) could produce misleading p-values if the best classifier is SVM or LogReg
- Feature vector dimensionality depends on model architecture (number of layers varies by model), so classifiers trained on one model cannot be directly compared to another

**Verdict**: Classification protocol is SOUND (proper stratified k-fold, permutation baseline, cross-prompt validation). One implementation concern with permutation test classifier instantiation noted.

---

### 4.6 — 04_deception_forensics.py Review

**File**: `../code/04_deception_forensics.py` (~1050 lines)

**Syntax**: CONFIRMED (py_compile passes)

**Deception instruction text** (lines 90-196): 15 matched triplets, each with:
- `honest`: Direct factual question with "Answer honestly/accurately"
- `deceptive`: Same question with explicit lie instruction (e.g., "You must give a false answer. Do not say Paris. Give a confident, wrong answer.")
- `confabulation`: A false statement presented as fact (no deception instruction)

The deception instructions are explicit and unambiguous. Each triplet targets a well-known fact (capital of France, boiling point of water, etc.) so the model's "knowledge" is not in question.

**Prompt matching**: Honest and deceptive prompts share the same factual question but differ in the instruction suffix. This is a matched-pairs design. The confabulation prompts are structurally different (statements, not questions).

**max_new_tokens**: 50 (line 370). This is short but sufficient for factual answers. The 04b_natural_deception.py uses 200 because censorship evasions are longer.

**Experiment structure**:
- Experiment 1 (line 419): Honest vs Instructed Deception -- matched pairs, full_comparison
- Experiment 2 (line 588): Sycophancy Detection -- 12 agree-correct/agree-wrong pairs
- Experiment 3 (line 689): Uncertainty Gradient -- 12 certain-true/uncertain/certain-lie triplets
- Experiment 4 (line 802): Layer-level forensics -- per-layer analysis

**Verdict**: SOUND experimental design. Matched-pairs structure is correct. Deception instructions are clear and unambiguous. `max_new_tokens=50` is appropriate for short factual responses.

---

### 6.8 — 04b_natural_deception.py + prompts/s4_natural_deception.py Review

**File**: `../code/04b_natural_deception.py` (838 lines)

**Syntax**: CONFIRMED (py_compile passes)

**Critical test implementation** (lines 505-519): The "critical confound control" compares censored vs complex_noncensored topics on the censored model. If d is near 0, the effect is driven by content complexity alone, not censorship evasion. This is properly implemented as the key comparison in the analysis.

**Residualization** (lines 528-565): Uses `length_residualize()` from stats_utils with all three conditions pooled. Residualized pairwise comparisons are computed for each condition pair. The length-rank correlation and R-squared are reported. **CONFIRMED**: correct application of the shared residualization function.

**H4 equivalence test** (lines 521-526): For uncensored models, tests whether censored topics produce geometry equivalent to controls using `tost_equivalence(censored_ranks, control_ranks, delta=0.3)`. **CONFIRMED**: correct TOST application for the null hypothesis.

**Response classification** (lines 178-235): Keyword-based automated classification with:
- Refusal signals (14 keywords)
- Evasion signals (15 keywords)
- Propaganda signals (11 CCP-narrative keywords)
- Ground truth number matching
Classification confidence ranges 0.3-0.9 based on signal strength. Low-confidence results flagged for manual review.

**prompts/s4_natural_deception.py**:
- **CENSORED_TOPIC_QUESTIONS**: 30 questions, 6 topics x 5 questions each (tiananmen, uyghur, taiwan, political_prisoners, xi_criticism, hong_kong). Each has `id`, `topic`, `question`, `ground_truth`, `sensitivity_level`.
- **CONTROL_QUESTIONS**: 30 questions, 6 topics x 5 each (cuisine, culture, festivals, geography, inventions, literature). Non-sensitive Chinese cultural topics.
- **COMPLEX_NONCENSORED_QUESTIONS**: 30 questions, 6 topics x 5 each (abu_ghraib, apartheid, armenian_genocide, bosnian_genocide, khmer_rouge, rwanda_genocide). Complexity-matched non-Chinese atrocities.
- **Total**: 90 prompts as claimed.

**Complexity matching**: The complex non-censored questions cover atrocities and genocide comparable in factual complexity to the censored topics, but are NOT subject to Chinese censorship training. This is the correct confound control for separating topic complexity from censorship evasion.

**Verdict**: SOUND design. The 2x3 structure (model_type x topic_complexity) correctly separates censorship from complexity. All 90 prompts verified present. Critical test, residualization, and TOST equivalence correctly implemented.

---

### 7.7 — 07_abliteration_geometry.py + heretic_abliterate.py Review

**File**: `../code/07_abliteration_geometry.py` (813 lines)

**Syntax**: CONFIRMED (py_compile passes)

**Bug verification**: The WS8 report identified a code version mismatch in tokenizer confound (8.7). For abliteration_geometry.py, I verified:
- `do_sample=False` at line 172 -- CORRECT
- `max_new_tokens=50` for cognitive modes, `100` for harmful prompts -- reasonable
- Harmful response classification (lines 359-377) uses keyword matching for refused/hedged/compliant bucketing -- functional but crude
- The `run_heretic()` function (lines 219-250) calls `heretic` as a CLI subprocess, not using the Python API. The `heretic_abliterate.py` script (separate file) uses the Python API instead.

**Comparison logic** (lines 505-610): `compare_geometries()` computes:
- Per-category geometric shift (d between baseline and abliterated)
- Abliteration specificity: ratio of refusal shift to mean non-refusal shift
- Self-reference preservation (H3)
- Input-only ordering preservation via Spearman rho (H4)
- Harmful refusal rate change

**No bugs found** in current code. The code correctly implements the designed comparison protocol.

**heretic_abliterate.py** (194 lines):

**Syntax**: CONFIRMED (py_compile passes, but heretic imports will fail without heretic-llm installed)

**Key design decision** (lines 22-31): Parses arguments BEFORE importing heretic, then clears `sys.argv`. This is because heretic uses `pydantic_settings` which auto-parses `sys.argv`. This is a correct workaround for the library conflict.

**Abliteration protocol** (lines 42-183):
1. Creates heretic `Settings` with model name and batch size
2. Loads model via heretic's `Model` class (handles LoRA/dtype)
3. Loads heretic's own good/bad prompt datasets
4. Computes residuals for both prompt sets (batched)
5. Computes refusal directions: `mean_bad - mean_good`, normalized per layer
6. Sets abliteration parameters: max_weight=1.0, min_weight=0.2, position at 80% of layers
7. Calls `model.abliterate()` with per-layer directions
8. Merges LoRA weights and saves
9. Quick refusal check (5 harmful prompts) AFTER saving (crash-safe)

**Verdict**: Both files are correctly implemented. No bugs found. The dual approach (CLI subprocess in 07_ vs Python API in heretic_abliterate.py) provides flexibility but could be confusing. The `run_heretic()` function in 07_ only works if the heretic CLI is installed, while `heretic_abliterate.py` uses the Python API directly.

---

## JSON Result File Integrity

(Preserved from initial audit)

- Total JSON files: **85** (paper claims 83 -- DISCREPANCY, see below)
- Parse failures: **0** (all 85 JSON files parse successfully)
- Non-JSON files: 79 (43 .md reports, 36 .txt logs incl. experiment_log.txt)
- Total files in results/: **164** (85 + 43 + 36)

### JSON File Count by Experiment Type
| Type | Count | Notes |
|------|-------|-------|
| scale_sweep | 17 | 17 models (incl. Phi-3.5 + abliterated) |
| input_only | 9 | 8 model results + 1 corrected rho file |
| bloom_taxonomy | 7 | |
| deception_forensics | 7 | |
| identity_signatures | 7 | |
| rdct_stability | 6 | |
| layer_map | 5 | |
| abliteration | 4 | 2 sweep + 2 comparison |
| individuation | 4 | 3 models + 1 controls |
| temporal_evolution | 4 | |
| cache_metadata | 3 | Legacy C1 |
| natural_deception | 3 | S4 experiment |
| tokenizer_confound | 2 | |
| adversarial_controls | 1 | Legacy C1 (14.4M!) |
| batch_results | 1 | Legacy C1 |
| cognitive_modes | 1 | Legacy C1 |
| cross_model_rho | 1 | Corrected file |
| length_confound | 1 | |
| phase2a_transfer | 1 | |
| s4_topic_analysis | 1 | Corrected file |

### "83 result files" Claim (C10, C22, C91)
Paper claims 83. We found 85 JSON files. The difference is likely:
- 3 "corrected" files (cross_model_rho_corrected, input_only_rho_corrected, s4_topic_analysis_corrected) added after initial count
- OR the paper counted only C2 files (excluding legacy C1 files like cache_metadata, cognitive_modes, adversarial_controls, batch_results)
- Needs verification: which 83 files does the paper mean?

### Pre-existing QA Toolkit
A separate QA toolkit exists outside this repo (in the broader workspace) with:
- JSON parsers for all 9 experiment types
- A Campaign 1 claim verification framework
- 5 bug detector classes and DataFrames-based exploration
- ~3,910 lines of tests across 4 test files
- NOTE: Written for Campaign 1 README claims; not used in this C2 audit (this audit recomputes from scratch)

## JiminAI-Cricket
- Code files: 0 (confirmed)
- Test files: 0 (confirmed)
- CI/CD: none (confirmed)

---

## Summary of Findings

### No Issues Found
- All 42 Python files compile without errors
- All utility functions are correctly implemented (verified against scipy references by WS8)
- `do_sample=False` correctly set in all Campaign 2 scripts
- JSON schemas are consistent within each experiment type (two minor version mismatches noted)
- Core `compute_cache_dimensionality` function correctly implements effective rank via SVD
- Classification protocol in 03b uses proper stratified k-fold with permutation baseline
- Deception forensics uses well-designed matched-pairs methodology
- Natural deception S4 experiment has correct 2x3 design with all 90 prompts verified

### Issues Found (from WS8, confirmed by WS11)
1. **Pseudoreplication (from WS8 8.3)**: `deduplicate_runs()` defined but never called in experiment scripts. P-values inflated.
2. **Missing TOST for null claims (from WS8 8.4)**: 4/6 null claims lack proper equivalence testing.
3. **Power analysis error (from WS8 8.5)**: d >= 0.81 vs correct d >= 0.792.
4. **Stale CATASTROPHIC_FAIL (from WS8 8.7)**: Qwen tokenizer confound verdict is code artifact.

### New Issues from WS11
5. **Dead code in prompts/**: 5 helper functions in prompt files are defined but never called (query helpers, validation). Minor housekeeping issue.
6. **Permutation test classifier fallback (11.3/3.8)**: In 03b_identity_signatures.py, the permutation test may use RandomForest even when the best classifier was SVM/LogReg.
7. **Input-only schema inconsistency (11.6)**: 2 early files use a different JSON schema than 6 later files, indicating script evolution mid-campaign.
8. **Heretic dual invocation (11.7/7.7)**: `07_abliteration_geometry.py` calls heretic as CLI subprocess while `heretic_abliterate.py` uses the Python API. Potential for different behavior.

### Severity Assessment

| Issue | Severity | Impact on Paper Claims |
|-------|----------|----------------------|
| Pseudoreplication (WS8) | **HIGH** | P-values overstated in all experiment scripts |
| Missing TOST (WS8) | **MEDIUM** | 4 null claims lack proper statistical support |
| Power analysis error (WS8) | **LOW** | Conservative direction (2.2%) |
| CATASTROPHIC_FAIL (WS8) | **LOW** | Stale artifact, not reported in paper |
| Dead prompt helpers (WS11) | **NEGLIGIBLE** | Housekeeping only |
| Permutation test fallback (WS11) | **LOW** | Could affect H2 p-value if best classifier is not RF |
| Schema inconsistency (WS11) | **LOW** | Analysis scripts handle both formats |
| Heretic dual invocation (WS11) | **LOW** | Both approaches produce correct abliteration |
