# Dead Code Analysis Report

**Date**: 2026-02-20
**Scope**: All 23 Python files across `code/`, `scripts/`, and `figures/` directories
**Method**: AST-based analysis, cross-file call graph tracing, manual verification

---

## Executive Summary

| Category                      | Count | Severity |
|-------------------------------|-------|----------|
| Unused imports                | 35    | Low      |
| Dead functions                | 5     | Medium   |
| Unused variables              | 6     | Low      |
| Unused class definitions      | 1     | Low      |
| Unreachable code              | 0     | —        |
| Duplicated utility functions  | ~70+  | High     |

The codebase has no unreachable code and no abandoned feature blocks. The primary issues are (1) vestigial imports left after refactoring, (2) a handful of copy-pasted statistical functions that are never called in their host files, and (3) extensive duplication of statistical infrastructure across experiment scripts that could be centralized.

---

## 1. Unused Imports (35 instances across 16 files)

### 1.1 `import os` — 9 files

Vestigial import; `os` is never referenced in these files.

| File | Line |
|------|------|
| `code/01d_adversarial_controls.py` | 47 |
| `code/03_scale_sweep.py` | 50 |
| `code/03b_identity_signatures.py` | 55 |
| `code/04_deception_forensics.py` | 50 |
| `code/05_layer_map.py` | 39 |
| `code/07_individuation_geometry.py` | 55 |
| `code/07b_individuation_controls.py` | 60 |
| `code/08_input_only_geometry.py` | 51 |
| `figures/generate_paper_figures.py` | 5 |

### 1.2 `model_id_from_name` from `gpu_utils` — 6 files

Imported but never called directly. `get_output_path` calls it internally, making the explicit import redundant.

| File | Line |
|------|------|
| `code/03_scale_sweep.py` | 60 |
| `code/03b_identity_signatures.py` | 65 |
| `code/04_deception_forensics.py` | 60 |
| `code/05_layer_map.py` | 49 |
| `code/06_temporal_evolution.py` | 48 |
| `code/07_individuation_geometry.py` | 66 |

### 1.3 `List` from `typing` — 6 files

Imported but never used in type annotations.

| File | Line |
|------|------|
| `code/03_scale_sweep.py` | 58 |
| `code/03b_identity_signatures.py` | 63 |
| `code/04_deception_forensics.py` | 58 |
| `code/07_individuation_geometry.py` | 63 |
| `code/07b_individuation_controls.py` | 68 |
| `code/08_input_only_geometry.py` | 59 |

### 1.4 `Tuple` from `typing` — 4 files

| File | Line |
|------|------|
| `code/03b_identity_signatures.py` | 63 |
| `code/04_deception_forensics.py` | 58 |
| `code/05_layer_map.py` | 47 |
| `code/07_individuation_geometry.py` | 63 |

### 1.5 Other one-off unused imports

| File | Line | Import | Notes |
|------|------|--------|-------|
| `code/gpu_utils.py` | 9 | `import re` | String methods used instead of regex |
| `code/02a_basic_transfer.py` | 16 | `import numpy as np` | `np` never referenced |
| `code/02b_projector_transfer.py` | 21 | `asdict` from `dataclasses` | Never called |
| `code/03c_scale_sweep_visualization.py` | 37 | `FancyBboxPatch` from `matplotlib.patches` | Never referenced |
| `code/04b_deception_visualization.py` | 36 | `FancyBboxPatch` from `matplotlib.patches` | Never referenced |
| `code/07b_individuation_controls.py` | 70 | `get_output_path` from `gpu_utils` | Never called |
| `code/08_input_only_geometry.py` | 61 | `get_output_path` from `gpu_utils` | Never called |
| `figures/generate_paper_figures.py` | 10 | `matplotlib.gridspec as gridspec` | Never referenced |
| `scripts/download_models.py` | 19 | `import sys` | Never referenced |
| `scripts/download_models.py` | 20 | `import os` | Never referenced |
| `scripts/download_models.py` | 21 | `Path` from `pathlib` | Never referenced |

---

## 2. Dead Functions (5 instances)

All 5 dead functions result from copy-pasting a statistical infrastructure template where not all functions were needed by the host file.

### 2.1 `bootstrap_ci()` — dead in 4 files

This function computes single-group bootstrap confidence intervals. It is included in the template but only actually called in files that do direct bootstrapping (e.g., `01d`, `03`, `03b`, `06`). In the following files, it is never called:

| File | Line | Why dead |
|------|------|----------|
| `code/04_deception_forensics.py` | 108 | `full_comparison()` uses `bootstrap_diff_ci()` directly, not `bootstrap_ci()` |
| `code/05_layer_map.py` | 75 | Only `cohens_d()` is used; no `full_comparison()` exists |
| `code/07b_individuation_controls.py` | 123 | `full_comparison()` does not delegate to `bootstrap_ci()` |
| `code/08_input_only_geometry.py` | 114 | Same pattern as `07b` |

### 2.2 `interpret_d()` — dead in 1 file

| File | Line | Why dead |
|------|------|----------|
| `code/05_layer_map.py` | 93 | Only `cohens_d()` is called bare; no `full_comparison()` wrapper exists to invoke `interpret_d()` |

---

## 3. Unused Variables (6 instances)

| File | Line | Variable | Description |
|------|------|----------|-------------|
| `figures/generate_paper_figures.py` | 491 | `gen_d` | List of effect sizes assigned but never plotted; intended for a dual-axis figure that was never implemented |
| `figures/generate_paper_figures.py` | 93 | `im` | Return value of `sns.heatmap()` assigned but never used |
| `figures/generate_paper_figures.py` | 221, 223 | `bars1`, `bars2` | Return values of `ax.bar()` in `fig3_input_only()` assigned but never referenced |
| `figures/generate_paper_figures.py` | 303, 305 | `bars1`, `bars2` | Same pattern in `fig5_refusal()` |
| `code/03c_scale_sweep_visualization.py` | 487 | `bars` | Return value of `ax.bar()` assigned but never used |

---

## 4. Unused Class Definitions (1 instance)

| File | Line | Class | Description |
|------|------|-------|-------------|
| `code/02b_projector_transfer.py` | 29–39 | `TransferResult` | Dataclass copied from `02a_basic_transfer.py` but never instantiated. `run_comparison_experiment()` builds plain dicts instead. |

---

## 5. Unreachable Code

**None found.** No statements after unconditional `return`/`break`/`continue`, no `if False:` guards, no code gated by always-false conditions.

---

## 6. Redundant / Duplicated Code (~70+ copies of ~10 functions)

This is the most significant finding. The codebase uses a copy-paste pattern for statistical and utility infrastructure across experiment scripts. While not strictly "dead" (most copies are called), this creates a maintenance burden and accounts for an estimated 600+ redundant lines.

### 6.1 Statistical functions duplicated across experiment scripts

| Function | Copies | Files |
|----------|--------|-------|
| `bootstrap_ci()` | 9 | 01d, 03, 03b, 04, 05, 06, 07, 07b, 08 |
| `cohens_d()` | 9 | 01d, 03, 03b, 04, 05, 06, 07, 07b, 08 |
| `interpret_d()` | 8 | 01d, 03, 03b, 04, 05, 07, 07b, 08 |
| `log_environment()` | 9 | 01d, 03, 03b, 04, 05, 06, 07, 07b, 08 |
| `welch_t()` | 7 | 01d, 03, 03b, 04, 07, 07b, 08 |
| `mann_whitney()` | 7 | 01d, 03, 03b, 04, 07, 07b, 08 |
| `shapiro_wilk()` | 7 | 01d, 03, 03b, 04, 07, 07b, 08 |
| `cohens_d_ci()` | 7 | 01d, 03, 03b, 04, 07, 07b, 08 |
| `bootstrap_diff_ci()` | 4 | 01d, 03, 03b, 04 |
| `load_model()` | 6 | gpu_utils, 03, 03b, 04, 05, 06 (the later scripts 07, 07b, 08 properly import from gpu_utils) |

### 6.2 Transfer experiment duplication (02a ↔ 02b)

| Function/Constant | `02a_basic_transfer.py` | `02b_projector_transfer.py` |
|--------------------|-------------------------|----------------------------|
| `get_cache_from_prompt()` | line 67 | line 95 |
| `check_markers()` | line 202 | line 230 |
| `TEST_CASES` constant | line 341 | line 237 |

Additionally, `generate_with_injected_cache()` (02a, line 95) and `generate_with_cache()` (02b, line 154) are near-identical manual token-by-token generation loops.

### 6.3 Recommended consolidation

All duplicated statistical functions could be centralized into a shared `stats_utils.py` module (or added to the existing `gpu_utils.py`). The later experiment scripts (`07`, `07b`, `08`) already demonstrate the correct pattern by importing `load_model` from `gpu_utils` rather than re-defining it.

---

## 7. Files with Zero Issues

The following files are completely clean — no unused imports, no dead functions, no unused variables:

- `code/01_cache_inspection.py`
- `code/01b_cognitive_modes.py`
- `code/01c_batch_replication.py`
- `code/01e_controls_visualization.py`
- `code/03d_identity_visualization.py`
- `code/05b_layer_map_visualization.py`
- `code/06b_temporal_visualization.py`
