# Dead Code & Script Correctness Analysis

**Analyzed**: All 22 Python scripts, 2 shell scripts, shared utilities, and results files.

---

## Category 1: Dead / Superseded Scripts (Safe to Remove)

These scripts were part of early exploration phases and have been fully superseded by later, more rigorous scripts. They are never referenced by the campaign runner (`cassidy_full_run.sh`), produce no results used by any other script, and represent dead code paths.

### `01_cache_inspection.py`
- **Status**: Dead code
- **Reason**: Phase 1 prototype. Hardcoded to TinyLlama with 4-bit quantization. Dumps cache metadata (shapes, dtypes) — pure structural inspection with no statistical analysis. Fully superseded by `01b_cognitive_modes.py` which runs the same model with a categorized prompt battery and computes cache statistics.
- **Uses relative path**: `Path("../results")` — breaks if run from any directory other than `code/`.

### `01b_cognitive_modes.py`
- **Status**: Dead code
- **Reason**: Phase 1.5 prototype. Hardcoded to TinyLlama 4-bit. Computes only norm/variance statistics (no SVD dimensionality, no statistical tests, no effect sizes). Fully superseded by `01c_batch_replication.py` (adds multi-run replication) and then by `03_scale_sweep.py` (adds all statistical infrastructure, SVD dimensionality, multi-model support).
- **Uses relative path**: `Path("../results")` — fragile.

### `01c_batch_replication.py`
- **Status**: Dead code
- **Reason**: Phase 1.5 batch runner. Hardcoded to TinyLlama 4-bit. Uses `do_sample=True` with `temperature=0.7` — introduces sampling variance that makes results non-reproducible (no seed control). The Cohen's d calculation in `compute_comparisons()` uses `pooled_std = sqrt((s1^2 + s2^2) / 2)` which is the average-of-variances formula, not the correct pooled-SD formula weighted by sample size. Fully superseded by `03_scale_sweep.py` which has correct Cohen's d and deterministic generation.
- **Uses relative path**: `Path("../results")` — fragile.
- **Existing results**: `batch_results_30runs.json` exists but is not consumed by any other script.

### `02a_basic_transfer.py`
- **Status**: Dead code (exploratory, no campaign results)
- **Reason**: Phase 2a cache injection experiment. Tests raw KV-cache transplantation between prompts. This was an early exploration of whether cache injection works — it largely doesn't (by design, this is expected). Fully superseded by `02b_projector_transfer.py` which adds the C2C projector-mediated approach.
- **Produces**: `phase2a_transfer_results.json` — exists in results, but not consumed by any downstream analysis or paper figure.
- **Uses relative path**: `Path("../results")`.

### `02b_projector_transfer.py`
- **Status**: Dead code (requires external dependency, never successfully run)
- **Reason**: Requires trained C2C projector checkpoints that don't exist in the repo. The `load_projectors()` function loads `projector_{idx}.pt` files from a checkpoint directory, but no such files exist. The `apply_projector()` function calls `projector(source_kv=..., target_kv=...)` on the loaded checkpoint, but the expected API of these projector objects is never defined — it assumes an external C2C framework provides the module. The shell script `phase2b_quickstart.sh` references the C2C training pipeline but it was never completed.
- **Uses relative path**: `Path("../results")`.
- **`cassidy_full_run.sh` Phase G**: Calls this script but notes "Medium and large projector configs need manual setup."

---

## Category 2: Visualization Scripts with Broken Default Paths

These scripts are functional but will fail when run with default arguments because they look for result files with names that don't match what the experiment scripts actually produce.

### `01e_controls_visualization.py`
- **Expected file**: `results/adversarial_controls_results.json`
- **Actual file**: `results/adversarial_controls_TinyLlama-1.1B_results.json`
- **Cause**: `01d_adversarial_controls.py` uses `get_output_path()` from `gpu_utils.py` which appends the model ID to the filename. The visualization script was written before this naming convention was adopted.
- **Fix**: Change default path to match actual output, or add model-ID-aware file discovery.

### `03c_scale_sweep_visualization.py`
- **Expected file**: `results/scale_sweep_results.json` (single aggregate file)
- **Actual files**: `results/scale_sweep_Qwen2.5-0.5B_results.json`, `results/scale_sweep_TinyLlama-1.1B_results.json`, etc. (per-model files)
- **Cause**: The visualization script was designed expecting a single combined results file, but `03_scale_sweep.py` produces per-model output files. The visualization script's `fig1_scale_heatmap()` function hardcodes individual model filenames (line 36-44) which correctly matches the per-model pattern, contradicting its own default `--results` argument.
- **The script has TWO loading strategies that conflict**: The `--results` default expects `scale_sweep_results.json`, but `fig1_scale_heatmap()` loads individual per-model files by name. The former will fail; the latter will work if the per-model files exist.

### `03d_identity_visualization.py`
- **Expected file**: `results/identity_signatures_results.json`
- **Actual files**: `results/identity_signatures_TinyLlama-1.1B_results.json`, `results/identity_signatures_Qwen2.5-7B_results.json`, etc.
- **Same cause**: Naming mismatch due to `get_output_path()` adding model IDs.

---

## Category 3: Shell Script CLI Argument Mismatches

### `phase2b_quickstart.sh`
- **Line 278**: Calls `03_scale_sweep.py` with `--num-runs 5` — but `03_scale_sweep.py` accepts `--runs`, not `--num-runs`. This will cause an **unrecognized argument error** and the script will fail.
- **Line 289**: Same `--num-runs` mismatch.
- **Line 301**: Same `--num-runs` mismatch.
- **Line 323**: Calls `03b_identity_signatures.py` with `--num-samples 10` — but `03b_identity_signatures.py` accepts `--runs`, not `--num-samples`. This will also **fail**.
- **Impact**: `phase2b_quickstart.sh` is completely non-functional for the scale sweep and identity steps. Only the preflight, train, and validate steps would work.

### `cassidy_full_run.sh`
- **Functional**: Uses correct `--runs` argument throughout. Works correctly.
- **Phase G**: Calls `02b_projector_transfer.py` without `--checkpoint` argument, which is required (`required=True`). This will fail with: `error: the following arguments are required: --checkpoint`. However, Phase G is documented as needing manual setup.

---

## Category 4: Duplicated Code (Not Dead, But Technical Debt)

### Duplicated `load_model()` functions
The following scripts define their own `load_model()` that duplicates `gpu_utils.load_model()`:
- `03_scale_sweep.py:570` — local `load_model()` (identical to gpu_utils version)
- `03b_identity_signatures.py:385` — local `load_model()`
- `04_deception_forensics.py:513` — local `load_model()`
- `05_layer_map.py:180` — local `load_model()`
- `06_temporal_evolution.py:201` — local `load_model()`

Scripts `07_individuation_geometry.py`, `07b_individuation_controls.py`, and `08_input_only_geometry.py` correctly import `load_model` from `gpu_utils`. The `gpu_utils.py` docstring says "Used by: 01d, 03, 03b, 04, 05, 06" — but 03, 03b, 04, 05, 06 only import `get_output_path` and `model_id_from_name` from it, not `load_model`.

### Duplicated statistical functions
Every experiment script (01d, 03, 03b, 04, 05, 06, 07, 07b, 08) contains its own copy of:
- `bootstrap_ci()`, `bootstrap_diff_ci()`, `welch_t()`, `mann_whitney()`, `shapiro_wilk()`, `cohens_d()`, `cohens_d_ci()`, `interpret_d()`, `holm_bonferroni()`, `full_comparison()`

These are identical across scripts. They should live in `gpu_utils.py` (or a separate `stats_utils.py`). This is not dead code — all copies are used — but it's a maintenance hazard where a bug fix in one script wouldn't propagate to others.

### Duplicated `log_environment()` functions
All experiment scripts (01d through 08) define their own `log_environment()` with minor variations. This could be centralized.

---

## Category 5: Functional Issues in Active Scripts

### `01d_adversarial_controls.py` — Control 4 (Semantic Gradient)
- **Issue**: Control 4 performs cache injection (transplanting a source cache into a target prompt) using the same manual generation loop as `02a_basic_transfer.py`. The experiment generates from a source prompt, captures the cache, then generates from a target prompt with the injected cache. It then checks for "marker" words.
- **Problem**: Cache injection is known not to work reliably (this is what the entire Phase 2a established). Control 4 is measuring whether injected caches carry semantic gradient — but the signal it's measuring may be noise, since raw cache injection doesn't preserve semantics. The control's validity depends on injection actually working.
- **Result data**: The experiment ran and produced results. The `gradient_tau` Kendall's tau measures whether transfer success degrades with semantic distance, which is valid as a relative comparison even if absolute transfer rates are low.

### `05_layer_map.py` — Experiment 3 (Selective Layer Transfer)
- **Issue**: Same cache injection approach — transfers subsets of cache layers between prompts and measures generation quality. The experiment zeros out non-selected layers and measures whether semantically-important layers preserve more information.
- **Problem**: Same as above. Cache injection with partial layers is even more fragile than full-cache injection. Results may reflect positional encoding corruption rather than semantic layer importance.

### `gpu_utils.py` — `model_id_from_name()` hardcoded TinyLlama case
- **Line 33-34**: `if name.startswith("TinyLlama-"): name = "TinyLlama-1.1B"` — This hardcodes the assumption that any TinyLlama model is 1.1B. If a different TinyLlama variant were used, the ID would be wrong. Minor issue given current usage.

---

## Category 6: Scripts Not Listed in README's Experiment Table

The README's "Experiment Scripts" table lists scripts `01d`, `03`, `03b`, `04`, `05`, `06`, `07`, `07b`, `08`. The following scripts exist but are not listed:

- `01_cache_inspection.py` — Dead (Phase 1 prototype)
- `01b_cognitive_modes.py` — Dead (Phase 1.5 prototype)
- `01c_batch_replication.py` — Dead (Phase 1.5 batch runner)
- `01e_controls_visualization.py` — Visualization for 01d (functional, with path bug noted above)
- `02a_basic_transfer.py` — Dead (Phase 2a prototype)
- `02b_projector_transfer.py` — Dead (requires missing checkpoints)
- `03c_scale_sweep_visualization.py` — Visualization for 03 (functional, with path bug noted above)
- `03d_identity_visualization.py` — Visualization for 03b (functional, with path bug noted above)
- `04b_deception_visualization.py` — Visualization for 04 (functional)
- `05b_layer_map_visualization.py` — Visualization for 05 (functional)
- `06b_temporal_visualization.py` — Visualization for 06 (functional)

The visualization scripts (01e, 03c, 03d, 04b, 05b, 06b) are not dead — they serve a real purpose — but some have the filename mismatch issues described in Category 2.

---

## Summary

| Category | Count | Action |
|----------|-------|--------|
| Dead/superseded scripts | 5 | `01`, `01b`, `01c`, `02a`, `02b` — safe to remove or archive |
| Viz scripts with broken defaults | 3 | `01e`, `03c`, `03d` — fix default result paths |
| Shell script arg mismatches | 1 | `phase2b_quickstart.sh` — `--num-runs` / `--num-samples` should be `--runs` |
| Duplicated `load_model()` | 5 | `03`, `03b`, `04`, `05`, `06` — should use `gpu_utils.load_model()` |
| Duplicated stats functions | 9 | All experiment scripts — should be centralized |
| Cache injection experiments | 2 | `01d` Control 4, `05` Experiment 3 — results may not measure what's intended |

**Critical priority**: Fix `phase2b_quickstart.sh` argument names (`--num-runs` → `--runs`, `--num-samples` → `--runs`). This script is completely broken for its main experiment steps.

**High priority**: Fix visualization script default paths to match actual result filenames.

**Medium priority**: Clean up dead scripts (01, 01b, 01c, 02a, 02b) to avoid confusion.

**Low priority**: Deduplicate `load_model()` and statistical functions into shared utilities.
