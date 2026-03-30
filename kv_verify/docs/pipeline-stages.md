# Pipeline Stages

## Stage Flow

```
0. environment    -> System info, model check, GPU detection
1. validation     -> Dataset quality gate (4 tiers)
2. prompt_gen     -> BLiMP-style minimal pairs (LLM or seed bank)
3. tokenization   -> Exact tokenizer validation
4. extraction     -> GPU inference + KV-cache features [GPU]
5. analysis       -> GroupKFold AUROC, FWL, permutation, bootstrap
6. falsification  -> Input-only AUROC, format baseline, confound checks
7. verdicts       -> Pre-registered criteria applied
8. report         -> Markdown report with global Holm correction
```

## Per-Stage Detail

### 0. Environment
- **Input**: PipelineConfig
- **Output**: System metadata (Python version, torch, CUDA, GPU name, VRAM)
- **GPU**: No

### 1. Validation (NEW)
- **Input**: Legacy datasets (hackathon JSONs) or generated prompts
- **Output**: DatasetReport per dataset, _validation_verdict
- **GPU**: No
- **Behavior**: FAIL halts pipeline unless --force. INCONCLUSIVE annotates downstream verdicts.

### 2. Prompt Generation
- **Input**: n_per_group, comparisons list
- **Output**: `output_dir/prompts/*.json` (PairSet format)
- **GPU**: Optional (LLM gap-fill mode uses GPU, seed bank fallback is CPU)
- **Two modes**: LLM gap-aware generation (GPU) or seed bank fallback (CPU, --skip-gpu)

### 3. Tokenization
- **Input**: Prompt JSON files
- **Output**: Token count validation, outlier list
- **GPU**: No
- **Uses**: Real tokenizer from lib/models.py (exact counts, not word-count proxies)

### 4. Extraction
- **Input**: Prompt JSON files + model
- **Output**: `output_dir/features/*.json`
- **GPU**: Yes (this is the only mandatory GPU stage)
- **Per-item caching**: Each item cached by content hash, interrupted runs resume
- **Remote**: When --remote is set, this stage runs on the remote machine

### 5. Analysis
- **Input**: Feature JSON files
- **Output**: `output_dir/results/{comparison}.json`
- **GPU**: No
- **Methods**: full_validation() = GroupKFold AUROC + bootstrap 95% CI + repeated CV + power analysis

### 6. Falsification
- **Input**: Feature JSON files
- **Output**: `output_dir/results/falsification.json`
- **GPU**: No
- **Tests**: Input-only AUROC, FWL residualization, format classifier baseline

### 7. Verdicts
- **Input**: Analysis + falsification results
- **Output**: `output_dir/results/verdicts.json`
- **GPU**: No
- **Uses**: CI lower bound (conservative) + actual power from analysis

### 8. Report
- **Input**: All results
- **Output**: `output_dir/final_report.md`
- **GPU**: No
- **Includes**: Global Holm-Bonferroni, per-claim verdicts, dataset quality annotations

## Caching

All stages use `@stage` decorator from `lib/tracking.py`. Stage results are cached to disk. Re-running a completed stage returns the cached result immediately. Delete `output_dir/cache/stage_{name}.json` to force re-run.

## Remote Execution

When `--remote config.yaml` is passed:
- Stages 0-1 and 5-8 run locally (CPU)
- Stage 4 (extraction) runs on the remote GPU via SSH + rsync
- Stage 2 with LLM mode also runs remotely
