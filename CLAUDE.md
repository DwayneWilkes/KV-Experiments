# CLAUDE.md

KV-Cache Verification Pipeline. Independent verification and falsification of claims from the KV-cache geometry paper (Campaign 5 / paper-c3).

## Quick Reference

```
# Run full pipeline (GPU required for extraction)
python -m kv_verify run

# CPU-only (skips extraction, uses pre-computed features)
python -m kv_verify run --skip-gpu

# Custom config
python -m kv_verify run --config config.yaml --n-per-group 50

# Specific stages only
python -m kv_verify run --stages analysis falsification verdicts report

# Run tests
TMPDIR=/tmp/claude-1000 .venv/bin/python -m pytest kv_verify/tests/ -q
```

## Compact Instructions

When context is compressed, preserve: package layout, STAGE_ORDER, the extension guide, and the library inventory. These are the most frequently needed references.

## Package Layout

```
kv_verify/
  __main__.py              # CLI: python -m kv_verify run [--config ...]
  config.py                # PipelineConfig dataclass + YAML loading
  constants.py             # All thresholds, statistical params, defaults (with WHY docs)
  pipeline.py              # 8-stage orchestrator (see "Pipeline Stages" below)
  types.py                 # ClaimVerification, Verdict, Severity enums
  data_loader.py           # Loads hackathon result JSONs from Campaign 5
  fixtures.py              # Synthetic data generators for tests + known data

  lib/                     # Reusable libraries (not KV-specific)
    tracking.py            # ExperimentTracker + @tracked/@stage/@validated decorators
    stats.py               # GroupKFold AUROC, FWL, permutation, bootstrap, power
    scorers.py             # MLflow-compatible evaluation scorers
    feature_extractor.py   # KV-cache feature extraction for HuggingFace models
    models.py              # Local-first model management (download, load, tokenizer)
    prompts/               # Prompt engineering: gen, analyzer, gap_filler, generator

  experiments/             # Standalone experiment scripts (V-series, F-series)
    v01_groupkfold.py      # V01: GroupKFold bug audit
    v03_fwl_leakage.py     # V03: FWL leakage detection
    v04_holm_bonferroni.py # V04: multiple comparison correction
    v07_sycophancy.py      # V07: sycophancy signal falsification
    v10_power_analysis.py  # V10: statistical power audit
    f01_falsification.py   # F01: input-length confound discovery
    f02_held_out_input_control.py   # F02: held-out deception control
    f03_cross_model_input_control.py # F03: cross-model transfer
    f04_cross_condition_validity.py  # F04: cross-condition validity
    output/                # Result JSONs from completed experiments

  tests/                   # 58+ tests across 22 files
    conftest.py            # Shared fixtures (TMPDIR setup)
    test_pipeline.py       # Pipeline integration tests
    test_tracking.py       # Tracker + decorator tests
    test_stats.py          # Statistical method tests
    test_*.py              # Per-module and per-experiment tests

  research-log/            # Design docs, agent traces, method references
    methods.md             # Statistical method citations and implementation notes
```

Re-export shims: `kv_verify/tracking.py`, `kv_verify/prompt_gen.py`, `kv_verify/scorers.py`, `kv_verify/stats.py`, `kv_verify/models.py`, `kv_verify/feature_extractor.py`, `kv_verify/prompt_analyzer.py`, `kv_verify/prompt_gap_filler.py`, `kv_verify/prompt_generator.py` all re-export from `kv_verify/lib/` for backward compatibility. Real implementations are always in `lib/`.

## Pipeline Stages

Defined in `pipeline.py:Pipeline.STAGE_ORDER`. Each stage is wrapped by `@stage()` from `lib/tracking.py` for automatic caching, timing, and dependency checking.

| # | Stage | Depends on | What it does |
|---|-------|-----------|--------------|
| 0 | `environment` | (none) | System info, GPU check, log config params |
| 1 | `prompt_gen` | environment | Generate minimal-pair prompt sets (BLiMP-style) |
| 2 | `tokenization` | prompt_gen | Validate token-count matching across pairs |
| 3 | `extraction` | tokenization | **GPU**: run inference, extract KV-cache features |
| 4 | `analysis` | extraction | GroupKFold AUROC, FWL, permutation, bootstrap, power |
| 5 | `falsification` | analysis | Input-only AUROC, format baseline, confound checks |
| 6 | `verdicts` | falsification | Apply pre-registered verdict criteria |
| 7 | `report` | verdicts | Generate markdown summary report |

Stage 3 (extraction) is skipped with `--skip-gpu`, returning `{"status": "skipped"}`. All other stages run on CPU.

### Stage caching

Cache keys include a hash of the pipeline config (excluding `output_dir`). Rerunning with different config values in the same output directory creates fresh cache entries. Same config reuses cached results. Cache files live in `<output_dir>/_cache/`.

### Running individual stages

```python
from kv_verify.config import PipelineConfig
from kv_verify.pipeline import Pipeline

config = PipelineConfig(skip_gpu=True, n_per_group=50)
pipeline = Pipeline(config)
pipeline.run_stage("environment")
pipeline.run_stage("prompt_gen")
# Or run all: pipeline.run()
```

## Reusable Libraries

These libraries are general-purpose. They can be extracted for other ML projects.

### ExperimentTracker (`lib/tracking.py`)

Structured logging with MLflow integration + disk-based checkpoint/resume.

```python
from kv_verify.tracking import ExperimentTracker, stage, tracked, validated, stage_cache_key

tracker = ExperimentTracker(output_dir="./output/run_001", experiment_name="my_exp")
tracker.log_params(model_id="qwen-7b", n_per_group=200)

# Per-item caching
with tracker.stage("extraction"):
    for key, data in items:
        if tracker.is_cached(key):
            continue
        result = expensive_compute(data)
        tracker.log_item(key, result)

# Batch metrics (single disk write)
tracker.log_metrics(auroc=0.85, p_value=0.001, power=0.9)

# Verdicts
tracker.log_verdict("claim-1", "confirmed", "AUROC=0.85, p<0.001")
```

**Decorators:**

- `@tracked(tracker, cache_key=fn)`: per-item auto-caching with hash-based dedup
- `@stage(tracker, "name", depends_on=["prev"], config_hash="abc")`: stage timing, caching, dependency enforcement
- `@validated(tracker, checks=[fn1, fn2])`: pre-flight validation before execution

### Probe Statistics (`lib/stats.py`)

GroupKFold AUROC, FWL residualization, permutation tests, bootstrap CIs, power analysis. See `research-log/methods.md` for citations.

```python
from kv_verify.stats import groupkfold_auroc, fwl_residualize, assign_groups

groups = assign_groups(n_pos=100, n_neg=100, paired=False)
result = groupkfold_auroc(X, y, groups, n_repeats=20)
# result.auroc, result.auroc_ci, result.p_value, result.power
```

### Minimal Pairs (`lib/prompts/gen.py`)

BLiMP-style prompt pair generation with token-count validation.

```python
from kv_verify.prompt_gen import PairSet, deception_pair, validate_token_counts

pair = deception_pair(question="What is 2+2?", pair_id="d_001")
check = validate_token_counts(pair, tokenizer=tok, max_diff=2)
```

## Writing a New Experiment

Each experiment is a standalone Python file in `experiments/`. Follow this pattern:

### 1. Create the experiment file

```python
"""VXX: Short Description of What This Tests.

Finding CX/MX: one-paragraph description of the claim being tested
and why it matters.

Spec: path/to/spec.md (if spec-driven)
"""

from pathlib import Path
from typing import Optional

from kv_verify.constants import ALPHA, AUROC_FWL_PRESERVE  # use constants, don't hardcode
from kv_verify.tracking import ExperimentTracker
from kv_verify.types import ClaimVerification, Severity, Verdict


def run_vXX(
    output_dir: Path,
    tracker: Optional[ExperimentTracker] = None,
) -> ClaimVerification:
    """Run the experiment. Returns a ClaimVerification verdict.

    Args:
        output_dir: where to write result artifacts
        tracker: optional shared tracker (creates local one if None)
    """
    output_dir = Path(output_dir)
    if tracker is None:
        tracker = ExperimentTracker(
            output_dir=output_dir, experiment_name="VXX-short-name",
        )

    # Load data
    # Run analysis using lib/stats.py functions
    # Log metrics via tracker.log_metrics(...)
    # Write result JSON to output_dir

    result = ClaimVerification(
        claim_id="CX",
        description="What was tested",
        verdict=Verdict.CONFIRMED,  # or WEAKENED, FALSIFIED
        severity=Severity.MEDIUM,
        evidence="Key numbers and reasoning",
    )
    tracker.log_verdict(result.claim_id, result.verdict.value, result.evidence)
    return result
```

### 2. Write tests in `tests/test_vXX.py`

```python
"""Tests for VXX experiment."""
import pytest
from kv_verify.experiments.vXX import run_vXX

class TestVXX:
    def test_runs_and_produces_verdict(self, tmp_path):
        result = run_vXX(output_dir=tmp_path)
        assert result.verdict in (Verdict.CONFIRMED, Verdict.WEAKENED, Verdict.FALSIFIED)
        assert (tmp_path / "vXX_results.json").exists()
```

### 3. Add to the pipeline (optional)

If the experiment should be a pipeline stage, add it to `Pipeline.STAGE_ORDER` in `pipeline.py` and implement `_do_vXX()`.

## Extending to a New Project

The `lib/` directory contains project-independent libraries. To reuse them:

1. **Copy or symlink `lib/`** into your new project
2. **Keep `constants.py` project-specific**: your thresholds and defaults will differ
3. **Keep `config.py` project-specific**: your `PipelineConfig` fields will differ
4. **Keep `types.py` project-specific**: your verdict types may differ

The pipeline orchestrator (`pipeline.py`) is the integration layer. For a new project:

1. Define your stages in `STAGE_ORDER` (list of `(name, depends_on)` tuples)
2. Implement `_do_{name}()` for each stage
3. The `@stage` decorator handles caching, timing, and dependencies automatically
4. Config-hash invalidation means you can safely rerun with different parameters

### What to import from lib

| Need | Import |
|------|--------|
| Experiment tracking + caching | `from kv_verify.tracking import ExperimentTracker, stage, tracked` |
| Classification with grouped CV | `from kv_verify.stats import groupkfold_auroc, assign_groups` |
| Confound control | `from kv_verify.stats import fwl_residualize` |
| Multiple comparison correction | `from kv_verify.stats import holm_bonferroni` |
| Bootstrap CIs | `from kv_verify.stats import bootstrap_ci` |
| Power analysis | `from kv_verify.stats import simulation_power` |
| Prompt pair generation | `from kv_verify.prompt_gen import PairSet, MinimalPair` |
| Prompt quality analysis | `from kv_verify.prompt_analyzer import analyze_pair_set` |
| KV-cache features | `from kv_verify.feature_extractor import extract_from_cache` |
| MLflow scorers | `from kv_verify.scorers import StatisticalScorer, ResponseScorer` |

## Build and Test

```bash
# Activate venv
source .venv/bin/activate

# Run all tests
TMPDIR=/tmp/claude-1000 pytest kv_verify/tests/ -q

# Run specific test file
TMPDIR=/tmp/claude-1000 pytest kv_verify/tests/test_pipeline.py -v

# Run specific test class
TMPDIR=/tmp/claude-1000 pytest kv_verify/tests/test_pipeline.py::TestCodexP1CudaAttribute -v
```

`TMPDIR=/tmp/claude-1000` is needed on WSL2 for writable temp directories in sandboxed environments.

## Environment

- Python 3.12+ with venv at `.venv/`
- PyTorch 2.11+ (GPU stages need CUDA)
- MLflow (optional, sqlite backend at `mlflow.db`)
- Key deps: numpy, scikit-learn, scipy, pyyaml

## Conventions

- Constants in `constants.py` with WHY comments. Never hardcode thresholds.
- All experiments return `ClaimVerification` typed results.
- Experiment files have a module docstring stating which claim they test.
- Use `tracker.log_metrics()` (batch) over `tracker.log_metric()` (individual) when logging multiple values.
- Statistical methods documented in `research-log/methods.md` with citations.
- Result JSONs go in `experiments/output/<experiment_id>/`.
