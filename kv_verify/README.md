# kv_verify

Verification pipeline for KV-cache experiment claims. Tests 14 paper claims through pre-registered experiments with falsification-first methodology.

## Quick Start

```bash
# CPU-only (uses stored features)
pip install .
python -m kv_verify run --skip-gpu

# With local GPU
pip install ".[gpu]"
python -m kv_verify run --n-per-group 5 --seed 42

# With remote GPU (SSH)
python -m kv_verify run --remote remote.yaml

# Validate a dataset
python -m kv_verify validate --dataset data.json --tier 2

# Dump default config
python -m kv_verify config --dump
```

## Architecture

9-stage pipeline with decorator-based caching, timing, and dependency checking:

```
environment -> validation -> prompt_gen -> tokenization -> extraction (GPU)
                                                              |
                                                              v
                                          analysis -> falsification -> verdicts -> report
```

### Key Modules

| Module | Purpose |
|--------|---------|
| `lib/dataset_validation.py` | 4-tier dataset quality gate (15 checks, modality-agnostic) |
| `lib/stats.py` | Statistical testing (GroupKFold, FWL, permutation, bootstrap, power) |
| `lib/models.py` | Local-first model management, configurable cache dir |
| `lib/remote.py` | SSH/RunPod backends for remote GPU execution |
| `lib/tracking.py` | MLflow + disk caching with @tracked/@stage/@validated decorators |
| `lib/final_report.py` | Global Holm-Bonferroni + claim verdict report |
| `lib/feature_extractor.py` | KV-cache feature extraction (norm, rank, entropy) |
| `constants.py` | All magic numbers centralized with documentation |
| `config.py` | PipelineConfig dataclass with YAML loading/export |

### Experiments

- **V-series** (V01-V13): Statistical methodology verification
- **F-series** (F01-F05): Falsification battery

See `docs/experiments.md` for the full catalog.

## Configuration

All parameters are in `config-default.yaml`. Override via YAML file or CLI flags:

```bash
python -m kv_verify run --config my_config.yaml --n-per-group 50 --seed 42
```

Environment variables:
- `KV_VERIFY_MODEL_DIR`: Model cache directory (default: /mnt/d/dev/models)

## Dependencies

- **Core** (CPU): numpy, scipy, scikit-learn, pyyaml
- **GPU**: torch, transformers, accelerate
- **Tracking**: mlflow

Install with: `pip install .` (core), `pip install ".[gpu]"` (with GPU), `pip install ".[all]"` (everything)

## Dataset Validation

The `validate` subcommand runs quality checks on any dataset:

```bash
python -m kv_verify validate --dataset data.json --tier 2 --output report.json
```

Tiers: 0 (smoke), 1 (standard), 2 (rigorous), 3 (regulatory). See `docs/dataset-validation.md`.
