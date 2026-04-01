"""CLI entry point for kv_verify pipeline.

Usage:
    python -m kv_verify run                          # run all stages
    python -m kv_verify run --config config.yaml     # custom config
    python -m kv_verify run --n-per-group 50         # override N
    python -m kv_verify run --skip-gpu               # CPU stages only
    python -m kv_verify run --stages analysis report  # specific stages
"""

import argparse
import sys
from pathlib import Path

from kv_verify.config import PipelineConfig
from kv_verify.pipeline import Pipeline


def main():
    parser = argparse.ArgumentParser(
        prog="kv_verify",
        description="KV-Cache Verification Pipeline",
    )
    sub = parser.add_subparsers(dest="command")

    run_parser = sub.add_parser("run", help="Run the verification pipeline")
    run_parser.add_argument("--config", type=Path, help="YAML config file")
    run_parser.add_argument("--n-per-group", type=int, help="Samples per condition")
    run_parser.add_argument("--model", type=str, help="Model ID")
    run_parser.add_argument("--skip-gpu", action="store_true", help="Skip GPU stages")
    run_parser.add_argument("--output-dir", type=Path, help="Output directory")
    run_parser.add_argument("--stages", nargs="*", help="Specific stages to run")
    run_parser.add_argument("--seed", type=int, help="Random seed")

    args = parser.parse_args()

    if args.command != "run":
        parser.print_help()
        sys.exit(1)

    # Build config
    if args.config and args.config.exists():
        config = PipelineConfig.from_yaml(args.config)
    else:
        config = PipelineConfig()

    # Apply CLI overrides
    if args.n_per_group:
        config.n_per_group = args.n_per_group
    if args.model:
        config.model_id = args.model
    if args.skip_gpu:
        config.skip_gpu = True
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.seed:
        config.seed = args.seed

    # Run pipeline
    pipeline = Pipeline(config)
    pipeline.run(stages=args.stages)


if __name__ == "__main__":
    main()
