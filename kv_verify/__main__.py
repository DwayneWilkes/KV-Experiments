"""CLI entry point for kv_verify pipeline.

Usage:
    python -m kv_verify run                          # run all stages
    python -m kv_verify run --config config.yaml     # custom config
    python -m kv_verify run --n-per-group 50         # override N
    python -m kv_verify run --skip-gpu               # CPU stages only
    python -m kv_verify run --stages analysis report  # specific stages
    python -m kv_verify validate --dataset data.json  # validate dataset
    python -m kv_verify validate --dataset d.json --tier 2  # rigorous
"""

import argparse
import json
import sys
from pathlib import Path

from kv_verify.config import PipelineConfig
from kv_verify.pipeline import Pipeline


def _run_validate(args):
    """Run standalone dataset validation."""
    from kv_verify.lib.dataset_validation import validate_dataset

    with open(args.dataset) as f:
        items = json.load(f)

    report = validate_dataset(items, tier=args.tier)

    # Print report
    print(f"Dataset Validation Report (Tier {report.tier})")
    print(f"  Verdict: {report.overall_verdict}")
    print(f"  Nominal N: {report.nominal_n}")
    if report.effective_n:
        print(f"  Effective N: {report.effective_n}")
    print(f"  Checks: {len(report.checks)}")
    for name, cr in report.checks.items():
        status = "PASS" if cr.passed else "FAIL"
        print(f"    [{status}] {name}: {cr.details or 'OK'}")
    if report.recommendations:
        print("  Recommendations:")
        for rec in report.recommendations:
            print(f"    - {rec}")
    if report.dataset_hash:
        print(f"  Dataset hash: {report.dataset_hash}")

    # Write JSON report if output specified
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"  Report saved to {args.output}")

    sys.exit(0 if report.overall_pass else 1)


def main():
    parser = argparse.ArgumentParser(
        prog="kv_verify",
        description="KV-Cache Verification Pipeline",
    )
    sub = parser.add_subparsers(dest="command")

    # --- run subcommand ---
    run_parser = sub.add_parser("run", help="Run the verification pipeline")
    run_parser.add_argument("--config", type=Path, help="YAML config file")
    run_parser.add_argument("--n-per-group", type=int, help="Samples per condition")
    run_parser.add_argument("--model", type=str, help="Model ID")
    run_parser.add_argument("--skip-gpu", action="store_true", help="Skip GPU stages")
    run_parser.add_argument("--output-dir", type=Path, help="Output directory")
    run_parser.add_argument("--stages", nargs="*", help="Specific stages to run")
    run_parser.add_argument("--seed", type=int, help="Random seed")

    # --- validate subcommand ---
    val_parser = sub.add_parser("validate", help="Validate a dataset for experiment quality")
    val_parser.add_argument("--dataset", type=Path, required=True, help="Path to dataset JSON")
    val_parser.add_argument("--tier", type=int, default=1, help="Validation tier (0-3, default 1)")
    val_parser.add_argument("--output", type=Path, help="Write JSON report to file")

    args = parser.parse_args()

    if args.command == "validate":
        _run_validate(args)
        return

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
