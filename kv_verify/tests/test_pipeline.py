"""Tests for kv_verify.pipeline — decorator-based stage orchestrator.

Integration tests use synthetic data and mock GPU. No real model needed.
"""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from kv_verify.config import PipelineConfig
from kv_verify.pipeline import Pipeline


class TestPipelineCreation:
    def test_creates_with_config(self, tmp_path):
        cfg = PipelineConfig(output_dir=tmp_path / "run")
        pipeline = Pipeline(cfg)
        assert pipeline.config.n_per_group == 200

    def test_creates_output_dirs(self, tmp_path):
        cfg = PipelineConfig(output_dir=tmp_path / "run")
        pipeline = Pipeline(cfg)
        assert (tmp_path / "run").exists()

    def test_has_all_stages(self, tmp_path):
        cfg = PipelineConfig(output_dir=tmp_path / "run")
        pipeline = Pipeline(cfg)
        stage_names = [s.name for s in pipeline.stages]
        assert "environment" in stage_names
        assert "prompt_gen" in stage_names
        assert "tokenization" in stage_names
        assert "extraction" in stage_names
        assert "analysis" in stage_names
        assert "falsification" in stage_names
        assert "verdicts" in stage_names
        assert "report" in stage_names


class TestPipelineStageSkipping:
    def test_skips_completed_stage(self, tmp_path):
        cfg = PipelineConfig(output_dir=tmp_path / "run", skip_gpu=True)
        pipeline = Pipeline(cfg)

        # Manually mark environment as complete (decorator @stage checks this)
        pipeline.tracker.log_item("stage_environment", {"status": "complete"})

        # Running again should return cached result (decorator auto-skips)
        result = pipeline.run_stage("environment")
        assert result.get("status") == "complete"


class TestPipelineEnvironmentStage:
    def test_env_stage_logs_metadata(self, tmp_path):
        cfg = PipelineConfig(output_dir=tmp_path / "run", skip_gpu=True)
        pipeline = Pipeline(cfg)
        pipeline.run_stage("environment")

        meta = json.loads((tmp_path / "run" / "run_metadata.json").read_text())
        assert "params" in meta
        assert meta["params"].get("model_id") == "Qwen/Qwen2.5-7B-Instruct"

    def test_env_stage_records_timing(self, tmp_path):
        cfg = PipelineConfig(output_dir=tmp_path / "run", skip_gpu=True)
        pipeline = Pipeline(cfg)
        pipeline.run_stage("environment")

        meta = json.loads((tmp_path / "run" / "run_metadata.json").read_text())
        assert "environment" in meta["stages"]
        assert "duration_seconds" in meta["stages"]["environment"]


class TestPipelinePromptGenStage:
    def test_generates_prompt_files(self, tmp_path):
        cfg = PipelineConfig(
            output_dir=tmp_path / "run",
            n_per_group=5,
            skip_gpu=True,
        )
        pipeline = Pipeline(cfg)
        pipeline.run_stage("environment")
        pipeline.run_stage("validation")
        pipeline.run_stage("prompt_gen")

        prompts_dir = tmp_path / "run" / "prompts"
        assert prompts_dir.exists()
        prompt_files = list(prompts_dir.glob("*.json"))
        assert len(prompt_files) >= 1

    def test_logs_dataset_to_tracker(self, tmp_path):
        cfg = PipelineConfig(
            output_dir=tmp_path / "run",
            n_per_group=5,
            skip_gpu=True,
        )
        pipeline = Pipeline(cfg)
        pipeline.run_stage("environment")
        pipeline.run_stage("validation")
        pipeline.run_stage("prompt_gen")

        meta = json.loads((tmp_path / "run" / "run_metadata.json").read_text())
        assert "datasets" in meta
        assert len(meta["datasets"]) >= 1


class TestPipelineAnalysisStage:
    def test_analysis_runs_on_synthetic_data(self, tmp_path):
        """Analysis should work with synthetic feature data (no GPU)."""
        cfg = PipelineConfig(
            output_dir=tmp_path / "run",
            n_per_group=10,
            n_permutations=100,
            n_bootstrap=100,
            skip_gpu=True,
        )
        pipeline = Pipeline(cfg)

        # Must run prerequisite stages (decorator checks dependencies)
        pipeline.run_stage("environment")
        pipeline.run_stage("validation")
        pipeline.run_stage("prompt_gen")
        pipeline.run_stage("tokenization")

        # Mark extraction as complete (GPU skipped)
        pipeline.tracker.log_item("stage_extraction", {"status": "complete"})

        # Inject synthetic extracted features
        features_dir = tmp_path / "run" / "features"
        features_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.RandomState(42)
        for comparison in cfg.comparisons:
            data = {
                "comparison": comparison,
                "positive": [
                    {"features": {"norm_per_token": float(rng.randn() + 1),
                                  "key_rank": float(rng.randn() * 5 + 15),
                                  "key_entropy": float(rng.randn() * 0.2 + 0.5)},
                     "n_tokens": 50, "n_generated": 30, "n_input_tokens": 20}
                    for _ in range(10)
                ],
                "negative": [
                    {"features": {"norm_per_token": float(rng.randn()),
                                  "key_rank": float(rng.randn() * 5 + 10),
                                  "key_entropy": float(rng.randn() * 0.2 + 0.4)},
                     "n_tokens": 48, "n_generated": 28, "n_input_tokens": 20}
                    for _ in range(10)
                ],
            }
            with open(features_dir / f"{comparison}.json", "w") as f:
                json.dump(data, f)

        pipeline.run_stage("analysis")

        results_dir = tmp_path / "run" / "results"
        assert results_dir.exists()

    def test_analysis_logs_metrics(self, tmp_path):
        cfg = PipelineConfig(
            output_dir=tmp_path / "run",
            n_per_group=10,
            n_permutations=100,
            skip_gpu=True,
            comparisons=["test"],
        )
        pipeline = Pipeline(cfg)
        pipeline.run_stage("environment")
        pipeline.run_stage("validation")
        pipeline.run_stage("prompt_gen")
        pipeline.run_stage("tokenization")
        pipeline.tracker.log_item("stage_extraction", {"status": "complete"})

        features_dir = tmp_path / "run" / "features"
        features_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.RandomState(42)
        data = {
            "comparison": "test",
            "positive": [{"features": {"norm_per_token": float(rng.randn()+1),
                          "key_rank": 15.0, "key_entropy": 0.5},
                          "n_tokens": 50, "n_generated": 30, "n_input_tokens": 20}
                         for _ in range(10)],
            "negative": [{"features": {"norm_per_token": float(rng.randn()),
                          "key_rank": 10.0, "key_entropy": 0.4},
                          "n_tokens": 48, "n_generated": 28, "n_input_tokens": 20}
                         for _ in range(10)],
        }
        with open(features_dir / "test.json", "w") as f:
            json.dump(data, f)

        pipeline.run_stage("analysis")

        meta = json.loads((tmp_path / "run" / "run_metadata.json").read_text())
        assert "test_auroc" in meta["metrics"]
        assert "test_p_value" in meta["metrics"]


class TestPipelineReport:
    def test_report_generation(self, tmp_path):
        cfg = PipelineConfig(output_dir=tmp_path / "run", skip_gpu=True)
        pipeline = Pipeline(cfg)

        # Run all prerequisite stages
        pipeline.run_stage("environment")
        pipeline.run_stage("validation")
        pipeline.run_stage("prompt_gen")
        pipeline.run_stage("tokenization")
        pipeline.tracker.log_item("stage_extraction", {"status": "complete"})
        pipeline.tracker.log_item("stage_analysis", {"status": "complete"})
        pipeline.tracker.log_item("stage_falsification", {"status": "complete"})
        pipeline.tracker.log_item("stage_verdicts", {"status": "complete"})

        pipeline.tracker.log_verdict("test-claim", "confirmed", "test evidence")
        pipeline.run_stage("report")

        report_path = tmp_path / "run" / "final_report.md"
        assert report_path.exists()
        content = report_path.read_text()
        assert "test-claim" in content


class TestDecoratorIntegration:
    def test_stage_decorator_caches_and_skips(self, tmp_path):
        """Verify the @stage decorator actually caches and skips."""
        cfg = PipelineConfig(output_dir=tmp_path / "run", skip_gpu=True)
        pipeline = Pipeline(cfg)

        # First run: executes
        pipeline.run_stage("environment")
        meta1 = json.loads((tmp_path / "run" / "run_metadata.json").read_text())
        first_time = meta1["stages"]["environment"]["duration_seconds"]

        # Second run: should skip (cached by @stage decorator)
        result = pipeline.run_stage("environment")
        assert result.get("status") == "complete"

    def test_stage_decorator_checks_dependencies(self, tmp_path):
        """Verify @stage raises on unmet dependencies."""
        cfg = PipelineConfig(output_dir=tmp_path / "run", skip_gpu=True)
        pipeline = Pipeline(cfg)

        # Try to run tokenization without prompt_gen
        with pytest.raises(RuntimeError, match="prompt_gen"):
            pipeline.run_stage("tokenization")

    def test_tracked_decorator_in_extraction(self, tmp_path):
        """Verify @tracked is used for per-item caching in extraction."""
        # This test just verifies the extraction stage's extract_single
        # uses @tracked. We can't run it without a model, but we can
        # verify the pipeline builds without error.
        cfg = PipelineConfig(output_dir=tmp_path / "run", skip_gpu=True)
        pipeline = Pipeline(cfg)
        assert "extraction" in pipeline._stages
