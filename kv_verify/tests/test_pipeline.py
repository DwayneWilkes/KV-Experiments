"""Tests for kv_verify.pipeline — stage orchestrator.

Integration tests use synthetic data and mock GPU. No real model needed.
"""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from kv_verify.config import PipelineConfig
from kv_verify.pipeline import Pipeline, StageStatus


class TestStageStatus:
    def test_not_started(self):
        assert StageStatus.NOT_STARTED.value == "not_started"

    def test_complete(self):
        assert StageStatus.COMPLETE.value == "complete"

    def test_skipped(self):
        assert StageStatus.SKIPPED.value == "skipped"


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

        # Manually mark a stage as complete
        pipeline.tracker.log_item("stage_environment", {"status": "complete"})

        # Stage should be skipped on run
        status = pipeline._check_stage_cache("environment")
        assert status == StageStatus.COMPLETE


class TestPipelineEnvironmentStage:
    def test_env_stage_logs_metadata(self, tmp_path):
        cfg = PipelineConfig(output_dir=tmp_path / "run", skip_gpu=True)
        pipeline = Pipeline(cfg)
        pipeline.run_stage("environment")

        meta = json.loads((tmp_path / "run" / "run_metadata.json").read_text())
        assert "params" in meta
        assert meta["params"].get("model_id") == "Qwen/Qwen2.5-7B-Instruct"


class TestPipelinePromptGenStage:
    def test_generates_prompt_files(self, tmp_path):
        cfg = PipelineConfig(
            output_dir=tmp_path / "run",
            n_per_group=5,  # small for testing
            skip_gpu=True,
        )
        pipeline = Pipeline(cfg)
        pipeline.run_stage("environment")
        pipeline.run_stage("prompt_gen")

        prompts_dir = tmp_path / "run" / "prompts"
        assert prompts_dir.exists()
        # Should have one file per comparison
        prompt_files = list(prompts_dir.glob("*.json"))
        assert len(prompt_files) >= 1


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
        pipeline.run_stage("environment")

        # Inject synthetic extracted features (simulating GPU stage output)
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


class TestPipelineReport:
    def test_report_generation(self, tmp_path):
        cfg = PipelineConfig(output_dir=tmp_path / "run", skip_gpu=True)
        pipeline = Pipeline(cfg)

        # Inject minimal verdict data
        pipeline.tracker.log_verdict("test-claim", "confirmed", "test evidence")
        pipeline.run_stage("report")

        report_path = tmp_path / "run" / "final_report.md"
        assert report_path.exists()
        content = report_path.read_text()
        assert "test-claim" in content
