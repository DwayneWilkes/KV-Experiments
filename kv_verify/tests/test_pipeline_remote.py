"""Tests for pipeline remote routing (Task 8.4)."""

from pathlib import Path

import pytest

from kv_verify.config import PipelineConfig
from kv_verify.lib.remote import RemoteConfig
from kv_verify.pipeline import Pipeline


class TestPipelineRemoteRouting:

    def test_remote_config_defaults_to_none(self, tmp_path):
        cfg = PipelineConfig(output_dir=tmp_path / "run", skip_gpu=True)
        pipeline = Pipeline(cfg)
        assert pipeline._remote_config is None

    def test_skip_gpu_extraction_returns_skipped(self, tmp_path):
        """Without GPU or remote, extraction should skip."""
        cfg = PipelineConfig(output_dir=tmp_path / "run", skip_gpu=True)
        pipeline = Pipeline(cfg)
        # Must run prerequisite stages before extraction
        pipeline.run_stage("environment")
        pipeline.run_stage("validation")
        pipeline.run_stage("prompt_gen")
        pipeline.run_stage("tokenization")
        result = pipeline.run_stage("extraction")
        assert result.get("status") == "skipped"
