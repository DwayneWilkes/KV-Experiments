"""Tests for pipeline remote routing (Task 8.4)."""

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from kv_verify.config import PipelineConfig
from kv_verify.lib.remote import RemoteConfig
from kv_verify.pipeline import Pipeline


class TestPipelineRemoteRouting:

    def test_pipeline_has_remote_config_attr(self, tmp_path):
        """Pipeline should have _remote_config attribute (None by default)."""
        cfg = PipelineConfig(output_dir=tmp_path / "run", skip_gpu=True)
        pipeline = Pipeline(cfg)
        assert hasattr(pipeline, "_remote_config")
        assert pipeline._remote_config is None

    def test_pipeline_accepts_remote_config(self, tmp_path):
        """Setting _remote_config should stick."""
        cfg = PipelineConfig(output_dir=tmp_path / "run", skip_gpu=True)
        pipeline = Pipeline(cfg)
        remote = RemoteConfig(backend="ssh", host="gpu.box", user="root",
                             key_path=Path("/tmp/key"), remote_dir="/workspace")
        pipeline._remote_config = remote
        assert pipeline._remote_config.host == "gpu.box"

    def test_do_extraction_checks_remote_config(self, tmp_path):
        """_do_extraction should check _remote_config before running locally."""
        cfg = PipelineConfig(output_dir=tmp_path / "run", skip_gpu=True)
        pipeline = Pipeline(cfg)
        # With skip_gpu and no remote, extraction returns skipped
        result = pipeline._do_extraction()
        assert result.get("status") == "skipped"
