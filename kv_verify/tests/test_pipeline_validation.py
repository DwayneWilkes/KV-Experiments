"""Tests for pipeline validation stage integration (Tasks 6.1-6.3)."""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from kv_verify.config import PipelineConfig
from kv_verify.pipeline import Pipeline


class TestValidationStageExists:
    """Task 6.1: validation stage in pipeline."""

    def test_validation_in_stage_order(self):
        config = PipelineConfig(skip_gpu=True)
        pipeline = Pipeline(config)
        stage_names = [s.name for s in pipeline.stages]
        assert "validation" in stage_names

    def test_validation_runs_before_prompt_gen(self):
        config = PipelineConfig(skip_gpu=True)
        pipeline = Pipeline(config)
        stage_names = [s.name for s in pipeline.stages]
        val_idx = stage_names.index("validation")
        gen_idx = stage_names.index("prompt_gen")
        assert val_idx < gen_idx


class TestValidationVerdictPropagation:
    """Task 6.2: INCONCLUSIVE annotation."""

    def test_pipeline_has_validation_verdict_attr(self):
        config = PipelineConfig(skip_gpu=True)
        pipeline = Pipeline(config)
        assert hasattr(pipeline, "_validation_verdict")
        assert pipeline._validation_verdict == "NOT_RUN"


class TestValidationHalt:
    """Task 6.3: FAIL halts pipeline unless --force."""

    def test_config_has_force_field(self):
        """PipelineConfig must have a force field for --force override."""
        config = PipelineConfig(skip_gpu=True, force=True)
        assert config.force is True

    def test_force_defaults_to_false(self):
        config = PipelineConfig(skip_gpu=True)
        assert config.force is False
