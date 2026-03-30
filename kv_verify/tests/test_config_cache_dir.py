"""Tests for PipelineConfig.model_cache_dir field (Task 1.2)."""

from pathlib import Path

import pytest
import yaml

from kv_verify.config import PipelineConfig


class TestModelCacheDir:

    def test_default_is_none(self):
        """Default model_cache_dir is None (uses env var / default resolution)."""
        config = PipelineConfig()
        assert config.model_cache_dir is None

    def test_set_via_constructor(self, tmp_path):
        config = PipelineConfig(model_cache_dir=tmp_path)
        assert config.model_cache_dir == tmp_path

    def test_from_yaml_with_cache_dir(self, tmp_path):
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text(yaml.dump({"model_cache_dir": "/workspace/models"}))
        config = PipelineConfig.from_yaml(yaml_path)
        assert config.model_cache_dir == Path("/workspace/models")

    def test_from_yaml_without_cache_dir(self, tmp_path):
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text(yaml.dump({"seed": 123}))
        config = PipelineConfig.from_yaml(yaml_path)
        assert config.model_cache_dir is None

    def test_to_dict_includes_cache_dir(self, tmp_path):
        config = PipelineConfig(model_cache_dir=tmp_path)
        d = config.to_dict()
        assert d["model_cache_dir"] == str(tmp_path)

    def test_to_dict_none_cache_dir(self):
        config = PipelineConfig()
        d = config.to_dict()
        assert d["model_cache_dir"] is None


class TestMlflowTrackingUri:

    def test_default_mlflow_inside_kv_verify(self):
        """MLflow db should default to inside kv_verify/ to be covered by nested gitignore."""
        config = PipelineConfig()
        assert "kv_verify/" in config.mlflow_tracking_uri
