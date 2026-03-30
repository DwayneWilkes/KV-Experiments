"""Tests for config YAML export (Task 11.1)."""

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from kv_verify.config import PipelineConfig


class TestToYaml:

    def test_round_trip(self, tmp_path):
        """from_yaml -> to_yaml -> from_yaml preserves all fields."""
        original = PipelineConfig(n_per_group=50, seed=123, skip_gpu=True)
        yaml_path = tmp_path / "config.yaml"
        original.to_yaml(yaml_path)

        loaded = PipelineConfig.from_yaml(yaml_path)
        assert loaded.n_per_group == 50
        assert loaded.seed == 123
        assert loaded.skip_gpu is True

    def test_output_is_valid_yaml(self, tmp_path):
        config = PipelineConfig()
        yaml_path = tmp_path / "config.yaml"
        config.to_yaml(yaml_path)
        data = yaml.safe_load(yaml_path.read_text())
        assert isinstance(data, dict)
        assert "seed" in data


class TestConfigDumpCLI:

    def test_config_dump_subcommand(self):
        result = subprocess.run(
            [sys.executable, "-m", "kv_verify", "config", "--dump"],
            capture_output=True, text=True, timeout=30,
            env={**__import__("os").environ, "PYTHONPATH": str(Path(__file__).resolve().parent.parent.parent)},
        )
        assert result.returncode == 0
        # Should output valid YAML
        data = yaml.safe_load(result.stdout)
        assert "seed" in data
        assert "model_id" in data
