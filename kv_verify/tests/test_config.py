"""Tests for kv_verify.config — pipeline configuration."""

from pathlib import Path

import yaml

from kv_verify.config import PipelineConfig


class TestPipelineConfig:
    def test_defaults(self):
        cfg = PipelineConfig()
        assert cfg.model_id == "Qwen/Qwen2.5-7B-Instruct"
        assert cfg.n_per_group == 200
        assert cfg.n_permutations == 10000
        assert cfg.n_bootstrap == 10000
        assert cfg.seed == 42
        assert cfg.temperature == 0.0
        assert cfg.max_new_tokens == 200

    def test_comparisons_default(self):
        cfg = PipelineConfig()
        assert "deception" in cfg.comparisons
        assert "refusal" in cfg.comparisons
        assert "impossibility" in cfg.comparisons

    def test_override(self):
        cfg = PipelineConfig(n_per_group=50, seed=123)
        assert cfg.n_per_group == 50
        assert cfg.seed == 123

    def test_output_dir_is_path(self):
        cfg = PipelineConfig()
        assert isinstance(cfg.output_dir, Path)

    def test_from_yaml(self, tmp_path):
        yaml_content = {
            "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "n_per_group": 10,
            "seed": 99,
        }
        yaml_path = tmp_path / "config.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_content, f)

        cfg = PipelineConfig.from_yaml(yaml_path)
        assert cfg.model_id == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        assert cfg.n_per_group == 10
        assert cfg.seed == 99
        # Unset fields keep defaults
        assert cfg.n_permutations == 10000

    def test_to_dict(self):
        cfg = PipelineConfig()
        d = cfg.to_dict()
        assert isinstance(d, dict)
        assert d["model_id"] == "Qwen/Qwen2.5-7B-Instruct"
        assert d["n_per_group"] == 200

    def test_mlflow_defaults(self):
        cfg = PipelineConfig()
        assert cfg.mlflow_tracking_uri == "file:./mlruns"
        assert cfg.mlflow_experiment == "kv-cache-verification"

    def test_skip_gpu(self):
        cfg = PipelineConfig(skip_gpu=True)
        assert cfg.skip_gpu is True
