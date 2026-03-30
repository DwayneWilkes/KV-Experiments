"""Tests for package structure and metadata (Task 1.3)."""

import importlib
from pathlib import Path

import pytest


class TestPackageMetadata:

    def test_version_exists(self):
        import kv_verify
        assert hasattr(kv_verify, "__version__")
        assert isinstance(kv_verify.__version__, str)

    def test_pyproject_toml_exists(self):
        """pyproject.toml must exist at the submodule root."""
        root = Path(__file__).resolve().parent.parent.parent
        assert (root / "pyproject.toml").exists()

    def test_main_is_importable(self):
        mod = importlib.import_module("kv_verify.__main__")
        assert hasattr(mod, "main")

    def test_core_deps_importable(self):
        """Core dependencies (no GPU) must be importable."""
        import numpy
        import scipy
        import sklearn
        import yaml

    def test_constants_importable(self):
        from kv_verify.constants import ALPHA, DEFAULT_SEED, N_PERMUTATIONS
        assert ALPHA == 0.05

    def test_config_importable(self):
        from kv_verify.config import PipelineConfig
        config = PipelineConfig()
        assert config.seed == 42
