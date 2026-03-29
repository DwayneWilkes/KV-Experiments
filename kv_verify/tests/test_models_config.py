"""Tests for configurable MODEL_CACHE_DIR (Task 1.1).

Verifies the resolution order: explicit parameter > env var > default.
"""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from kv_verify.lib.models import (
    _get_snapshot_path,
    _resolve_model_id,
    _set_cache_dir,
    get_cache_dir,
    is_downloaded,
)


class TestGetCacheDir:
    """Test the 3-level resolution: explicit > env var > default."""

    def test_default_when_no_env_no_explicit(self):
        """Default fallback when nothing is set."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("KV_VERIFY_MODEL_DIR", None)
            result = get_cache_dir()
            assert isinstance(result, Path)
            # Default should be the hardcoded path
            assert result == Path("/mnt/d/dev/models")

    def test_env_var_overrides_default(self, tmp_path):
        """KV_VERIFY_MODEL_DIR env var takes precedence over default."""
        with patch.dict(os.environ, {"KV_VERIFY_MODEL_DIR": str(tmp_path)}):
            result = get_cache_dir()
            assert result == tmp_path

    def test_explicit_overrides_env_var(self, tmp_path):
        """Explicit cache_dir parameter wins over env var."""
        other = tmp_path / "other"
        other.mkdir()
        with patch.dict(os.environ, {"KV_VERIFY_MODEL_DIR": str(tmp_path)}):
            result = get_cache_dir(cache_dir=other)
            assert result == other

    def test_explicit_overrides_default(self, tmp_path):
        """Explicit cache_dir parameter wins over default."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("KV_VERIFY_MODEL_DIR", None)
            result = get_cache_dir(cache_dir=tmp_path)
            assert result == tmp_path

    def test_returns_path_object(self, tmp_path):
        """Always returns a Path, even if string is passed."""
        result = get_cache_dir(cache_dir=str(tmp_path))
        assert isinstance(result, Path)


class TestSetCacheDir:
    """_set_cache_dir should use get_cache_dir resolution."""

    def test_sets_hf_home_from_env_var(self, tmp_path):
        with patch.dict(os.environ, {"KV_VERIFY_MODEL_DIR": str(tmp_path)}):
            _set_cache_dir()
            assert os.environ["HF_HOME"] == str(tmp_path)

    def test_sets_hf_home_from_explicit(self, tmp_path):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("KV_VERIFY_MODEL_DIR", None)
            _set_cache_dir(cache_dir=tmp_path)
            assert os.environ["HF_HOME"] == str(tmp_path)


class TestIsDownloadedWithCacheDir:
    """is_downloaded should respect cache_dir parameter."""

    def test_not_found_in_custom_dir(self, tmp_path):
        """Model not found when cache_dir is empty."""
        assert is_downloaded("qwen", cache_dir=tmp_path) is False

    def test_found_in_custom_dir(self, tmp_path):
        """Model found when snapshot exists in custom cache_dir."""
        model_dir = tmp_path / "models--Qwen--Qwen2.5-7B-Instruct" / "snapshots" / "abc123"
        model_dir.mkdir(parents=True)
        assert is_downloaded("qwen", cache_dir=tmp_path) is True


class TestGetSnapshotPathWithCacheDir:
    """_get_snapshot_path should respect cache_dir parameter."""

    def test_finds_snapshot_in_custom_dir(self, tmp_path):
        snapshot = tmp_path / "models--Qwen--Qwen2.5-7B-Instruct" / "snapshots" / "abc123"
        snapshot.mkdir(parents=True)
        result = _get_snapshot_path("qwen", cache_dir=tmp_path)
        assert result == snapshot

    def test_raises_for_missing_model(self, tmp_path):
        with pytest.raises(RuntimeError, match="No snapshots"):
            _get_snapshot_path("qwen", cache_dir=tmp_path)
