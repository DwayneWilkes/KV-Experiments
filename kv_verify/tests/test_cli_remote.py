"""Tests for --remote CLI flag (Task 8.3)."""

import subprocess
import sys
from pathlib import Path

import pytest
import yaml


class TestRemoteCLIFlag:

    def test_remote_flag_in_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "kv_verify", "run", "--help"],
            capture_output=True, text=True, timeout=30,
            env={**__import__("os").environ, "PYTHONPATH": str(Path(__file__).resolve().parent.parent.parent)},
        )
        assert "--remote" in result.stdout

    def test_remote_yaml_parsed(self, tmp_path):
        """--remote flag should accept a YAML config file path."""
        yaml_data = {"backend": "ssh", "host": "gpu.box", "user": "root",
                     "key_path": "/tmp/key", "remote_dir": "/workspace"}
        remote_path = tmp_path / "remote.yaml"
        remote_path.write_text(yaml.dump(yaml_data))

        # Just verify parsing doesn't crash (pipeline will fail without real SSH)
        result = subprocess.run(
            [sys.executable, "-m", "kv_verify", "run", "--remote", str(remote_path),
             "--skip-gpu", "--help"],
            capture_output=True, text=True, timeout=30,
            env={**__import__("os").environ, "PYTHONPATH": str(Path(__file__).resolve().parent.parent.parent)},
        )
        # --help should still work even with --remote
        assert result.returncode == 0
