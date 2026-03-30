"""Tests for validate CLI subcommand (Task 6.4)."""

import json
import subprocess
import sys
from pathlib import Path

import pytest


def _write_dataset(tmp_path, items):
    """Helper: write items to a JSON file."""
    path = tmp_path / "test_data.json"
    path.write_text(json.dumps(items))
    return path


class TestValidateCLI:

    def test_validate_subcommand_exists(self, tmp_path):
        """python -m kv_verify validate --help should work."""
        result = subprocess.run(
            [sys.executable, "-m", "kv_verify", "validate", "--help"],
            capture_output=True, text=True, timeout=30,
            cwd=str(tmp_path),
            env={**__import__("os").environ, "PYTHONPATH": str(Path(__file__).resolve().parent.parent.parent)},
        )
        assert result.returncode == 0
        assert "validate" in result.stdout.lower() or "dataset" in result.stdout.lower()

    def test_validate_runs_on_json_file(self, tmp_path):
        """Validate a simple JSON dataset file."""
        items = [
            {"condition": "A", "prompt": f"question about topic {i}", "features": {"n_tokens": 50 + i}}
            for i in range(20)
        ] + [
            {"condition": "B", "prompt": f"question about subject {i}", "features": {"n_tokens": 52 + i}}
            for i in range(20)
        ]
        dataset_path = _write_dataset(tmp_path, items)

        result = subprocess.run(
            [sys.executable, "-m", "kv_verify", "validate", "--dataset", str(dataset_path), "--tier", "0"],
            capture_output=True, text=True, timeout=30,
            cwd=str(tmp_path),
            env={**__import__("os").environ, "PYTHONPATH": str(Path(__file__).resolve().parent.parent.parent)},
        )
        assert result.returncode == 0
        assert "PASS" in result.stdout or "pass" in result.stdout.lower()
