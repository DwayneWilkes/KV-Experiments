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
        assert "--dataset" in result.stdout

    def test_validate_passes_on_clean_dataset(self, tmp_path):
        """Tier 0 validation on a balanced dataset should exit 0."""
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
        # Exit code 0 = PASS, 1 = FAIL/INCONCLUSIVE
        assert result.returncode == 0
