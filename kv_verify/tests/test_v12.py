"""Tests for V12: System Prompt Residualization (Task 10.5)."""

import json
from pathlib import Path

import pytest

from kv_verify.experiments.v12_system_prompt import run_v12
from kv_verify.types import Verdict


class TestRunV12:

    def test_produces_valid_verdict(self, tmp_path):
        result = run_v12(output_dir=tmp_path / "v12")
        assert result.verdict in (Verdict.CONFIRMED, Verdict.WEAKENED, Verdict.FALSIFIED)

    def test_result_json_has_required_fields(self, tmp_path):
        out = tmp_path / "v12"
        run_v12(output_dir=out)
        data = json.loads((out / "v12_results.json").read_text())
        assert "system_prompts_identical" in data
        assert "verdict" in data
        assert "experiment" in data
        assert data["experiment"] == "V12"

    def test_system_prompt_analysis_performed(self, tmp_path):
        out = tmp_path / "v12"
        result = run_v12(output_dir=out)
        data = json.loads((out / "v12_results.json").read_text())
        # Must determine whether system prompts are identical or different
        assert isinstance(data["system_prompts_identical"], bool)
        # If identical, verdict should be CONFIRMED per pre-registration
        if data["system_prompts_identical"]:
            assert result.verdict == Verdict.CONFIRMED
