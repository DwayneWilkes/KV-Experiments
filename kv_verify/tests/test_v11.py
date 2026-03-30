"""Tests for V11: Feature Ablation via Permutation Importance (Task 10.4)."""

import json
from pathlib import Path

import numpy as np
import pytest

from kv_verify.experiments.v11_feature_ablation import run_v11
from kv_verify.types import Verdict


class TestRunV11:

    def test_produces_result(self, tmp_path):
        result = run_v11(output_dir=tmp_path / "v11")
        assert result is not None
        assert hasattr(result, "verdict")

    def test_verdict_is_valid(self, tmp_path):
        result = run_v11(output_dir=tmp_path / "v11")
        assert result.verdict in (Verdict.CONFIRMED, Verdict.WEAKENED, Verdict.FALSIFIED)

    def test_reports_per_feature_importance(self, tmp_path):
        result = run_v11(output_dir=tmp_path / "v11")
        evidence = result.evidence_summary
        assert "norm_per_token" in evidence or "key_rank" in evidence or "key_entropy" in evidence

    def test_writes_result_json(self, tmp_path):
        out = tmp_path / "v11"
        run_v11(output_dir=out)
        result_path = out / "v11_results.json"
        assert result_path.exists()
        data = json.loads(result_path.read_text())
        assert "feature_importance" in data
