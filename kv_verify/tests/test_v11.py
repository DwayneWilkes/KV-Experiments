"""Tests for V11: Feature Ablation via Permutation Importance (Task 10.4)."""

import json
from pathlib import Path

import numpy as np
import pytest

from kv_verify.experiments.v11_feature_ablation import run_v11
from kv_verify.types import ClaimVerification, Verdict


class TestRunV11:

    def test_returns_claim_verification_with_valid_verdict(self, tmp_path):
        result = run_v11(output_dir=tmp_path / "v11")
        assert isinstance(result, ClaimVerification)
        assert result.verdict in (Verdict.CONFIRMED, Verdict.WEAKENED, Verdict.FALSIFIED)

    def test_result_json_has_all_features(self, tmp_path):
        out = tmp_path / "v11"
        run_v11(output_dir=out)
        data = json.loads((out / "v11_results.json").read_text())
        importance = data["feature_importance"]
        assert "norm_per_token" in importance
        assert "key_rank" in importance
        assert "key_entropy" in importance
        # Each importance should be a number
        for v in importance.values():
            assert isinstance(v, float)

    def test_result_json_has_per_comparison(self, tmp_path):
        out = tmp_path / "v11"
        run_v11(output_dir=out)
        data = json.loads((out / "v11_results.json").read_text())
        assert data["n_comparisons"] > 0
        assert len(data["per_comparison"]) == data["n_comparisons"]
