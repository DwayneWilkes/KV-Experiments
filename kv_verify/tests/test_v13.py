"""Tests for V13: Matched-Scale Transfer Control (Task 10.6)."""

import json
from pathlib import Path

import pytest

from kv_verify.experiments.v13_matched_scale import run_v13
from kv_verify.types import Verdict


class TestRunV13:

    def test_produces_valid_verdict(self, tmp_path):
        result = run_v13(output_dir=tmp_path / "v13")
        assert result.verdict in (Verdict.CONFIRMED, Verdict.WEAKENED, Verdict.FALSIFIED)

    def test_result_json_has_required_structure(self, tmp_path):
        out = tmp_path / "v13"
        run_v13(output_dir=out)
        data = json.loads((out / "v13_results.json").read_text())
        assert "model_pairs" in data
        assert "mean_transfer_auroc" in data
        assert isinstance(data["mean_transfer_auroc"], (int, float))
        assert "experiment" in data
        assert data["experiment"] == "V13"

    def test_result_pairs_have_correct_fields(self, tmp_path):
        out = tmp_path / "v13"
        run_v13(output_dir=out)
        data = json.loads((out / "v13_results.json").read_text())
        # If data was available, pairs should have structured fields
        for pair in data["model_pairs"]:
            assert "train_model" in pair
            assert "test_model" in pair
            assert "raw_auroc" in pair
            assert isinstance(pair["raw_auroc"], float)
