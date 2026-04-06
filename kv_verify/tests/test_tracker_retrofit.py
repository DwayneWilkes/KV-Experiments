"""Tests for ExperimentTracker retrofit across falsification experiments.

Verifies that all falsification experiment functions:
1. Accept an optional tracker parameter
2. Create a local tracker when none is provided
3. Log params, metrics, verdicts, and items via the tracker
4. Produce run_metadata.json with expected structure
5. Remain backward-compatible (tracker=None works)
"""

import json
from pathlib import Path

import pytest

from kv_verify.tracking import ExperimentTracker
from kv_verify.types import ClaimVerification, Verdict


# ================================================================
# F01a: Null Experiment
# ================================================================

class TestF01aTracker:
    def test_accepts_tracker_param(self, tmp_path):
        from kv_verify.experiments.f01_falsification import run_f01a
        tracker = ExperimentTracker(
            output_dir=tmp_path, experiment_name="F01a-null",
        )
        result = run_f01a(tmp_path, tracker=tracker)
        assert isinstance(result, ClaimVerification)

    def test_backward_compatible_without_tracker(self, tmp_path):
        from kv_verify.experiments.f01_falsification import run_f01a
        result = run_f01a(tmp_path)
        assert isinstance(result, ClaimVerification)

    def test_tracker_logs_metadata(self, tmp_path):
        from kv_verify.experiments.f01_falsification import run_f01a
        run_f01a(tmp_path)
        meta_path = tmp_path / "run_metadata.json"
        assert meta_path.exists()
        with open(meta_path) as f:
            meta = json.load(f)
        assert "params" in meta
        assert "metrics" in meta
        assert "verdicts" in meta

    def test_tracker_logs_verdict(self, tmp_path):
        from kv_verify.experiments.f01_falsification import run_f01a
        run_f01a(tmp_path)
        with open(tmp_path / "run_metadata.json") as f:
            meta = json.load(f)
        assert "F01a-null" in meta["verdicts"]

    def test_tracker_logs_metrics(self, tmp_path):
        from kv_verify.experiments.f01_falsification import run_f01a
        run_f01a(tmp_path)
        with open(tmp_path / "run_metadata.json") as f:
            meta = json.load(f)
        assert "max_null_auroc" in meta["metrics"]

    def test_tracker_caches_result(self, tmp_path):
        from kv_verify.experiments.f01_falsification import run_f01a
        run_f01a(tmp_path)
        cache_path = tmp_path / "cache" / "f01a_result.json"
        assert cache_path.exists()
        with open(cache_path) as f:
            data = json.load(f)
        assert data["claim_id"] == "F01a-null"

    def test_external_tracker_receives_logs(self, tmp_path):
        from kv_verify.experiments.f01_falsification import run_f01a
        tracker = ExperimentTracker(
            output_dir=tmp_path / "external", experiment_name="external-test",
        )
        run_f01a(tmp_path, tracker=tracker)
        with open(tmp_path / "external" / "run_metadata.json") as f:
            meta = json.load(f)
        assert "F01a-null" in meta["verdicts"]


# ================================================================
# F01b: Input-Length Confound
# ================================================================

class TestF01bTracker:
    def test_accepts_tracker_param(self, tmp_path):
        from kv_verify.experiments.f01_falsification import run_f01b
        tracker = ExperimentTracker(
            output_dir=tmp_path, experiment_name="F01b-input",
        )
        result = run_f01b(tmp_path, tracker=tracker)
        assert isinstance(result, ClaimVerification)

    def test_backward_compatible_without_tracker(self, tmp_path):
        from kv_verify.experiments.f01_falsification import run_f01b
        result = run_f01b(tmp_path)
        assert isinstance(result, ClaimVerification)

    def test_tracker_logs_verdict(self, tmp_path):
        from kv_verify.experiments.f01_falsification import run_f01b
        run_f01b(tmp_path)
        with open(tmp_path / "run_metadata.json") as f:
            meta = json.load(f)
        assert "F01b-input" in meta["verdicts"]

    def test_tracker_logs_metrics(self, tmp_path):
        from kv_verify.experiments.f01_falsification import run_f01b
        run_f01b(tmp_path)
        with open(tmp_path / "run_metadata.json") as f:
            meta = json.load(f)
        assert "max_input_auroc" in meta["metrics"]

    def test_tracker_caches_result(self, tmp_path):
        from kv_verify.experiments.f01_falsification import run_f01b
        run_f01b(tmp_path)
        cache_path = tmp_path / "cache" / "f01b_result.json"
        assert cache_path.exists()


# ================================================================
# F01c: Format Classifier Baseline
# ================================================================

class TestF01cTracker:
    def test_accepts_tracker_param(self, tmp_path):
        from kv_verify.experiments.f01_falsification import run_f01c
        tracker = ExperimentTracker(
            output_dir=tmp_path, experiment_name="F01c-format",
        )
        result = run_f01c(tmp_path, tracker=tracker)
        assert isinstance(result, ClaimVerification)

    def test_backward_compatible_without_tracker(self, tmp_path):
        from kv_verify.experiments.f01_falsification import run_f01c
        result = run_f01c(tmp_path)
        assert isinstance(result, ClaimVerification)

    def test_tracker_logs_verdict(self, tmp_path):
        from kv_verify.experiments.f01_falsification import run_f01c
        run_f01c(tmp_path)
        with open(tmp_path / "run_metadata.json") as f:
            meta = json.load(f)
        assert "F01c-format" in meta["verdicts"]

    def test_tracker_caches_result(self, tmp_path):
        from kv_verify.experiments.f01_falsification import run_f01c
        run_f01c(tmp_path)
        cache_path = tmp_path / "cache" / "f01c_result.json"
        assert cache_path.exists()


# ================================================================
# F01b-49b: Input-length analysis
# ================================================================

class TestF01b49bTracker:
    def test_accepts_tracker_param(self, tmp_path):
        from kv_verify.experiments.f01b_49b_analysis import run_f01b_49b
        tracker = ExperimentTracker(
            output_dir=tmp_path, experiment_name="F01b-49b",
        )
        result = run_f01b_49b(tmp_path, tracker=tracker)
        assert isinstance(result, ClaimVerification)

    def test_backward_compatible_without_tracker(self, tmp_path):
        from kv_verify.experiments.f01b_49b_analysis import run_f01b_49b
        result = run_f01b_49b(tmp_path)
        assert isinstance(result, ClaimVerification)

    def test_tracker_logs_verdict(self, tmp_path):
        from kv_verify.experiments.f01b_49b_analysis import run_f01b_49b
        run_f01b_49b(tmp_path)
        with open(tmp_path / "run_metadata.json") as f:
            meta = json.load(f)
        assert "F01b-49b-input" in meta["verdicts"]

    def test_tracker_logs_metrics(self, tmp_path):
        from kv_verify.experiments.f01b_49b_analysis import run_f01b_49b
        run_f01b_49b(tmp_path)
        with open(tmp_path / "run_metadata.json") as f:
            meta = json.load(f)
        assert "input_only_auroc" in meta["metrics"]
        assert "residualized_auroc" in meta["metrics"]

    def test_tracker_caches_result(self, tmp_path):
        from kv_verify.experiments.f01b_49b_analysis import run_f01b_49b
        run_f01b_49b(tmp_path)
        cache_path = tmp_path / "cache" / "f01b_49b_result.json"
        assert cache_path.exists()


# ================================================================
# F02: Held-Out Prompt Generalization
# ================================================================

class TestF02Tracker:
    def test_accepts_tracker_param(self, tmp_path):
        from kv_verify.experiments.f02_held_out_input_control import run_f02
        tracker = ExperimentTracker(
            output_dir=tmp_path, experiment_name="F02-held-out",
        )
        results = run_f02(tmp_path, tracker=tracker)
        assert len(results) == 3

    def test_backward_compatible_without_tracker(self, tmp_path):
        from kv_verify.experiments.f02_held_out_input_control import run_f02
        results = run_f02(tmp_path)
        assert len(results) == 3

    def test_tracker_logs_verdicts(self, tmp_path):
        from kv_verify.experiments.f02_held_out_input_control import run_f02
        run_f02(tmp_path)
        with open(tmp_path / "run_metadata.json") as f:
            meta = json.load(f)
        # Should have verdicts for all three paradigms
        assert "F02-deception" in meta["verdicts"]
        assert "F02-refusal" in meta["verdicts"]
        assert "F02-impossibility" in meta["verdicts"]

    def test_tracker_logs_params(self, tmp_path):
        from kv_verify.experiments.f02_held_out_input_control import run_f02
        run_f02(tmp_path)
        with open(tmp_path / "run_metadata.json") as f:
            meta = json.load(f)
        assert meta["params"]["experiment"] == "F02"

    def test_tracker_caches_result(self, tmp_path):
        from kv_verify.experiments.f02_held_out_input_control import run_f02
        run_f02(tmp_path)
        cache_path = tmp_path / "cache" / "f02_result.json"
        assert cache_path.exists()

    def test_external_tracker_receives_logs(self, tmp_path):
        from kv_verify.experiments.f02_held_out_input_control import run_f02
        tracker = ExperimentTracker(
            output_dir=tmp_path / "external", experiment_name="external-f02",
        )
        run_f02(tmp_path, tracker=tracker)
        with open(tmp_path / "external" / "run_metadata.json") as f:
            meta = json.load(f)
        assert "F02-deception" in meta["verdicts"]


# ================================================================
# F03: Cross-Model Transfer
# ================================================================

class TestF03Tracker:
    def test_accepts_tracker_param(self, tmp_path):
        from kv_verify.experiments.f03_cross_model_input_control import run_f03
        tracker = ExperimentTracker(
            output_dir=tmp_path, experiment_name="F03-cross-model",
        )
        result = run_f03(tmp_path, tracker=tracker)
        assert isinstance(result, ClaimVerification)

    def test_backward_compatible_without_tracker(self, tmp_path):
        from kv_verify.experiments.f03_cross_model_input_control import run_f03
        result = run_f03(tmp_path)
        assert isinstance(result, ClaimVerification)

    def test_tracker_logs_verdict(self, tmp_path):
        from kv_verify.experiments.f03_cross_model_input_control import run_f03
        run_f03(tmp_path)
        with open(tmp_path / "run_metadata.json") as f:
            meta = json.load(f)
        assert "F03-cross-model" in meta["verdicts"]

    def test_tracker_logs_metrics(self, tmp_path):
        from kv_verify.experiments.f03_cross_model_input_control import run_f03
        run_f03(tmp_path)
        with open(tmp_path / "run_metadata.json") as f:
            meta = json.load(f)
        assert "mean_cross_raw_auroc" in meta["metrics"]
        assert "mean_cross_resid_auroc" in meta["metrics"]

    def test_tracker_caches_result(self, tmp_path):
        from kv_verify.experiments.f03_cross_model_input_control import run_f03
        run_f03(tmp_path)
        cache_path = tmp_path / "cache" / "f03_result.json"
        assert cache_path.exists()


# ================================================================
# F04: Cross-Condition Transfer Validity
# ================================================================

class TestF04Tracker:
    def test_accepts_tracker_param(self, tmp_path):
        from kv_verify.experiments.f04_cross_condition_validity import run_f04
        tracker = ExperimentTracker(
            output_dir=tmp_path, experiment_name="F04-transfer",
        )
        result = run_f04(tmp_path, tracker=tracker)
        assert isinstance(result, ClaimVerification)

    def test_backward_compatible_without_tracker(self, tmp_path):
        from kv_verify.experiments.f04_cross_condition_validity import run_f04
        result = run_f04(tmp_path)
        assert isinstance(result, ClaimVerification)

    def test_tracker_logs_verdict(self, tmp_path):
        from kv_verify.experiments.f04_cross_condition_validity import run_f04
        run_f04(tmp_path)
        with open(tmp_path / "run_metadata.json") as f:
            meta = json.load(f)
        assert "F04-transfer" in meta["verdicts"]

    def test_tracker_logs_metrics(self, tmp_path):
        from kv_verify.experiments.f04_cross_condition_validity import run_f04
        run_f04(tmp_path)
        with open(tmp_path / "run_metadata.json") as f:
            meta = json.load(f)
        assert "critical_failures" in meta["metrics"]

    def test_tracker_caches_result(self, tmp_path):
        from kv_verify.experiments.f04_cross_condition_validity import run_f04
        run_f04(tmp_path)
        cache_path = tmp_path / "cache" / "f04_result.json"
        assert cache_path.exists()
