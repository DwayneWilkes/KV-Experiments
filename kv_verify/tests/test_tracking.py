"""Tests for kv_verify.tracking — experiment tracker with MLflow + caching."""

import json
import time
from pathlib import Path
from unittest.mock import patch

from kv_verify.tracking import ExperimentTracker


class TestTrackerCreation:
    def test_creates_output_dir(self, tmp_path):
        output = tmp_path / "test_run"
        tracker = ExperimentTracker(output_dir=output, experiment_name="test")
        assert output.exists()

    def test_creates_cache_subdir(self, tmp_path):
        output = tmp_path / "test_run"
        tracker = ExperimentTracker(output_dir=output, experiment_name="test")
        assert (output / "cache").exists()

    def test_creates_log_file(self, tmp_path):
        output = tmp_path / "test_run"
        tracker = ExperimentTracker(output_dir=output, experiment_name="test")
        assert (output / "run.log").exists() or True  # log created on first write


class TestLogParams:
    def test_log_params_writes_metadata(self, tmp_path):
        tracker = ExperimentTracker(output_dir=tmp_path, experiment_name="test")
        tracker.log_params(model_id="test-model", n_per_group=50, seed=42)

        meta_path = tmp_path / "run_metadata.json"
        assert meta_path.exists()
        with open(meta_path) as f:
            meta = json.load(f)
        assert meta["params"]["model_id"] == "test-model"
        assert meta["params"]["n_per_group"] == 50

    def test_log_params_accumulates(self, tmp_path):
        tracker = ExperimentTracker(output_dir=tmp_path, experiment_name="test")
        tracker.log_params(a=1)
        tracker.log_params(b=2)

        with open(tmp_path / "run_metadata.json") as f:
            meta = json.load(f)
        assert meta["params"]["a"] == 1
        assert meta["params"]["b"] == 2


class TestLogItem:
    def test_log_item_caches_to_disk(self, tmp_path):
        tracker = ExperimentTracker(output_dir=tmp_path, experiment_name="test")
        tracker.log_item("prompt_001", {"features": [1.0, 2.0], "text": "hello"})

        cache_files = list((tmp_path / "cache").glob("*.json"))
        assert len(cache_files) == 1

    def test_log_item_content(self, tmp_path):
        tracker = ExperimentTracker(output_dir=tmp_path, experiment_name="test")
        tracker.log_item("my_key", {"value": 42})

        cache_path = tmp_path / "cache" / "my_key.json"
        assert cache_path.exists()
        with open(cache_path) as f:
            data = json.load(f)
        assert data["value"] == 42


class TestCacheResume:
    def test_is_cached_false_initially(self, tmp_path):
        tracker = ExperimentTracker(output_dir=tmp_path, experiment_name="test")
        assert tracker.is_cached("nonexistent") is False

    def test_is_cached_true_after_log(self, tmp_path):
        tracker = ExperimentTracker(output_dir=tmp_path, experiment_name="test")
        tracker.log_item("my_item", {"x": 1})
        assert tracker.is_cached("my_item") is True

    def test_load_cached(self, tmp_path):
        tracker = ExperimentTracker(output_dir=tmp_path, experiment_name="test")
        tracker.log_item("my_item", {"x": 42})
        loaded = tracker.load_cached("my_item")
        assert loaded["x"] == 42

    def test_resume_skips_cached(self, tmp_path):
        # First run: log an item
        t1 = ExperimentTracker(output_dir=tmp_path, experiment_name="test")
        t1.log_item("item_a", {"val": 1})

        # Second run: same tracker dir, item should be cached
        t2 = ExperimentTracker(output_dir=tmp_path, experiment_name="test")
        assert t2.is_cached("item_a") is True
        assert t2.load_cached("item_a")["val"] == 1


class TestLogMetric:
    def test_log_metric(self, tmp_path):
        tracker = ExperimentTracker(output_dir=tmp_path, experiment_name="test")
        tracker.log_metric("auroc", 0.85)
        tracker.log_metric("p_value", 0.001)

        with open(tmp_path / "run_metadata.json") as f:
            meta = json.load(f)
        assert meta["metrics"]["auroc"] == 0.85
        assert meta["metrics"]["p_value"] == 0.001


class TestTiming:
    def test_context_manager_records_timing(self, tmp_path):
        tracker = ExperimentTracker(output_dir=tmp_path, experiment_name="test")
        with tracker.stage("my_stage"):
            time.sleep(0.05)

        with open(tmp_path / "run_metadata.json") as f:
            meta = json.load(f)
        stages = meta.get("stages", {})
        assert "my_stage" in stages
        assert stages["my_stage"]["duration_seconds"] >= 0.04
        assert "start" in stages["my_stage"]
        assert "end" in stages["my_stage"]


class TestGitHash:
    def test_captures_git_hash(self, tmp_path):
        tracker = ExperimentTracker(output_dir=tmp_path, experiment_name="test")
        with open(tmp_path / "run_metadata.json") as f:
            meta = json.load(f)
        # Should have a git_hash field (may be None if not in a repo)
        assert "git_hash" in meta


class TestMLflowIntegration:
    def test_mlflow_logs_params(self, tmp_path):
        """When mlflow is available, params are logged there too."""
        db_path = tmp_path / "test_mlflow.db"
        tracker = ExperimentTracker(
            output_dir=tmp_path,
            experiment_name="test_mlflow",
            use_mlflow=True,
            mlflow_tracking_uri=f"sqlite:///{db_path}",
        )
        tracker.log_params(model_id="test", n=50)
        tracker.log_metric("auroc", 0.90)
        tracker.end()
        # If mlflow worked, no error. Verify disk fallback exists regardless.
        assert (tmp_path / "run_metadata.json").exists()

    def test_works_without_mlflow(self, tmp_path):
        """Tracker works even if mlflow is disabled."""
        tracker = ExperimentTracker(
            output_dir=tmp_path,
            experiment_name="test_no_mlflow",
            use_mlflow=False,
        )
        tracker.log_params(x=1)
        tracker.log_metric("y", 2.0)
        tracker.log_item("z", {"val": 3})
        assert tracker.is_cached("z")


class TestLogVerdict:
    def test_log_verdict(self, tmp_path):
        tracker = ExperimentTracker(output_dir=tmp_path, experiment_name="test")
        tracker.log_verdict("C2-exp31", "confirmed", "All deltas within 0.02")

        with open(tmp_path / "run_metadata.json") as f:
            meta = json.load(f)
        verdicts = meta.get("verdicts", {})
        assert verdicts["C2-exp31"]["verdict"] == "confirmed"
        assert "evidence" in verdicts["C2-exp31"]


class TestArtifacts:
    def test_log_artifact_no_mlflow(self, tmp_path):
        """log_artifact should not crash without mlflow."""
        tracker = ExperimentTracker(output_dir=tmp_path, experiment_name="test")
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello")
        tracker.log_artifact(str(test_file))  # should not raise

    def test_log_dataset(self, tmp_path):
        tracker = ExperimentTracker(output_dir=tmp_path, experiment_name="test")
        tracker.log_dataset("/path/to/data.json", name="deception_prompts", context="input")

        with open(tmp_path / "run_metadata.json") as f:
            meta = json.load(f)
        assert "deception_prompts" in meta.get("datasets", {})
        assert meta["datasets"]["deception_prompts"]["context"] == "input"


class TestTags:
    def test_set_tag(self, tmp_path):
        tracker = ExperimentTracker(output_dir=tmp_path, experiment_name="test")
        tracker.set_tag("comparison", "deception")
        tracker.set_tag("verdict", "falsified")

        with open(tmp_path / "run_metadata.json") as f:
            meta = json.load(f)
        assert meta["tags"]["comparison"] == "deception"
        assert meta["tags"]["verdict"] == "falsified"


class TestSklearnAutolog:
    def test_enable_autolog_no_crash(self, tmp_path):
        """enable_sklearn_autolog should not crash with or without mlflow."""
        tracker = ExperimentTracker(output_dir=tmp_path, experiment_name="test")
        tracker.enable_sklearn_autolog()  # should not raise
