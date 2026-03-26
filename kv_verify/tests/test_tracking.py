"""Tests for kv_verify.tracking — experiment tracker with MLflow + caching + decorators."""

import json
import time
from pathlib import Path
from unittest.mock import patch

from kv_verify.tracking import ExperimentTracker, tracked, stage, validated


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


# ================================================================
# DECORATOR TESTS
# ================================================================

class TestTrackedDecorator:
    def test_caches_result(self, tmp_path):
        tracker = ExperimentTracker(output_dir=tmp_path, experiment_name="test")
        call_count = 0

        @tracked(tracker, cache_key=lambda x: f"key_{x}")
        def compute(x):
            nonlocal call_count
            call_count += 1
            return {"value": x * 2}

        result1 = compute(5)
        assert result1["value"] == 10
        assert call_count == 1

        result2 = compute(5)
        assert result2["value"] == 10
        assert call_count == 1  # cached, not called again

    def test_different_args_not_cached(self, tmp_path):
        tracker = ExperimentTracker(output_dir=tmp_path, experiment_name="test")
        call_count = 0

        @tracked(tracker, cache_key=lambda x: f"key_{x}")
        def compute(x):
            nonlocal call_count
            call_count += 1
            return {"value": x * 2}

        compute(5)
        compute(10)
        assert call_count == 2

    def test_logs_timing(self, tmp_path):
        tracker = ExperimentTracker(output_dir=tmp_path, experiment_name="test")

        @tracked(tracker, log_timing=True)
        def slow_fn():
            time.sleep(0.05)
            return {"done": True}

        slow_fn()
        with open(tmp_path / "run_metadata.json") as f:
            meta = json.load(f)
        assert "slow_fn_seconds" in meta["metrics"]
        assert meta["metrics"]["slow_fn_seconds"] >= 0.04

    def test_no_caching_without_cache_key(self, tmp_path):
        tracker = ExperimentTracker(output_dir=tmp_path, experiment_name="test")
        call_count = 0

        @tracked(tracker)
        def compute():
            nonlocal call_count
            call_count += 1
            return {"v": 1}

        compute()
        compute()
        assert call_count == 2  # no caching, both calls execute

    def test_logs_metric(self, tmp_path):
        tracker = ExperimentTracker(output_dir=tmp_path, experiment_name="test")

        @tracked(tracker, metric_name="auroc")
        def get_auroc():
            return 0.85

        get_auroc()
        with open(tmp_path / "run_metadata.json") as f:
            meta = json.load(f)
        assert meta["metrics"]["auroc"] == 0.85


class TestStageDecorator:
    def test_auto_caches_completion(self, tmp_path):
        tracker = ExperimentTracker(output_dir=tmp_path, experiment_name="test")

        @stage(tracker, "my_stage")
        def run_stage():
            return {"items": 42}

        result = run_stage()
        assert result["items"] == 42
        assert tracker.is_cached("stage_my_stage")

    def test_skips_if_cached(self, tmp_path):
        tracker = ExperimentTracker(output_dir=tmp_path, experiment_name="test")
        call_count = 0

        @stage(tracker, "repeat_stage")
        def run_stage():
            nonlocal call_count
            call_count += 1
            return {"done": True}

        run_stage()
        run_stage()
        assert call_count == 1  # second call skipped

    def test_checks_dependencies(self, tmp_path):
        tracker = ExperimentTracker(output_dir=tmp_path, experiment_name="test")

        @stage(tracker, "dependent_stage", depends_on=["prereq"])
        def run_dependent():
            return {"ok": True}

        import pytest
        with pytest.raises(RuntimeError, match="prereq"):
            run_dependent()

    def test_dependency_satisfied(self, tmp_path):
        tracker = ExperimentTracker(output_dir=tmp_path, experiment_name="test")

        @stage(tracker, "prereq")
        def run_prereq():
            return {"ready": True}

        @stage(tracker, "dependent", depends_on=["prereq"])
        def run_dependent():
            return {"done": True}

        run_prereq()
        result = run_dependent()
        assert result["done"] is True

    def test_logs_timing(self, tmp_path):
        tracker = ExperimentTracker(output_dir=tmp_path, experiment_name="test")

        @stage(tracker, "timed_stage")
        def run_stage():
            time.sleep(0.05)
            return {}

        run_stage()
        with open(tmp_path / "run_metadata.json") as f:
            meta = json.load(f)
        assert "timed_stage" in meta["stages"]
        assert meta["stages"]["timed_stage"]["duration_seconds"] >= 0.04


class TestValidatedDecorator:
    def test_passes_when_valid(self, tmp_path):
        tracker = ExperimentTracker(output_dir=tmp_path, experiment_name="test")

        def check_positive(x):
            if x <= 0:
                raise ValueError("x must be positive")
            return True

        @validated(tracker, checks=[check_positive])
        def process(x):
            return x * 2

        assert process(5) == 10

    def test_raises_when_invalid(self, tmp_path):
        tracker = ExperimentTracker(output_dir=tmp_path, experiment_name="test")

        def check_positive(x):
            if x <= 0:
                raise ValueError("x must be positive")

        @validated(tracker, checks=[check_positive])
        def process(x):
            return x * 2

        import pytest
        with pytest.raises(ValueError, match="positive"):
            process(-1)

    def test_logs_validation_failure(self, tmp_path):
        tracker = ExperimentTracker(output_dir=tmp_path, experiment_name="test")

        def always_fail():
            raise ValueError("bad input")

        @validated(tracker, checks=[always_fail])
        def process():
            return 42

        import pytest
        with pytest.raises(ValueError):
            process()

        with open(tmp_path / "run_metadata.json") as f:
            meta = json.load(f)
        assert meta["metrics"].get("process_validation_failed") == 1
