"""Tests for dataset validation core (Task 2.1).

Tests the entry point, report structure, and basic validation flow.
"""

import json

import numpy as np
import pytest

from kv_verify.lib.dataset_validation import (
    CheckResult,
    DatasetReport,
    validate_dataset,
)


def _make_items(n_per_condition=20, conditions=("A", "B")):
    """Helper: create a valid balanced dataset."""
    items = []
    for cond in conditions:
        for i in range(n_per_condition):
            items.append({
                "condition": cond,
                "prompt": f"This is a {cond.lower()} prompt number {i} about various topics.",
                "features": {
                    "norm_per_token": float(np.random.default_rng(i).normal(1.0, 0.1)),
                    "key_rank": float(np.random.default_rng(i + 100).normal(5.0, 0.5)),
                    "key_entropy": float(np.random.default_rng(i + 200).normal(2.0, 0.2)),
                    "n_tokens": 50 + i,
                },
            })
    return items


class TestValidateDatasetEntryPoint:

    def test_empty_list_returns_fail(self):
        report = validate_dataset([], tier=0)
        assert report.overall_verdict == "FAIL"
        assert not report.overall_pass

    def test_valid_dataset_returns_pass(self):
        items = _make_items(20)
        report = validate_dataset(items, tier=0)
        assert report.overall_pass
        assert report.overall_verdict == "PASS"

    def test_returns_dataset_report(self):
        items = _make_items(10)
        report = validate_dataset(items, tier=0)
        assert isinstance(report, DatasetReport)

    def test_default_tier_is_1(self):
        items = _make_items(20)
        report = validate_dataset(items)
        assert report.tier == 1

    def test_tier_0_runs_fewer_checks(self):
        items = _make_items(20)
        report_0 = validate_dataset(items, tier=0)
        report_1 = validate_dataset(items, tier=1)
        assert len(report_0.checks) <= len(report_1.checks)

    def test_standalone_no_pipeline_imports(self):
        """validate_dataset must not import from pipeline or experiments."""
        import kv_verify.lib.dataset_validation as mod
        source = open(mod.__file__).read()
        assert "from kv_verify.pipeline" not in source
        assert "from kv_verify.experiments" not in source
        assert "from kv_verify.tracking" not in source


class TestCheckResult:

    def test_has_required_fields(self):
        cr = CheckResult(name="test", passed=True, tier=0, metrics={"x": 1}, details="ok")
        assert cr.name == "test"
        assert cr.passed is True
        assert cr.tier == 0
        assert cr.metrics == {"x": 1}
        assert cr.details == "ok"


class TestDatasetReport:

    def test_report_fields_have_correct_types(self):
        report = validate_dataset(_make_items(10), tier=0)
        assert isinstance(report.schema_version, str)
        assert isinstance(report.tier, int)
        assert isinstance(report.overall_pass, bool)
        assert report.overall_verdict in ("PASS", "INCONCLUSIVE", "FAIL")
        assert isinstance(report.checks, dict)
        assert isinstance(report.nominal_n, dict)
        assert len(report.timestamp) > 0
        assert len(report.config_hash) > 0

    def test_nominal_n_per_condition(self):
        items = _make_items(15, conditions=("X", "Y"))
        report = validate_dataset(items, tier=0)
        assert report.nominal_n == {"X": 15, "Y": 15}

    def test_json_round_trip(self):
        items = _make_items(10)
        report = validate_dataset(items, tier=0)
        json_str = json.dumps(report.to_dict())
        loaded = json.loads(json_str)
        assert loaded["overall_verdict"] == report.overall_verdict
        assert loaded["schema_version"] == report.schema_version
        assert loaded["tier"] == report.tier

    def test_single_condition_fails(self):
        items = [{"condition": "A", "prompt": "test", "features": {"n_tokens": 50}}] * 5
        report = validate_dataset(items, tier=0)
        assert report.overall_verdict == "FAIL"
