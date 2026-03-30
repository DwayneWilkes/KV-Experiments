"""Tests for pre-registration compatibility check (Task 5.3)."""

import pytest

from kv_verify.lib.dataset_validation import validate_dataset


from kv_verify.tests.conftest import make_item as _item

class TestPreregCompatibility:

    def test_matching_plan_passes(self):
        items = [_item("A", f"a{i}") for i in range(20)]
        items += [_item("B", f"b{i}") for i in range(20)]
        plan = {"planned_n": {"A": 20, "B": 20}, "alpha": 0.05}
        report = validate_dataset(items, tier=3, config_overrides={"prereg_plan": plan})
        cr = report.checks.get("prereg_compatibility")
        assert cr is not None
        assert cr.passed

    def test_deviation_documented(self):
        items = [_item("A", f"a{i}") for i in range(18)]
        items += [_item("B", f"b{i}") for i in range(20)]
        plan = {"planned_n": {"A": 20, "B": 20}, "alpha": 0.05}
        report = validate_dataset(items, tier=3, config_overrides={"prereg_plan": plan})
        cr = report.checks.get("prereg_compatibility")
        assert cr is not None
        assert not cr.passed
        assert "A" in str(cr.metrics.get("deviations", []))

    def test_no_plan_skips(self):
        items = [_item("A", f"a{i}") for i in range(20)]
        items += [_item("B", f"b{i}") for i in range(20)]
        report = validate_dataset(items, tier=3)
        cr = report.checks.get("prereg_compatibility")
        assert cr is not None
        assert cr.metrics.get("skipped") is True
