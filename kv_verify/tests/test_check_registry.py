"""Tests for @check decorator and registry (Task 2.2)."""

import pytest

from kv_verify.lib.dataset_validation import (
    _CHECKS,
    _run_checks,
    check,
    list_checks,
)


class TestCheckDecorator:

    def test_registered_checks_exist(self):
        """At least the Tier 0 checks should be registered."""
        names = {c["name"] for c in _CHECKS}
        assert "structural" in names
        assert "duplicates" in names
        assert "class_balance" in names

    def test_check_has_tier(self):
        structural = next(c for c in _CHECKS if c["name"] == "structural")
        assert structural["tier"] == 0

    def test_check_has_callable(self):
        structural = next(c for c in _CHECKS if c["name"] == "structural")
        assert callable(structural["fn"])


class TestListChecks:

    def test_list_tier_0(self):
        checks = list_checks(tier=0)
        assert all(c["tier"] <= 0 for c in checks)
        assert len(checks) >= 3  # structural, duplicates, class_balance

    def test_list_tier_1_includes_tier_0(self):
        checks_0 = list_checks(tier=0)
        checks_1 = list_checks(tier=1)
        assert len(checks_1) >= len(checks_0)


class TestRunChecks:

    def test_tier_0_only_runs_tier_0(self):
        items = [
            {"condition": "A", "prompt": "hello", "features": {"n_tokens": 50}},
            {"condition": "B", "prompt": "world", "features": {"n_tokens": 55}},
        ] * 5
        results = _run_checks(items, tier=0, config={"balance_ratio_threshold": 2.0, "condition_field": "condition"}, shared={})
        for cr in results.values():
            assert cr.tier == 0
