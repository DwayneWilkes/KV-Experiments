"""Tests for Tier 0 checks: structural, duplicates, class_balance (Tasks 2.3-2.5)."""

import json

import numpy as np
import pytest

from kv_verify.lib.dataset_validation import validate_dataset


from kv_verify.tests.conftest import make_item as _item

class TestStructuralCheck:

    def test_valid_items_pass(self):
        items = [_item("A"), _item("B")] * 5
        report = validate_dataset(items, tier=0)
        assert report.checks["structural"].passed

    def test_missing_condition_label(self):
        items = [{"prompt": "no condition", "features": {"n_tokens": 50}}] * 5
        items += [_item("B")] * 5
        report = validate_dataset(items, tier=0)
        assert not report.checks["structural"].passed
        assert "missing" in report.checks["structural"].details

    def test_non_numeric_features(self):
        items = [_item("A")] * 5 + [_item("B")] * 5
        items[0]["features"]["bad"] = "not a number"
        report = validate_dataset(items, tier=0)
        assert not report.checks["structural"].passed


class TestDuplicatesCheck:

    def test_no_duplicates_pass(self):
        items = [_item("A", f"prompt {i}") for i in range(10)]
        items += [_item("B", f"prompt {i+10}") for i in range(10)]
        report = validate_dataset(items, tier=0)
        assert report.checks["duplicates"].passed

    def test_within_condition_duplicates_flagged(self):
        items = [_item("A", "same prompt")] * 5 + [_item("B", f"unique {i}") for i in range(5)]
        report = validate_dataset(items, tier=0)
        assert not report.checks["duplicates"].passed
        metrics = report.checks["duplicates"].metrics
        assert len(metrics["within_condition"]) > 0
        assert metrics["adjusted_n"]["A"] < 5

    def test_cross_condition_duplicates_flagged_as_leakage(self):
        items = [_item("A", "leaked prompt")] * 3
        items += [_item("B", "leaked prompt")] * 3
        items += [_item("B", "unique")] * 2
        report = validate_dataset(items, tier=0)
        assert not report.checks["duplicates"].passed
        assert report.checks["duplicates"].metrics["leakage_risk"]


class TestClassBalanceCheck:

    def test_balanced_passes(self):
        items = [_item("A", f"a{i}") for i in range(20)]
        items += [_item("B", f"b{i}") for i in range(20)]
        report = validate_dataset(items, tier=0)
        assert report.checks["class_balance"].passed
        assert report.checks["class_balance"].metrics["ratio"] == 1.0

    def test_5_to_1_fails(self):
        items = [_item("A", f"a{i}") for i in range(50)]
        items += [_item("B", f"b{i}") for i in range(10)]
        report = validate_dataset(items, tier=0)
        assert not report.checks["class_balance"].passed
        assert report.checks["class_balance"].metrics["ratio"] == 5.0


class TestVerdictLogic:

    def test_tier0_fail_gives_fail_verdict(self):
        """Missing condition labels -> structural fails -> FAIL."""
        items = [{"prompt": "no cond", "features": {}}] * 5
        items += [_item("B")] * 5
        report = validate_dataset(items, tier=0)
        assert report.overall_verdict == "FAIL"

    def test_all_pass_gives_pass_verdict(self):
        items = [_item("A", f"a{i}") for i in range(20)]
        items += [_item("B", f"b{i}") for i in range(20)]
        report = validate_dataset(items, tier=0)
        assert report.overall_verdict == "PASS"
