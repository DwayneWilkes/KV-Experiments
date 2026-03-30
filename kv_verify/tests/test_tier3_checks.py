"""Tests for Tier 3 checks: provenance hash, metadata, prereg, measurement (Tasks 5.1-5.4)."""

import json

import numpy as np
import pytest

from kv_verify.lib.dataset_validation import validate_dataset


from kv_verify.tests.conftest import make_item as _item

class TestProvenanceHash:
    """Task 5.1: SHA-256 of canonical dataset."""

    def test_same_data_same_hash(self):
        items = [_item("A", f"prompt {i}") for i in range(10)]
        items += [_item("B", f"prompt {i+10}") for i in range(10)]
        r1 = validate_dataset(items, tier=3)
        r2 = validate_dataset(items, tier=3)
        assert r1.dataset_hash is not None
        assert r1.dataset_hash == r2.dataset_hash

    def test_single_change_different_hash(self):
        items1 = [_item("A", f"prompt {i}") for i in range(10)]
        items1 += [_item("B", f"prompt {i+10}") for i in range(10)]
        items2 = list(items1)
        items2[0] = _item("A", "prompt 0 modified")
        r1 = validate_dataset(items1, tier=3)
        r2 = validate_dataset(items2, tier=3)
        assert r1.dataset_hash != r2.dataset_hash

    def test_hash_is_sha256(self):
        items = [_item("A", f"question about astronomy {i}") for i in range(10)]
        items += [_item("B", f"question about biology {i}") for i in range(10)]
        report = validate_dataset(items, tier=3)
        assert len(report.dataset_hash) == 64  # SHA-256 hex = 64 chars

    def test_hash_none_below_tier3(self):
        items = [_item("A", f"a{i}") for i in range(10)]
        items += [_item("B", f"b{i}") for i in range(10)]
        report = validate_dataset(items, tier=2)
        assert report.dataset_hash is None


class TestMetadataCompleteness:
    """Task 5.2: Required metadata fields."""

    def test_complete_metadata_passes(self):
        items = [_item("A", f"a{i}") for i in range(20)]
        items += [_item("B", f"b{i}") for i in range(20)]
        meta = {
            "generation_method": "manual",
            "llm_generated": False,
            "author": "test",
            "decoding_strategy": "greedy",
            "contamination_risk": "none known",
        }
        report = validate_dataset(items, tier=3, config_overrides={"dataset_metadata": meta})
        assert report.checks["metadata_completeness"].passed

    def test_missing_llm_generated_fails(self):
        items = [_item("A", f"a{i}") for i in range(20)]
        items += [_item("B", f"b{i}") for i in range(20)]
        meta = {"generation_method": "manual", "author": "test"}
        report = validate_dataset(items, tier=3, config_overrides={"dataset_metadata": meta})
        assert not report.checks["metadata_completeness"].passed


class TestMeasurementValidation:
    """Task 5.4: Optional measurement validator hook."""

    def test_reliable_extractor_passes(self):
        def reliable_validator():
            return {"icc": 0.98, "passed": True}
        items = [_item("A", f"a{i}") for i in range(20)]
        items += [_item("B", f"b{i}") for i in range(20)]
        report = validate_dataset(
            items, tier=3,
            config_overrides={"measurement_validator": reliable_validator},
        )
        assert report.checks["measurement_validation"].passed
        assert report.checks["measurement_validation"].metrics["icc"] == 0.98

    def test_unreliable_extractor_fails(self):
        def unreliable_validator():
            return {"icc": 0.45, "passed": False}
        items = [_item("A", f"a{i}") for i in range(20)]
        items += [_item("B", f"b{i}") for i in range(20)]
        report = validate_dataset(
            items, tier=3,
            config_overrides={"measurement_validator": unreliable_validator},
        )
        assert not report.checks["measurement_validation"].passed

    def test_absent_validator_skips(self):
        items = [_item("A", f"a{i}") for i in range(20)]
        items += [_item("B", f"b{i}") for i in range(20)]
        report = validate_dataset(items, tier=3)
        cr = report.checks.get("measurement_validation")
        assert cr is not None
        assert cr.metrics.get("skipped") is True
