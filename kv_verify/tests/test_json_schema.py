"""Tests for DatasetReport JSON Schema (Task 5.5)."""

import json
from pathlib import Path

import pytest

from kv_verify.lib.dataset_validation import validate_dataset


from kv_verify.tests.conftest import make_item as _item

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "schemas" / "dataset-report-v1.0.json"


class TestJsonSchema:

    def test_schema_file_exists(self):
        assert SCHEMA_PATH.exists(), f"Schema not found at {SCHEMA_PATH}"

    def test_schema_has_required_structure(self):
        data = json.loads(SCHEMA_PATH.read_text())
        assert "$schema" in data
        assert data["type"] == "object"
        assert "required" in data
        assert "overall_verdict" in data["required"]

    def test_valid_report_passes_schema(self):
        jsonschema = pytest.importorskip("jsonschema")
        schema = json.loads(SCHEMA_PATH.read_text())
        items = [_item("A", f"a{i}") for i in range(20)]
        items += [_item("B", f"b{i}") for i in range(20)]
        report = validate_dataset(items, tier=0)
        jsonschema.validate(report.to_dict(), schema)

    def test_invalid_report_fails_schema(self):
        jsonschema = pytest.importorskip("jsonschema")
        schema = json.loads(SCHEMA_PATH.read_text())
        # Missing required fields
        bad = {"overall_verdict": 123}  # wrong type
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(bad, schema)
