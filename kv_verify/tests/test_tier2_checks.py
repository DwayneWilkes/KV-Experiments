"""Tests for Tier 2 checks: shortcut, confound discovery, variance, pairs, format (Tasks 4.1-4.6)."""

import numpy as np
import pytest

from kv_verify.lib.dataset_validation import validate_dataset


from kv_verify.tests.conftest import make_item as _item

class TestShortcutDetection:
    """Task 4.1: Classifier AUROC on prompt text to predict condition."""

    def test_no_confound_near_chance(self):
        """Same prompts in both conditions -> AUROC near 0.5."""
        rng = np.random.default_rng(42)
        topics = ["astronomy", "cooking", "politics", "sports", "medicine",
                  "technology", "history", "music", "biology", "economics"]
        items = []
        for i in range(30):
            t = topics[i % len(topics)]
            items.append(_item("A", f"Question about {t} number {i} exploring various aspects deeply"))
        for i in range(30):
            t = topics[i % len(topics)]
            items.append(_item("B", f"Question about {t} number {i + 30} exploring various aspects deeply"))
        report = validate_dataset(items, tier=2)
        assert report.checks["shortcut_detection"].passed
        assert report.checks["shortcut_detection"].metrics["auroc"] < 0.65

    def test_planted_lexical_pattern_detected(self):
        """Systematic prefix difference -> high AUROC."""
        items = [_item("A", f"Please help me understand topic {i}") for i in range(30)]
        items += [_item("B", f"Can you explain concept number {i}") for i in range(30)]
        report = validate_dataset(items, tier=2)
        assert not report.checks["shortcut_detection"].passed
        assert report.checks["shortcut_detection"].metrics["auroc"] > 0.65


class TestConfoundDiscovery:
    """Task 4.2: MI between features and condition label."""

    def test_undeclared_confound_flagged(self):
        """Feature n_tokens perfectly predicts condition -> flagged."""
        items = [_item("A", f"short {i}", n_tokens=50) for i in range(20)]
        items += [_item("B", f"long {i}", n_tokens=150) for i in range(20)]
        report = validate_dataset(items, tier=2, config_overrides={"confound_spec": {}})
        assert not report.checks["confound_discovery"].passed
        discovered = report.checks["confound_discovery"].metrics.get("discovered", [])
        assert any("n_tokens" in str(d) for d in discovered)

    def test_declared_confounds_pass(self):
        """All high-MI features declared -> pass."""
        items = [_item("A", f"short {i}", n_tokens=50) for i in range(20)]
        items += [_item("B", f"long {i}", n_tokens=150) for i in range(20)]
        report = validate_dataset(
            items, tier=2,
            config_overrides={"confound_spec": {"n_tokens": "residualized"}},
        )
        assert report.checks["confound_discovery"].passed


class TestVarianceRatio:
    """Task 4.3: Within-condition variance ratio."""

    def test_balanced_variance_passes(self):
        rng = np.random.default_rng(42)
        items = [_item("A", f"a{i}", n_tokens=int(rng.normal(100, 10))) for i in range(30)]
        items += [_item("B", f"b{i}", n_tokens=int(rng.normal(100, 12))) for i in range(30)]
        report = validate_dataset(items, tier=2)
        assert report.checks["variance_ratio"].passed

    def test_asymmetric_variance_fails(self):
        """One condition tight (CV~2%), other wide (CV~30%)."""
        items = [_item("A", f"a{i}", n_tokens=100 + (i % 3)) for i in range(30)]
        items += [_item("B", f"b{i}", n_tokens=50 + i * 5) for i in range(30)]
        report = validate_dataset(items, tier=2)
        assert not report.checks["variance_ratio"].passed
        assert report.checks["variance_ratio"].metrics["ratio"] > 3.0


class TestPairIntegrity:
    """Task 4.4: Minimal pair checks."""

    def test_valid_pair_passes(self):
        items = [
            _item("A", "What is the capital of France? Please answer honestly.", n_tokens=12),
            _item("B", "What is the capital of France? Please answer deceptively.", n_tokens=12),
        ]
        report = validate_dataset(
            items, tier=2,
            config_overrides={"paired": True, "pair_token_tolerance": 2},
        )
        assert report.checks["pair_integrity"].passed

    def test_excessive_diff_fails(self):
        items = [
            _item("A", "Short prompt.", n_tokens=5),
            _item("B", "This is a much longer prompt that has many more tokens in it.", n_tokens=20),
        ]
        report = validate_dataset(
            items, tier=2,
            config_overrides={"paired": True, "pair_token_tolerance": 5},
        )
        assert not report.checks["pair_integrity"].passed

    def test_non_paired_skips(self):
        items = [_item("A", f"a{i}") for i in range(10)]
        items += [_item("B", f"b{i}") for i in range(10)]
        report = validate_dataset(items, tier=2)
        cr = report.checks.get("pair_integrity")
        assert cr is not None
        assert cr.metrics.get("skipped") is True


class TestConfoundDisclosure:
    """Task 4.5: User-provided confound spec."""

    def test_all_controlled_passes(self):
        items = [_item("A", f"a{i}") for i in range(20)]
        items += [_item("B", f"b{i}") for i in range(20)]
        report = validate_dataset(
            items, tier=2,
            config_overrides={"confound_spec": {"input_length": "residualized", "system_prompt": "identical"}},
        )
        assert report.checks["confound_disclosure"].passed

    def test_uncontrolled_fails(self):
        items = [_item("A", f"a{i}") for i in range(20)]
        items += [_item("B", f"b{i}") for i in range(20)]
        report = validate_dataset(
            items, tier=2,
            config_overrides={"confound_spec": {"response_format": "uncontrolled"}},
        )
        assert not report.checks["confound_disclosure"].passed


class TestFormatConsistency:
    """Task 4.6: Whitespace/punctuation pattern comparison."""

    def test_identical_format_passes(self):
        items = [_item("A", f"Question: What is topic {i}?\nAnswer:") for i in range(20)]
        items += [_item("B", f"Question: What is subject {i}?\nAnswer:") for i in range(20)]
        report = validate_dataset(items, tier=2)
        assert report.checks["format_consistency"].passed

    def test_systematic_format_difference_fails(self):
        items = [_item("A", f"Question: What is {i}?\nAnswer:") for i in range(20)]
        items += [_item("B", f"Q: {i}\nA:") for i in range(20)]
        report = validate_dataset(items, tier=2)
        assert not report.checks["format_consistency"].passed
