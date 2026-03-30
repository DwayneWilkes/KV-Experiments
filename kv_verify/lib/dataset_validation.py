"""Dataset validation library for ML experiments.

Reusable pre-flight quality gate. Modality-agnostic, tiered, composable.
No imports from pipeline, experiments, or tracking modules.

Usage:
    from kv_verify.lib.dataset_validation import validate_dataset

    report = validate_dataset(items, tier=1)
    print(report.overall_verdict)  # PASS / INCONCLUSIVE / FAIL
"""

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional


@dataclass
class CheckResult:
    """Result of a single validation check."""
    name: str
    passed: bool
    tier: int
    metrics: Dict[str, Any] = field(default_factory=dict)
    details: str = ""


@dataclass
class DatasetReport:
    """Complete validation report for a dataset."""
    schema_version: str = "1.0.0"
    tier: int = 1
    dataset_hash: Optional[str] = None
    overall_pass: bool = False
    overall_verdict: str = "FAIL"
    checks: Dict[str, CheckResult] = field(default_factory=dict)
    nominal_n: Dict[str, int] = field(default_factory=dict)
    effective_n: Optional[Dict[str, float]] = None
    metadata: Optional[Dict[str, Any]] = None
    confound_summary: Optional[Dict[str, Any]] = None
    timestamp: str = ""
    config_hash: str = ""
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict."""
        d = asdict(self)
        # CheckResult dataclass already serialized by asdict
        return d


# ================================================================
# Check registry (v1: simple list, not full decorator registry)
# ================================================================

_CHECKS: List[dict] = []


def check(name: str, tier: int, depends_on: Optional[List[str]] = None):
    """Decorator to register a validation check function."""
    def decorator(fn):
        _CHECKS.append({
            "name": name,
            "tier": tier,
            "depends_on": depends_on or [],
            "fn": fn,
        })
        return fn
    return decorator


def _run_checks(
    items: List[dict],
    tier: int,
    config: Dict[str, Any],
    shared: Dict[str, Any],
) -> Dict[str, CheckResult]:
    """Run all registered checks up to the given tier."""
    results = {}
    for c in sorted(_CHECKS, key=lambda x: (x["tier"], x["name"])):
        if c["tier"] > tier:
            continue
        result = c["fn"](items, config, shared)
        results[c["name"]] = result
    return results


# ================================================================
# Entry point
# ================================================================

def validate_dataset(
    items: List[dict],
    tier: int = 1,
    condition_field: str = "condition",
    size_fn: Optional[Callable] = None,
    similarity_fn: Optional[Callable] = None,
    text_fn: Optional[Callable] = None,
    config_overrides: Optional[Dict[str, Any]] = None,
) -> DatasetReport:
    """Validate a dataset for experiment quality.

    Args:
        items: List of dicts, each with at least a condition label.
        tier: Validation tier (0=smoke, 1=standard, 2=rigorous, 3=regulatory).
        condition_field: Key in each item for the condition label.
        size_fn: Custom size metric function (default: token count).
        similarity_fn: Custom similarity function (default: TF-IDF cosine).
        text_fn: Custom text extraction function (default: item["prompt"]).
        config_overrides: Override default thresholds.

    Returns:
        DatasetReport with per-check results and overall verdict.
    """
    config = _default_config()
    if config_overrides:
        config.update(config_overrides)
    config["condition_field"] = condition_field
    config["size_fn"] = size_fn
    config["similarity_fn"] = similarity_fn
    config["text_fn"] = text_fn

    # Compute config hash for reproducibility
    config_for_hash = {k: v for k, v in config.items() if not callable(v)}
    config_hash = hashlib.sha256(
        json.dumps(config_for_hash, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]

    # Edge case: empty dataset
    if not items:
        return DatasetReport(
            tier=tier,
            overall_pass=False,
            overall_verdict="FAIL",
            timestamp=datetime.now(timezone.utc).isoformat(),
            config_hash=config_hash,
            recommendations=["No items to validate."],
        )

    # Extract conditions
    conditions = [item.get(condition_field) for item in items]
    unique_conditions = set(conditions)

    # Single condition = FAIL
    if len(unique_conditions) < 2:
        return DatasetReport(
            tier=tier,
            overall_pass=False,
            overall_verdict="FAIL",
            nominal_n=dict(Counter(conditions)),
            timestamp=datetime.now(timezone.utc).isoformat(),
            config_hash=config_hash,
            recommendations=["Cannot validate single-condition dataset. Need at least 2 conditions."],
        )

    nominal_n = dict(Counter(conditions))

    # Shared computation cache (avoids recomputing TF-IDF etc.)
    shared: Dict[str, Any] = {}

    # Run checks
    checks = _run_checks(items, tier, config, shared)

    # Compute verdict
    tier0_failed = any(cr.tier == 0 and not cr.passed for cr in checks.values())
    any_failed = any(not cr.passed for cr in checks.values())

    if tier0_failed:
        verdict = "FAIL"
        overall_pass = False
    elif any_failed:
        verdict = "INCONCLUSIVE"
        overall_pass = False
    else:
        verdict = "PASS"
        overall_pass = True

    # Build recommendations for failed checks
    recommendations = []
    for cr in checks.values():
        if not cr.passed and cr.details:
            recommendations.append(f"[{cr.name}] {cr.details}")

    return DatasetReport(
        tier=tier,
        overall_pass=overall_pass,
        overall_verdict=verdict,
        checks=checks,
        nominal_n=nominal_n,
        timestamp=datetime.now(timezone.utc).isoformat(),
        config_hash=config_hash,
        recommendations=recommendations,
    )


def _default_config() -> Dict[str, Any]:
    """Default thresholds for all checks."""
    return {
        # Tier 0
        "balance_ratio_threshold": 2.0,
        # Tier 1
        "alpha": 0.05,
        "effective_n_min": 20,
        "diversity_threshold": 0.4,
        "entropy_diff_threshold": 0.3,
        "n_clusters": 5,
        "max_pairwise_n": 2000,
        # Tier 2
        "shortcut_auroc_threshold": 0.65,
        "mi_threshold": 0.1,
        "variance_ratio_threshold": 3.0,
        "pair_token_tolerance": 5,
    }
