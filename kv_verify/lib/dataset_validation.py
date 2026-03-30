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


def list_checks(tier: int = 3) -> List[dict]:
    """List all registered checks up to the given tier."""
    return [
        {"name": c["name"], "tier": c["tier"], "depends_on": c["depends_on"]}
        for c in sorted(_CHECKS, key=lambda x: (x["tier"], x["name"]))
        if c["tier"] <= tier
    ]


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
        "required_fields": ["condition"],
        "feature_fields_numeric": True,
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


# ================================================================
# TIER 0 CHECKS
# ================================================================

@check(name="structural", tier=0)
def _check_structural(items: List[dict], config: Dict, shared: Dict) -> CheckResult:
    """Verify required fields exist and feature values are numeric."""
    cond_field = config.get("condition_field", "condition")
    issues = []

    for i, item in enumerate(items):
        if cond_field not in item:
            issues.append(f"Item {i}: missing '{cond_field}' field")
        if "features" in item and config.get("feature_fields_numeric", True):
            for k, v in item["features"].items():
                if not isinstance(v, (int, float)):
                    issues.append(f"Item {i}: features['{k}'] is {type(v).__name__}, expected numeric")

    return CheckResult(
        name="structural",
        passed=len(issues) == 0,
        tier=0,
        metrics={"n_issues": len(issues)},
        details="; ".join(issues[:5]) if issues else "",
    )


@check(name="duplicates", tier=0)
def _check_duplicates(items: List[dict], config: Dict, shared: Dict) -> CheckResult:
    """Detect exact duplicates within and across conditions."""
    cond_field = config.get("condition_field", "condition")

    # Build content fingerprints per item
    def _fingerprint(item: dict) -> str:
        prompt = item.get("prompt", "")
        feats = item.get("features", {})
        return f"{prompt}|{json.dumps(feats, sort_keys=True)}"

    # Group by condition
    by_condition: Dict[str, List[str]] = {}
    all_fps: List[tuple] = []  # (fingerprint, condition, index)
    for i, item in enumerate(items):
        cond = item.get(cond_field, "")
        fp = _fingerprint(item)
        by_condition.setdefault(cond, []).append(fp)
        all_fps.append((fp, cond, i))

    # Within-condition duplicates
    within_dupes = []
    for cond, fps in by_condition.items():
        seen = {}
        for idx, fp in enumerate(fps):
            if fp in seen:
                within_dupes.append({"condition": cond, "indices": [seen[fp], idx]})
            else:
                seen[fp] = idx

    # Cross-condition duplicates
    cross_dupes = []
    fp_to_cond: Dict[str, List[tuple]] = {}
    for fp, cond, idx in all_fps:
        fp_to_cond.setdefault(fp, []).append((cond, idx))
    for fp, entries in fp_to_cond.items():
        conds = {e[0] for e in entries}
        if len(conds) > 1:
            cross_dupes.append({"fingerprint": fp[:32], "conditions": list(conds)})

    n_within = len(within_dupes)
    n_cross = len(cross_dupes)
    total_dupes = n_within + n_cross

    details = ""
    if n_within > 0:
        details += f"{n_within} within-condition duplicate(s). "
    if n_cross > 0:
        details += f"{n_cross} cross-condition duplicate(s) (leakage risk). "

    return CheckResult(
        name="duplicates",
        passed=total_dupes == 0,
        tier=0,
        metrics={
            "n_exact": total_dupes,
            "within_condition": within_dupes[:10],
            "cross_condition": cross_dupes[:10],
            "leakage_risk": n_cross > 0,
            "adjusted_n": {
                cond: len(set(fps)) for cond, fps in by_condition.items()
            },
        },
        details=details,
    )


@check(name="class_balance", tier=0)
def _check_class_balance(items: List[dict], config: Dict, shared: Dict) -> CheckResult:
    """Check class balance across conditions."""
    cond_field = config.get("condition_field", "condition")
    threshold = config.get("balance_ratio_threshold", 2.0)

    counts = Counter(item.get(cond_field) for item in items)
    if not counts:
        return CheckResult(name="class_balance", passed=False, tier=0, details="No items")

    max_count = max(counts.values())
    min_count = min(counts.values())
    ratio = max_count / max(min_count, 1)

    return CheckResult(
        name="class_balance",
        passed=ratio <= threshold,
        tier=0,
        metrics={
            "counts": dict(counts),
            "ratio": round(ratio, 2),
            "threshold": threshold,
        },
        details=f"Imbalance ratio {ratio:.1f}:1 exceeds {threshold}:1 threshold." if ratio > threshold else "",
    )
