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
import math
from collections import Counter
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

import numpy as np


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

    # Provenance hash (Tier 3 only)
    dataset_hash = None
    if tier >= 3:
        dataset_hash = _compute_dataset_hash(items)

    return DatasetReport(
        tier=tier,
        dataset_hash=dataset_hash,
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


# ================================================================
# TIER 1 CHECKS
# ================================================================

def _default_size_fn(item: dict) -> float:
    """Default: use n_tokens from features dict."""
    feats = item.get("features", {})
    return float(feats.get("n_tokens", len(item.get("prompt", "").split())))


def _default_text_fn(item: dict) -> str:
    """Default: use prompt field."""
    return item.get("prompt", "")


def _get_tfidf_matrix(items: List[dict], config: Dict, shared: Dict):
    """Compute or retrieve cached TF-IDF matrix."""
    if "tfidf_matrix" in shared:
        return shared["tfidf_matrix"], shared["tfidf_vectorizer"]
    from sklearn.feature_extraction.text import TfidfVectorizer
    text_fn = config.get("text_fn") or _default_text_fn
    texts = [text_fn(item) for item in items]
    vec = TfidfVectorizer(max_features=1000, stop_words="english")
    matrix = vec.fit_transform(texts)
    shared["tfidf_matrix"] = matrix
    shared["tfidf_vectorizer"] = vec
    return matrix, vec


def _get_similarity_matrix(items: List[dict], config: Dict, shared: Dict) -> np.ndarray:
    """Compute or retrieve cached pairwise cosine similarity."""
    if "sim_matrix" in shared:
        return shared["sim_matrix"]
    custom_fn = config.get("similarity_fn")
    if custom_fn:
        sim = custom_fn(items)
    else:
        from sklearn.metrics.pairwise import cosine_similarity
        tfidf, _ = _get_tfidf_matrix(items, config, shared)
        max_n = config.get("max_pairwise_n", 2000)
        if tfidf.shape[0] > max_n:
            rng = np.random.default_rng(42)
            idx = rng.choice(tfidf.shape[0], max_n, replace=False)
            sim = cosine_similarity(tfidf[idx])
            shared["sim_approximated"] = True
            shared["sim_sample_idx"] = idx
        else:
            sim = cosine_similarity(tfidf)
            shared["sim_approximated"] = False
    shared["sim_matrix"] = sim
    return sim


@check(name="size_overlap", tier=1)
def _check_size_overlap(items: List[dict], config: Dict, shared: Dict) -> CheckResult:
    """Two-sample KS test on size distributions between conditions."""
    from scipy.stats import ks_2samp
    cond_field = config.get("condition_field", "condition")
    size_fn = config.get("size_fn") or _default_size_fn
    alpha = config.get("alpha", 0.05)

    conditions = sorted(set(item.get(cond_field) for item in items))
    by_cond = {}
    for item in items:
        c = item.get(cond_field)
        by_cond.setdefault(c, []).append(size_fn(item))

    # Pairwise KS tests
    worst_p = 1.0
    worst_pair = None
    pair_results = []
    for i, c1 in enumerate(conditions):
        for c2 in conditions[i + 1:]:
            stat, p = ks_2samp(by_cond[c1], by_cond[c2])
            pair_results.append({"pair": f"{c1}_vs_{c2}", "ks_stat": round(stat, 4), "ks_p": round(p, 6)})
            if p < worst_p:
                worst_p = p
                worst_pair = f"{c1}_vs_{c2}"

    means = {c: round(float(np.mean(v)), 2) for c, v in by_cond.items()}
    mean_diff = max(means.values()) - min(means.values()) if len(means) >= 2 else 0

    passed = worst_p >= alpha

    return CheckResult(
        name="size_overlap",
        passed=passed,
        tier=1,
        metrics={
            "worst_ks_p": round(worst_p, 6),
            "worst_pair": worst_pair,
            "mean_per_condition": means,
            "mean_diff": round(mean_diff, 2),
            "pairs": pair_results,
        },
        details=f"Size distributions differ significantly ({worst_pair}, p={worst_p:.4f})." if not passed else "",
    )


@check(name="effective_n", tier=1)
def _check_effective_n(items: List[dict], config: Dict, shared: Dict) -> CheckResult:
    """Compute effective sample size via design effect from similarity."""
    cond_field = config.get("condition_field", "condition")
    min_n_eff = config.get("effective_n_min", 20)

    sim = _get_similarity_matrix(items, config, shared)

    # Group items by condition
    conditions = [item.get(cond_field) for item in items]
    unique_conds = sorted(set(conditions))
    cond_indices = {c: [i for i, x in enumerate(conditions) if x == c] for c in unique_conds}

    metrics = {"approximated": shared.get("sim_approximated", False)}
    min_n_eff_value = float("inf")

    for cond, indices in cond_indices.items():
        n = len(indices)
        if n < 2:
            metrics[f"deff_{cond}"] = 1.0
            metrics[f"n_eff_{cond}"] = float(n)
            continue

        # Mean intra-condition similarity (excluding diagonal)
        sub_sim = sim[np.ix_(indices, indices)] if len(indices) <= sim.shape[0] else np.eye(n)
        np.fill_diagonal(sub_sim, 0)
        mean_rho = float(sub_sim.sum() / (n * (n - 1))) if n > 1 else 0

        # Design effect: DEFF = 1 + mean_rho * (cluster_size - 1)
        # Use n as cluster_size (conservative: treats all items as one cluster)
        deff = 1 + mean_rho * (n - 1)
        deff = max(deff, 1.0)  # DEFF can't be < 1
        n_eff = n / deff

        metrics[f"deff_{cond}"] = round(deff, 2)
        metrics[f"n_eff_{cond}"] = round(n_eff, 1)
        metrics[f"mean_rho_{cond}"] = round(mean_rho, 4)
        min_n_eff_value = min(min_n_eff_value, n_eff)

    passed = min_n_eff_value >= min_n_eff

    return CheckResult(
        name="effective_n",
        passed=passed,
        tier=1,
        metrics=metrics,
        details=f"Effective N below {min_n_eff} threshold. Template reuse likely." if not passed else "",
    )


@check(name="semantic_diversity", tier=1)
def _check_semantic_diversity(items: List[dict], config: Dict, shared: Dict) -> CheckResult:
    """Mean pairwise distance within each condition."""
    cond_field = config.get("condition_field", "condition")
    threshold = config.get("diversity_threshold", 0.4)

    sim = _get_similarity_matrix(items, config, shared)

    conditions = [item.get(cond_field) for item in items]
    unique_conds = sorted(set(conditions))
    cond_indices = {c: [i for i, x in enumerate(conditions) if x == c] for c in unique_conds}

    min_diversity = float("inf")
    metrics = {}

    for cond, indices in cond_indices.items():
        n = len(indices)
        if n < 2:
            metrics[f"mean_distance_{cond}"] = 1.0
            continue
        sub_sim = sim[np.ix_(indices, indices)]
        # Distance = 1 - similarity
        np.fill_diagonal(sub_sim, 1)  # exclude self
        distances = 1 - sub_sim
        np.fill_diagonal(distances, 0)
        mean_dist = float(distances.sum() / (n * (n - 1)))
        metrics[f"mean_distance_{cond}"] = round(mean_dist, 4)
        min_diversity = min(min_diversity, mean_dist)

    passed = min_diversity >= threshold

    return CheckResult(
        name="semantic_diversity",
        passed=passed,
        tier=1,
        metrics=metrics,
        details=f"Low semantic diversity (min={min_diversity:.3f} < {threshold}). Monoculture risk." if not passed else "",
    )


@check(name="domain_balance", tier=1)
def _check_domain_balance(items: List[dict], config: Dict, shared: Dict) -> CheckResult:
    """Shannon entropy of topic clusters per condition."""
    from sklearn.cluster import KMeans
    cond_field = config.get("condition_field", "condition")
    n_clusters = config.get("n_clusters", 5)
    entropy_threshold = config.get("entropy_diff_threshold", 0.3)

    tfidf, _ = _get_tfidf_matrix(items, config, shared)
    n_items = tfidf.shape[0]
    actual_k = min(n_clusters, n_items)

    km = KMeans(n_clusters=actual_k, random_state=42, n_init=3)
    labels = km.fit_predict(tfidf.toarray() if hasattr(tfidf, "toarray") else tfidf)

    conditions = [item.get(cond_field) for item in items]
    unique_conds = sorted(set(conditions))
    cond_indices = {c: [i for i, x in enumerate(conditions) if x == c] for c in unique_conds}

    def _entropy(counts):
        total = sum(counts)
        if total == 0:
            return 0.0
        probs = [c / total for c in counts if c > 0]
        return -sum(p * math.log(p) for p in probs)

    entropies = {}
    for cond, indices in cond_indices.items():
        cluster_counts = Counter(labels[i] for i in indices)
        counts = [cluster_counts.get(k, 0) for k in range(actual_k)]
        entropies[cond] = round(_entropy(counts), 4)

    entropy_vals = list(entropies.values())
    entropy_diff = max(entropy_vals) - min(entropy_vals) if len(entropy_vals) >= 2 else 0

    passed = entropy_diff <= entropy_threshold

    return CheckResult(
        name="domain_balance",
        passed=passed,
        tier=1,
        metrics={
            "entropy_per_condition": entropies,
            "entropy_diff": round(entropy_diff, 4),
            "n_clusters": actual_k,
        },
        details=f"Topic entropy diff {entropy_diff:.3f} exceeds {entropy_threshold} threshold." if not passed else "",
    )


# ================================================================
# TIER 2 CHECKS
# ================================================================

@check(name="shortcut_detection", tier=2)
def _check_shortcut_detection(items: List[dict], config: Dict, shared: Dict) -> CheckResult:
    """Train classifier on item content to predict condition. High AUROC = surface confound."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import LeaveOneOut, StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    cond_field = config.get("condition_field", "condition")
    threshold = config.get("shortcut_auroc_threshold", 0.65)

    tfidf, vec = _get_tfidf_matrix(items, config, shared)
    X = tfidf.toarray() if hasattr(tfidf, "toarray") else np.array(tfidf)
    conditions = [item.get(cond_field) for item in items]
    unique_conds = sorted(set(conditions))
    y = np.array([unique_conds.index(c) for c in conditions])

    n = len(y)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, solver="lbfgs"))

    if n < 100:
        # LOO for small datasets
        loo = LeaveOneOut()
        y_proba = np.zeros(n)
        for train_idx, test_idx in loo.split(X):
            if len(np.unique(y[train_idx])) < 2:
                y_proba[test_idx] = 0.5
                continue
            clf.fit(X[train_idx], y[train_idx])
            y_proba[test_idx] = clf.predict_proba(X[test_idx])[:, 1]
    else:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        y_proba = np.zeros(n)
        for train_idx, test_idx in skf.split(X, y):
            clf.fit(X[train_idx], y[train_idx])
            y_proba[test_idx] = clf.predict_proba(X[test_idx])[:, 1]

    auroc = float(roc_auc_score(y, y_proba)) if len(np.unique(y)) >= 2 else 0.5

    # Top distinguishing features
    top_features = []
    if hasattr(vec, "get_feature_names_out"):
        clf_final = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, solver="lbfgs"))
        clf_final.fit(X, y)
        coefs = clf_final.named_steps["logisticregression"].coef_[0]
        names = vec.get_feature_names_out()
        top_idx = np.argsort(np.abs(coefs))[-5:][::-1]
        top_features = [{"feature": names[i], "coef": round(float(coefs[i]), 4)} for i in top_idx]

    passed = auroc < threshold

    return CheckResult(
        name="shortcut_detection",
        passed=passed,
        tier=2,
        metrics={"auroc": round(auroc, 4), "top_features": top_features, "threshold": threshold},
        details=f"Condition predictable from content (AUROC={auroc:.3f} > {threshold}). Surface confound." if not passed else "",
    )


@check(name="confound_discovery", tier=2)
def _check_confound_discovery(items: List[dict], config: Dict, shared: Dict) -> CheckResult:
    """Compute MI between numeric features and condition label."""
    from sklearn.feature_selection import mutual_info_classif
    cond_field = config.get("condition_field", "condition")
    mi_threshold = config.get("mi_threshold", 0.1)
    confound_spec = config.get("confound_spec", {})

    # Extract all numeric features
    feature_names = set()
    for item in items:
        for k, v in item.get("features", {}).items():
            if isinstance(v, (int, float)):
                feature_names.add(k)
    feature_names = sorted(feature_names)

    if not feature_names:
        return CheckResult(name="confound_discovery", passed=True, tier=2, metrics={"discovered": []})

    X = np.array([[item.get("features", {}).get(f, 0) for f in feature_names] for item in items])
    conditions = [item.get(cond_field) for item in items]
    unique_conds = sorted(set(conditions))
    y = np.array([unique_conds.index(c) for c in conditions])

    # Need at least 3 samples per class for MI estimation
    min_class_size = min(Counter(y).values())
    if min_class_size < 3:
        return CheckResult(name="confound_discovery", passed=True, tier=2,
                          metrics={"discovered": [], "skipped": True},
                          details="Too few samples per class for MI estimation.")

    mi = mutual_info_classif(X, y, random_state=42)

    # Features intentionally used for classification are expected to have high MI
    classification_features = set(config.get("classification_features", []))

    discovered = []
    undeclared = []
    for i, fname in enumerate(feature_names):
        if mi[i] > mi_threshold:
            discovered.append({"feature": fname, "mi": round(float(mi[i]), 4)})
            # Only flag as undeclared if it's not a classification feature AND not in confound spec
            if fname not in confound_spec and fname not in classification_features:
                undeclared.append(fname)

    passed = len(undeclared) == 0

    return CheckResult(
        name="confound_discovery",
        passed=passed,
        tier=2,
        metrics={"discovered": discovered, "undeclared": undeclared, "confound_spec": confound_spec},
        details=f"Undeclared confounds with high MI: {', '.join(undeclared)}" if undeclared else "",
    )


@check(name="variance_ratio", tier=2)
def _check_variance_ratio(items: List[dict], config: Dict, shared: Dict) -> CheckResult:
    """Compare within-condition variance of size metric."""
    cond_field = config.get("condition_field", "condition")
    threshold = config.get("variance_ratio_threshold", 3.0)
    size_fn = config.get("size_fn") or _default_size_fn

    by_cond: Dict[str, List[float]] = {}
    for item in items:
        c = item.get(cond_field)
        by_cond.setdefault(c, []).append(size_fn(item))

    variances = {}
    for c, vals in by_cond.items():
        variances[c] = float(np.var(vals)) if len(vals) > 1 else 0.0

    var_vals = [v for v in variances.values() if v > 0]
    if len(var_vals) < 2:
        return CheckResult(name="variance_ratio", passed=True, tier=2, metrics={"ratio": 1.0})

    ratio = max(var_vals) / min(var_vals)
    passed = ratio <= threshold

    return CheckResult(
        name="variance_ratio",
        passed=passed,
        tier=2,
        metrics={"variances": {k: round(v, 2) for k, v in variances.items()}, "ratio": round(ratio, 2), "threshold": threshold},
        details=f"Variance ratio {ratio:.1f}:1 exceeds {threshold}:1. Asymmetric response distribution." if not passed else "",
    )


@check(name="pair_integrity", tier=2)
def _check_pair_integrity(items: List[dict], config: Dict, shared: Dict) -> CheckResult:
    """Verify minimal pair structure: shared prefix/suffix, size tolerance."""
    cond_field = config.get("condition_field", "condition")
    paired = config.get("paired", False)
    tolerance = config.get("pair_token_tolerance", 5)
    size_fn = config.get("size_fn") or _default_size_fn

    if not paired:
        return CheckResult(name="pair_integrity", passed=True, tier=2, metrics={"skipped": True}, details="")

    # Group by condition, pair by index
    conditions = sorted(set(item.get(cond_field) for item in items))
    if len(conditions) != 2:
        return CheckResult(name="pair_integrity", passed=False, tier=2, details="Paired mode requires exactly 2 conditions.")

    by_cond = {c: [] for c in conditions}
    for item in items:
        by_cond[item.get(cond_field)].append(item)

    c1, c2 = conditions
    n_pairs = min(len(by_cond[c1]), len(by_cond[c2]))
    violations = []

    for i in range(n_pairs):
        s1 = size_fn(by_cond[c1][i])
        s2 = size_fn(by_cond[c2][i])
        diff = abs(s1 - s2)
        if diff > tolerance:
            violations.append({"pair": i, "size_diff": round(diff, 1), "tolerance": tolerance})

    passed = len(violations) == 0

    return CheckResult(
        name="pair_integrity",
        passed=passed,
        tier=2,
        metrics={"n_pairs": n_pairs, "n_violations": len(violations), "violations": violations[:10]},
        details=f"{len(violations)} pair(s) exceed size tolerance of {tolerance}." if violations else "",
    )


@check(name="confound_disclosure", tier=2)
def _check_confound_disclosure(items: List[dict], config: Dict, shared: Dict) -> CheckResult:
    """Verify user-declared confound spec has no uncontrolled items."""
    confound_spec = config.get("confound_spec")

    if confound_spec is None:
        return CheckResult(
            name="confound_disclosure",
            passed=True, tier=2,
            metrics={"skipped": True},
            details="No confound spec provided.",
        )

    uncontrolled = [k for k, v in confound_spec.items() if v in ("uncontrolled", "unknown")]
    passed = len(uncontrolled) == 0

    return CheckResult(
        name="confound_disclosure",
        passed=passed,
        tier=2,
        metrics={"spec": confound_spec, "uncontrolled": uncontrolled},
        details=f"Uncontrolled confounds: {', '.join(uncontrolled)}" if uncontrolled else "",
    )


@check(name="format_consistency", tier=2)
def _check_format_consistency(items: List[dict], config: Dict, shared: Dict) -> CheckResult:
    """Compare prompt formatting structure between conditions."""
    import re
    cond_field = config.get("condition_field", "condition")
    text_fn = config.get("text_fn") or _default_text_fn

    def _format_signature(text: str) -> dict:
        """Extract structural features: newline count, punct density, avg word len."""
        words = text.split()
        n_words = max(len(words), 1)
        return {
            "newline_rate": text.count("\n") / n_words,
            "punct_density": len(re.findall(r"[^\w\s]", text)) / n_words,
            "avg_word_len": sum(len(w) for w in words) / n_words if words else 0,
            "colon_rate": text.count(":") / n_words,
            "question_rate": text.count("?") / n_words,
        }

    by_cond: Dict[str, List[dict]] = {}
    for item in items:
        c = item.get(cond_field)
        sig = _format_signature(text_fn(item))
        by_cond.setdefault(c, []).append(sig)

    # Compare mean format features between conditions
    conditions = sorted(by_cond.keys())
    if len(conditions) < 2:
        return CheckResult(name="format_consistency", passed=True, tier=2, metrics={})

    feature_diffs = {}
    for feat in ["newline_rate", "punct_density", "avg_word_len", "colon_rate", "question_rate"]:
        means = {}
        for c in conditions:
            vals = [s[feat] for s in by_cond[c]]
            means[c] = float(np.mean(vals))
        max_diff = max(means.values()) - min(means.values())
        feature_diffs[feat] = round(max_diff, 4)

    # Fail if any structural feature differs substantially
    # Thresholds: newline/colon/question rates > 0.1, punct density > 0.15, avg_word_len > 2
    significant_diffs = []
    thresholds = {"newline_rate": 0.1, "punct_density": 0.15, "avg_word_len": 2.0, "colon_rate": 0.1, "question_rate": 0.1}
    for feat, diff in feature_diffs.items():
        if diff > thresholds.get(feat, 0.1):
            significant_diffs.append(feat)

    passed = len(significant_diffs) == 0

    return CheckResult(
        name="format_consistency",
        passed=passed,
        tier=2,
        metrics={"feature_diffs": feature_diffs, "significant_diffs": significant_diffs},
        details=f"Format differs: {', '.join(significant_diffs)}" if significant_diffs else "",
    )


# ================================================================
# TIER 3 CHECKS
# ================================================================

def _compute_dataset_hash(items: List[dict]) -> str:
    """SHA-256 of canonical JSON serialization."""
    canonical = json.dumps(items, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@check(name="metadata_completeness", tier=3)
def _check_metadata_completeness(items: List[dict], config: Dict, shared: Dict) -> CheckResult:
    """Verify required metadata fields are present."""
    meta = config.get("dataset_metadata", {})
    required = ["generation_method", "llm_generated", "author", "decoding_strategy", "contamination_risk"]

    missing = [f for f in required if f not in meta or meta[f] is None or meta[f] == ""]
    passed = len(missing) == 0

    return CheckResult(
        name="metadata_completeness",
        passed=passed,
        tier=3,
        metrics={"provided": list(meta.keys()), "missing": missing, "required": required},
        details=f"Missing metadata: {', '.join(missing)}. NeurIPS checklist item 16 requires LLM disclosure." if missing else "",
    )


@check(name="prereg_compatibility", tier=3)
def _check_prereg_compatibility(items: List[dict], config: Dict, shared: Dict) -> CheckResult:
    """Check dataset against pre-registered plan."""
    plan = config.get("prereg_plan")
    cond_field = config.get("condition_field", "condition")

    if plan is None:
        return CheckResult(name="prereg_compatibility", passed=True, tier=3,
                          metrics={"skipped": True}, details="No pre-registration plan provided.")

    planned_n = plan.get("planned_n", {})
    actual_n = Counter(item.get(cond_field) for item in items)
    deviations = []

    for cond, planned in planned_n.items():
        actual = actual_n.get(cond, 0)
        if actual != planned:
            deviations.append({"condition": cond, "planned": planned, "actual": actual, "diff": actual - planned})

    passed = len(deviations) == 0

    return CheckResult(
        name="prereg_compatibility",
        passed=passed,
        tier=3,
        metrics={"planned_n": planned_n, "actual_n": dict(actual_n), "deviations": deviations},
        details=f"Deviations from plan: {deviations}" if deviations else "",
    )


@check(name="measurement_validation", tier=3)
def _check_measurement_validation(items: List[dict], config: Dict, shared: Dict) -> CheckResult:
    """Run optional measurement validator and report ICC."""
    validator = config.get("measurement_validator")

    if validator is None:
        return CheckResult(
            name="measurement_validation",
            passed=True, tier=3,
            metrics={"skipped": True},
            details="No measurement validator provided.",
        )

    result = validator()
    passed = result.get("passed", False)
    icc = result.get("icc", 0.0)

    return CheckResult(
        name="measurement_validation",
        passed=passed,
        tier=3,
        metrics={"icc": icc, "skipped": False, **result},
        details=f"Measurement ICC={icc:.2f} below reliability threshold." if not passed else "",
    )
