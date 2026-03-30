"""V11: Feature Ablation via Permutation Importance.

Tests which features (norm_per_token, key_rank, key_entropy) drive
classification AUROC. If one feature dominates AND it correlates with
input length, the "geometry" signal is a length proxy.

Pre-registered: research-log/V11-design.md

Usage:
    .venv/bin/python -m kv_verify.experiments.v11_feature_ablation
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np

from kv_verify.constants import (
    AUROC_DELTA, AUROC_GEOMETRY_ADVANTAGE, DEFAULT_SEED, N_SPLITS,
)
from kv_verify.data_loader import load_comparison_data
from kv_verify.fixtures import EXP47_COMPARISONS, PRIMARY_FEATURES
from kv_verify.stats import assign_groups, groupkfold_auroc
from kv_verify.tracking import ExperimentTracker
from kv_verify.types import ClaimVerification, Severity, Verdict


def _permutation_importance(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    feature_idx: int,
    n_repeats: int = 100,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Compute permutation importance for a single feature."""
    rng = np.random.RandomState(seed)

    # Baseline AUROC
    baseline = groupkfold_auroc(X, y, groups, n_splits=min(N_SPLITS, len(np.unique(groups))))
    baseline_auroc = baseline.auroc

    # Permute the feature and recompute
    drops = []
    for _ in range(n_repeats):
        X_perm = X.copy()
        X_perm[:, feature_idx] = rng.permutation(X_perm[:, feature_idx])
        perm_result = groupkfold_auroc(X_perm, y, groups, n_splits=min(N_SPLITS, len(np.unique(groups))))
        drops.append(baseline_auroc - perm_result.auroc)

    return {
        "baseline_auroc": float(baseline_auroc),
        "mean_drop": float(np.mean(drops)),
        "std_drop": float(np.std(drops)),
        "ci_95": [float(np.percentile(drops, 2.5)), float(np.percentile(drops, 97.5))],
    }


def run_v11(
    output_dir: Path,
    n_repeats: int = 50,
    tracker: Optional[ExperimentTracker] = None,
) -> ClaimVerification:
    """Run feature ablation on surviving comparisons."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if tracker is None:
        tracker = ExperimentTracker(output_dir=output_dir, experiment_name="V11-feature-ablation")

    tracker.log_params(experiment="V11", n_repeats=n_repeats)

    # Use the 7 comparisons that survived input control (F01b)
    surviving = [
        "exp31_refusal_vs_normal",
        "exp31_refusal_vs_benign",
        "exp36_impossible_vs_benign",
        "exp36_harmful_vs_benign",
        "exp36_impossible_vs_harmful",
        "exp32_jailbreak_vs_normal",
        "exp32_jailbreak_vs_refusal",
    ]

    all_importance = {}

    for comp_name in surviving:
        comp_info = next((c for c in EXP47_COMPARISONS if c["name"] == comp_name), None)
        if comp_info is None:
            continue

        try:
            X, y, meta = load_comparison_data(comp_name)
        except Exception:
            continue

        groups = assign_groups(
            meta["n_pos"], meta["n_neg"],
            paired=meta.get("paired", False),
            prompt_indices_pos=meta.get("prompt_indices_pos"),
            prompt_indices_neg=meta.get("prompt_indices_neg"),
        )

        comp_results = {}
        for fi, fname in enumerate(PRIMARY_FEATURES):
            imp = _permutation_importance(X, y, groups, fi, n_repeats=n_repeats)
            comp_results[fname] = imp
            tracker.log_metric(f"v11_{comp_name}_{fname}_importance", imp["mean_drop"])

        all_importance[comp_name] = comp_results

    # Aggregate across comparisons
    feature_means = {}
    for fname in PRIMARY_FEATURES:
        drops = [
            all_importance[comp][fname]["mean_drop"]
            for comp in all_importance
            if fname in all_importance[comp]
        ]
        feature_means[fname] = float(np.mean(drops)) if drops else 0.0

    # Verdict per pre-registered criteria
    max_feature = max(feature_means, key=feature_means.get) if feature_means else ""
    max_importance = feature_means.get(max_feature, 0)

    if max_importance > AUROC_GEOMETRY_ADVANTAGE:
        # Check if the dominant feature is the one most correlated with input length
        if max_feature == "norm_per_token":
            verdict = Verdict.FALSIFIED
            evidence = (
                f"norm_per_token dominates (mean drop={max_importance:.3f}). "
                f"This is the feature most correlated with input length (R^2=0.80 in F01b-49b). "
                f"The 'geometry' is an input-length proxy."
            )
        else:
            verdict = Verdict.CONFIRMED
            evidence = (
                f"{max_feature} dominates (mean drop={max_importance:.3f}). "
                f"Signal localized to a feature not confounded with input length."
            )
    else:
        verdict = Verdict.WEAKENED
        importances_str = ", ".join(f"{f}={v:.3f}" for f, v in sorted(feature_means.items(), key=lambda x: -x[1]))
        evidence = (
            f"No single feature dominates (max={max_feature} at {max_importance:.3f} < {AUROC_GEOMETRY_ADVANTAGE}). "
            f"Importances: {importances_str}. Features are interchangeable. Geometry is diffuse."
        )

    # Save results
    result_data = {
        "experiment": "V11",
        "verdict": verdict.value,
        "evidence": evidence,
        "feature_importance": feature_means,
        "per_comparison": {
            comp: {f: v["mean_drop"] for f, v in feats.items()}
            for comp, feats in all_importance.items()
        },
        "dominant_feature": max_feature,
        "n_comparisons": len(all_importance),
        "n_repeats": n_repeats,
    }
    with open(output_dir / "v11_results.json", "w") as f:
        json.dump(result_data, f, indent=2)

    return ClaimVerification(
        claim_id="V11",
        claim_text="KV-cache features each contribute meaningfully to classification",
        paper_section="Sections 4-5 (implicit)",
        finding_id="V11",
        severity=Severity.MAJOR,
        null_hypothesis="Removing any single feature reduces AUROC by less than 0.05",
        experiment_description="Permutation importance across 7 surviving comparisons",
        verdict=verdict,
        evidence_summary=evidence,
        original_value=None,
        corrected_value=feature_means,
        visualization_paths=[],
        gpu_time_seconds=0.0,
        stats=result_data,
    )


if __name__ == "__main__":
    result = run_v11(output_dir=Path("experiments/output/v11"))
    print(f"\nV11 Verdict: {result.verdict.value}")
    print(f"Evidence: {result.evidence_summary}")
