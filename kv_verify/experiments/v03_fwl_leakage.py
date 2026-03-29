"""V03: FWL Leakage Test.

Finding C4: fwl_residualize() at 48_fwl_residualization.py:296-327 fits OLS
on the ENTIRE dataset before CV splitting. Test-fold confound values leak
into the regression coefficients.

This experiment runs four FWL conditions per comparison:
  1. Full-dataset linear FWL (original, buggy — leakage)
  2. Within-fold linear FWL (corrected)
  3. Polynomial FWL degree 2 (nonlinear confound removal)
  4. Polynomial FWL degree 3 (nonlinear confound removal)

Pre-registered pass/fail:
  - Within-fold changes AUROC > 0.05: C4 CONFIRMED (leakage is real)
  - Polynomial FWL collapses 3+ of 7 surviving comparisons: FALSIFIED
  - Polynomial FWL preserves all 7 surviving comparisons: STRENGTHENED

Spec: research-log/V03-design.md
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from kv_verify.constants import AUROC_DELTA, AUROC_FWL_COLLAPSE, AUROC_FWL_PRESERVE, FWL_COLLAPSE_COUNT
from kv_verify.data_loader import load_comparison_data
from kv_verify.fixtures import EXP47_COMPARISONS
from kv_verify.stats import (
    assign_groups,
    fwl_nonlinear,
    fwl_residualize,
    groupkfold_auroc,
)
from kv_verify.tracking import ExperimentTracker
from kv_verify.types import ClaimVerification, Severity, Verdict


# Comparisons excluded from the polynomial collapse count.
# These are already known to be confounded or non-significant.
EXCLUDED_COMPARISONS = {
    "exp39_sycophancy",        # Already known length-confounded (M2)
    "exp18b_deception",        # FWL-both already known to fail at n=10
    "exp32_jailbreak_vs_refusal",  # Already non-significant
}

# Thresholds (pre-registered, do NOT adjust) — sourced from constants
LEAKAGE_THRESHOLD = AUROC_DELTA
COLLAPSE_AUROC = AUROC_FWL_COLLAPSE
PRESERVE_AUROC = AUROC_FWL_PRESERVE
COLLAPSE_COUNT_THRESHOLD = FWL_COLLAPSE_COUNT


def _run_comparison(
    name: str,
    comp_info: dict,
) -> Dict:
    """Run all four FWL conditions for a single comparison.

    Returns a dict with all AUROCs and R-squared values.
    """
    X, y, meta = load_comparison_data(name)
    Z = meta["confounds"]

    # Build correct groups (C2 fix)
    if meta["paired"]:
        groups = assign_groups(
            n_pos=meta["n_pos"],
            n_neg=meta["n_neg"],
            paired=True,
            prompt_indices_pos=meta["prompt_indices_pos"],
            prompt_indices_neg=meta["prompt_indices_neg"],
        )
    else:
        groups = assign_groups(
            n_pos=meta["n_pos"],
            n_neg=meta["n_neg"],
            paired=False,
        )

    # Condition 1: Full-dataset linear FWL (original buggy mode)
    result_full = groupkfold_auroc(
        X, y, groups,
        fwl_confounds=Z,
        fwl_within_fold=False,
        feature_names=meta["feature_names"],
    )
    auroc_fwl_full = result_full.auroc

    # Condition 2: Within-fold linear FWL (corrected)
    result_within = groupkfold_auroc(
        X, y, groups,
        fwl_confounds=Z,
        fwl_within_fold=True,
        feature_names=meta["feature_names"],
    )
    auroc_fwl_within = result_within.auroc

    # Condition 3: Polynomial FWL degree 2
    # Residualize full dataset first, then classify residuals
    X_poly2, r2_poly2 = fwl_nonlinear(X, Z, degree=2)
    result_poly2 = groupkfold_auroc(
        X_poly2, y, groups,
        fwl_confounds=None,  # already residualized
        feature_names=meta["feature_names"],
    )
    auroc_poly2 = result_poly2.auroc

    # Condition 4: Polynomial FWL degree 3
    X_poly3, r2_poly3 = fwl_nonlinear(X, Z, degree=3)
    result_poly3 = groupkfold_auroc(
        X_poly3, y, groups,
        fwl_confounds=None,  # already residualized
        feature_names=meta["feature_names"],
    )
    auroc_poly3 = result_poly3.auroc

    # R-squared for linear FWL (from full-dataset residualization for comparison)
    _, r2_linear = fwl_residualize(X, Z, within_fold=False)

    return {
        "auroc_fwl_full": float(auroc_fwl_full) if not np.isnan(auroc_fwl_full) else float("nan"),
        "auroc_fwl_within": float(auroc_fwl_within) if not np.isnan(auroc_fwl_within) else float("nan"),
        "auroc_poly2": float(auroc_poly2) if not np.isnan(auroc_poly2) else float("nan"),
        "auroc_poly3": float(auroc_poly3) if not np.isnan(auroc_poly3) else float("nan"),
        "r_squared": {
            "linear": [float(v) for v in r2_linear],
            "poly2": [float(v) for v in r2_poly2],
            "poly3": [float(v) for v in r2_poly3],
        },
    }


def _leakage_detected(auroc_full: float, auroc_within: float) -> bool:
    """Check if leakage is detected: |full - within| > threshold."""
    if np.isnan(auroc_full) or np.isnan(auroc_within):
        return False
    return abs(auroc_full - auroc_within) > LEAKAGE_THRESHOLD


def _per_comparison_verdict(
    name: str,
    auroc_fwl_full: float,
    auroc_fwl_within: float,
) -> Verdict:
    """Determine verdict for a single comparison's leakage test."""
    if _leakage_detected(auroc_fwl_full, auroc_fwl_within):
        return Verdict.CONFIRMED
    return Verdict.INDETERMINATE


def _polynomial_verdict(
    comparisons: List[Dict],
) -> str:
    """Compute overall polynomial verdict across surviving comparisons.

    Returns one of: "falsified", "strengthened", "weakened".
    """
    surviving = [
        c for c in comparisons
        if c["name"] not in EXCLUDED_COMPARISONS
    ]

    n_collapsed = sum(
        1 for c in surviving
        if not np.isnan(c["auroc_poly2"]) and c["auroc_poly2"] < COLLAPSE_AUROC
    )
    n_preserved = sum(
        1 for c in surviving
        if not np.isnan(c["auroc_poly2"]) and c["auroc_poly2"] > PRESERVE_AUROC
    )

    if n_collapsed >= COLLAPSE_COUNT_THRESHOLD:
        return "falsified"
    if n_preserved == len(surviving):
        return "strengthened"
    return "weakened"


def run_v03(
    output_dir: Path,
    n_permutations: int = 200,
    tracker: Optional[ExperimentTracker] = None,
) -> List[ClaimVerification]:
    """Run FWL leakage test on all 10 Exp 47 comparisons.

    For each comparison, runs four FWL conditions (full-dataset linear,
    within-fold linear, polynomial degree 2, polynomial degree 3) and
    checks for leakage and signal collapse.

    Args:
        output_dir: Directory for result JSON output.
        n_permutations: Reserved for future permutation testing. Currently
            unused since V03 compares FWL conditions via AUROC difference
            rather than permutation null distributions.
        tracker: ExperimentTracker for logging. If None, creates a local one.

    Returns a list of ClaimVerification, one per comparison.
    """
    output_dir = Path(output_dir)

    # Use provided tracker or create a local one
    if tracker is None:
        tracker = ExperimentTracker(
            output_dir=output_dir, experiment_name="V03-fwl-leakage",
        )

    tracker.log_params(
        experiment="V03", finding="C4",
        leakage_threshold=LEAKAGE_THRESHOLD,
        collapse_auroc=COLLAPSE_AUROC,
        preserve_auroc=PRESERVE_AUROC,
        collapse_count_threshold=COLLAPSE_COUNT_THRESHOLD,
        n_comparisons=len(EXP47_COMPARISONS),
    )
    tracker.set_tag("experiment", "V03")
    tracker.set_tag("finding", "C4")

    results: List[ClaimVerification] = []
    comparison_records: List[Dict] = []

    for comp in EXP47_COMPARISONS:
        name = comp["name"]

        # Run all four FWL conditions
        condition_results = _run_comparison(name, comp)

        auroc_full = condition_results["auroc_fwl_full"]
        auroc_within = condition_results["auroc_fwl_within"]
        auroc_poly2 = condition_results["auroc_poly2"]
        auroc_poly3 = condition_results["auroc_poly3"]

        leakage = _leakage_detected(auroc_full, auroc_within)
        verdict = _per_comparison_verdict(name, auroc_full, auroc_within)

        delta = abs(auroc_full - auroc_within) if not (
            np.isnan(auroc_full) or np.isnan(auroc_within)
        ) else float("nan")

        evidence = (
            f"FWL AUROC full={auroc_full:.3f}, within={auroc_within:.3f} "
            f"(delta={delta:.3f}). "
            f"Poly2={auroc_poly2:.3f}, poly3={auroc_poly3:.3f}."
        )
        if leakage:
            evidence += " Leakage DETECTED."

        claim_id = f"C4-{name}"

        cv = ClaimVerification(
            claim_id=claim_id,
            claim_text=f"FWL residualization for {name} is free of leakage",
            paper_section="Section 3.2",
            finding_id="C4",
            severity=Severity.CRITICAL,
            null_hypothesis=(
                f"Within-fold FWL produces the same AUROC as "
                f"full-dataset FWL for {name}"
            ),
            experiment_description=(
                f"Compare full-dataset vs within-fold FWL, plus polynomial "
                f"FWL at degree 2 and 3, for {name}"
            ),
            verdict=verdict,
            evidence_summary=evidence,
            original_value=f"AUROC={auroc_full:.3f} (full-dataset FWL)",
            corrected_value=f"AUROC={auroc_within:.3f} (within-fold FWL)",
            visualization_paths=[],
            gpu_time_seconds=0.0,
            stats={
                "auroc_fwl_full": auroc_full,
                "auroc_fwl_within": auroc_within,
                "auroc_poly2": auroc_poly2,
                "auroc_poly3": auroc_poly3,
                "r_squared": condition_results["r_squared"],
                "leakage_detected": leakage,
                "delta": delta,
                "comparison_name": name,
                "excluded_from_poly_count": name in EXCLUDED_COMPARISONS,
            },
        )
        results.append(cv)

        comparison_records.append({
            "name": name,
            "claim_id": claim_id,
            "auroc_fwl_full": auroc_full,
            "auroc_fwl_within": auroc_within,
            "auroc_poly2": auroc_poly2,
            "auroc_poly3": auroc_poly3,
            "delta": delta,
            "leakage_detected": leakage,
            "verdict": verdict.value,
            "excluded_from_poly_count": name in EXCLUDED_COMPARISONS,
            "r_squared": condition_results["r_squared"],
        })

    # Overall verdicts
    any_leakage = any(c["leakage_detected"] for c in comparison_records)
    leakage_verdict = "confirmed" if any_leakage else "not_detected"

    surviving = [
        c for c in comparison_records
        if c["name"] not in EXCLUDED_COMPARISONS
    ]
    n_collapsed = sum(
        1 for c in surviving
        if not np.isnan(c["auroc_poly2"]) and c["auroc_poly2"] < COLLAPSE_AUROC
    )
    poly_verdict = _polynomial_verdict(comparison_records)

    # --- Log metrics ---
    n_leakage = sum(1 for c in comparison_records if c["leakage_detected"])
    tracker.log_metric("n_leakage_detected", n_leakage)
    tracker.log_metric("n_collapsed_poly2", n_collapsed)
    tracker.log_metric("n_surviving", len(surviving))

    # --- Log per-comparison verdicts ---
    for r in results:
        tracker.log_verdict(r.claim_id, r.verdict.value, r.evidence_summary)

    # --- Handle NaN for JSON serialization ---
    def _nan_to_none(obj):
        """Convert NaN floats to None for JSON compatibility."""
        if isinstance(obj, float) and np.isnan(obj):
            return None
        if isinstance(obj, dict):
            return {k: _nan_to_none(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_nan_to_none(v) for v in obj]
        return obj

    # --- Cache result via tracker ---
    tracker.log_item("v03_result", _nan_to_none({
        "experiment": "V03_FWL_Leakage",
        "finding_id": "C4",
        "leakage_verdict": leakage_verdict,
        "polynomial_verdict": poly_verdict,
        "n_comparisons": len(comparison_records),
        "n_surviving": len(surviving),
        "n_collapsed_poly2": n_collapsed,
        "leakage_threshold": LEAKAGE_THRESHOLD,
        "collapse_auroc_threshold": COLLAPSE_AUROC,
        "preserve_auroc_threshold": PRESERVE_AUROC,
        "comparisons": comparison_records,
    }))

    return results
