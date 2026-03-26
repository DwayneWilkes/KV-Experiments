"""V1: GroupKFold Bug Detection.

Finding C2: ``groups = np.array(list(range(n_pos)) + list(range(n_neg)))``
at ``49_expanded_validation.py:773`` assigns overlapping group IDs across
classes. For non-paired experiments (31, 32, 33, 36), unrelated prompts
share group IDs. For same-prompt experiments (18b, 39), the overlap is
accidentally correct.

Sub-experiments:
    One per Exp 47 comparison (10 total). Each comparison runs with both
    buggy and fixed group assignments, then compares AUROC and significance.

Hypothesis:
    H0: Correcting group assignment does not change any AUROC by more than
    0.02 or flip any significance result.
    H1: At least one comparison changes AUROC by more than 0.05 or flips
    significance after correction.

Expected outcomes:
    Non-paired comparisons may show AUROC changes because overlapping group
    IDs cause GroupKFold to place correlated samples in the same fold.
    Same-prompt paired comparisons should be unaffected because the overlap
    is accidentally correct.

Pre-registered pass/fail (from V01-design.md):
    - abs(AUROC delta) > 0.05: WEAKENED for that comparison
    - Significance flip (either direction): WEAKENED
    - All AUROCs within 0.02 and no flips: CONFIRMED

Spec: verification-pipeline/experiments/V01-groupkfold.md
"""

from pathlib import Path
from typing import List, Optional

import numpy as np

from kv_verify.data_loader import load_comparison_data
from kv_verify.fixtures import EXP47_COMPARISONS
from kv_verify.stats import assign_groups, groupkfold_auroc, holm_bonferroni, permutation_test
from kv_verify.tracking import ExperimentTracker
from kv_verify.types import ClaimVerification, Severity, Verdict


# Pre-registered thresholds (V01-design.md)
AUROC_DELTA_WEAKENED = 0.05
ALPHA = 0.05


def _make_buggy_groups(n_pos: int, n_neg: int) -> np.ndarray:
    """Reproduce the original buggy group assignment.

    From 49_expanded_validation.py:773:
        groups = np.array(list(range(n_pos)) + list(range(n_neg)))

    This creates overlapping group IDs: groups 0..n_pos-1 for positive
    class and 0..n_neg-1 for negative class. GroupKFold then treats
    pos sample i and neg sample i as the same group, leaking information.
    """
    return np.array(list(range(n_pos)) + list(range(n_neg)))


def run_v01(
    output_dir: Path,
    n_permutations: int = 10000,
    tracker: Optional[ExperimentTracker] = None,
) -> List[ClaimVerification]:
    """Run GroupKFold bug detection on all 10 Exp 47 comparisons.

    For each comparison:
    1. Load per-item features from hackathon JSONs.
    2. Run with BUGGY groups (overlapping IDs) + sample-level permutation.
    3. Run with FIXED groups (correct assignment) + group-level permutation.
    4. Compare AUROCs and significance.
    5. Apply Holm-Bonferroni correction to fixed p-values.

    Args:
        output_dir: Directory for result JSON.
        n_permutations: Number of permutations for significance test.
            Default 10000 for production, use 200 for tests.
        tracker: ExperimentTracker for logging. If None, creates a local one.

    Returns:
        List of ClaimVerification, one per comparison.
    """
    output_dir = Path(output_dir)

    # Use provided tracker or create a local one
    if tracker is None:
        tracker = ExperimentTracker(
            output_dir=output_dir, experiment_name="V01-groupkfold",
        )

    tracker.log_params(
        experiment="V01", finding="C2",
        n_permutations=n_permutations, alpha=ALPHA,
        auroc_delta_threshold=AUROC_DELTA_WEAKENED, seed=42,
        n_comparisons=len(EXP47_COMPARISONS),
    )
    tracker.set_tag("experiment", "V01")
    tracker.set_tag("finding", "C2")

    results: List[ClaimVerification] = []
    comparison_records: List[dict] = []

    # Collect fixed p-values for Holm-Bonferroni
    fixed_p_values: List[float] = []

    for comp in EXP47_COMPARISONS:
        name = comp["name"]
        n_pos = comp["n_pos"]
        n_neg = comp["n_neg"]
        paired = comp["paired"]

        # Load data
        X, y, meta = load_comparison_data(name)

        # --- Buggy groups ---
        groups_buggy = _make_buggy_groups(n_pos, n_neg)

        result_buggy = groupkfold_auroc(X, y, groups_buggy)
        auroc_buggy = result_buggy.auroc

        perm_buggy = permutation_test(
            X, y, groups_buggy,
            n_permutations=n_permutations,
            group_level=False,  # original behavior: sample-level
            seed=42,
        )
        p_buggy = perm_buggy["p_value"]

        # --- Fixed groups ---
        groups_fixed = assign_groups(
            n_pos, n_neg,
            paired=paired,
            prompt_indices_pos=meta.get("prompt_indices_pos"),
            prompt_indices_neg=meta.get("prompt_indices_neg"),
        )

        result_fixed = groupkfold_auroc(X, y, groups_fixed)
        auroc_fixed = result_fixed.auroc

        perm_fixed = permutation_test(
            X, y, groups_fixed,
            n_permutations=n_permutations,
            group_level=True,  # corrected behavior: group-level
            seed=42,
        )
        p_fixed = perm_fixed["p_value"]
        fixed_p_values.append(p_fixed)

        # --- Compute delta and significance ---
        auroc_delta = auroc_fixed - auroc_buggy

        sig_buggy = p_buggy < ALPHA
        sig_fixed = p_fixed < ALPHA
        significance_flipped = sig_buggy != sig_fixed

        # Record for JSON
        n_groups_buggy = len(np.unique(groups_buggy))
        n_groups_fixed = len(np.unique(groups_fixed))

        comparison_records.append({
            "name": name,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "paired": paired,
            "n_groups_buggy": n_groups_buggy,
            "n_groups_fixed": n_groups_fixed,
            "auroc_buggy": auroc_buggy,
            "auroc_fixed": auroc_fixed,
            "auroc_delta": auroc_delta,
            "p_value_buggy": p_buggy,
            "p_value_fixed": p_fixed,
            "sig_buggy": sig_buggy,
            "sig_fixed": sig_fixed,
            "significance_flipped": significance_flipped,
        })

    # --- Holm-Bonferroni on fixed p-values ---
    corrected = holm_bonferroni(fixed_p_values, alpha=ALPHA)
    for i, rec in enumerate(comparison_records):
        rec["p_value_fixed_corrected"] = corrected[i]["corrected_p"]
        rec["sig_fixed_corrected"] = corrected[i]["reject_null"]
        # Re-check significance flip against corrected p
        sig_fixed_corrected = corrected[i]["reject_null"]
        sig_buggy_raw = rec["sig_buggy"]
        rec["significance_flipped_corrected"] = sig_buggy_raw != sig_fixed_corrected

    # --- Build ClaimVerification objects ---
    for rec in comparison_records:
        # Determine per-comparison verdict
        delta = rec["auroc_delta"]
        flipped = rec["significance_flipped"]

        if abs(delta) > AUROC_DELTA_WEAKENED:
            verdict = Verdict.WEAKENED
        elif flipped:
            verdict = Verdict.WEAKENED
        else:
            verdict = Verdict.CONFIRMED

        # Evidence summary
        parts = [
            f"AUROC buggy={rec['auroc_buggy']:.4f}, "
            f"fixed={rec['auroc_fixed']:.4f}, "
            f"delta={delta:+.4f}.",
        ]
        if abs(delta) > AUROC_DELTA_WEAKENED:
            parts.append(f"Delta exceeds 0.05 threshold.")
        if flipped:
            direction = "gained" if rec["sig_fixed"] else "lost"
            parts.append(f"Significance {direction} after fix.")
        if not (abs(delta) > AUROC_DELTA_WEAKENED or flipped):
            parts.append("Within pre-registered tolerance.")

        cv = ClaimVerification(
            claim_id=f"C2-{rec['name']}",
            claim_text="GroupKFold prevents prompt leakage across CV folds",
            paper_section="Section 4.1",
            finding_id="C2",
            severity=Severity.CRITICAL,
            null_hypothesis=(
                "Correcting group assignment does not change AUROC "
                "by more than 0.02 or flip significance"
            ),
            experiment_description=(
                f"Compare buggy vs fixed GroupKFold groups for {rec['name']}"
            ),
            verdict=verdict,
            evidence_summary=" ".join(parts),
            original_value=rec["auroc_buggy"],
            corrected_value=rec["auroc_fixed"],
            visualization_paths=[],
            gpu_time_seconds=0.0,
            stats={
                "auroc_buggy": rec["auroc_buggy"],
                "auroc_fixed": rec["auroc_fixed"],
                "auroc_delta": rec["auroc_delta"],
                "p_value_buggy": rec["p_value_buggy"],
                "p_value_fixed": rec["p_value_fixed"],
                "p_value_fixed_corrected": rec["p_value_fixed_corrected"],
                "significance_flipped": rec["significance_flipped"],
                "n_groups_buggy": rec["n_groups_buggy"],
                "n_groups_fixed": rec["n_groups_fixed"],
            },
        )
        results.append(cv)

    # --- Overall verdict ---
    any_weakened = any(r.verdict == Verdict.WEAKENED for r in results)
    overall_verdict = Verdict.WEAKENED if any_weakened else Verdict.CONFIRMED

    # --- Log metrics ---
    n_weakened = sum(1 for r in results if r.verdict == Verdict.WEAKENED)
    n_confirmed = sum(1 for r in results if r.verdict == Verdict.CONFIRMED)
    tracker.log_metric("n_weakened", n_weakened)
    tracker.log_metric("n_confirmed", n_confirmed)
    tracker.log_metric("n_comparisons", len(results))

    # --- Log per-comparison verdicts ---
    for r in results:
        tracker.log_verdict(r.claim_id, r.verdict.value, r.evidence_summary)

    # --- Convert numpy types for JSON serialization ---
    def _convert_record(rec: dict) -> dict:
        """Convert numpy types in a comparison record to native Python."""
        out = {}
        for k, v in rec.items():
            if isinstance(v, (np.integer,)):
                out[k] = int(v)
            elif isinstance(v, (np.floating,)):
                out[k] = float(v)
            elif isinstance(v, np.ndarray):
                out[k] = v.tolist()
            elif isinstance(v, np.bool_):
                out[k] = bool(v)
            else:
                out[k] = v
        return out

    # --- Cache result via tracker ---
    tracker.log_item("v01_result", {
        "experiment": "V01_GroupKFold_Bug_Detection",
        "finding_id": "C2",
        "overall_verdict": overall_verdict.value,
        "parameters": {
            "n_permutations": n_permutations,
            "alpha": ALPHA,
            "auroc_delta_threshold": AUROC_DELTA_WEAKENED,
            "seed": 42,
        },
        "comparisons": [_convert_record(rec) for rec in comparison_records],
    })

    return results
