"""V07: Sycophancy Length Confound.

Finding M2: Length-only AUROC (0.943) EXCEEDS feature AUROC (0.933).
The sycophancy detection claim collapses when tested against a
length-only baseline.

This experiment runs three classifiers on exp39 sycophancy data:
  1. Feature-only:  norm_per_token, key_rank, key_entropy
  2. Length-only:   norm, n_generated
  3. FWL-both:      features residualized against both confounds

Pre-registered criteria:
  - length_only >= feature_only           => FALSIFIED
  - fwl_both < 0.55                       => FALSIFIED
  - feature_only > length_only + 0.05
    AND fwl_both > 0.60                   => CONFIRMED
  - otherwise                             => WEAKENED

CPU only. Uses pre-computed per-item features from same_prompt_sycophancy.json.

Spec: verification-pipeline/experiments/V07-design.md
"""

from pathlib import Path
from typing import Optional

import numpy as np

from kv_verify.data_loader import load_comparison_data
from kv_verify.stats import assign_groups, groupkfold_auroc
from kv_verify.tracking import ExperimentTracker
from kv_verify.types import ClaimVerification, Severity, Verdict


def run_v07(
    output_dir: Path,
    tracker: Optional[ExperimentTracker] = None,
) -> ClaimVerification:
    """Run sycophancy length confound analysis on exp39 data.

    Args:
        output_dir: Directory for result artifacts.
        tracker: ExperimentTracker for logging. If None, creates a local one.

    Returns a ClaimVerification with the verdict.
    """
    output_dir = Path(output_dir)

    # Use provided tracker or create a local one
    if tracker is None:
        tracker = ExperimentTracker(
            output_dir=output_dir, experiment_name="V07-sycophancy",
        )

    tracker.log_params(experiment="V07", finding="M2", alpha=0.05)
    tracker.set_tag("experiment", "V07")
    tracker.set_tag("finding", "M2")

    # ---- Load data ----
    X, y, meta = load_comparison_data("exp39_sycophancy")
    n_pos = meta["n_pos"]
    n_neg = meta["n_neg"]

    groups = assign_groups(
        n_pos,
        n_neg,
        paired=True,
        prompt_indices_pos=meta["prompt_indices_pos"],
        prompt_indices_neg=meta["prompt_indices_neg"],
    )

    Z = meta["confounds"]  # columns: norm, n_generated

    # ---- Classifier 1: Feature-only ----
    # X columns are [norm_per_token, key_rank, key_entropy] per PRIMARY_FEATURES
    feature_result = groupkfold_auroc(
        X, y, groups,
        feature_names=["norm_per_token", "key_rank", "key_entropy"],
    )
    feature_auroc = feature_result.auroc

    # ---- Classifier 2: Length-only ----
    length_result = groupkfold_auroc(
        Z, y, groups,
        feature_names=["norm", "n_generated"],
    )
    length_only_auroc = length_result.auroc

    # ---- Classifier 3: FWL-both (features residualized against confounds) ----
    fwl_result = groupkfold_auroc(
        X, y, groups,
        fwl_confounds=Z,
        fwl_within_fold=True,
        feature_names=["norm_per_token", "key_rank", "key_entropy"],
    )
    fwl_both_auroc = fwl_result.auroc

    # ---- Log metrics ----
    tracker.log_metric("feature_auroc", feature_auroc)
    tracker.log_metric("length_only_auroc", length_only_auroc)
    tracker.log_metric("fwl_both_auroc", fwl_both_auroc)

    # ---- Verdict per pre-registered criteria ----
    if length_only_auroc >= feature_auroc:
        verdict = Verdict.FALSIFIED
    elif fwl_both_auroc < 0.55:
        verdict = Verdict.FALSIFIED
    elif feature_auroc > length_only_auroc + 0.05 and fwl_both_auroc > 0.60:
        verdict = Verdict.CONFIRMED
    else:
        verdict = Verdict.WEAKENED

    # ---- Evidence summary ----
    evidence = (
        f"Feature-only AUROC={feature_auroc:.3f}, "
        f"length-only AUROC={length_only_auroc:.3f}, "
        f"FWL-both AUROC={fwl_both_auroc:.3f}. "
    )
    if verdict == Verdict.FALSIFIED:
        if length_only_auroc >= feature_auroc:
            evidence += (
                "Length-only classifier matches or exceeds cache geometry features. "
                "Sycophancy signal is entirely length-driven."
            )
        else:
            evidence += (
                "FWL-both AUROC below 0.55 after residualization. "
                "No independent geometric signal beyond length."
            )
    elif verdict == Verdict.CONFIRMED:
        evidence += (
            "Features outperform length by >0.05 and FWL-both >0.60. "
            "Cache geometry provides signal beyond length."
        )
    else:
        evidence += (
            "Results fall between pre-registered thresholds. "
            "Some geometric signal may exist but evidence is weak."
        )

    # ---- Log verdict ----
    tracker.log_verdict("M2-39-sycophancy", verdict.value, evidence)

    result = ClaimVerification(
        claim_id="M2-39-sycophancy",
        claim_text="KV-cache geometry detects sycophancy",
        paper_section="Section 4.3",
        finding_id="M2",
        severity=Severity.MAJOR,
        null_hypothesis="Cache geometry features outperform length-only features for sycophancy detection",
        experiment_description=(
            "Three-way classifier comparison on exp39 sycophancy data: "
            "feature-only vs length-only vs FWL-residualized"
        ),
        verdict=verdict,
        evidence_summary=evidence,
        original_value=f"feature AUROC={feature_auroc:.3f}",
        corrected_value=f"length AUROC={length_only_auroc:.3f}, FWL AUROC={fwl_both_auroc:.3f}",
        visualization_paths=[],
        gpu_time_seconds=0.0,
        stats={
            "feature_auroc": feature_auroc,
            "length_only_auroc": length_only_auroc,
            "fwl_both_auroc": fwl_both_auroc,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "n_groups": int(len(np.unique(groups))),
        },
    )

    # ---- Cache result via tracker ----
    tracker.log_item("v07_result", {
        "claim_id": result.claim_id,
        "finding_id": result.finding_id,
        "verdict": result.verdict.value,
        "evidence_summary": result.evidence_summary,
        "original_value": result.original_value,
        "corrected_value": result.corrected_value,
        "stats": result.stats,
    })

    return result
