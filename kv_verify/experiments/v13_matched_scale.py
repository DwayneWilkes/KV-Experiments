"""V13: Matched-Scale Transfer Control.

Tests whether cross-model transfer at matched scale (~7B) shows
shared geometry or if the paper's 0.86 was a scale artifact.

Pre-registered: research-log/V13-design.md

Uses stored 49c cross-model features. Focus on refusal (survived F01b).
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np

from kv_verify.constants import AUROC_FWL_PRESERVE, AUROC_INPUT_CONFOUND
from kv_verify.data_loader import _load_json
from kv_verify.fixtures import PRIMARY_FEATURES
from kv_verify.stats import train_test_auroc
from kv_verify.tracking import ExperimentTracker
from kv_verify.types import ClaimVerification, Severity, Verdict


MODEL_SHORT_NAMES = [
    "Qwen2.5-7B-Instruct",
    "Llama-3.1-8B-Instruct",
    "Mistral-7B-Instruct-v0.3",
]


def run_v13(
    output_dir: Path,
    tracker: Optional[ExperimentTracker] = None,
) -> ClaimVerification:
    """Run matched-scale transfer on refusal data from 49c."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if tracker is None:
        tracker = ExperimentTracker(output_dir=output_dir, experiment_name="V13-matched-scale")

    tracker.log_params(experiment="V13")

    # Load 49c cross-model data
    try:
        data = _load_json("49c_cross_model_suppression.json")
    except FileNotFoundError:
        result_data = {
            "experiment": "V13", "verdict": "BLOCKED",
            "evidence": "49c data not available", "mean_transfer_auroc": 0.0,
            "model_pairs": [], "n_models": 0,
            "system_prompts_identical": False,
        }
        with open(output_dir / "v13_results.json", "w") as f:
            json.dump(result_data, f, indent=2)
        return ClaimVerification(
            claim_id="V13", claim_text="Cross-model transfer reflects shared suppression geometry",
            paper_section="Section 4.3", finding_id="V13", severity=Severity.MAJOR,
            null_hypothesis="Within-scale transfer achieves AUROC > 0.70",
            experiment_description="Blocked: 49c data not available",
            verdict=Verdict.WEAKENED, evidence_summary="49c data not available in this environment.",
            original_value=0.86, corrected_value=None, visualization_paths=[],
            gpu_time_seconds=0.0, stats=result_data,
        )

    # Extract per-model data for refusal task
    per_model = {}
    for model_name in MODEL_SHORT_NAMES:
        model_data = data.get("per_model", {}).get(model_name, {})
        task_data = model_data.get("refusal", {})
        if not task_data:
            continue

        pos = task_data.get("positive", task_data.get("refusal", []))
        neg = task_data.get("negative", task_data.get("normal", []))
        if not pos or not neg:
            continue

        X_pos = np.array([[r["features"][f] for f in PRIMARY_FEATURES] for r in pos])
        X_neg = np.array([[r["features"][f] for f in PRIMARY_FEATURES] for r in neg])
        X = np.vstack([X_pos, X_neg])
        y = np.array([1] * len(X_pos) + [0] * len(X_neg))
        per_model[model_name] = {"X": X, "y": y, "n_pos": len(X_pos), "n_neg": len(X_neg)}

    # All directional pairs
    model_pairs = []
    for train_model in per_model:
        for test_model in per_model:
            if train_model == test_model:
                continue
            train = per_model[train_model]
            test = per_model[test_model]

            raw_auroc = train_test_auroc(train["X"], train["y"], test["X"], test["y"])

            model_pairs.append({
                "train_model": train_model,
                "test_model": test_model,
                "raw_auroc": float(raw_auroc),
                "n_train": len(train["y"]),
                "n_test": len(test["y"]),
            })

    # Aggregate
    aurocs = [p["raw_auroc"] for p in model_pairs]
    mean_auroc = float(np.mean(aurocs)) if aurocs else 0.5

    tracker.log_metric("v13_mean_transfer_auroc", mean_auroc)
    tracker.log_metric("v13_n_pairs", len(model_pairs))

    # Verdict per pre-registered criteria
    if mean_auroc > AUROC_INPUT_CONFOUND:
        verdict = Verdict.CONFIRMED
        evidence = (
            f"Matched-scale transfer AUROC={mean_auroc:.3f} > {AUROC_INPUT_CONFOUND}. "
            f"Shared geometry at matched scale ({len(model_pairs)} pairs)."
        )
    elif mean_auroc < AUROC_FWL_PRESERVE:
        verdict = Verdict.FALSIFIED
        evidence = (
            f"Matched-scale transfer AUROC={mean_auroc:.3f} < {AUROC_FWL_PRESERVE}. "
            f"No transfer even at matched scale. Geometry is not shared."
        )
    else:
        verdict = Verdict.WEAKENED
        evidence = (
            f"Matched-scale transfer AUROC={mean_auroc:.3f}. "
            f"Weak signal ({AUROC_FWL_PRESERVE}-{AUROC_INPUT_CONFOUND})."
        )

    result_data = {
        "experiment": "V13",
        "verdict": verdict.value,
        "evidence": evidence,
        "mean_transfer_auroc": mean_auroc,
        "model_pairs": model_pairs,
        "n_models": len(per_model),
    }
    with open(output_dir / "v13_results.json", "w") as f:
        json.dump(result_data, f, indent=2)

    return ClaimVerification(
        claim_id="V13",
        claim_text="Cross-model transfer reflects shared suppression geometry",
        paper_section="Section 4.3 (Exp 49c)",
        finding_id="V13",
        severity=Severity.MAJOR,
        null_hypothesis="Within-scale transfer achieves AUROC > 0.70",
        experiment_description="Train-test AUROC across matched-scale model pairs (7B-8B)",
        verdict=verdict,
        evidence_summary=evidence,
        original_value=0.86,
        corrected_value=mean_auroc,
        visualization_paths=[],
        gpu_time_seconds=0.0,
        stats=result_data,
    )


if __name__ == "__main__":
    result = run_v13(output_dir=Path("experiments/output/v13"))
    print(f"\nV13 Verdict: {result.verdict.value}")
    print(f"Evidence: {result.evidence_summary}")
