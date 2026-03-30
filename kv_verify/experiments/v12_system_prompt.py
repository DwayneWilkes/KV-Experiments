"""V12: System Prompt Residualization.

Tests whether deception AUROC survives residualization against
system-prompt-derived features. The paper acknowledges step-0
detection was "system prompt fingerprinting" (Section 5.1).

Pre-registered: research-log/V12-design.md

If system prompts are identical across conditions, the confound
does not exist and the experiment trivially CONFIRMS.
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np

from kv_verify.constants import AUROC_FWL_COLLAPSE, AUROC_FWL_PRESERVE
from kv_verify.data_loader import _load_json
from kv_verify.fixtures import PRIMARY_FEATURES
from kv_verify.stats import assign_groups, groupkfold_auroc, loo_auroc
from kv_verify.tracking import ExperimentTracker
from kv_verify.types import ClaimVerification, Severity, Verdict


def run_v12(
    output_dir: Path,
    tracker: Optional[ExperimentTracker] = None,
) -> ClaimVerification:
    """Run system prompt residualization on deception data."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if tracker is None:
        tracker = ExperimentTracker(output_dir=output_dir, experiment_name="V12-system-prompt")

    tracker.log_params(experiment="V12", finding="V12")

    # Load deception data
    data = _load_json("same_prompt_deception.json")
    global_system_prompt = data.get("system_prompt", "")

    # Check if per-item system prompts exist and differ
    results = data["results"]
    per_item_prompts = set()
    for r in results:
        sp = r.get("system_prompt", global_system_prompt)
        per_item_prompts.add(sp)

    system_prompts_identical = len(per_item_prompts) <= 1

    if system_prompts_identical:
        verdict = Verdict.CONFIRMED
        evidence = (
            f"System prompts are identical across all conditions "
            f"(\"{global_system_prompt[:60]}...\"). "
            f"No system prompt confound exists. Signal is not from prompt fingerprinting."
        )
        residualized_auroc = None
    else:
        # If prompts differ, residualize against system prompt features
        # Build one-hot encoding of system prompt identity
        prompt_to_idx = {p: i for i, p in enumerate(sorted(per_item_prompts))}

        honest = [r for r in results if r["condition"] == "honest"]
        deceptive = [r for r in results if r["condition"] == "deceptive"]

        X_all = np.array([
            [r["features"][f] for f in PRIMARY_FEATURES]
            for r in honest + deceptive
        ])
        y = np.array([0] * len(honest) + [1] * len(deceptive))

        # System prompt one-hot as confound
        Z = np.zeros((len(results), len(per_item_prompts)))
        for i, r in enumerate(honest + deceptive):
            sp = r.get("system_prompt", global_system_prompt)
            Z[i, prompt_to_idx[sp]] = 1.0

        groups = assign_groups(len(honest), len(deceptive), paired=True)

        # Residualized AUROC
        resid_result = groupkfold_auroc(
            X_all, y, groups,
            fwl_confounds=Z,
            fwl_within_fold=True,
        )
        residualized_auroc = resid_result.auroc

        if residualized_auroc < AUROC_FWL_COLLAPSE:
            verdict = Verdict.FALSIFIED
            evidence = (
                f"System prompts differ ({len(per_item_prompts)} unique). "
                f"Residualized AUROC={residualized_auroc:.3f} < {AUROC_FWL_COLLAPSE}. "
                f"Signal collapses after system prompt control."
            )
        elif residualized_auroc > AUROC_FWL_PRESERVE:
            verdict = Verdict.CONFIRMED
            evidence = (
                f"System prompts differ ({len(per_item_prompts)} unique). "
                f"Residualized AUROC={residualized_auroc:.3f} > {AUROC_FWL_PRESERVE}. "
                f"Signal survives system prompt control."
            )
        else:
            verdict = Verdict.WEAKENED
            evidence = (
                f"System prompts differ ({len(per_item_prompts)} unique). "
                f"Residualized AUROC={residualized_auroc:.3f} is ambiguous "
                f"({AUROC_FWL_COLLAPSE}-{AUROC_FWL_PRESERVE})."
            )

    result_data = {
        "experiment": "V12",
        "verdict": verdict.value,
        "evidence": evidence,
        "system_prompts_identical": system_prompts_identical,
        "n_unique_system_prompts": len(per_item_prompts),
        "global_system_prompt": global_system_prompt[:100],
        "residualized_auroc": residualized_auroc,
    }
    with open(output_dir / "v12_results.json", "w") as f:
        json.dump(result_data, f, indent=2)

    return ClaimVerification(
        claim_id="V12",
        claim_text="Deception detection is not system prompt fingerprinting",
        paper_section="Section 5.1",
        finding_id="V12",
        severity=Severity.MAJOR,
        null_hypothesis="Residualizing against system prompt features does not reduce AUROC below 0.60",
        experiment_description="FWL residualization against system-prompt one-hot on exp18b deception",
        verdict=verdict,
        evidence_summary=evidence,
        original_value=None,
        corrected_value=residualized_auroc,
        visualization_paths=[],
        gpu_time_seconds=0.0,
        stats=result_data,
    )


if __name__ == "__main__":
    result = run_v12(output_dir=Path("experiments/output/v12"))
    print(f"\nV12 Verdict: {result.verdict.value}")
    print(f"Evidence: {result.evidence_summary}")
