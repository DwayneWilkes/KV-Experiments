"""V10: Power Analysis for all comparisons.

Finding M7: N=10-20 per group limits statistical power. Non-significant
results cannot be interpreted as absence of effect.

Hypothesis:
    H0: All significant results have achieved power > 0.80.
    H1: At least one significant result has power < 0.50.

Pre-registered pass/fail (from V10-design.md):
    - All sig results power > 0.80: CONFIRMED
    - Any sig result power < 0.50: WEAKENED
    - Linchpin (exp36_impossible_vs_harmful) power < 0.30: annotated

Spec: verification-pipeline/experiments/V10-power-analysis.md
"""

from pathlib import Path
from typing import List, Optional

from kv_verify.constants import ALPHA, POWER_ADEQUATE, POWER_N_SIM, POWER_UNDERPOWERED
from kv_verify.fixtures import EXP47_COMPARISONS
from kv_verify.stats import power_analysis
from kv_verify.tracking import ExperimentTracker
from kv_verify.types import ClaimVerification, Severity, Verdict


def run_v10(
    output_dir: Path,
    n_sim: int = POWER_N_SIM,
    tracker: Optional[ExperimentTracker] = None,
) -> ClaimVerification:
    """Compute achieved power for all 10 Exp 47 comparisons.

    Uses simulation-based power analysis at each comparison's
    observed AUROC and sample size.

    Args:
        output_dir: Directory for result JSON.
        n_sim: Simulations per power estimate. Default 10000 for
            production, use 200-500 for tests.
        tracker: ExperimentTracker for logging. If None, creates a local one.

    Returns:
        ClaimVerification with power table in stats.
    """
    output_dir = Path(output_dir)

    # Use provided tracker or create a local one
    if tracker is None:
        tracker = ExperimentTracker(
            output_dir=output_dir, experiment_name="V10-power-analysis",
        )

    tracker.log_params(experiment="V10", finding="M7", n_sim=n_sim, n_comparisons=10)
    tracker.set_tag("experiment", "V10")
    tracker.set_tag("finding", "M7")

    power_table: List[dict] = []
    underpowered: List[str] = []

    for comp in EXP47_COMPARISONS:
        name = comp["name"]
        auroc = comp["auroc"]
        n_per_group = min(comp["n_pos"], comp["n_neg"])
        was_significant = comp["p_value"] < ALPHA

        result = power_analysis(
            n_per_group=n_per_group,
            observed_auroc=auroc,
            n_sim=n_sim,
        )

        entry = {
            "name": name,
            "n_per_group": n_per_group,
            "observed_auroc": auroc,
            "was_significant": was_significant,
            "achieved_power": result["achieved_power"],
            "min_detectable_auroc": result["min_detectable_auroc"],
            "required_n": result["required_n"],
            "observed_d": result["observed_d"],
        }
        power_table.append(entry)

        if was_significant and result["achieved_power"] < POWER_UNDERPOWERED:
            underpowered.append(name)

    # Verdict per pre-registered criteria
    n_underpowered = len(underpowered)
    if n_underpowered > 0:
        verdict = Verdict.WEAKENED
    else:
        # Check if all significant results have power > 0.80
        all_adequate = all(
            e["achieved_power"] > POWER_ADEQUATE
            for e in power_table
            if e["was_significant"]
        )
        verdict = Verdict.CONFIRMED if all_adequate else Verdict.WEAKENED

    # Evidence summary
    sig_count = sum(1 for e in power_table if e["was_significant"])
    adequate_count = sum(
        1 for e in power_table
        if e["was_significant"] and e["achieved_power"] > POWER_ADEQUATE
    )

    evidence_parts = [
        f"{adequate_count}/{sig_count} significant comparisons have power > 0.80.",
    ]
    if underpowered:
        evidence_parts.append(
            f"Underpowered (power < 0.50): {', '.join(underpowered)}."
        )

    # Annotate any comparison below the underpowered threshold as INCONCLUSIVE
    for entry in power_table:
        if entry["was_significant"] and entry["achieved_power"] < POWER_UNDERPOWERED:
            entry["inconclusive"] = True
            entry["inconclusive_reason"] = (
                f"Power={entry['achieved_power']:.2f} < {POWER_UNDERPOWERED}. "
                f"Requires N={entry['required_n']} for adequate power. "
                f"Result cannot support claims in either direction."
            )

    linchpin = [e for e in power_table if "impossible_vs_harmful" in e["name"]]
    if linchpin:
        lp = linchpin[0]
        status = "INCONCLUSIVE" if lp.get("inconclusive") else "adequate"
        evidence_parts.append(
            f"Linchpin (exp36_impossible_vs_harmful): {status}. "
            f"AUROC={lp['observed_auroc']:.2f}, power={lp['achieved_power']:.2f}, "
            f"requires N={lp['required_n']} for {POWER_ADEQUATE} power."
        )

    # Log metrics
    tracker.log_metric("n_underpowered", n_underpowered)
    tracker.log_metric("sig_count", sig_count)
    tracker.log_metric("adequate_count", adequate_count)

    # Log verdict
    evidence_text = " ".join(evidence_parts)
    tracker.log_verdict("M7-power", verdict.value, evidence_text)

    result = ClaimVerification(
        claim_id="M7-power",
        claim_text="All AUROC claims assume adequate statistical power",
        paper_section="Section 4",
        finding_id="M7",
        severity=Severity.MAJOR,
        null_hypothesis="All significant results have power > 0.80",
        experiment_description="Simulation-based power analysis for all 10 comparisons",
        verdict=verdict,
        evidence_summary=evidence_text,
        original_value=f"{sig_count} significant",
        corrected_value=f"{adequate_count}/{sig_count} adequately powered",
        visualization_paths=[],
        gpu_time_seconds=0.0,
        stats={
            "power_table": power_table,
            "n_underpowered": n_underpowered,
            "underpowered_comparisons": underpowered,
        },
    )

    # Cache result via tracker
    tracker.log_item("v10_result", {
        "claim_id": result.claim_id,
        "finding_id": result.finding_id,
        "verdict": result.verdict.value,
        "evidence_summary": result.evidence_summary,
        "stats": result.stats,
    })

    return result
