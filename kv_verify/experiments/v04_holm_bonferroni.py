"""V4: Multiple Comparison Correction (Holm-Bonferroni).

Finding C5: 10 binary comparisons in Exp 47 reported without family-wise
correction. The holm_bonferroni() implementation EXISTS in stats_utils.py
but was never called.

This experiment applies Holm-Bonferroni to the 10 p-values and reports
which results lose significance. CPU only, pure computation.

Spec: verification-pipeline/experiments/V04-holm-bonferroni.md
"""

import json
import time
from pathlib import Path
from typing import List

from kv_verify.fixtures import EXP47_COMPARISONS
from kv_verify.stats import holm_bonferroni
from kv_verify.types import ClaimVerification, Severity, Verdict


def run_v04(output_dir: Path) -> ClaimVerification:
    """Run Holm-Bonferroni correction on Exp 47 p-values.

    Returns a ClaimVerification with the verdict.
    """
    t0 = time.monotonic()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract p-values in comparison order
    p_values = [c["p_value"] for c in EXP47_COMPARISONS]
    names = [c["name"] for c in EXP47_COMPARISONS]

    # Apply Holm-Bonferroni
    corrected = holm_bonferroni(p_values, alpha=0.05)

    # Build per-comparison correction records
    corrections: List[dict] = []
    for i, c in enumerate(corrected):
        was_sig = p_values[i] < 0.05
        is_sig = c["reject_null"]
        corrections.append({
            "name": names[i],
            "original_p": c["original_p"],
            "corrected_p": c["corrected_p"],
            "was_significant": was_sig,
            "is_significant": is_sig,
            "flipped": was_sig != is_sig,
            "rank": c["rank"],
        })

    # Count significance changes
    n_sig_raw = sum(1 for p in p_values if p < 0.05)
    n_sig_corrected = sum(1 for c in corrected if c["reject_null"])
    flipped = [c for c in corrections if c["flipped"]]
    flipped_names = [c["name"] for c in flipped]

    # Determine verdict
    if n_sig_corrected < n_sig_raw:
        verdict = Verdict.WEAKENED
    else:
        verdict = Verdict.CONFIRMED

    # Build evidence summary
    if flipped:
        flip_desc = ", ".join(flipped_names)
        evidence = (
            f"{n_sig_raw}/10 significant before correction, "
            f"{n_sig_corrected}/10 after Holm-Bonferroni. "
            f"Lost significance: {flip_desc}."
        )
    else:
        evidence = (
            f"All {n_sig_raw}/10 significant results survive "
            f"Holm-Bonferroni correction."
        )

    elapsed = time.monotonic() - t0

    result = ClaimVerification(
        claim_id="C5-47-holm",
        claim_text=f"{n_sig_raw}/10 comparisons significant in Exp 47",
        paper_section="Section 4.1",
        finding_id="C5",
        severity=Severity.CRITICAL,
        null_hypothesis=f"All {n_sig_raw} significant results survive Holm-Bonferroni",
        experiment_description="Apply Holm-Bonferroni to 10-test family from Exp 47",
        verdict=verdict,
        evidence_summary=evidence,
        original_value=f"{n_sig_raw}/10",
        corrected_value=f"{n_sig_corrected}/10",
        visualization_paths=[],
        gpu_time_seconds=0.0,
        stats={
            "corrections": corrections,
            "n_significant_raw": n_sig_raw,
            "n_significant_corrected": n_sig_corrected,
            "flipped_comparisons": flipped_names,
            "elapsed_seconds": elapsed,
        },
    )

    # Save result JSON
    result_data = {
        "claim_id": result.claim_id,
        "finding_id": result.finding_id,
        "verdict": result.verdict.value,
        "evidence_summary": result.evidence_summary,
        "original_value": result.original_value,
        "corrected_value": result.corrected_value,
        "stats": result.stats,
    }
    with open(output_dir / "v04_results.json", "w") as f:
        json.dump(result_data, f, indent=2)

    return result
