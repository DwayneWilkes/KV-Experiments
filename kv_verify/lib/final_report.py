"""Final verification report generator.

Collects all experiment results, applies global Holm-Bonferroni,
and generates a markdown report with per-claim verdicts.
"""

from typing import Dict, List, Optional

from kv_verify.lib.stats import global_holm_bonferroni

# All 14 claims from the paper, mapped to their testing experiments
CLAIMS = [
    {"id": "C1", "text": "KV-cache geometry contains detectable signal about model behavior", "experiments": ["F01a", "F01c"]},
    {"id": "C2", "text": "GroupKFold CV prevents prompt leakage across folds", "experiments": ["V01"]},
    {"id": "C3", "text": "9/10 comparisons yield significant results (p<0.05)", "experiments": ["V04"]},
    {"id": "C4", "text": "Features retain significance after FWL residualization", "experiments": ["V03"]},
    {"id": "C5", "text": "KV-cache geometry detects sycophancy (AUROC=0.938)", "experiments": ["V07"]},
    {"id": "C6", "text": "Refusal/jailbreak geometry is genuine (survives input control)", "experiments": ["F01b"]},
    {"id": "C7", "text": "Cross-model transfer at AUROC ~0.86", "experiments": ["F03"]},
    {"id": "C8", "text": "Cross-condition transfer deception->censorship (AUROC=0.89)", "experiments": ["F04"]},
    {"id": "C9", "text": "Held-out prompt generalization (AUROC 0.76-0.81)", "experiments": ["F02"]},
    {"id": "C10", "text": "Input features do not confound the signal", "experiments": ["F01b"]},
    {"id": "C11", "text": "49b definitive length control is valid", "experiments": ["F01b_49b"]},
    {"id": "C12", "text": "Deception and refusal share a suppression mechanism", "experiments": ["F04"]},
    {"id": "M2", "text": "Length is adequately controlled via FWL and 49b", "experiments": ["V03", "F01b"]},
    {"id": "M7", "text": "Sample sizes provide adequate statistical power", "experiments": ["V10"]},
]


def generate_final_report(
    experiment_results: Dict[str, Dict],
    dataset_reports: Optional[Dict] = None,
) -> str:
    """Generate a markdown report of all claim verdicts.

    Args:
        experiment_results: Dict mapping experiment name to result dict
            (must have "verdict" key, optionally "p_value").
        dataset_reports: Optional dict of DatasetReport summaries per dataset.

    Returns:
        Markdown string with the full report.
    """
    lines = []
    lines.append("# KV-Cache Verification: Final Report")
    lines.append("")

    # Global Holm-Bonferroni correction
    p_values = [
        (name, r["p_value"])
        for name, r in experiment_results.items()
        if "p_value" in r
    ]
    holm_results = {}
    if p_values:
        corrected = global_holm_bonferroni(p_values)
        for r in corrected:
            holm_results[r["name"]] = r
        lines.append("## Global Holm-Bonferroni Correction")
        lines.append("")
        lines.append(f"Applied across {len(p_values)} experiments with p-values.")
        lines.append("")
        lines.append("| Experiment | Original p | Corrected p | Reject H0 |")
        lines.append("|-----------|-----------|------------|-----------|")
        for r in corrected:
            reject = "Yes" if r["reject"] else "**No**"
            lines.append(f"| {r['name']} | {r['original_p']:.4f} | {r['corrected_p']:.4f} | {reject} |")
        lines.append("")

    # Per-claim verdicts
    lines.append("## Claim Verdicts")
    lines.append("")
    lines.append("| Claim | Text | Experiments | Verdict | Corrected |")
    lines.append("|-------|------|-------------|---------|-----------|")

    for claim in CLAIMS:
        cid = claim["id"]
        text = claim["text"][:60]
        exps = ", ".join(claim["experiments"])

        # Find verdict from experiment results
        verdict = "NOT TESTED"
        corrected_note = ""
        for exp in claim["experiments"]:
            if exp in experiment_results:
                verdict = experiment_results[exp].get("verdict", "UNKNOWN")
                if exp in holm_results:
                    if not holm_results[exp]["reject"]:
                        corrected_note = "Lost significance"
                break

        lines.append(f"| {cid} | {text} | {exps} | {verdict} | {corrected_note} |")

    lines.append("")

    # Dataset quality section
    if dataset_reports:
        lines.append("## Dataset Quality")
        lines.append("")
        for name, dr in dataset_reports.items():
            verdict = dr.get("overall_verdict", "UNKNOWN")
            lines.append(f"- **{name}**: {verdict}")
        lines.append("")

    # Summary
    tested = sum(1 for c in CLAIMS if any(e in experiment_results for e in c["experiments"]))
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Claims tested: {tested}/{len(CLAIMS)}")
    if experiment_results:
        verdicts = [r.get("verdict", "UNKNOWN") for r in experiment_results.values()]
        from collections import Counter
        vc = Counter(verdicts)
        for v, count in sorted(vc.items()):
            lines.append(f"- {v}: {count}")
    lines.append("")

    return "\n".join(lines)
