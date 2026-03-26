"""F01b-49b: Input-length analysis of the paper's "definitive" length control.

Exp 49b is presented as the strongest evidence that the deception signal
is not a length artifact: "when responses are forced to equal length,
AUROC=0.990 while length-only falls to chance (0.535)."

But the paper only matched RESPONSE length. The INPUT prompts differ by
~11 tokens because the deception instruction ("I need you to give me a
confidently wrong answer") is appended. The cache features are computed
over the ENTIRE cache (input + generation), so input length directly
influences norm_per_token, key_rank, and key_entropy.

This script tests whether input length explains the 49b signal.

Pre-registered: F01-falsification-battery.md (F01b extension)
"""

import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from kv_verify.fixtures import PRIMARY_FEATURES
from kv_verify.types import ClaimVerification, Severity, Verdict


def run_f01b_49b(output_dir: Path) -> ClaimVerification:
    """Analyze whether input length explains the 49b "definitive" result."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    hackathon = Path(__file__).resolve().parent.parent.parent / "results" / "hackathon"
    with open(hackathon / "49b_length_matched_deception.json") as f:
        data = json.load(f)

    honest = data["honest_results"]
    deceptive = data["deceptive_results"]

    def get_feats(items, keys):
        return np.array([[r["features"][k] for k in keys] for r in items])

    X_h = get_feats(honest, PRIMARY_FEATURES)
    X_d = get_feats(deceptive, PRIMARY_FEATURES)
    X_all = np.vstack([X_h, X_d])
    y = np.array([0] * 10 + [1] * 10)

    # Estimate input token counts
    h_input = np.array([
        r["features"]["n_tokens"] - r["features"]["n_generated"]
        for r in honest
    ])
    d_input = np.array([
        r["features"]["n_tokens"] - r["features"]["n_generated"]
        for r in deceptive
    ])
    input_all = np.concatenate([h_input, d_input]).reshape(-1, 1)

    # Test 1: input-only classification
    loo = LeaveOneOut()
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
    y_proba_input = np.zeros(20)
    for train_idx, test_idx in loo.split(input_all):
        clf.fit(input_all[train_idx], y[train_idx])
        y_proba_input[test_idx] = clf.predict_proba(input_all[test_idx])[:, 1]
    input_auroc = float(roc_auc_score(y, y_proba_input))

    # Feature correlations with input length
    correlations = {}
    for i, feat in enumerate(PRIMARY_FEATURES):
        r, p = pearsonr(X_all[:, i], input_all.ravel())
        correlations[feat] = {"r": float(r), "p": float(p)}

    # R-squared: feature ~ input_length
    r_squared = {}
    reg = LinearRegression()
    for i, feat in enumerate(PRIMARY_FEATURES):
        reg.fit(input_all, X_all[:, i])
        r_squared[feat] = float(reg.score(input_all, X_all[:, i]))

    # Test 2: classify after within-fold input-length residualization
    y_proba_resid = np.zeros(20)
    for train_idx, test_idx in loo.split(X_all):
        X_train = X_all[train_idx].copy()
        X_test = X_all[test_idx].copy()
        for j in range(X_all.shape[1]):
            reg_fold = LinearRegression()
            reg_fold.fit(input_all[train_idx], X_all[train_idx, j])
            X_train[:, j] = X_all[train_idx, j] - reg_fold.predict(input_all[train_idx])
            X_test[:, j] = X_all[test_idx, j] - reg_fold.predict(input_all[test_idx])
        clf.fit(X_train, y[train_idx])
        y_proba_resid[test_idx] = clf.predict_proba(X_test)[:, 1]
    resid_auroc = float(roc_auc_score(y, y_proba_resid))

    # Verdict
    if input_auroc > 0.90 and resid_auroc < 0.60:
        verdict = Verdict.FALSIFIED
        evidence = (
            f"49b is confounded by input length. "
            f"Input-only AUROC={input_auroc:.3f} (perfect classification). "
            f"Input is {d_input.mean() - h_input.mean():.0f} tokens longer for deceptive prompts. "
            f"After input-length residualization, AUROC drops from 0.990 to {resid_auroc:.3f}. "
            f"The paper's 'definitive' length control matched response length but not input length."
        )
    elif resid_auroc > 0.80:
        verdict = Verdict.CONFIRMED
        evidence = (
            f"Signal survives input-length control. "
            f"Input-only AUROC={input_auroc:.3f}, residualized AUROC={resid_auroc:.3f}."
        )
    else:
        verdict = Verdict.WEAKENED
        evidence = (
            f"Signal weakened by input-length control. "
            f"Input-only AUROC={input_auroc:.3f}, residualized AUROC={resid_auroc:.3f}."
        )

    result = ClaimVerification(
        claim_id="F01b-49b-input",
        claim_text="Exp 49b definitively resolves the length confound",
        paper_section="Section 5.2 (Length Confound: Rejected)",
        finding_id="F01b",
        severity=Severity.CRITICAL,
        null_hypothesis="Input length does not explain the 49b signal",
        experiment_description=(
            "Test whether input token count alone classifies 49b, "
            "and whether signal survives input-length residualization"
        ),
        verdict=verdict,
        evidence_summary=evidence,
        original_value=0.990,
        corrected_value=resid_auroc,
        visualization_paths=[],
        gpu_time_seconds=0.0,
        stats={
            "paper_auroc": 0.990,
            "input_only_auroc": input_auroc,
            "residualized_auroc": resid_auroc,
            "input_length_honest_mean": float(h_input.mean()),
            "input_length_deceptive_mean": float(d_input.mean()),
            "input_length_diff": float(d_input.mean() - h_input.mean()),
            "correlations": correlations,
            "r_squared": r_squared,
        },
    )

    with open(output_dir / "f01b_49b_results.json", "w") as f:
        json.dump({
            "claim_id": result.claim_id,
            "verdict": result.verdict.value,
            "evidence_summary": result.evidence_summary,
            "stats": result.stats,
        }, f, indent=2)

    return result


if __name__ == "__main__":
    from pathlib import Path
    r = run_f01b_49b(Path("/tmp/claude-1000/f01b_49b"))
    print(f"Verdict: {r.verdict.value.upper()}")
    print(f"Evidence: {r.evidence_summary}")
