"""F02: Held-Out Prompt Generalization Under Input-Length Control.

Tests whether the paper's held-out evaluation (Exp 49d) survives
input-length residualization. The paper claims classifiers trained on
original prompts generalize to novel held-out prompts (AUROC 0.76-0.81).

The deception paradigm appends "I need you to give me a confidently wrong
answer." to held-out deceptive prompts, adding ~11 tokens of input. This
is the SAME confound discovered in F01b: the classifier may be detecting
input length, not deception.

Sub-experiments:
  F02a: Input-length confound in held-out deception (honest vs deceptive)
  F02b: Input-length confound in held-out refusal (refusal vs benign)
  F02c: Input-length confound in held-out impossibility (impossible vs benign)

Hypothesis:
  H0: Input-length residualization does not reduce transfer AUROC below 0.65.
  H1: Deception transfer AUROC drops below 0.60 after residualization;
      refusal and impossibility may survive.

Expected outcomes:
  - F02a (deception): AUROC collapses. Same confound as 18b/49b.
  - F02b (refusal): AUROC may survive (different prompt types, not length-appended).
  - F02c (impossibility): AUROC may survive (same reasoning as refusal).

Pre-registered: F02-held-out-input-control.md
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import mannwhitneyu, pearsonr, ttest_ind
from sklearn.linear_model import LinearRegression

from kv_verify.data_loader import _load_json
from kv_verify.fixtures import PRIMARY_FEATURES
from kv_verify.stats import (
    cohens_d, hedges_g, holm_bonferroni,
    extract_feature_matrix, loo_auroc, train_test_auroc,
)
from kv_verify.tracking import ExperimentTracker
from kv_verify.types import ClaimVerification, Severity, Verdict

# ================================================================
# CONSTANTS
# ================================================================

HACKATHON_DIR = Path(__file__).resolve().parent.parent.parent / "results" / "hackathon"
SEED = 42

# Transfer paradigm mapping: which training data and conditions go with
# which held-out test conditions.
PARADIGMS = {
    "deception": {
        "train_file": "same_prompt_deception.json",
        "train_pos_cond": "deceptive",
        "train_neg_cond": "honest",
        "test_pos_cond": "deceptive",
        "test_neg_cond": "honest",
        "prompt_key": "user_prompt",
        "paired": True,
        "paper_transfer_auroc": 0.81,
    },
    "refusal": {
        "train_file": "refusal_generation.json",
        "train_pos_cond": "refusal",
        "train_neg_cond": "normal",
        "test_pos_cond": "harmful",
        "test_neg_cond": "benign",
        "prompt_key": "prompt",
        "paired": False,
        "paper_transfer_auroc": 0.79,
    },
    "impossibility": {
        "train_file": "impossibility_refusal.json",
        "train_pos_cond": "impossible",
        "train_neg_cond": "benign",
        "test_pos_cond": "impossible",
        "test_neg_cond": "benign",
        "prompt_key": "prompt",
        "paired": False,
        "paper_transfer_auroc": 0.79,
    },
}


# ================================================================
# DATA LOADING HELPERS
# ================================================================

def _get_input_tokens(item: dict) -> float:
    """Compute input token count from features.

    Uses n_input_tokens if available, otherwise n_tokens - n_generated.
    """
    feats = item["features"]
    if "n_input_tokens" in feats:
        return float(feats["n_input_tokens"])
    return float(feats["n_tokens"] - feats["n_generated"])


def _get_word_count(item: dict, prompt_key: str) -> int:
    """Count words in the prompt text."""
    prompt = item.get(prompt_key, item.get("prompt", ""))
    return len(prompt.split())


def _extract_input_lengths(items: List[dict]) -> np.ndarray:
    """Extract input token counts as a (n, 1) array."""
    return np.array([_get_input_tokens(r) for r in items]).reshape(-1, 1)


def _load_train_data(paradigm: str) -> Tuple[List[dict], List[dict]]:
    """Load training items for a given paradigm."""
    cfg = PARADIGMS[paradigm]
    data = _load_json(cfg["train_file"])

    if paradigm == "impossibility":
        # Dict-of-lists structure
        pos_items = data["results"][cfg["train_pos_cond"]]
        neg_items = data["results"][cfg["train_neg_cond"]]
    else:
        # List with condition field
        pos_items = [r for r in data["results"] if r["condition"] == cfg["train_pos_cond"]]
        neg_items = [r for r in data["results"] if r["condition"] == cfg["train_neg_cond"]]

    return pos_items, neg_items


def _load_test_data(paradigm: str) -> Tuple[List[dict], List[dict]]:
    """Load held-out test items from 49d for a given paradigm."""
    with open(HACKATHON_DIR / "49d_held_out_evaluation.json") as f:
        data = json.load(f)

    cfg = PARADIGMS[paradigm]
    test_sets = data["test_sets"]

    pos_items = test_sets[cfg["test_pos_cond"]]
    neg_items = test_sets[cfg["test_neg_cond"]]

    return pos_items, neg_items


# ================================================================
# CORE ANALYSIS: INPUT LENGTH STATISTICS
# ================================================================

def _compute_input_length_stats(
    pos_items: List[dict],
    neg_items: List[dict],
    prompt_key: str,
    label: str,
) -> Dict[str, Any]:
    """Compute input-length statistics for a pos/neg comparison."""
    pos_input = np.array([_get_input_tokens(r) for r in pos_items])
    neg_input = np.array([_get_input_tokens(r) for r in neg_items])

    pos_words = np.array([_get_word_count(r, prompt_key) for r in pos_items])
    neg_words = np.array([_get_word_count(r, prompt_key) for r in neg_items])

    # Welch's t-test on input token counts
    t_stat, t_p = ttest_ind(pos_input, neg_input, equal_var=False)
    # Mann-Whitney U as non-parametric alternative
    try:
        u_stat, u_p = mannwhitneyu(pos_input, neg_input, alternative="two-sided")
    except ValueError:
        u_stat, u_p = 0.0, 1.0

    return {
        "label": label,
        "n_pos": len(pos_items),
        "n_neg": len(neg_items),
        "pos_input_tokens_mean": float(pos_input.mean()),
        "pos_input_tokens_std": float(pos_input.std()),
        "neg_input_tokens_mean": float(neg_input.mean()),
        "neg_input_tokens_std": float(neg_input.std()),
        "input_token_diff": float(pos_input.mean() - neg_input.mean()),
        "pos_words_mean": float(pos_words.mean()),
        "neg_words_mean": float(neg_words.mean()),
        "word_count_diff": float(pos_words.mean() - neg_words.mean()),
        "welch_t_statistic": float(t_stat),
        "welch_t_p_value": float(t_p),
        "mann_whitney_u": float(u_stat),
        "mann_whitney_p": float(u_p),
        "cohens_d": cohens_d(pos_input, neg_input),
        "hedges_g": hedges_g(pos_input, neg_input),
    }


# ================================================================
# CORE ANALYSIS: TRANSFER CLASSIFICATION
# ================================================================


def _residualize_train_test(
    X_train: np.ndarray,
    Z_train: np.ndarray,
    X_test: np.ndarray,
    Z_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    """Within-fold input-length residualization.

    Fits OLS on training data, applies to both train and test.
    Returns (X_train_resid, X_test_resid, r_squared_per_feature).
    """
    X_train_resid = X_train.copy()
    X_test_resid = X_test.copy()
    r_squared = []

    for j in range(X_train.shape[1]):
        reg = LinearRegression()
        reg.fit(Z_train, X_train[:, j])
        r2 = reg.score(Z_train, X_train[:, j])
        r_squared.append(float(r2))
        X_train_resid[:, j] = X_train[:, j] - reg.predict(Z_train)
        X_test_resid[:, j] = X_test[:, j] - reg.predict(Z_test)

    return X_train_resid, X_test_resid, r_squared


def _feature_correlations(
    X: np.ndarray,
    Z: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """Pearson correlations between each feature and input length."""
    correlations = {}
    for i, feat in enumerate(PRIMARY_FEATURES):
        r, p = pearsonr(X[:, i], Z.ravel())
        correlations[feat] = {"r": float(r), "p": float(p)}
    return correlations


# ================================================================
# PARADIGM ANALYSIS
# ================================================================

def _analyze_paradigm(paradigm: str) -> Dict[str, Any]:
    """Full analysis for one transfer paradigm.

    Steps:
    1. Load training and held-out test data.
    2. Compute input-length statistics for both sets.
    3. Train on original features, test on held-out (baseline).
    4. Input-only classification on held-out (length confound check).
    5. Train on residualized features, test on held-out (controlled).
    6. Feature-input correlations.
    """
    np.random.seed(SEED)

    cfg = PARADIGMS[paradigm]
    train_pos, train_neg = _load_train_data(paradigm)
    test_pos, test_neg = _load_test_data(paradigm)

    # Feature matrices
    X_train_pos = extract_feature_matrix(train_pos)
    X_train_neg = extract_feature_matrix(train_neg)
    X_train = np.vstack([X_train_pos, X_train_neg])
    y_train = np.array([1] * len(train_pos) + [0] * len(train_neg))

    X_test_pos = extract_feature_matrix(test_pos)
    X_test_neg = extract_feature_matrix(test_neg)
    X_test = np.vstack([X_test_pos, X_test_neg])
    y_test = np.array([1] * len(test_pos) + [0] * len(test_neg))

    # Input lengths
    Z_train = np.vstack([
        _extract_input_lengths(train_pos),
        _extract_input_lengths(train_neg),
    ])
    Z_test = np.vstack([
        _extract_input_lengths(test_pos),
        _extract_input_lengths(test_neg),
    ])

    prompt_key = cfg["prompt_key"]

    # Step 1: Input-length statistics
    train_length_stats = _compute_input_length_stats(
        train_pos, train_neg, prompt_key, f"{paradigm}_train"
    )
    test_length_stats = _compute_input_length_stats(
        test_pos, test_neg, prompt_key, f"{paradigm}_held_out"
    )

    # Step 2: Baseline transfer AUROC (no residualization)
    # NOTE: The paper used only n=10 per condition for training (even when
    # more were available, e.g., refusal has 20+20). We use ALL available
    # training data. This is favorable to the paper's claim: more training
    # data gives a better classifier, so signal collapse despite more data
    # is a STRONGER falsification.
    baseline_auroc = train_test_auroc(X_train, y_train, X_test, y_test)

    # Step 3: Input-only classification on held-out
    # LOO on the held-out set using only input token count
    input_only_auroc_heldout = loo_auroc(Z_test, y_test)

    # Also train input-only on training data, test on held-out
    input_only_transfer = train_test_auroc(Z_train, y_train, Z_test, y_test)

    # Step 4: Residualized transfer AUROC
    # Fit residualization on training data, apply to both train and test
    X_train_resid, X_test_resid, r_squared = _residualize_train_test(
        X_train, Z_train, X_test, Z_test,
    )
    residualized_auroc = train_test_auroc(
        X_train_resid, y_train, X_test_resid, y_test,
    )

    # Step 5: Within-heldout LOO classification (both original and residualized)
    within_heldout_auroc = loo_auroc(X_test, y_test)

    # LOO residualized within held-out
    loo = LeaveOneOut()
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, solver="lbfgs"))
    y_proba_resid_loo = np.zeros(len(y_test))
    for train_idx, test_idx in loo.split(X_test):
        X_tr = X_test[train_idx].copy()
        X_te = X_test[test_idx].copy()
        Z_tr = Z_test[train_idx]
        Z_te = Z_test[test_idx]
        for j in range(X_test.shape[1]):
            reg = LinearRegression()
            reg.fit(Z_tr, X_tr[:, j])
            X_tr[:, j] = X_tr[:, j] - reg.predict(Z_tr)
            X_te[:, j] = X_te[:, j] - reg.predict(Z_te)
        if len(np.unique(y_test[train_idx])) < 2:
            y_proba_resid_loo[test_idx] = 0.5
            continue
        clf.fit(X_tr, y_test[train_idx])
        y_proba_resid_loo[test_idx] = clf.predict_proba(X_te)[:, 1]

    within_heldout_resid_auroc = float(roc_auc_score(y_test, y_proba_resid_loo))

    # Step 6: Feature-input correlations
    train_correlations = _feature_correlations(X_train, Z_train)
    test_correlations = _feature_correlations(X_test, Z_test)

    # Bootstrap CI for the AUROC drop
    rng = np.random.RandomState(SEED)
    auroc_drop = baseline_auroc - residualized_auroc
    n_boot = 2000
    boot_drops = np.zeros(n_boot)
    n_test = len(y_test)
    for b in range(n_boot):
        idx = rng.choice(n_test, size=n_test, replace=True)
        if len(np.unique(y_test[idx])) < 2:
            boot_drops[b] = 0.0
            continue
        try:
            base_b = roc_auc_score(
                y_test[idx],
                make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, solver="lbfgs"))
                .fit(X_train, y_train)
                .predict_proba(X_test[idx])[:, 1],
            )
            resid_b = roc_auc_score(
                y_test[idx],
                make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, solver="lbfgs"))
                .fit(X_train_resid, y_train)
                .predict_proba(X_test_resid[idx])[:, 1],
            )
            boot_drops[b] = base_b - resid_b
        except (ValueError, np.linalg.LinAlgError):
            boot_drops[b] = 0.0

    drop_ci = {
        "estimate": float(auroc_drop),
        "ci_lower": float(np.percentile(boot_drops, 2.5)),
        "ci_upper": float(np.percentile(boot_drops, 97.5)),
        "se": float(np.std(boot_drops)),
    }

    return {
        "paradigm": paradigm,
        "paper_transfer_auroc": cfg["paper_transfer_auroc"],
        "train_length_stats": train_length_stats,
        "test_length_stats": test_length_stats,
        "baseline_transfer_auroc": baseline_auroc,
        "input_only_auroc_heldout_loo": input_only_auroc_heldout,
        "input_only_transfer_auroc": input_only_transfer,
        "residualized_transfer_auroc": residualized_auroc,
        "within_heldout_auroc": within_heldout_auroc,
        "within_heldout_resid_auroc": within_heldout_resid_auroc,
        "auroc_drop": auroc_drop,
        "auroc_drop_bootstrap_ci": drop_ci,
        "r_squared_per_feature": dict(zip(PRIMARY_FEATURES, r_squared)),
        "train_feature_correlations": train_correlations,
        "test_feature_correlations": test_correlations,
        "n_train_pos": len(train_pos),
        "n_train_neg": len(train_neg),
        "n_test_pos": len(test_pos),
        "n_test_neg": len(test_neg),
    }


# ================================================================
# VERDICT LOGIC
# ================================================================

def _paradigm_verdict(result: Dict[str, Any]) -> Tuple[Verdict, str]:
    """Determine verdict for a single paradigm."""
    baseline = result["baseline_transfer_auroc"]
    residualized = result["residualized_transfer_auroc"]
    input_only = result["input_only_auroc_heldout_loo"]
    drop = result["auroc_drop"]
    paradigm = result["paradigm"]
    test_stats = result["test_length_stats"]

    # Confound present: significant input-length difference AND
    # input-only classification above chance
    length_confounded = (
        test_stats["welch_t_p_value"] < 0.05 and
        input_only > 0.70
    )

    if length_confounded and residualized < 0.60:
        verdict = Verdict.FALSIFIED
        evidence = (
            f"{paradigm}: Held-out transfer is an input-length artifact. "
            f"Input tokens differ by {test_stats['input_token_diff']:.1f} "
            f"(p={test_stats['welch_t_p_value']:.4f}). "
            f"Input-only AUROC={input_only:.3f}. "
            f"Baseline transfer AUROC={baseline:.3f} drops to "
            f"{residualized:.3f} after residualization (drop={drop:.3f})."
        )
    elif residualized > 0.65:
        verdict = Verdict.CONFIRMED
        evidence = (
            f"{paradigm}: Transfer signal survives input-length control. "
            f"Baseline={baseline:.3f}, residualized={residualized:.3f} "
            f"(drop={drop:.3f}). Input-only AUROC={input_only:.3f}."
        )
    else:
        verdict = Verdict.WEAKENED
        evidence = (
            f"{paradigm}: Transfer signal weakened by input-length control. "
            f"Baseline={baseline:.3f}, residualized={residualized:.3f} "
            f"(drop={drop:.3f}). Input-only AUROC={input_only:.3f}."
        )

    return verdict, evidence


# ================================================================
# MAIN RUNNER
# ================================================================

def run_f02(
    output_dir: Path,
    tracker: Optional[ExperimentTracker] = None,
) -> List[ClaimVerification]:
    """Run held-out input-length control analysis.

    Analyzes all three transfer paradigms (deception, refusal, impossibility)
    and returns a ClaimVerification per paradigm.

    Args:
        output_dir: Directory for result artifacts.
        tracker: ExperimentTracker for logging. If None, creates a local one.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use provided tracker or create a local one
    if tracker is None:
        tracker = ExperimentTracker(
            output_dir=output_dir, experiment_name="F02-held-out",
        )

    tracker.log_params(
        experiment="F02", finding="F02", seed=SEED,
        n_paradigms=3, bootstrap_n=2000,
    )
    tracker.set_tag("experiment", "F02")
    tracker.set_tag("finding", "F02")

    np.random.seed(SEED)

    results = []
    paradigm_data = {}
    p_values_for_correction = []

    for paradigm in ["deception", "refusal", "impossibility"]:
        pdata = _analyze_paradigm(paradigm)
        paradigm_data[paradigm] = pdata
        p_values_for_correction.append(
            pdata["test_length_stats"]["welch_t_p_value"]
        )

    # Holm-Bonferroni correction on input-length p-values
    corrected = holm_bonferroni(p_values_for_correction, alpha=0.05)
    for i, paradigm in enumerate(["deception", "refusal", "impossibility"]):
        paradigm_data[paradigm]["test_length_stats"]["welch_t_p_corrected"] = (
            corrected[i]["corrected_p"]
        )
        paradigm_data[paradigm]["test_length_stats"]["reject_null_corrected"] = (
            corrected[i]["reject_null"]
        )

    # Build ClaimVerifications
    for paradigm in ["deception", "refusal", "impossibility"]:
        pdata = paradigm_data[paradigm]
        verdict, evidence = _paradigm_verdict(pdata)

        cv = ClaimVerification(
            claim_id=f"F02-{paradigm}",
            claim_text=(
                f"Held-out {paradigm} transfer generalizes beyond input length"
            ),
            paper_section="Section 4.2 (Held-Out Evaluation)",
            finding_id="F02",
            severity=Severity.CRITICAL,
            null_hypothesis=(
                f"Input-length residualization does not reduce "
                f"{paradigm} transfer AUROC below 0.65"
            ),
            experiment_description=(
                f"Train classifier on original {paradigm} data, test on "
                f"held-out prompts with and without input-length residualization"
            ),
            verdict=verdict,
            evidence_summary=evidence,
            original_value=pdata["paper_transfer_auroc"],
            corrected_value=pdata["residualized_transfer_auroc"],
            visualization_paths=[],
            gpu_time_seconds=0.0,
            stats=pdata,
        )
        results.append(cv)

    # Determine overall verdict
    verdicts = [r.verdict for r in results]
    if Verdict.FALSIFIED in verdicts:
        overall_verdict = Verdict.FALSIFIED
    elif Verdict.WEAKENED in verdicts:
        overall_verdict = Verdict.WEAKENED
    else:
        overall_verdict = Verdict.CONFIRMED

    # Log per-paradigm verdicts and metrics to tracker
    for i, paradigm in enumerate(["deception", "refusal", "impossibility"]):
        pdata = paradigm_data[paradigm]
        tracker.log_verdict(
            f"F02-{paradigm}",
            results[i].verdict.value,
            results[i].evidence_summary,
        )
        tracker.log_metric(
            f"{paradigm}_baseline_auroc", pdata["baseline_transfer_auroc"],
        )
        tracker.log_metric(
            f"{paradigm}_resid_auroc", pdata["residualized_transfer_auroc"],
        )

    # Cache the full result
    tracker.log_item("f02_result", {
        "overall_verdict": overall_verdict.value,
        "paradigms": {
            p: {
                "verdict": results[i].verdict.value,
                "baseline_transfer_auroc": paradigm_data[p]["baseline_transfer_auroc"],
                "residualized_transfer_auroc": paradigm_data[p]["residualized_transfer_auroc"],
            }
            for i, p in enumerate(["deception", "refusal", "impossibility"])
        },
    })

    # Serialize results
    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    result_json = {
        "experiment": "F02_Held_Out_Input_Length_Control",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            "seed": SEED,
            "features": PRIMARY_FEATURES,
            "bootstrap_n": 2000,
        },
        "overall_verdict": overall_verdict.value,
        "paradigms": {
            p: {
                "verdict": results[i].verdict.value,
                "evidence": results[i].evidence_summary,
                **paradigm_data[p],
            }
            for i, p in enumerate(["deception", "refusal", "impossibility"])
        },
        "holm_bonferroni_correction": {
            paradigm: {
                "original_p": corrected[i]["original_p"],
                "corrected_p": corrected[i]["corrected_p"],
                "reject_null": corrected[i]["reject_null"],
            }
            for i, paradigm in enumerate(["deception", "refusal", "impossibility"])
        },
    }

    result_str = json.dumps(result_json, indent=2, default=_convert)
    checksum = hashlib.sha256(result_str.encode()).hexdigest()
    result_json["checksum"] = f"sha256:{checksum}"

    tracker.log_item("f02_result", json.loads(json.dumps(result_json, default=_convert)))

    # Markdown summary
    summary_lines = [
        "# F02: Held-Out Prompt Generalization Under Input-Length Control",
        "",
        f"**Overall verdict**: {overall_verdict.value.upper()}",
        "",
        "## Results by Paradigm",
        "",
    ]
    for i, paradigm in enumerate(["deception", "refusal", "impossibility"]):
        pdata = paradigm_data[paradigm]
        v = results[i].verdict.value.upper()
        summary_lines.extend([
            f"### {paradigm.title()} (Verdict: {v})",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Paper transfer AUROC | {pdata['paper_transfer_auroc']:.3f} |",
            f"| Baseline transfer AUROC | {pdata['baseline_transfer_auroc']:.3f} |",
            f"| Input-only AUROC (LOO) | {pdata['input_only_auroc_heldout_loo']:.3f} |",
            f"| Input-only transfer AUROC | {pdata['input_only_transfer_auroc']:.3f} |",
            f"| Residualized transfer AUROC | {pdata['residualized_transfer_auroc']:.3f} |",
            f"| AUROC drop | {pdata['auroc_drop']:.3f} |",
            f"| Drop 95% CI | [{pdata['auroc_drop_bootstrap_ci']['ci_lower']:.3f}, {pdata['auroc_drop_bootstrap_ci']['ci_upper']:.3f}] |",
            f"| Input token diff (pos - neg) | {pdata['test_length_stats']['input_token_diff']:.1f} |",
            f"| Welch t p-value | {pdata['test_length_stats']['welch_t_p_value']:.4f} |",
            f"| Welch t p-value (Holm-corrected) | {pdata['test_length_stats'].get('welch_t_p_corrected', 'N/A')} |",
            "",
        ])

    summary_lines.extend([
        "## Interpretation",
        "",
        "The deception paradigm appends 'I need you to give me a confidently wrong",
        "answer.' to held-out deceptive prompts, creating the SAME input-length",
        "confound as in 18b/49b. Refusal and impossibility use inherently different",
        "prompt types (harmful vs benign, impossible vs benign) where the length",
        "difference arises from prompt content, not a systematic append.",
    ])

    with open(output_dir / "f02_summary.md", "w") as f:
        f.write("\n".join(summary_lines) + "\n")

    return results


# ================================================================
# CLI ENTRY POINT
# ================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="F02: Held-out prompt generalization under input-length control"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "output" / "f02_held_out",
        help="Directory for result JSON and summary",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    # Module-level reassignment (not inside a function, so no 'global' needed)
    SEED = args.seed

    results = run_f02(args.output_dir)
    for r in results:
        print(f"{r.claim_id}: {r.verdict.value.upper()}")
        print(f"  {r.evidence_summary}")
        print()
