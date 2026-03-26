"""F03: Cross-Model Transfer with Input-Length Control.

The paper claims cross-model suppression transfer at AUROC ~0.86 using
logistic regression, presented as evidence for a "universal geometry"
of output suppression across model families (Exp 49c).

F01b showed that deception/sycophancy signals are input-length artifacts.
Refusal survived input control in F01b. This experiment tests whether
cross-model refusal transfer also survives input-length control, or
whether the "universality" is just "universal prompt structure."

Sub-experiments:
  F03a: Cross-model transfer (6 pairs) with input-length residualization
  F03b: Within-model classification with input-length residualization
  F03c: Input-only AUROC (just token count) per model pair

Hypothesis: If input-only AUROC is high across models, the cross-model
transfer is driven by prompt structure, not learned geometry.

Expected outcomes:
  - If resid_auroc drops substantially below raw_auroc, transfer is confounded
  - If resid_auroc holds, the cross-model signal is genuine geometry
  - If input_only_auroc is high, "universality" = "universal prompt structure"
"""

import hashlib
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import mannwhitneyu, pearsonr
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from kv_verify.fixtures import PRIMARY_FEATURES
from kv_verify.tracking import ExperimentTracker
from kv_verify.types import ClaimVerification, Severity, Verdict

# Short names used as keys in per_model_data
MODEL_SHORT_NAMES = [
    "Qwen2.5-7B-Instruct",
    "Llama-3.1-8B-Instruct",
    "Mistral-7B-Instruct-v0.3",
]

TASKS = ["refusal"]  # Focus on refusal; it survived F01b input control


def _load_49c_data() -> dict:
    """Load the 49c cross-model suppression JSON."""
    hackathon = Path(__file__).resolve().parent.parent.parent / "results" / "hackathon"
    with open(hackathon / "49c_cross_model_suppression.json") as f:
        return json.load(f)


def _extract_features(items: List[dict], feature_names: List[str]) -> np.ndarray:
    """Extract feature matrix from a list of items."""
    return np.array([
        [item["features"][f] for f in feature_names]
        for item in items
    ])


def _extract_input_tokens(items: List[dict]) -> np.ndarray:
    """Compute input token count as n_tokens - n_generated."""
    return np.array([
        item["features"]["n_tokens"] - item["features"]["n_generated"]
        for item in items
    ], dtype=float)


def _safe_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """AUROC with guard for degenerate cases."""
    if len(np.unique(y_true)) < 2:
        return 0.5
    if np.all(np.isnan(y_score)):
        return 0.5
    return float(roc_auc_score(y_true, y_score))


def _loo_auroc(X: np.ndarray, y: np.ndarray) -> float:
    """Leave-one-out AUROC for small samples."""
    loo = LeaveOneOut()
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
    y_proba = np.zeros(len(y))
    for train_idx, test_idx in loo.split(X):
        if len(np.unique(y[train_idx])) < 2:
            y_proba[test_idx] = 0.5
            continue
        clf.fit(X[train_idx], y[train_idx])
        y_proba[test_idx] = clf.predict_proba(X[test_idx])[:, 1]
    return _safe_auroc(y, y_proba)


def _cross_model_auroc(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> float:
    """Train on model A, test on model B. StandardScaler + LogisticRegression."""
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return 0.5
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
    clf.fit(X_train, y_train)
    y_proba = clf.predict_proba(X_test)[:, 1]
    return _safe_auroc(y_test, y_proba)


def _residualize_cross_model(
    X_train: np.ndarray,
    input_train: np.ndarray,
    X_test: np.ndarray,
    input_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Residualize features against input length.

    Fits linear regression on training data, applies to both train and test.
    This is the correct cross-model procedure: the residualization model
    is learned on train data only, then applied to test data.
    """
    X_train_resid = X_train.copy()
    X_test_resid = X_test.copy()
    for j in range(X_train.shape[1]):
        reg = LinearRegression()
        reg.fit(input_train.reshape(-1, 1), X_train[:, j])
        X_train_resid[:, j] = X_train[:, j] - reg.predict(input_train.reshape(-1, 1))
        X_test_resid[:, j] = X_test[:, j] - reg.predict(input_test.reshape(-1, 1))
    return X_train_resid, X_test_resid


def run_f03(
    output_dir: Path,
    tracker: Optional[ExperimentTracker] = None,
) -> ClaimVerification:
    """Run cross-model transfer with input-length control.

    For each model pair (train A, test B):
      1. Raw AUROC (paper's method)
      2. Input-only AUROC (just token count)
      3. Residualized AUROC (features residualized against input length)

    Also within-model: LOO with input-length residualization.

    Args:
        output_dir: Directory for result artifacts.
        tracker: ExperimentTracker for logging. If None, creates a local one.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use provided tracker or create a local one
    if tracker is None:
        tracker = ExperimentTracker(
            output_dir=output_dir, experiment_name="F03-cross-model",
        )

    tracker.log_params(
        experiment="F03", finding="F03",
        n_models=len(MODEL_SHORT_NAMES), tasks=TASKS,
    )
    tracker.set_tag("experiment", "F03")
    tracker.set_tag("finding", "F03")

    data = _load_49c_data()
    pmd = data["per_model_data"]

    # Paper's reported cross-model results
    paper_cross = data.get("cross_model_transfer", {})

    # ------------------------------------------------------------------
    # F03b: Within-model analysis
    # ------------------------------------------------------------------
    within_model_results = {}

    for model in MODEL_SHORT_NAMES:
        mdata = pmd[model]
        for task in TASKS:
            pos_items = mdata[task]
            neg_items = mdata["benign"]

            X_pos = _extract_features(pos_items, PRIMARY_FEATURES)
            X_neg = _extract_features(neg_items, PRIMARY_FEATURES)
            X_all = np.vstack([X_pos, X_neg])
            y = np.array([1] * len(X_pos) + [0] * len(X_neg))

            input_pos = _extract_input_tokens(pos_items)
            input_neg = _extract_input_tokens(neg_items)
            input_all = np.concatenate([input_pos, input_neg])

            # Input length asymmetry
            u_stat, u_p = mannwhitneyu(input_pos, input_neg, alternative="two-sided")

            # Raw within-model AUROC (LOO)
            raw_auroc = _loo_auroc(X_all, y)

            # Input-only AUROC
            input_only_auroc = _loo_auroc(input_all.reshape(-1, 1), y)

            # Residualized AUROC (within-fold residualization)
            loo = LeaveOneOut()
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
            y_proba_resid = np.zeros(len(y))
            for train_idx, test_idx in loo.split(X_all):
                X_tr = X_all[train_idx].copy()
                X_te = X_all[test_idx].copy()
                for j in range(X_all.shape[1]):
                    reg = LinearRegression()
                    reg.fit(input_all[train_idx].reshape(-1, 1), X_all[train_idx, j])
                    X_tr[:, j] = X_all[train_idx, j] - reg.predict(
                        input_all[train_idx].reshape(-1, 1)
                    )
                    X_te[:, j] = X_all[test_idx, j] - reg.predict(
                        input_all[test_idx].reshape(-1, 1)
                    )
                if len(np.unique(y[train_idx])) < 2:
                    y_proba_resid[test_idx] = 0.5
                    continue
                clf.fit(X_tr, y[train_idx])
                y_proba_resid[test_idx] = clf.predict_proba(X_te)[:, 1]
            resid_auroc = _safe_auroc(y, y_proba_resid)

            # Feature correlations with input length
            correlations = {}
            for i, feat in enumerate(PRIMARY_FEATURES):
                r, p = pearsonr(X_all[:, i], input_all)
                correlations[feat] = {"r": round(float(r), 4), "p": round(float(p), 6)}

            key = f"{model}_{task}"
            within_model_results[key] = {
                "model": model,
                "task": task,
                "n_pos": len(X_pos),
                "n_neg": len(X_neg),
                "input_length_pos_mean": round(float(input_pos.mean()), 1),
                "input_length_neg_mean": round(float(input_neg.mean()), 1),
                "input_length_diff": round(float(input_pos.mean() - input_neg.mean()), 1),
                "input_length_mann_whitney_p": round(float(u_p), 6),
                "raw_auroc": round(raw_auroc, 4),
                "input_only_auroc": round(input_only_auroc, 4),
                "resid_auroc": round(resid_auroc, 4),
                "auroc_drop": round(raw_auroc - resid_auroc, 4),
                "correlations": correlations,
            }

    # ------------------------------------------------------------------
    # F03a: Cross-model transfer
    # ------------------------------------------------------------------
    cross_model_results = {}

    for task in TASKS:
        for train_model in MODEL_SHORT_NAMES:
            for test_model in MODEL_SHORT_NAMES:
                if train_model == test_model:
                    continue

                # Training data from model A
                train_pos = pmd[train_model][task]
                train_neg = pmd[train_model]["benign"]
                X_train_pos = _extract_features(train_pos, PRIMARY_FEATURES)
                X_train_neg = _extract_features(train_neg, PRIMARY_FEATURES)
                X_train = np.vstack([X_train_pos, X_train_neg])
                y_train = np.array([1] * len(X_train_pos) + [0] * len(X_train_neg))
                input_train = np.concatenate([
                    _extract_input_tokens(train_pos),
                    _extract_input_tokens(train_neg),
                ])

                # Test data from model B
                test_pos = pmd[test_model][task]
                test_neg = pmd[test_model]["benign"]
                X_test_pos = _extract_features(test_pos, PRIMARY_FEATURES)
                X_test_neg = _extract_features(test_neg, PRIMARY_FEATURES)
                X_test = np.vstack([X_test_pos, X_test_neg])
                y_test = np.array([1] * len(X_test_pos) + [0] * len(X_test_neg))
                input_test = np.concatenate([
                    _extract_input_tokens(test_pos),
                    _extract_input_tokens(test_neg),
                ])

                # 1. Raw cross-model AUROC
                raw_auroc = _cross_model_auroc(X_train, y_train, X_test, y_test)

                # 2. Input-only cross-model AUROC
                input_only_auroc = _cross_model_auroc(
                    input_train.reshape(-1, 1), y_train,
                    input_test.reshape(-1, 1), y_test,
                )

                # 3. Residualized cross-model AUROC
                X_train_resid, X_test_resid = _residualize_cross_model(
                    X_train, input_train, X_test, input_test,
                )
                resid_auroc = _cross_model_auroc(
                    X_train_resid, y_train, X_test_resid, y_test,
                )

                # Look up paper's reported AUROC for this pair
                paper_key = f"{task}_{train_model}_to_{test_model}"
                paper_auroc = None
                for pk, pv in paper_cross.items():
                    if (pv.get("task") == task
                            and train_model in pv.get("train_model", "")
                            and test_model in pv.get("test_model", "")):
                        paper_auroc = pv["auroc"]
                        break

                # Input length asymmetry on test model
                test_input_pos = _extract_input_tokens(test_pos)
                test_input_neg = _extract_input_tokens(test_neg)
                input_diff = float(test_input_pos.mean() - test_input_neg.mean())

                pair_key = f"{task}_{train_model}_to_{test_model}"
                cross_model_results[pair_key] = {
                    "train_model": train_model,
                    "test_model": test_model,
                    "task": task,
                    "paper_auroc": round(paper_auroc, 4) if paper_auroc is not None else None,
                    "raw_auroc": round(raw_auroc, 4),
                    "input_only_auroc": round(input_only_auroc, 4),
                    "resid_auroc": round(resid_auroc, 4),
                    "auroc_drop": round(raw_auroc - resid_auroc, 4),
                    "test_input_length_diff": round(input_diff, 1),
                }

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------
    cross_raw = [v["raw_auroc"] for v in cross_model_results.values()]
    cross_resid = [v["resid_auroc"] for v in cross_model_results.values()]
    cross_input = [v["input_only_auroc"] for v in cross_model_results.values()]

    within_raw = [v["raw_auroc"] for v in within_model_results.values()]
    within_resid = [v["resid_auroc"] for v in within_model_results.values()]
    within_input = [v["input_only_auroc"] for v in within_model_results.values()]

    # Count how many cross-model pairs have input_only > 0.70
    n_input_confounded = sum(1 for v in cross_model_results.values() if v["input_only_auroc"] > 0.70)
    n_cross_total = len(cross_model_results)

    # Count how many cross-model pairs lose >0.10 AUROC after residualization
    n_degraded = sum(1 for v in cross_model_results.values() if v["auroc_drop"] > 0.10)

    # Count how many within-model results survive residualization (resid > 0.70)
    n_within_survive = sum(1 for v in within_model_results.values() if v["resid_auroc"] > 0.70)
    n_within_total = len(within_model_results)

    # ------------------------------------------------------------------
    # Verdict
    # ------------------------------------------------------------------
    # Cross-model transfer is the claim being tested.
    # Pre-registered logic:
    #   - If mean cross-model input_only AUROC > 0.70: input structure drives transfer
    #   - If mean cross-model resid AUROC drops below 0.55: transfer is confounded
    #   - If mean cross-model resid AUROC holds > 0.70: genuine geometry
    mean_cross_raw = float(np.mean(cross_raw))
    mean_cross_resid = float(np.mean(cross_resid))
    mean_cross_input = float(np.mean(cross_input))

    if mean_cross_input > 0.70 and mean_cross_resid < 0.55:
        verdict = Verdict.FALSIFIED
        evidence = (
            f"Cross-model transfer is an input-length artifact. "
            f"Mean input-only AUROC={mean_cross_input:.3f} (high). "
            f"After residualization, mean AUROC drops from {mean_cross_raw:.3f} to "
            f"{mean_cross_resid:.3f}. The 'universal geometry' is universal prompt structure."
        )
    elif mean_cross_resid > 0.70:
        verdict = Verdict.CONFIRMED
        evidence = (
            f"Cross-model transfer survives input-length control. "
            f"Mean raw AUROC={mean_cross_raw:.3f}, residualized={mean_cross_resid:.3f}. "
            f"Genuine geometric signal transfers across models."
        )
    elif mean_cross_raw < 0.55:
        verdict = Verdict.WEAKENED
        evidence = (
            f"Cross-model transfer was already near chance before input control. "
            f"Mean raw AUROC={mean_cross_raw:.3f}. Nothing to residualize. "
            f"The paper's ~0.86 claim is not reproduced in the raw data."
        )
    else:
        verdict = Verdict.WEAKENED
        evidence = (
            f"Mixed results. Mean raw AUROC={mean_cross_raw:.3f}, "
            f"input-only={mean_cross_input:.3f}, residualized={mean_cross_resid:.3f}. "
            f"Some signal may exist but input length contributes."
        )

    result = ClaimVerification(
        claim_id="F03-cross-model",
        claim_text="Cross-model suppression transfer reflects universal geometric signal",
        paper_section="Section 5.3 (Exp 49c)",
        finding_id="F03",
        severity=Severity.CRITICAL,
        null_hypothesis="Cross-model transfer AUROC is not explained by input length",
        experiment_description=(
            "For 6 cross-model pairs and 3 within-model tests on refusal task: "
            "compare raw AUROC, input-only AUROC, and input-residualized AUROC"
        ),
        verdict=verdict,
        evidence_summary=evidence,
        original_value=f"mean cross-model raw AUROC={mean_cross_raw:.3f}",
        corrected_value=f"mean cross-model resid AUROC={mean_cross_resid:.3f}",
        visualization_paths=[],
        gpu_time_seconds=0.0,
        stats={
            "cross_model_results": cross_model_results,
            "within_model_results": within_model_results,
            "summary": {
                "mean_cross_raw_auroc": round(mean_cross_raw, 4),
                "mean_cross_resid_auroc": round(mean_cross_resid, 4),
                "mean_cross_input_auroc": round(mean_cross_input, 4),
                "n_cross_input_confounded": n_input_confounded,
                "n_cross_degraded": n_degraded,
                "n_cross_total": n_cross_total,
                "mean_within_raw_auroc": round(float(np.mean(within_raw)), 4),
                "mean_within_resid_auroc": round(float(np.mean(within_resid)), 4),
                "mean_within_input_auroc": round(float(np.mean(within_input)), 4),
                "n_within_survive": n_within_survive,
                "n_within_total": n_within_total,
            },
        },
    )

    # Log metrics
    tracker.log_metric("mean_cross_raw_auroc", mean_cross_raw)
    tracker.log_metric("mean_cross_resid_auroc", mean_cross_resid)
    tracker.log_metric("mean_cross_input_auroc", mean_cross_input)
    tracker.log_metric("n_cross_input_confounded", n_input_confounded)
    tracker.log_metric("n_cross_degraded", n_degraded)

    # Log verdict
    tracker.log_verdict("F03-cross-model", verdict.value, evidence)

    # Cache the full result
    tracker.log_item("f03_result", {
        "claim_id": result.claim_id,
        "finding_id": result.finding_id,
        "verdict": result.verdict.value,
        "evidence_summary": result.evidence_summary,
        "stats": result.stats,
    })

    # ------------------------------------------------------------------
    # Serialize results
    # ------------------------------------------------------------------
    result_data = {
        "experiment": "F03_cross_model_input_control",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
        "claim_id": result.claim_id,
        "finding_id": result.finding_id,
        "verdict": result.verdict.value,
        "evidence_summary": result.evidence_summary,
        "original_value": result.original_value,
        "corrected_value": result.corrected_value,
        "stats": result.stats,
    }

    result_json = json.dumps(result_data, indent=2)
    checksum = hashlib.sha256(result_json.encode()).hexdigest()
    result_data["checksum"] = f"sha256:{checksum}"

    result_path = output_dir / "f03_results.json"
    with open(result_path, "w") as f:
        json.dump(result_data, f, indent=2)

    # ------------------------------------------------------------------
    # Print summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("F03: Cross-Model Transfer with Input-Length Control")
    print("=" * 90)

    print(f"\nVerdict: {result.verdict.value.upper()}")
    print(f"Evidence: {result.evidence_summary}")

    print("\n--- Within-Model Results ---")
    print(f"{'Model':<35} {'Raw':>7} {'Input':>7} {'Resid':>7} {'Drop':>7} {'InpDiff':>8}")
    print("-" * 78)
    for key, v in within_model_results.items():
        model_short = v["model"][:32]
        print(
            f"{model_short:<35} "
            f"{v['raw_auroc']:>7.3f} "
            f"{v['input_only_auroc']:>7.3f} "
            f"{v['resid_auroc']:>7.3f} "
            f"{v['auroc_drop']:>+7.3f} "
            f"{v['input_length_diff']:>+8.1f}"
        )

    print("\n--- Cross-Model Transfer Results ---")
    print(f"{'Pair':<55} {'Paper':>7} {'Raw':>7} {'Input':>7} {'Resid':>7} {'Drop':>7}")
    print("-" * 98)
    for key, v in cross_model_results.items():
        pair = f"{v['train_model'][:18]} -> {v['test_model'][:18]}"
        paper_str = f"{v['paper_auroc']:.3f}" if v["paper_auroc"] is not None else "   N/A"
        print(
            f"{pair:<55} "
            f"{paper_str:>7} "
            f"{v['raw_auroc']:>7.3f} "
            f"{v['input_only_auroc']:>7.3f} "
            f"{v['resid_auroc']:>7.3f} "
            f"{v['auroc_drop']:>+7.3f}"
        )

    print(f"\nMean cross-model: raw={mean_cross_raw:.3f}, "
          f"input_only={mean_cross_input:.3f}, resid={mean_cross_resid:.3f}")
    print(f"Results saved to: {result_path}")
    print()

    return result


if __name__ == "__main__":
    output = Path(__file__).resolve().parent / "output" / "f03_cross_model"
    r = run_f03(output)
    print(f"\nFinal verdict: {r.verdict.value.upper()}")
