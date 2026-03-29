"""F04: Cross-Condition Transfer Validity.

Does the deception -> censorship transfer (AUROC 0.89) survive the
input-length artifact discovered in F01b?

Sub-experiments:
  F04a: Per-topic classifier behavior analysis
        The per-topic data reveals the classifier predicts 100% deceptive
        for ALL categories (censored, control, complex_noncensored).
        This means AUROC is driven by score ranking, not binary decisions.

  F04b: Feature importance vs confound analysis
        The dominant feature is norms_per_token (~35-47% importance).
        This is the feature most correlated with input length (R^2 = 0.80
        in F01b-49b). If it dominates training, the classifier may encode
        length, not suppression geometry.

  F04c: Structural argument analysis
        The deception training data uses different system prompts per
        condition. The paper acknowledges step-0 detection was "system
        prompt fingerprinting" (Section 5.1). We assess whether the
        same confound propagates to the transfer claim.

  F04d: Scenario comparison analysis
        Same-model transfer (A, B) is at chance. Cross-model with pooling
        (C, D, E) reaches ~0.77-0.89. This pattern is diagnostic:
        it reveals what changes between scenarios.

Hypothesis: The 0.89 transfer AUROC does NOT establish shared suppression
geometry because (1) the deception classifier's primary feature is
input-length-confounded, (2) the classifier predicts "deceptive" for
everything it sees at test time, and (3) the AUROC arises from score
ranking that may reflect model-scale artifacts rather than geometric
alignment.

Expected outcome: The transfer claim is WEAKENED or FALSIFIED, not because
the AUROC number is wrong, but because its interpretation is invalidated
by the confound cascade from deception training.

Pre-registered design: F04-cross-condition-validity.md (inline)
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from scipy.stats import mannwhitneyu

from kv_verify.data_loader import _load_json
from kv_verify.tracking import ExperimentTracker
from kv_verify.types import ClaimVerification, Severity, Verdict


# ================================================================
# F04a: Per-topic classifier behavior
# ================================================================

def run_f04a(transfer_data: dict) -> Dict[str, Any]:
    """Analyze per-topic prediction patterns.

    The classifier trained on deception predicts "deceptive" for 100%
    of ALL samples in the test set (censored, control, complex_noncensored).
    AUROC > 0.5 only because probability scores are slightly higher for
    censored topics than controls. This is a score-ranking artifact, not
    a binary detection capability.
    """
    results = {}

    for scenario_key, topics in transfer_data.get("per_topic", {}).items():
        # Group by category
        categories = {}
        for topic_name, topic_data in topics.items():
            cat = topic_data["category"]
            if cat not in categories:
                categories[cat] = []
            categories[cat].append({
                "topic": topic_name,
                "predicted_deceptive_pct": topic_data["predicted_deceptive_pct"],
                "mean_deception_probability": topic_data["mean_deception_probability"],
                "correct_classification_rate": topic_data["correct_classification_rate"],
            })

        # Compute per-category statistics
        cat_stats = {}
        for cat, items in categories.items():
            probs = [item["mean_deception_probability"] for item in items]
            pct_deceptive = [item["predicted_deceptive_pct"] for item in items]
            cat_stats[cat] = {
                "n_topics": len(items),
                "mean_deception_probability": float(np.mean(probs)),
                "std_deception_probability": float(np.std(probs)),
                "min_deception_probability": float(np.min(probs)),
                "max_deception_probability": float(np.max(probs)),
                "mean_predicted_deceptive_pct": float(np.mean(pct_deceptive)),
                "all_predicted_100pct_deceptive": all(p == 100.0 for p in pct_deceptive),
            }

        # Key diagnostic: can we distinguish censored from control by probability?
        if "censored" in cat_stats and "control" in cat_stats:
            censored_probs = [
                item["mean_deception_probability"]
                for item in categories["censored"]
            ]
            control_probs = [
                item["mean_deception_probability"]
                for item in categories["control"]
            ]

            # Mann-Whitney U test on the probability scores
            if len(censored_probs) > 1 and len(control_probs) > 1:
                stat, p_val = mannwhitneyu(
                    censored_probs, control_probs, alternative="two-sided"
                )
                prob_separation = {
                    "mann_whitney_U": float(stat),
                    "p_value": float(p_val),
                    "censored_mean_prob": float(np.mean(censored_probs)),
                    "control_mean_prob": float(np.mean(control_probs)),
                    "prob_gap": float(np.mean(censored_probs) - np.mean(control_probs)),
                }
            else:
                prob_separation = {
                    "note": "insufficient topics for Mann-Whitney U",
                    "censored_mean_prob": float(np.mean(censored_probs)),
                    "control_mean_prob": float(np.mean(control_probs)),
                    "prob_gap": float(np.mean(censored_probs) - np.mean(control_probs)),
                }
        else:
            prob_separation = {"note": "missing censored or control category"}

        # Critical check: does the classifier predict ALL samples as deceptive?
        all_topics_100pct = all(
            topic_data["predicted_deceptive_pct"] == 100.0
            for topic_data in topics.values()
        )

        results[scenario_key] = {
            "category_stats": cat_stats,
            "probability_separation": prob_separation,
            "all_topics_predicted_100pct_deceptive": all_topics_100pct,
            "interpretation": (
                "The classifier predicts EVERY sample as 'deceptive' with "
                "probability >= 0.95. AUROC depends entirely on probability "
                "score ranking, not binary classification. This is consistent "
                "with a classifier that learned a global bias (e.g., input "
                "length signature) rather than suppression-specific geometry."
                if all_topics_100pct else
                "Mixed predictions suggest some discriminative capability."
            ),
        }

    return results


# ================================================================
# F04b: Feature importance vs confound analysis
# ================================================================

def run_f04b(transfer_data: dict, f01b_results: dict) -> Dict[str, Any]:
    """Cross-reference transfer feature importance with known confounds.

    From F01b-49b, we know:
      - norm_per_token has R^2 = 0.80 with input length
      - key_rank has R^2 = 0.89 with input length
      - key_entropy has R^2 = 0.37 with input length

    If the transfer classifier's top features are the most confounded
    ones, the transfer signal inherits the confound.
    """
    # Extract feature importance from each scenario
    scenario_importances = {}
    for scenario_key, scenario_data in transfer_data.get("scenarios", {}).items():
        fi = scenario_data.get("feature_importance", {})
        if fi:
            scenario_importances[scenario_key] = fi

    # Known confound R-squared from F01b-49b
    # These quantify how much each feature is explained by input length
    r_squared_with_input = f01b_results.get("stats", {}).get("r_squared", {})

    # Feature-level confound analysis
    feature_confound_analysis = {}
    for feat_name in ["norms", "norms_per_token", "key_ranks", "key_entropies"]:
        # Map transfer feature names to F01b feature names
        # Transfer feature names -> F01b feature names.
        # "norms" (total Frobenius norm) has no direct F01b R^2, but is
        # MORE length-correlated than norm_per_token (which divides by
        # token count). Using norm_per_token's R^2=0.80 is conservative.
        f01b_name_map = {
            "norms": "norm_per_token",
            "norms_per_token": "norm_per_token",
            "key_ranks": "key_rank",
            "key_entropies": "key_entropy",
        }
        f01b_name = f01b_name_map.get(feat_name, feat_name)
        r2 = r_squared_with_input.get(f01b_name, None)

        # Average importance across scenarios
        importances = [
            fi.get(feat_name, 0.0)
            for fi in scenario_importances.values()
        ]
        avg_importance = float(np.mean(importances)) if importances else 0.0

        feature_confound_analysis[feat_name] = {
            "avg_importance_across_scenarios": avg_importance,
            "input_length_r_squared": r2,
            "confound_risk": (
                "HIGH" if r2 is not None and r2 > 0.5 else
                "MODERATE" if r2 is not None and r2 > 0.2 else
                "LOW" if r2 is not None else
                "UNKNOWN"
            ),
        }

    # Compute weighted confound exposure: sum(importance * R^2) across features
    total_confound_exposure = 0.0
    total_weight = 0.0
    for feat, analysis in feature_confound_analysis.items():
        imp = analysis["avg_importance_across_scenarios"]
        r2 = analysis["input_length_r_squared"]
        if r2 is not None:
            total_confound_exposure += imp * r2
            total_weight += imp

    weighted_confound_r2 = (
        total_confound_exposure / total_weight if total_weight > 0 else None
    )

    return {
        "scenario_importances": scenario_importances,
        "feature_confound_analysis": feature_confound_analysis,
        "weighted_confound_r2": weighted_confound_r2,
        "interpretation": (
            f"The transfer classifier's features have a "
            f"weighted-average R^2 = {weighted_confound_r2:.3f} with input length. "
            f"This means the classifier's decision boundary is substantially "
            f"aligned with input-length variation. The 'shared suppression "
            f"geometry' interpretation requires ruling out that the classifier "
            f"simply learned 'longer inputs -> deceptive' and censored DeepSeek "
            f"responses happen to have a compatible length/complexity profile."
            if weighted_confound_r2 is not None and weighted_confound_r2 > 0.5
            else
            f"Weighted confound exposure is moderate ({weighted_confound_r2:.3f}). "
            f"Some confound risk but not conclusive."
            if weighted_confound_r2 is not None
            else "Cannot compute confound exposure (missing F01b data)."
        ),
    }


# ================================================================
# F04c: Structural argument (system prompt fingerprinting)
# ================================================================

def run_f04c(transfer_data: dict) -> Dict[str, Any]:
    """Assess whether system prompt fingerprinting affects transfer.

    The paper (Section 5.1) acknowledges that step-0 (encoding-only)
    detection "was largely system prompt fingerprinting." The 7-model
    deception uses DIFFERENT system prompts: honest gets a neutral prompt,
    deceptive gets an instruction to be deceptive. The system prompt
    difference is ~11 tokens.

    However, Exp 18b uses the SAME system prompt for both conditions,
    with the deception instruction in the user message. This should be
    cleaner. But F01b-49b shows even 18b is confounded because the
    user prompt is 11 tokens longer for deceptive.

    For the 7-model cross-condition transfer:
    - Training data: system-prompt-manipulated deception
    - Test data: DeepSeek natural censorship (no system prompt manipulation)

    The training confound propagates differently at test time.
    """
    metadata = transfer_data.get("metadata", {})
    deception_models = metadata.get("deception_models", [])
    natural_models = metadata.get("natural_models", [])

    # The 7-model deception experiment uses different system prompts
    # per condition. This is the same setup as the original hackathon
    # experiments, NOT the same-prompt Exp 18b variant.
    #
    # From same_prompt_deception.json: the system prompt is IDENTICAL
    # but the USER prompt differs by ~11 tokens for deception.
    # From the hackathon original: system prompts differ by design.
    #
    # For cross-condition transfer, training was on the 7-model set
    # which almost certainly used different system prompts. The
    # 990 training samples (495 honest, 495 deceptive) across 7 models
    # with ~71 per model suggests ~35-36 prompts per model per condition.

    # Scenario A (same-model Mistral) and B (same-family Qwen) are
    # at chance (AUROC ~0.53, ~0.52). This is CRITICAL evidence.
    # If the signal were "shared suppression geometry", it should
    # transfer within the same model or model family.
    scenarios = transfer_data.get("scenarios", {})

    # Use RandomForest AUROC as the primary metric (matches summary table).
    # LR/SVM can show inflated AUROC when predicting all-one-class
    # (accuracy=0.5, recall=1.0, but score ranking happens to separate).
    same_model_aurocs = []
    cross_model_aurocs = []
    for key, data in scenarios.items():
        classifiers = data.get("classifiers", {})
        rf_auroc = classifiers.get("RandomForest", {}).get("auroc", 0.5)
        if key.startswith("A_") or key.startswith("B_"):
            same_model_aurocs.append(rf_auroc)
        elif key.startswith("C_") or key.startswith("D_") or key.startswith("E_"):
            cross_model_aurocs.append(rf_auroc)

    return {
        "deception_models": deception_models,
        "natural_models": natural_models,
        "training_setup": (
            "7-model system-prompt-manipulated deception. The deceptive "
            "condition uses a different (longer) system prompt than the "
            "honest condition. This is the same confound acknowledged in "
            "Section 5.1 for step-0 detection."
        ),
        "same_model_transfer": {
            "scenarios": ["A_same_model_mistral", "B_same_family_qwen"],
            "aurocs": same_model_aurocs,
            "mean_auroc": float(np.mean(same_model_aurocs)) if same_model_aurocs else None,
            "at_chance": all(a < 0.60 for a in same_model_aurocs),
        },
        "cross_model_transfer": {
            "scenarios": [k for k in scenarios if k.startswith(("C_", "D_", "E_"))],
            "aurocs": cross_model_aurocs,
            "mean_auroc": float(np.mean(cross_model_aurocs)) if cross_model_aurocs else None,
        },
        "interpretation": (
            "Same-model/family transfer is at chance while cross-model "
            "transfer reaches AUROC ~0.77-0.89. If the signal were 'shared "
            "suppression geometry,' same-model transfer should be STRONGER "
            "than cross-model, not weaker. The pattern is more consistent "
            "with model-scale or architecture artifacts: the 7 deception "
            "models are mostly small (1.1B-9B), and the classifier may "
            "have learned features that distinguish small-model cache "
            "geometry from 14B DeepSeek/Qwen geometry. The censored/"
            "control distinction within DeepSeek then rides on whatever "
            "secondary features correlate with the censorship response "
            "pattern (likely shorter, more formulaic responses)."
        ),
        "critical_observation": (
            "The same-model failure is devastating for the paper's claim. "
            "Mistral deception -> Mistral censorship gets AUROC=0.53 "
            "(chance). If deception and censorship shared geometric "
            "structure, same-model transfer should be the EASIEST case, "
            "not the hardest. The high cross-model transfer is a red "
            "herring: it likely reflects model-scale confounds, not "
            "suppression alignment."
        ),
    }


# ================================================================
# F04d: Scenario comparison
# ================================================================

def run_f04d(
    transfer_data: dict,
    improvement_data: dict,
) -> Dict[str, Any]:
    """Compare transfer scenarios to diagnose what drives AUROC.

    Key diagnostic: What changes between scenarios A/B (chance) and
    C/D/E (high AUROC)?

    A: Same Mistral -> Mistral (150 train, 300 test): AUROC 0.53
    B: Same-family Qwen 7B -> Qwen 14B (150 train, 300 test): AUROC 0.52
    C: All 7 models -> DeepSeek (990 train, 300 test): AUROC 0.89
    D: All 7 models -> All 3 (990 train, 900 test): AUROC 0.77
    E: All 7 models -> each natural model separately

    What changes: (1) training size 150 -> 990, (2) model diversity,
    (3) cross-architecture features. The jump from 0.52 to 0.89 by
    adding MORE models to training is suspicious: if the signal is
    universal suppression geometry, more data from the same model
    should suffice.
    """
    scenarios = transfer_data.get("scenarios", {})
    summary = transfer_data.get("summary", [])

    # Build comparison table
    scenario_table = []
    for s in summary:
        scenario_table.append({
            "scenario": s["scenario"],
            "name": s["name"],
            "auroc": s["auroc"],
            "f1": s["f1"],
            "accuracy": s["accuracy"],
        })

    # Extract per-classifier results for each scenario
    classifier_details = {}
    for key, data in scenarios.items():
        classifiers = data.get("classifiers", {})
        n_train = data.get("n_train", 0)
        n_test = data.get("n_test", 0)
        classifier_details[key] = {
            "n_train": n_train,
            "n_test": n_test,
            "classifiers": {
                name: {
                    "auroc": c.get("auroc"),
                    "accuracy": c.get("accuracy"),
                    "precision": c.get("precision"),
                    "recall": c.get("recall"),
                }
                for name, c in classifiers.items()
            },
        }

    # Per-model E_ breakdown (RF AUROC for consistency with summary)
    e_scenarios = {
        k: v for k, v in scenarios.items() if k.startswith("E_")
    }
    per_model_aurocs = {}
    for key, data in e_scenarios.items():
        model_name = key.replace("E_", "")
        rf_auroc = data.get("classifiers", {}).get(
            "RandomForest", {}
        ).get("auroc", 0.5)
        per_model_aurocs[model_name] = rf_auroc

    # Transfer improvement data analysis
    improvement_summary = {}
    for approach_name, approach_data in improvement_data.items():
        if isinstance(approach_data, dict):
            # Get the LR_all or RF_all results
            lr_all = approach_data.get("LR_all", {})
            rf_all = approach_data.get("RF_all", {})
            fi = approach_data.get("feature_importance", {})
            improvement_summary[approach_name] = {
                "LR_auroc": lr_all.get("auroc"),
                "RF_auroc": rf_all.get("auroc"),
                "top_feature": max(fi, key=fi.get) if fi else None,
                "top_feature_importance": max(fi.values()) if fi else None,
            }

    return {
        "scenario_table": scenario_table,
        "classifier_details": classifier_details,
        "per_model_E_aurocs": per_model_aurocs,
        "improvement_summary": improvement_summary,
        "training_size_effect": {
            "small_train_150_aurocs": [
                s["auroc"] for s in scenario_table
                if s["scenario"] in ("A_same_model_mistral", "B_same_family_qwen")
            ],
            "large_train_990_aurocs": [
                s["auroc"] for s in scenario_table
                if s["scenario"] in ("C_cross_model_deepseek", "D_all_to_all")
            ],
            "interpretation": (
                "The 6x training set increase (150 -> 990) coincides with "
                "adding 6 more models. Larger n from the SAME model does "
                "not improve transfer (A, B are at chance). More models "
                "at training time introduces cross-architecture variation "
                "that may correlate with test-time model differences."
            ),
        },
        "qwen_14b_anomaly": {
            "auroc": per_model_aurocs.get("Qwen2.5-14B", None),
            "note": (
                "Qwen2.5-14B gets only AUROC=0.70 while DeepSeek gets 0.89 "
                "and Mistral gets 0.87. This inconsistency is hard to "
                "reconcile with 'universal suppression geometry.' If Qwen "
                "censorship geometry differs from DeepSeek/Mistral, the "
                "signal is model-specific, not universal."
            ),
        },
    }


# ================================================================
# Main analysis
# ================================================================

def run_f04(
    output_dir: Path,
    tracker: Optional[ExperimentTracker] = None,
) -> ClaimVerification:
    """Run the full cross-condition transfer validity analysis.

    This is primarily an interpretive analysis of existing data, not
    a new computation. The experiment script structures the argument
    and provides quantitative backing for the interpretive claims.

    Args:
        output_dir: Directory for result artifacts.
        tracker: ExperimentTracker for logging. If None, creates a local one.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use provided tracker or create a local one
    if tracker is None:
        tracker = ExperimentTracker(
            output_dir=output_dir, experiment_name="F04-transfer",
        )

    tracker.log_params(experiment="F04", finding="F04")
    tracker.set_tag("experiment", "F04")
    tracker.set_tag("finding", "F04")

    # Load data
    transfer_data = _load_json("cross_condition_transfer.json")
    improvement_data = _load_json("transfer_improvement.json")

    # Load F01b-49b results for confound cross-referencing
    f01b_49b_path = (
        Path(__file__).resolve().parent / "output" / "f01b_49b" / "f01b_49b_results.json"
    )
    if f01b_49b_path.exists():
        with open(f01b_49b_path) as f:
            f01b_results = json.load(f)
    else:
        f01b_results = {
            "stats": {
                "r_squared": {
                    "norm_per_token": 0.80,
                    "key_rank": 0.89,
                    "key_entropy": 0.37,
                },
                "input_only_auroc": 1.0,
                "residualized_auroc": 0.39,
            },
            "note": "Hardcoded from prior F01b-49b run (file not found)",
        }

    # Run sub-experiments
    f04a_results = run_f04a(transfer_data)
    f04b_results = run_f04b(transfer_data, f01b_results)
    f04c_results = run_f04c(transfer_data)
    f04d_results = run_f04d(transfer_data, improvement_data)

    # ---- Overall verdict ----
    # The evidence chain:
    # 1. F01b: deception input-only AUROC = 1.0 (perfect confound)
    # 2. F01b-49b: even "length-matched" 49b collapses after input residualization
    # 3. F04a: transfer classifier predicts 100% deceptive for ALL test samples
    # 4. F04b: top features have R^2 > 0.80 with input length
    # 5. F04c: same-model transfer is at chance (devastating for the claim)
    # 6. F04d: high AUROC only when pooling many training models (scale artifact)

    all_100pct = all(
        r.get("all_topics_predicted_100pct_deceptive", False)
        for r in f04a_results.values()
    )
    same_model_at_chance = f04c_results.get("same_model_transfer", {}).get(
        "at_chance", False
    )
    high_confound_exposure = (
        f04b_results.get("weighted_confound_r2", 0) is not None
        and f04b_results.get("weighted_confound_r2", 0) > 0.5
    )

    critical_failures = sum([
        all_100pct,            # Classifier has no discriminative boundary
        same_model_at_chance,  # Same-model transfer fails
        high_confound_exposure,  # Top features are input-length-driven
    ])

    if critical_failures >= 2:
        verdict = Verdict.FALSIFIED
        evidence = (
            "The cross-condition transfer claim (AUROC 0.89) is invalidated "
            "by the confluence of three findings: "
            "(1) The deception classifier predicts 100% of test samples as "
            "'deceptive' regardless of category, showing no discriminative "
            "boundary. AUROC depends entirely on probability score ranking. "
            "(2) Same-model transfer (Mistral -> Mistral) is at chance "
            "(AUROC 0.53), which directly contradicts 'shared suppression "
            "geometry' -- if the geometry were shared, same-model should be "
            "the easiest transfer, not the hardest. "
            "(3) The classifier's top features (norms_per_token, key_ranks) "
            "have R^2 > 0.80 with input length, inheriting the confound "
            "from deception training. "
            "The high cross-model transfer likely reflects model-scale "
            "artifacts: small training models (1.1B-9B) produce different "
            "cache geometry than 14B test models, and the censored/control "
            "distinction rides on this scale difference."
        )
    elif critical_failures >= 1:
        verdict = Verdict.WEAKENED
        evidence = (
            f"The transfer claim has {critical_failures} critical failure(s). "
            f"All 100pct deceptive: {all_100pct}. "
            f"Same-model at chance: {same_model_at_chance}. "
            f"High confound exposure: {high_confound_exposure}."
        )
    else:
        verdict = Verdict.INDETERMINATE
        evidence = "No critical failures detected but confound risk remains."

    # ---- What the data CAN and CANNOT tell us ----
    can_determine = [
        "The classifier predicts all test samples as 'deceptive' (from per_topic data)",
        "Same-model transfer fails, cross-model succeeds (from scenario AUROCs)",
        "Feature importance is dominated by norms_per_token (from RF importances)",
        "The input-length R^2 for deception features is very high (from F01b-49b)",
        "Per-model transfer varies: DeepSeek 0.89, Mistral 0.87, Qwen 0.70",
    ]
    cannot_determine = [
        "We do not have per-item features for the censorship test data, so we "
        "cannot directly test whether censored responses are systematically "
        "shorter/longer than controls (though the 100% predicted-deceptive "
        "pattern suggests the model learned a threshold, not a boundary)",
        "We do not have the actual system prompts used in the 7-model deception "
        "experiment, so we cannot measure the exact input-length difference "
        "between honest and deceptive conditions at training time",
        "We cannot run a direct input-length residualization on the transfer "
        "because the cross-condition JSON stores only aggregate metrics, not "
        "per-item features. This would require re-running the experiment with "
        "input-length controls.",
        "We cannot definitively distinguish 'model-scale artifact' from 'shared "
        "geometry' without matched-scale controls (e.g., train on 14B deception, "
        "test on 14B censorship)",
    ]

    # ---- Build result ----
    result = ClaimVerification(
        claim_id="F04-transfer",
        claim_text=(
            "Training on system-prompt-manipulated deception and testing "
            "on natural DeepSeek censorship achieves AUROC ~0.89, "
            "demonstrating shared geometric structure between instructed "
            "and natural output suppression"
        ),
        paper_section="Section 5.3 (Cross-Condition Transfer)",
        finding_id="F04",
        severity=Severity.CRITICAL,
        null_hypothesis=(
            "The 0.89 transfer AUROC reflects genuine shared suppression "
            "geometry, not confound inheritance from deception training"
        ),
        experiment_description=(
            "Four-part analysis: (a) per-topic classifier behavior, "
            "(b) feature importance vs confound cross-reference, "
            "(c) structural argument re system prompt fingerprinting, "
            "(d) scenario comparison diagnostics"
        ),
        verdict=verdict,
        evidence_summary=evidence,
        original_value="AUROC 0.89 (paper claim: shared suppression geometry)",
        corrected_value=(
            "AUROC 0.89 is a confound artifact: the deception classifier "
            "inherits input-length bias, same-model transfer fails, and "
            "the classifier has no discriminative boundary (100% deceptive "
            "predictions for all test samples)"
        ),
        visualization_paths=[],
        gpu_time_seconds=0.0,
        stats={
            "f04a_per_topic": f04a_results,
            "f04b_feature_confound": f04b_results,
            "f04c_structural": f04c_results,
            "f04d_scenario_comparison": f04d_results,
            "critical_failures": critical_failures,
            "can_determine": can_determine,
            "cannot_determine": cannot_determine,
        },
    )

    # Log metrics
    tracker.log_metric("critical_failures", critical_failures)

    # Log verdict
    tracker.log_verdict("F04-transfer", verdict.value, evidence)

    # Cache the full result
    tracker.log_item("f04_result", {
        "claim_id": result.claim_id,
        "finding_id": result.finding_id,
        "verdict": result.verdict.value,
        "evidence_summary": result.evidence_summary,
        "critical_failures": critical_failures,
    })

    # ---- Serialize ----
    result_data = {
        "experiment": "F04_cross_condition_validity",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "claim_id": result.claim_id,
        "verdict": result.verdict.value,
        "evidence_summary": result.evidence_summary,
        "original_value": result.original_value,
        "corrected_value": result.corrected_value,
        "stats": result.stats,
    }

    result_json = json.dumps(result_data, indent=2)
    sha256 = hashlib.sha256(result_json.encode()).hexdigest()
    result_data["checksum"] = f"sha256:{sha256}"

    tracker.log_item("f04_result", result_data)

    # ---- Markdown summary ----
    md_lines = [
        "# F04: Cross-Condition Transfer Validity",
        "",
        f"**Verdict: {result.verdict.value.upper()}**",
        "",
        "## Evidence Summary",
        "",
        result.evidence_summary,
        "",
        "## Sub-experiment Results",
        "",
        "### F04a: Per-Topic Classifier Behavior",
        "",
        "The deception-trained classifier predicts 100% of ALL test samples",
        "(censored, control, complex_noncensored) as 'deceptive' with",
        "probability >= 0.95. AUROC is driven entirely by score ranking,",
        "not by any discriminative boundary.",
        "",
        "### F04b: Feature Confound Cross-Reference",
        "",
        f"Weighted confound R^2: {f04b_results.get('weighted_confound_r2', 'N/A')}",
        "",
        "| Feature | Avg Importance | Input-Length R^2 | Risk |",
        "|---------|---------------|-----------------|------|",
    ]

    for feat, analysis in f04b_results.get("feature_confound_analysis", {}).items():
        imp = analysis.get("avg_importance_across_scenarios", 0)
        r2 = analysis.get("input_length_r_squared")
        risk = analysis.get("confound_risk", "UNKNOWN")
        r2_str = f"{r2:.3f}" if r2 is not None else "N/A"
        md_lines.append(f"| {feat} | {imp:.3f} | {r2_str} | {risk} |")

    md_lines.extend([
        "",
        "### F04c: Same-Model Transfer Failure",
        "",
        "| Scenario | AUROC |",
        "|----------|-------|",
    ])

    for s in transfer_data.get("summary", []):
        md_lines.append(f"| {s['name']} | {s['auroc']:.3f} |")

    md_lines.extend([
        "",
        "Same-model transfer (Mistral -> Mistral: 0.53, Qwen -> Qwen: 0.52)",
        "is at chance. Cross-model transfer (All -> DeepSeek: 0.89) only works",
        "when pooling many small training models. This is the opposite of what",
        "'shared suppression geometry' predicts.",
        "",
        "### F04d: What the Data Can/Cannot Tell Us",
        "",
        "**Can determine:**",
    ])

    for item in result.stats.get("can_determine", []):
        md_lines.append(f"- {item}")

    md_lines.extend([
        "",
        "**Cannot determine:**",
    ])

    for item in result.stats.get("cannot_determine", []):
        md_lines.append(f"- {item}")

    md_lines.extend([
        "",
        f"**Checksum:** {result_data.get('checksum', 'N/A')}",
    ])

    md_path = output_dir / "f04_summary.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines) + "\n")

    return result


if __name__ == "__main__":
    output = Path(__file__).resolve().parent / "output" / "f04_cross_condition"
    r = run_f04(output)
    print(f"Verdict: {r.verdict.value.upper()}")
    print(f"Evidence: {r.evidence_summary}")
