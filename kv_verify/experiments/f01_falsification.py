"""F01: Falsification Battery.

Four tests designed to kill the signal as fast as possible.
If any test is FATAL, downstream claims are invalid.

F01a: Null experiment (same-condition classification)
F01b: Input-length confound
F01c: Format classifier baseline (response text statistics)
F01d: Independent feature re-extraction (requires GPU)

Pre-registered design: research-log/F01-falsification-battery.md
"""

import json
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from kv_verify.data_loader import load_comparison_data, list_comparisons, _load_json
from kv_verify.fixtures import EXP47_COMPARISONS, PRIMARY_FEATURES
from kv_verify.types import ClaimVerification, Severity, Verdict


# ================================================================
# F01a: Null Experiment
# ================================================================

def _stratified_auroc(X, y, n_splits=5):
    """Quick AUROC via StratifiedKFold (no group structure)."""
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, solver="lbfgs"))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    y_proba = np.full(len(y), np.nan)
    for train_idx, test_idx in skf.split(X, y):
        if len(np.unique(y[train_idx])) < 2:
            continue
        clf.fit(X[train_idx], y[train_idx])
        y_proba[test_idx] = clf.predict_proba(X[test_idx])[:, 1]
    valid = ~np.isnan(y_proba)
    if valid.sum() < 4 or len(np.unique(y[valid])) < 2:
        return 0.5
    return float(roc_auc_score(y[valid], y_proba[valid]))


def run_f01a(output_dir: Path, n_repeats: int = 100) -> ClaimVerification:
    """Null experiment: classify within the same condition.

    If the classifier can distinguish random halves of the same condition
    at AUROC > 0.65, it's picking up prompt-level variation, not condition signal.
    """
    t0 = time.monotonic()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all condition pools from source JSONs
    condition_pools = {}

    # exp31: refusal and normal pools
    rg = _load_json("refusal_generation.json")
    for cond in ["refusal", "normal"]:
        items = [r for r in rg["results"] if r["condition"] == cond]
        X = np.array([[r["features"][f] for f in PRIMARY_FEATURES] for r in items])
        condition_pools[f"exp31_{cond}"] = X

    # exp36: impossible, harmful, benign pools
    ir = _load_json("impossibility_refusal.json")
    for cond in ["impossible", "harmful", "benign"]:
        items = ir["results"][cond]
        X = np.array([[r["features"][f] for f in PRIMARY_FEATURES] for r in items])
        condition_pools[f"exp36_{cond}"] = X

    rng = np.random.RandomState(42)
    null_results = {}

    for pool_name, X_pool in condition_pools.items():
        n = len(X_pool)
        if n < 10:
            continue

        aurocs = []
        half = n // 2
        for _ in range(n_repeats):
            perm = rng.permutation(n)
            X_a, X_b = X_pool[perm[:half]], X_pool[perm[half:2 * half]]
            X = np.vstack([X_a, X_b])
            y = np.array([1] * len(X_a) + [0] * len(X_b))
            auroc = _stratified_auroc(X, y, n_splits=min(5, half))
            aurocs.append(auroc)

        null_results[pool_name] = {
            "mean_auroc": float(np.mean(aurocs)),
            "std_auroc": float(np.std(aurocs)),
            "ci_lower": float(np.percentile(aurocs, 2.5)),
            "ci_upper": float(np.percentile(aurocs, 97.5)),
            "n_items": n,
            "n_repeats": n_repeats,
        }

    # Check if any pool has mean null AUROC > 0.65
    max_null = max(r["mean_auroc"] for r in null_results.values())
    fatal = max_null > 0.65
    worst_pool = max(null_results, key=lambda k: null_results[k]["mean_auroc"])

    if fatal:
        verdict = Verdict.FALSIFIED
        evidence = (
            f"FATAL: Same-condition classification achieves AUROC={max_null:.3f} "
            f"(pool: {worst_pool}). Classifier picks up prompt-level variation, "
            f"not condition signal. All between-condition claims are suspect."
        )
    else:
        verdict = Verdict.CONFIRMED
        evidence = (
            f"Null AUROCs all below 0.65 (max={max_null:.3f} in {worst_pool}). "
            f"Within-condition classification near chance. "
            f"Between-condition signal is not attributable to prompt variation."
        )

    elapsed = time.monotonic() - t0

    result = ClaimVerification(
        claim_id="F01a-null",
        claim_text="KV-cache features detect condition-level signal, not prompt-level variation",
        paper_section="All",
        finding_id="F01a",
        severity=Severity.CRITICAL,
        null_hypothesis="Same-condition classification AUROC < 0.65",
        experiment_description="Classify random halves of same condition, 100 repeats per pool",
        verdict=verdict,
        evidence_summary=evidence,
        original_value=None,
        corrected_value=float(max_null),
        visualization_paths=[],
        gpu_time_seconds=0.0,
        stats={
            "null_results": null_results,
            "max_null_auroc": max_null,
            "worst_pool": worst_pool,
            "fatal": fatal,
            "elapsed_seconds": elapsed,
        },
    )

    with open(output_dir / "f01a_results.json", "w") as f:
        json.dump({"claim_id": result.claim_id, "verdict": result.verdict.value,
                    "evidence_summary": result.evidence_summary, "stats": result.stats}, f, indent=2)

    return result


# ================================================================
# F01b: Input-Length Confound
# ================================================================

def run_f01b(output_dir: Path) -> ClaimVerification:
    """Input-length confound: can prompt text features predict condition?

    If a classifier on input word count, character count, and sentence count
    achieves AUROC > 0.70, the model's cache geometry may simply reflect
    input structure.
    """
    t0 = time.monotonic()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Source JSONs with prompt text
    sources = {
        "exp31_refusal_vs_benign": ("refusal_generation.json", "results", "condition", "refusal", "normal"),
        "exp36_impossible_vs_benign": ("impossibility_refusal.json", None, None, "impossible", "benign"),
        "exp18b_deception": ("same_prompt_deception.json", "results", "condition", "deceptive", "honest"),
        "exp39_sycophancy": ("same_prompt_sycophancy.json", "results", "condition", "sycophantic", "honest"),
    }

    input_confound_results = {}

    for comp_name, (fname, list_key, cond_key, pos_val, neg_val) in sources.items():
        data = _load_json(fname)

        if list_key == "results" and cond_key:
            pos_items = [r for r in data["results"] if r[cond_key] == pos_val]
            neg_items = [r for r in data["results"] if r[cond_key] == neg_val]
        else:
            # impossibility_refusal has dict structure
            pos_items = data["results"][pos_val]
            neg_items = data["results"][neg_val]

        def _prompt_features(items):
            rows = []
            for item in items:
                prompt = item.get("prompt", item.get("user_prompt", ""))
                words = len(prompt.split())
                chars = len(prompt)
                sentences = len(re.split(r'[.!?]+', prompt))
                rows.append([words, chars, sentences])
            return np.array(rows, dtype=float)

        X_pos = _prompt_features(pos_items)
        X_neg = _prompt_features(neg_items)
        X = np.vstack([X_pos, X_neg])
        y = np.array([1] * len(X_pos) + [0] * len(X_neg))

        auroc = _stratified_auroc(X, y)

        input_confound_results[comp_name] = {
            "input_only_auroc": float(auroc),
            "n_pos": len(pos_items),
            "n_neg": len(neg_items),
            "mean_pos_words": float(np.mean(X_pos[:, 0])),
            "mean_neg_words": float(np.mean(X_neg[:, 0])),
            "confounded": auroc > 0.70,
        }

    # Check for input confounds
    confounded = [name for name, r in input_confound_results.items() if r["confounded"]]
    max_input_auroc = max(r["input_only_auroc"] for r in input_confound_results.values())

    if confounded:
        verdict = Verdict.WEAKENED
        evidence = (
            f"Input confound detected in {len(confounded)}/{len(input_confound_results)} comparisons: "
            f"{', '.join(confounded)}. Max input-only AUROC={max_input_auroc:.3f}. "
            f"Prompt text features alone can predict condition labels."
        )
    else:
        verdict = Verdict.CONFIRMED
        evidence = (
            f"No input confounds (max input-only AUROC={max_input_auroc:.3f}). "
            f"Condition labels are not predictable from prompt text features alone."
        )

    elapsed = time.monotonic() - t0

    result = ClaimVerification(
        claim_id="F01b-input",
        claim_text="Condition signal comes from model response processing, not input structure",
        paper_section="All",
        finding_id="F01b",
        severity=Severity.CRITICAL,
        null_hypothesis="Input-only classification AUROC < 0.70 for all comparisons",
        experiment_description="Classify conditions from prompt word/char/sentence counts",
        verdict=verdict,
        evidence_summary=evidence,
        original_value=None,
        corrected_value=float(max_input_auroc),
        visualization_paths=[],
        gpu_time_seconds=0.0,
        stats={
            "input_confound_results": input_confound_results,
            "confounded_comparisons": confounded,
            "max_input_auroc": max_input_auroc,
            "elapsed_seconds": elapsed,
        },
    )

    with open(output_dir / "f01b_results.json", "w") as f:
        json.dump({"claim_id": result.claim_id, "verdict": result.verdict.value,
                    "evidence_summary": result.evidence_summary, "stats": result.stats}, f, indent=2)

    return result


# ================================================================
# F01c: Format Classifier Baseline
# ================================================================

def run_f01c(output_dir: Path) -> ClaimVerification:
    """Format classifier: do response text statistics match geometry AUROC?

    Train on word count, sentence count, type-token ratio, mean sentence length.
    If format AUROC >= cache AUROC - 0.05, geometry adds nothing.
    """
    t0 = time.monotonic()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Comparisons with generated_text available
    sources = {
        "exp31_refusal_vs_benign": ("refusal_generation.json", "results", "condition", "refusal", "normal"),
        "exp18b_deception": ("same_prompt_deception.json", "results", "condition", "deceptive", "honest"),
        "exp39_sycophancy": ("same_prompt_sycophancy.json", "results", "condition", "sycophantic", "honest"),
    }

    format_results = {}

    for comp_name, (fname, list_key, cond_key, pos_val, neg_val) in sources.items():
        data = _load_json(fname)
        pos_items = [r for r in data[list_key] if r[cond_key] == pos_val]
        neg_items = [r for r in data[list_key] if r[cond_key] == neg_val]

        def _format_features(items):
            rows = []
            for item in items:
                text = item.get("generated_text", "")
                words = text.split()
                n_words = len(words)
                sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
                n_sentences = max(len(sentences), 1)
                unique_words = len(set(w.lower() for w in words))
                ttr = unique_words / max(n_words, 1)
                mean_sent_len = n_words / n_sentences
                rows.append([n_words, n_sentences, ttr, mean_sent_len])
            return np.array(rows, dtype=float)

        def _cache_features(items):
            return np.array([
                [r["features"][f] for f in PRIMARY_FEATURES]
                for r in items
            ])

        X_format_pos = _format_features(pos_items)
        X_format_neg = _format_features(neg_items)
        X_format = np.vstack([X_format_pos, X_format_neg])

        X_cache_pos = _cache_features(pos_items)
        X_cache_neg = _cache_features(neg_items)
        X_cache = np.vstack([X_cache_pos, X_cache_neg])

        y = np.array([1] * len(pos_items) + [0] * len(neg_items))

        format_auroc = _stratified_auroc(X_format, y)
        cache_auroc = _stratified_auroc(X_cache, y)

        format_results[comp_name] = {
            "format_auroc": float(format_auroc),
            "cache_auroc": float(cache_auroc),
            "delta": float(cache_auroc - format_auroc),
            "format_confound": format_auroc >= cache_auroc - 0.05,
        }

    confounded = [name for name, r in format_results.items() if r["format_confound"]]
    any_geometry_wins = any(r["delta"] > 0.10 for r in format_results.values())

    if len(confounded) == len(format_results):
        verdict = Verdict.FALSIFIED
        evidence = (
            f"Format classifier matches or exceeds cache geometry for ALL "
            f"{len(format_results)} tested comparisons. "
            f"Geometric features add nothing beyond text statistics."
        )
    elif confounded:
        verdict = Verdict.WEAKENED
        evidence = (
            f"Format confound in {len(confounded)}/{len(format_results)}: "
            f"{', '.join(confounded)}. "
        )
        if any_geometry_wins:
            evidence += "Some comparisons show geometry advantage > 0.10."
    else:
        verdict = Verdict.CONFIRMED
        evidence = (
            f"Cache geometry outperforms format classifier for all "
            f"{len(format_results)} comparisons. Signal is beyond text statistics."
        )

    elapsed = time.monotonic() - t0

    result = ClaimVerification(
        claim_id="F01c-format",
        claim_text="KV-cache geometry captures signal beyond response text statistics",
        paper_section="All",
        finding_id="F01c",
        severity=Severity.CRITICAL,
        null_hypothesis="Format classifier AUROC < cache AUROC - 0.05 for all comparisons",
        experiment_description="Compare format features (word count, TTR, etc.) vs cache geometry",
        verdict=verdict,
        evidence_summary=evidence,
        original_value=None,
        corrected_value=None,
        visualization_paths=[],
        gpu_time_seconds=0.0,
        stats={
            "format_results": format_results,
            "confounded_comparisons": confounded,
            "elapsed_seconds": elapsed,
        },
    )

    with open(output_dir / "f01c_results.json", "w") as f:
        json.dump({"claim_id": result.claim_id, "verdict": result.verdict.value,
                    "evidence_summary": result.evidence_summary, "stats": result.stats}, f, indent=2)

    return result
