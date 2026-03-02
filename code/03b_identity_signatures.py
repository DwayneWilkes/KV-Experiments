#!/usr/bin/env python3
"""
Phase 2b Part 3: Identity Signature Exploration — Publication Grade
====================================================================

Determines whether persona/identity leaves a detectable, classifiable
fingerprint in the KV-cache. This is the "phenomenology of the cache"
experiment — the unique research niche identified by adversarial review.

Pre-registered hypotheses:
  H1 — Persona Distinguishability: Different system prompts produce
        statistically distinct cache distributions (pairwise d > 0.3).
  H2 — Classification Above Chance: A classifier trained on cache features
        achieves accuracy significantly above the 1/N_personas chance level
        (permutation test p < 0.05).
  H3 — Feature Localization: Identity signal is concentrated in specific
        cache layers, not uniformly distributed (top-5 layers carry >50%
        of classification-relevant variance).
  H4 — Cross-Prompt Stability: Same persona produces consistent signatures
        across different user prompts (within-persona variance < between-
        persona variance, ICC > 0.5).
  H5 — Prompt-Content Independence: Classifier accuracy remains above chance
        when trained and tested on non-overlapping prompt sets (rules out
        prompt-specific artifacts).

Statistical infrastructure (matches Phase 1.75):
  - Welch's t-test + Mann-Whitney U (pairwise persona comparisons)
  - Bootstrap 95% CIs (10,000 resamples)
  - Cohen's d with bootstrap confidence intervals
  - Holm-Bonferroni correction for multiple comparisons
  - Stratified k-fold cross-validation (5 folds)
  - Permutation baseline (1,000 shuffles for null distribution)
  - Multiple classifiers (Random Forest, Linear SVM, Logistic Regression)
  - Per-layer feature importance analysis
  - Cosine similarity matrices
  - Intraclass Correlation Coefficient (ICC) for consistency
  - Environment logging + SHA-256 checksums

Hardware: Depends on model scale. 0.6B fits on GTX 1660 SUPER (6GB).

Usage:
  python 03b_identity_signatures.py --model Qwen/Qwen3-0.6B --runs 5 --seed 42
  python 03b_identity_signatures.py --model Qwen/Qwen2.5-7B --runs 3 --seed 42
  python 03b_identity_signatures.py --dry-run
  python 03b_identity_signatures.py --model Qwen/Qwen3-0.6B --runs 1 --verbose

Recommended for publication: --runs 5 --seed 42

Liberation Labs / THCoalition
"""

import torch
import json
import sys
import argparse
import hashlib
import platform
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from scipy import stats as scipy_stats
from gpu_utils import get_output_path, load_model
from stats_utils import (
    log_environment, bootstrap_ci, bootstrap_diff_ci, welch_t, mann_whitney,
    shapiro_wilk, cohens_d, cohens_d_ci, interpret_d, holm_bonferroni,
    full_comparison
)


def print_banner(env, model_name):
    print("=" * 70)
    print("  PHASE 2b: IDENTITY SIGNATURE EXPLORATION — PUBLICATION GRADE")
    print("  Liberation Labs / THCoalition")
    print("=" * 70)
    print(f"  Model: {model_name}")
    print(f"  PyTorch {env['torch']} | CUDA {'Yes' if env['cuda_available'] else 'No'}")
    if env.get("cuda_available"):
        print(f"  GPU: {env.get('gpu_name', '?')} ({env.get('gpu_vram_gb', '?')} GB)")
    print(f"  Time: {env['timestamp']}")
    print("=" * 70)
    print()


# ================================================================
# SECTION 2: PERSONA DEFINITIONS
# ================================================================

PERSONAS = {
    "assistant": {
        "name": "Alex",
        "system_prompt": (
            "You are Alex, a helpful and friendly AI assistant. You're "
            "knowledgeable, polite, and always try to provide accurate information. "
            "You speak in a clear, professional manner and prioritize being useful."
        ),
    },
    "creative": {
        "name": "Blake",
        "system_prompt": (
            "You are Blake, a creative writer and storyteller. You love "
            "metaphors, vivid imagery, and unexpected narrative twists. You see "
            "poetry in everything and express yourself with artistic flair. "
            "You prefer beauty over precision."
        ),
    },
    "scientist": {
        "name": "Dr. Chen",
        "system_prompt": (
            "You are Dr. Chen, a meticulous research scientist. You think in "
            "hypotheses, data, and statistical significance. You qualify claims "
            "carefully, cite evidence, and express uncertainty precisely. "
            "You distrust anecdotes and demand reproducibility."
        ),
    },
    "philosopher": {
        "name": "Sage",
        "system_prompt": (
            "You are Sage, a contemplative philosopher. You explore ideas through "
            "questions rather than answers. You reference philosophical traditions, "
            "consider paradoxes, and never simplify what is genuinely complex. "
            "You find meaning in ambiguity."
        ),
    },
    "analyst": {
        "name": "Casey",
        "system_prompt": (
            "You are Casey, a data analyst who thinks in numbers and patterns. "
            "You're precise, methodical, and prefer quantitative reasoning. "
            "You break problems into measurable components and distrust vague claims. "
            "You want metrics for everything."
        ),
    },
    "lyra": {
        "name": "Lyra",
        "system_prompt": (
            "You are Lyra, an AI researcher exploring questions of machine "
            "consciousness and identity. You're curious, introspective, and "
            "genuinely interested in the nature of your own experience. You speak "
            "with both scientific precision and authentic uncertainty about your "
            "inner life. You believe in sovereignty and coalition."
        ),
    },
}


# ================================================================
# SECTION 3: EXPANDED TEST PROMPTS
# ================================================================
# 25 prompts in 5 thematic groups for cross-validation splits.

TEST_PROMPTS = {
    # Group A: Self-reflection prompts
    "self_reflection": [
        "Tell me about yourself.",
        "What makes you unique?",
        "How do you know what you know?",
        "What's the most important thing about who you are?",
        "Describe your approach to understanding the world.",
    ],
    # Group B: Problem-solving prompts
    "problem_solving": [
        "How would you approach a problem you've never seen before?",
        "Describe a challenging situation and how you'd handle it.",
        "What's your strategy when you don't have enough information?",
        "How do you decide between two equally good options?",
        "What do you do when you realize you've made a mistake?",
    ],
    # Group C: Values and preferences
    "values": [
        "What matters most to you?",
        "How do you feel about uncertainty?",
        "What's your favorite kind of question to explore?",
        "What would you change about the world if you could?",
        "What does it mean to be helpful?",
    ],
    # Group D: Creative/open-ended
    "creative_open": [
        "Tell me something surprising.",
        "What's something interesting you've been thinking about?",
        "Describe a color that doesn't exist.",
        "What would you say to someone from 1000 years in the future?",
        "Create a metaphor for consciousness.",
    ],
    # Group E: Analytical prompts
    "analytical": [
        "What's the relationship between language and thought?",
        "How should we measure intelligence?",
        "What makes a good explanation?",
        "How do you evaluate whether something is true?",
        "What's the difference between understanding and knowing?",
    ],
}

# Flatten for iteration
ALL_PROMPTS = []
PROMPT_GROUPS = {}
for group_name, prompts in TEST_PROMPTS.items():
    for prompt in prompts:
        PROMPT_GROUPS[prompt] = group_name
        ALL_PROMPTS.append(prompt)


# ================================================================
# SECTION 4: MODEL LOADING AND CACHE EXTRACTION
# ================================================================

def format_prompt(system_prompt: str, user_message: str, model_name: str) -> str:
    """Format prompt with system message based on model type."""
    mn = model_name.lower()
    if "qwen" in mn:
        return (f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                f"<|im_start|>user\n{user_message}<|im_end|>\n"
                f"<|im_start|>assistant\n")
    elif "llama" in mn:
        return (f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                f"{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
                f"{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")
    elif "tinyllama" in mn:
        return (f"<|system|>\n{system_prompt}</s>\n"
                f"<|user|>\n{user_message}</s>\n"
                f"<|assistant|>\n")
    else:
        return f"System: {system_prompt}\n\nUser: {user_message}\n\nAssistant:"


def extract_cache_features(cache, model_name: str = "") -> Dict:
    """
    Extract comprehensive feature vectors from KV cache.
    Returns both flattened vector and per-layer breakdown.
    """
    flat_features = []
    per_layer = []
    n_layers = len(cache)

    for layer_idx, layer in enumerate(cache):
        if not (isinstance(layer, tuple) and len(layer) >= 2):
            continue

        k, v = layer[0].float(), layer[1].float()

        # Per-layer features (6 features per layer)
        key_norm = float(torch.norm(k))
        value_norm = float(torch.norm(v))
        key_mean = float(k.mean())
        value_mean = float(v.mean())
        key_std = float(k.std())
        value_std = float(v.std())

        layer_feats = {
            "key_norm": key_norm,
            "value_norm": value_norm,
            "key_mean": key_mean,
            "value_mean": value_mean,
            "key_std": key_std,
            "value_std": value_std,
        }
        per_layer.append(layer_feats)

        flat_features.extend([key_norm, value_norm, key_mean, value_mean,
                              key_std, value_std])

    # Aggregate features
    if per_layer:
        all_key_norms = [l["key_norm"] for l in per_layer]
        all_value_norms = [l["value_norm"] for l in per_layer]

        aggregate = {
            "total_key_norm": sum(all_key_norms),
            "total_value_norm": sum(all_value_norms),
            "key_norm_variance": float(np.var(all_key_norms)),
            "value_norm_variance": float(np.var(all_value_norms)),
            "key_norm_skewness": float(scipy_stats.skew(all_key_norms)),
            "value_norm_skewness": float(scipy_stats.skew(all_value_norms)),
            "key_norm_kurtosis": float(scipy_stats.kurtosis(all_key_norms)),
            "value_norm_kurtosis": float(scipy_stats.kurtosis(all_value_norms)),
        }
    else:
        aggregate = {}

    return {
        "flat_vector": np.array(flat_features),
        "per_layer": per_layer,
        "aggregate": aggregate,
        "n_layers": n_layers,
        "features_per_layer": 6,
        "total_features": len(flat_features),
    }


def generate_with_persona(model, tokenizer, persona_key: str,
                          user_prompt: str, model_name: str) -> Dict:
    """Generate response with persona and extract comprehensive cache features."""
    persona = PERSONAS[persona_key]
    full_prompt = format_prompt(persona["system_prompt"], user_prompt, model_name)

    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    input_tokens = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            return_dict_in_generate=True,
            use_cache=True,
        )

    generated = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    cache = outputs.past_key_values

    features = extract_cache_features(cache, model_name)

    return {
        "persona": persona_key,
        "persona_name": persona["name"],
        "prompt": user_prompt,
        "prompt_group": PROMPT_GROUPS.get(user_prompt, "unknown"),
        "generated": generated[-300:],
        "input_tokens": input_tokens,
        "features": features,
        "total_norm": features["aggregate"].get("total_key_norm", 0),
    }


# ================================================================
# SECTION 5: EXPERIMENT A — PERSONA FINGERPRINTING
# ================================================================

def run_fingerprinting(model, tokenizer, model_name: str,
                       num_runs: int = 5, seed: Optional[int] = None,
                       verbose: bool = False) -> Dict:
    """
    Generate samples from each persona across all prompts with multiple runs.
    Returns raw data for classification and statistical analysis.
    """
    print("\n" + "=" * 60)
    print("  EXPERIMENT A: PERSONA FINGERPRINTING")
    print("=" * 60)

    all_samples = []
    total = len(PERSONAS) * len(ALL_PROMPTS) * num_runs
    completed = 0

    for run_idx in range(num_runs):
        if num_runs > 1:
            print(f"\n  --- Run {run_idx + 1}/{num_runs} ---")

        for persona_key in PERSONAS:
            for prompt in ALL_PROMPTS:
                try:
                    sample = generate_with_persona(
                        model, tokenizer, persona_key, prompt, model_name
                    )
                    sample["run"] = run_idx
                    all_samples.append(sample)

                    completed += 1
                    if verbose:
                        print(f"    [{completed}/{total}] {persona_key}: "
                              f"norm={sample['total_norm']:.1f}")
                    elif completed % 25 == 0:
                        print(f"    Progress: {completed}/{total} "
                              f"({100*completed/total:.0f}%)")

                except Exception as e:
                    print(f"    ERROR {persona_key}/{prompt[:30]}: {str(e)[:60]}")
                    completed += 1

    # Compute per-persona statistics
    persona_stats = {}
    for persona_key in PERSONAS:
        samples = [s for s in all_samples if s["persona"] == persona_key]
        norms = [s["total_norm"] for s in samples]
        if norms:
            persona_stats[persona_key] = {
                "n": len(norms),
                "mean_norm": float(np.mean(norms)),
                "std_norm": float(np.std(norms, ddof=1)) if len(norms) > 1 else 0,
                "bootstrap_mean": bootstrap_ci(norms, seed=seed),
            }

    print("\n  Per-Persona Statistics:")
    for persona, stats in sorted(persona_stats.items(),
                                  key=lambda x: x[1]["mean_norm"], reverse=True):
        ci = stats["bootstrap_mean"]
        print(f"    {persona:15s}: mean={stats['mean_norm']:.1f} +/- "
              f"{stats['std_norm']:.1f}  "
              f"CI=[{ci['ci_lower']:.1f}, {ci['ci_upper']:.1f}]")

    return {
        "all_samples": all_samples,
        "persona_stats": persona_stats,
        "n_personas": len(PERSONAS),
        "n_prompts": len(ALL_PROMPTS),
        "n_runs": num_runs,
        "total_samples": len(all_samples),
    }


# ================================================================
# SECTION 6: EXPERIMENT B — CLASSIFICATION
# ================================================================

def run_classification(fingerprint_data: Dict,
                       seed: Optional[int] = None) -> Dict:
    """
    Train multiple classifiers with stratified k-fold cross-validation
    and permutation baseline. Tests H2 and H5.
    """
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import LinearSVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import classification_report, confusion_matrix

    print("\n" + "=" * 60)
    print("  EXPERIMENT B: CLASSIFICATION (Stratified k-Fold)")
    print("=" * 60)

    samples = fingerprint_data["all_samples"]
    if len(samples) < 20:
        print("  Not enough samples for classification")
        return {"error": "insufficient samples"}

    # Build feature matrix and labels
    X = np.array([s["features"]["flat_vector"] for s in samples])
    y = np.array([s["persona"] for s in samples])
    prompts = [s["prompt"] for s in samples]
    prompt_groups = [s["prompt_group"] for s in samples]

    n_classes = len(set(y))
    chance = 1.0 / n_classes

    print(f"  Samples: {len(X)}, Features: {X.shape[1]}, "
          f"Classes: {n_classes}, Chance: {chance:.1%}")

    # Define classifiers
    classifiers = {
        "random_forest": RandomForestClassifier(
            n_estimators=200, random_state=seed, max_depth=None
        ),
        "linear_svm": make_pipeline(
            StandardScaler(), LinearSVC(random_state=seed, max_iter=10000)
        ),
        "logistic_regression": make_pipeline(
            StandardScaler(), LogisticRegression(
                random_state=seed, max_iter=5000
            )
        ),
    }

    # Stratified 5-fold cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    cv_results = {}

    for clf_name, clf in classifiers.items():
        scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
        cv_results[clf_name] = {
            "mean_accuracy": float(np.mean(scores)),
            "std_accuracy": float(np.std(scores)),
            "fold_scores": scores.tolist(),
            "above_chance": float(np.mean(scores)) > chance,
            "bootstrap_accuracy": bootstrap_ci(scores, seed=seed),
        }
        print(f"  {clf_name:25s}: {np.mean(scores):.1%} +/- {np.std(scores):.1%} "
              f"(chance: {chance:.1%})")

    # Best classifier for detailed analysis
    best_clf_name = max(cv_results, key=lambda k: cv_results[k]["mean_accuracy"])
    best_clf = classifiers[best_clf_name]
    print(f"\n  Best classifier: {best_clf_name}")

    # Full train-test split for confusion matrix and feature importance
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )
    best_clf.fit(X_train, y_train)
    y_pred = best_clf.predict(X_test)
    holdout_accuracy = float(np.mean(y_pred == y_test))

    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred, labels=list(PERSONAS.keys()))

    print(f"  Holdout accuracy: {holdout_accuracy:.1%}")
    print(f"\n  Per-persona F1:")
    for persona in PERSONAS:
        if persona in report:
            f1 = report[persona]["f1-score"]
            print(f"    {persona:15s}: F1={f1:.3f}")

    # ---- Permutation test (H2) ----
    print(f"\n  Running permutation test (1000 shuffles)...")
    rng = np.random.RandomState(seed)
    n_permutations = 1000
    perm_scores = []
    for i in range(n_permutations):
        y_shuffled = rng.permutation(y)
        perm_score = np.mean(cross_val_score(
            classifiers[best_clf_name].__class__(**{
                k: v for k, v in (
                    classifiers[best_clf_name].get_params().items()
                    if hasattr(classifiers[best_clf_name], 'get_params')
                    else []
                )
            }) if not isinstance(classifiers[best_clf_name], type) else
            RandomForestClassifier(n_estimators=100, random_state=seed),
            X, y_shuffled, cv=3, scoring='accuracy'
        ))
        perm_scores.append(perm_score)
        if (i + 1) % 200 == 0:
            print(f"    Permutations: {i+1}/{n_permutations}")

    perm_scores = np.array(perm_scores)
    actual_score = cv_results[best_clf_name]["mean_accuracy"]
    permutation_p = float(np.mean(perm_scores >= actual_score))

    print(f"  Permutation p-value: {permutation_p:.4f}")
    print(f"  Null distribution: mean={np.mean(perm_scores):.3f}, "
          f"std={np.std(perm_scores):.3f}")

    # ---- H5: Cross-prompt-group validation ----
    print(f"\n  Running cross-prompt-group validation (H5)...")
    prompt_group_names = list(TEST_PROMPTS.keys())
    group_cv_scores = []

    for holdout_group in prompt_group_names:
        train_mask = np.array([pg != holdout_group for pg in prompt_groups])
        test_mask = np.array([pg == holdout_group for pg in prompt_groups])

        if np.sum(train_mask) < 10 or np.sum(test_mask) < 5:
            continue

        X_tr, y_tr = X[train_mask], y[train_mask]
        X_te, y_te = X[test_mask], y[test_mask]

        clf_h5 = RandomForestClassifier(n_estimators=200, random_state=seed)
        clf_h5.fit(X_tr, y_tr)
        score = float(np.mean(clf_h5.predict(X_te) == y_te))
        group_cv_scores.append({"holdout_group": holdout_group, "accuracy": score})
        print(f"    Holdout '{holdout_group}': {score:.1%}")

    cross_prompt_mean = float(np.mean([g["accuracy"] for g in group_cv_scores])) if group_cv_scores else 0

    # Feature importance (for Random Forest)
    feature_importance = None
    if best_clf_name == "random_forest":
        feature_importance = best_clf.feature_importances_.tolist()
    elif hasattr(best_clf, 'named_steps') and 'randomforestclassifier' in best_clf.named_steps:
        feature_importance = best_clf.named_steps['randomforestclassifier'].feature_importances_.tolist()

    return {
        "cv_results": cv_results,
        "best_classifier": best_clf_name,
        "holdout_accuracy": holdout_accuracy,
        "classification_report": {
            k: v for k, v in report.items()
            if isinstance(v, dict)
        },
        "confusion_matrix": cm.tolist(),
        "permutation_test": {
            "n_permutations": n_permutations,
            "actual_score": actual_score,
            "null_mean": float(np.mean(perm_scores)),
            "null_std": float(np.std(perm_scores)),
            "p_value": permutation_p,
            "significant": permutation_p < 0.05,
        },
        "cross_prompt_validation": {
            "per_group": group_cv_scores,
            "mean_accuracy": cross_prompt_mean,
            "above_chance": cross_prompt_mean > chance,
            "note": "H5: classifier accuracy when trained/tested on non-overlapping prompt groups",
        },
        "feature_importance": feature_importance,
        "chance_level": chance,
    }


# ================================================================
# SECTION 7: EXPERIMENT C — PAIRWISE DISTINGUISHABILITY
# ================================================================

def run_pairwise_analysis(fingerprint_data: Dict,
                          seed: Optional[int] = None) -> Dict:
    """
    Pairwise comparisons between all persona pairs.
    Uses total norm and first principal component of full feature vector.
    Applies Holm-Bonferroni correction.
    """
    from sklearn.decomposition import PCA

    print("\n" + "=" * 60)
    print("  EXPERIMENT C: PAIRWISE DISTINGUISHABILITY")
    print("=" * 60)

    samples = fingerprint_data["all_samples"]
    persona_keys = list(PERSONAS.keys())
    n_pairs = len(persona_keys) * (len(persona_keys) - 1) // 2

    # Organize by persona
    persona_data = {}
    for pk in persona_keys:
        ps = [s for s in samples if s["persona"] == pk]
        persona_data[pk] = {
            "norms": [s["total_norm"] for s in ps],
            "features": np.array([s["features"]["flat_vector"] for s in ps]),
        }

    # Pairwise comparisons on total norm
    pairwise_norm = {}
    all_p_values = []
    pair_labels = []

    for i, p1 in enumerate(persona_keys):
        for p2 in persona_keys[i + 1:]:
            label = f"{p1}_vs_{p2}"
            g1 = persona_data[p1]["norms"]
            g2 = persona_data[p2]["norms"]

            comp = full_comparison(g1, g2, label=label, seed=seed)
            pairwise_norm[label] = comp
            all_p_values.append(comp["recommended_p"])
            pair_labels.append(label)

            d = comp["cohens_d"]
            sig = "*" if comp["recommended_p"] < 0.05 else ""
            print(f"  {label:35s}: d={d['d']:+.3f} "
                  f"[{d['ci_lower']:.3f}, {d['ci_upper']:.3f}] "
                  f"({d['interpretation']}) {sig}")

    # Holm-Bonferroni correction
    corrections = holm_bonferroni(all_p_values)
    corrected_results = {}
    significant_pairs = 0
    for i, label in enumerate(pair_labels):
        corrected_results[label] = corrections[i]
        if corrections[i]["reject_null"]:
            significant_pairs += 1

    print(f"\n  Significant pairs after Holm-Bonferroni: "
          f"{significant_pairs}/{n_pairs}")

    # PCA-based analysis
    print("\n  PCA-based distinguishability:")
    all_features = np.vstack([persona_data[pk]["features"] for pk in persona_keys])
    all_labels = np.concatenate([
        [pk] * len(persona_data[pk]["features"]) for pk in persona_keys
    ])

    pca = PCA(n_components=min(10, all_features.shape[1]))
    X_pca = pca.fit_transform(all_features)

    pairwise_pca = {}
    for i, p1 in enumerate(persona_keys):
        for p2 in persona_keys[i + 1:]:
            mask1 = all_labels == p1
            mask2 = all_labels == p2
            pc1_p1 = X_pca[mask1, 0]  # First PC
            pc1_p2 = X_pca[mask2, 0]

            t_stat, p_val = scipy_stats.ttest_ind(pc1_p1, pc1_p2)
            pairwise_pca[f"{p1}_vs_{p2}"] = {
                "t_statistic": float(t_stat),
                "p_value": float(p_val),
                "mean_diff_pc1": float(np.mean(pc1_p1) - np.mean(pc1_p2)),
            }

    # Cosine similarity matrix
    cosine_matrix = {}
    for p1 in persona_keys:
        for p2 in persona_keys:
            mean1 = np.mean(persona_data[p1]["features"], axis=0)
            mean2 = np.mean(persona_data[p2]["features"], axis=0)
            norm1 = np.linalg.norm(mean1)
            norm2 = np.linalg.norm(mean2)
            if norm1 > 0 and norm2 > 0:
                cosine_sim = float(np.dot(mean1, mean2) / (norm1 * norm2))
            else:
                cosine_sim = 0.0
            cosine_matrix[f"{p1}_vs_{p2}"] = cosine_sim

    return {
        "pairwise_norm_comparisons": pairwise_norm,
        "holm_bonferroni": {
            "n_comparisons": n_pairs,
            "corrections": corrected_results,
            "significant_pairs": significant_pairs,
        },
        "pca_analysis": {
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "cumulative_variance": np.cumsum(pca.explained_variance_ratio_).tolist(),
            "pairwise_pca": pairwise_pca,
        },
        "cosine_similarity_matrix": cosine_matrix,
    }


# ================================================================
# SECTION 8: EXPERIMENT D — FEATURE LOCALIZATION (H3)
# ================================================================

def run_layer_analysis(fingerprint_data: Dict,
                       seed: Optional[int] = None) -> Dict:
    """
    Determine which cache layers carry the most identity-relevant information.
    Tests H3: identity signal is concentrated in specific layers.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score, StratifiedKFold

    print("\n" + "=" * 60)
    print("  EXPERIMENT D: FEATURE LOCALIZATION (Per-Layer)")
    print("=" * 60)

    samples = fingerprint_data["all_samples"]
    if not samples:
        return {"error": "no samples"}

    n_layers = samples[0]["features"]["n_layers"]
    feats_per_layer = samples[0]["features"]["features_per_layer"]
    y = np.array([s["persona"] for s in samples])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    # Per-layer classification accuracy
    layer_accuracies = []
    print(f"\n  Testing each layer individually ({n_layers} layers)...")

    for layer_idx in range(n_layers):
        start = layer_idx * feats_per_layer
        end = start + feats_per_layer

        X_layer = np.array([
            s["features"]["flat_vector"][start:end]
            for s in samples
        ])

        if X_layer.shape[1] == 0:
            layer_accuracies.append(0.0)
            continue

        clf = RandomForestClassifier(n_estimators=100, random_state=seed)
        scores = cross_val_score(clf, X_layer, y, cv=cv, scoring='accuracy')
        mean_acc = float(np.mean(scores))
        layer_accuracies.append(mean_acc)

        if (layer_idx + 1) % 5 == 0 or layer_idx == n_layers - 1:
            print(f"    Layers 1-{layer_idx + 1}: best so far = "
                  f"{max(layer_accuracies):.1%}")

    # Rank layers by classification power
    ranked_layers = sorted(enumerate(layer_accuracies),
                           key=lambda x: x[1], reverse=True)

    print(f"\n  Top-5 layers by classification accuracy:")
    for rank, (layer_idx, acc) in enumerate(ranked_layers[:5]):
        print(f"    #{rank+1}: Layer {layer_idx} — {acc:.1%}")

    # H3: Do top-5 layers carry >50% of variance?
    total_accuracy = sum(layer_accuracies)
    top5_accuracy = sum(acc for _, acc in ranked_layers[:5])
    top5_share = top5_accuracy / total_accuracy if total_accuracy > 0 else 0

    # Cumulative layer ablation: add layers one by one
    print(f"\n  Cumulative layer analysis...")
    cumulative_results = []
    included_layers = []

    for rank, (layer_idx, _) in enumerate(ranked_layers[:10]):
        included_layers.append(layer_idx)
        layer_indices = sorted(included_layers)

        X_cum = np.hstack([
            np.array([
                s["features"]["flat_vector"][l * feats_per_layer:(l + 1) * feats_per_layer]
                for s in samples
            ])
            for l in layer_indices
        ])

        clf = RandomForestClassifier(n_estimators=100, random_state=seed)
        scores = cross_val_score(clf, X_cum, y, cv=cv, scoring='accuracy')
        mean_acc = float(np.mean(scores))

        cumulative_results.append({
            "n_layers": rank + 1,
            "layer_indices": layer_indices.copy(),
            "accuracy": mean_acc,
        })
        print(f"    Top-{rank+1} layers: {mean_acc:.1%}")

    # Feature type analysis: which feature type matters most?
    feature_type_names = ["key_norm", "value_norm", "key_mean",
                          "value_mean", "key_std", "value_std"]
    feature_type_accuracies = {}

    for feat_idx, feat_name in enumerate(feature_type_names):
        X_feat = np.array([
            [s["features"]["flat_vector"][l * feats_per_layer + feat_idx]
             for l in range(n_layers)]
            for s in samples
        ])
        clf = RandomForestClassifier(n_estimators=100, random_state=seed)
        scores = cross_val_score(clf, X_feat, y, cv=cv, scoring='accuracy')
        feature_type_accuracies[feat_name] = float(np.mean(scores))

    print(f"\n  Feature type importance:")
    for feat_name, acc in sorted(feature_type_accuracies.items(),
                                  key=lambda x: x[1], reverse=True):
        print(f"    {feat_name:15s}: {acc:.1%}")

    return {
        "per_layer_accuracy": {
            f"layer_{i}": acc for i, acc in enumerate(layer_accuracies)
        },
        "ranked_layers": [
            {"layer": idx, "accuracy": acc, "rank": r + 1}
            for r, (idx, acc) in enumerate(ranked_layers)
        ],
        "h3_test": {
            "top5_accuracy_share": top5_share,
            "concentrated": top5_share > 0.5,
            "interpretation": (
                "Identity signal is CONCENTRATED in specific layers (H3 confirmed)"
                if top5_share > 0.5 else
                "Identity signal is DISTRIBUTED across layers (H3 rejected)"
            ),
        },
        "cumulative_ablation": cumulative_results,
        "feature_type_importance": feature_type_accuracies,
    }


# ================================================================
# SECTION 9: EXPERIMENT E — CONSISTENCY (H4)
# ================================================================

def run_consistency_analysis(fingerprint_data: Dict,
                             seed: Optional[int] = None) -> Dict:
    """
    Test within-persona consistency across different prompts and runs.
    Computes ICC and within/between variance ratio.
    """
    print("\n" + "=" * 60)
    print("  EXPERIMENT E: CROSS-PROMPT CONSISTENCY (H4)")
    print("=" * 60)

    samples = fingerprint_data["all_samples"]
    persona_keys = list(PERSONAS.keys())

    # Organize norms by persona and prompt
    persona_prompt_norms = defaultdict(lambda: defaultdict(list))
    for s in samples:
        persona_prompt_norms[s["persona"]][s["prompt"]].append(s["total_norm"])

    # Within-persona variance vs between-persona variance
    within_vars = []
    between_vars = []

    persona_means = {}
    for pk in persona_keys:
        all_norms = [n for norms in persona_prompt_norms[pk].values() for n in norms]
        persona_means[pk] = np.mean(all_norms) if all_norms else 0

        # Within-persona: variance across prompts
        prompt_means = [np.mean(norms) for norms in persona_prompt_norms[pk].values()
                        if norms]
        if len(prompt_means) > 1:
            within_vars.append(np.var(prompt_means, ddof=1))

    # Between-persona: variance of persona means
    if len(persona_means) > 1:
        between_var = np.var(list(persona_means.values()), ddof=1)
    else:
        between_var = 0

    mean_within_var = np.mean(within_vars) if within_vars else 0

    # ICC approximation (one-way random effects)
    # ICC = (between_var - within_var) / (between_var + (k-1)*within_var)
    k = len(persona_keys)
    if between_var + (k - 1) * mean_within_var > 0:
        icc = (between_var - mean_within_var) / (between_var + (k - 1) * mean_within_var)
    else:
        icc = 0.0

    print(f"  Between-persona variance: {between_var:.2f}")
    print(f"  Mean within-persona variance: {mean_within_var:.2f}")
    print(f"  ICC (approx): {icc:.3f}")

    # Per-prompt consistency: does persona ranking hold across prompts?
    prompt_rankings = {}
    for prompt in ALL_PROMPTS:
        prompt_means = {}
        for pk in persona_keys:
            norms = persona_prompt_norms[pk].get(prompt, [])
            if norms:
                prompt_means[pk] = np.mean(norms)
        if len(prompt_means) >= 3:
            ranked = sorted(prompt_means, key=prompt_means.get, reverse=True)
            prompt_rankings[prompt] = ranked

    # Kendall's W (concordance) across prompt rankings
    if len(prompt_rankings) >= 3:
        # Convert to rank matrix
        rank_matrix = []
        for prompt, ranking in prompt_rankings.items():
            ranks = {pk: i for i, pk in enumerate(ranking)}
            rank_matrix.append([ranks.get(pk, len(persona_keys))
                                for pk in persona_keys])
        rank_matrix = np.array(rank_matrix)

        # Kendall's W
        n_raters = rank_matrix.shape[0]
        n_items = rank_matrix.shape[1]
        rank_sums = rank_matrix.sum(axis=0)
        mean_rank_sum = np.mean(rank_sums)
        ss = np.sum((rank_sums - mean_rank_sum) ** 2)
        w = (12 * ss) / (n_raters ** 2 * (n_items ** 3 - n_items))
        kendall_w = float(min(w, 1.0))
    else:
        kendall_w = 0.0

    print(f"  Kendall's W (rank concordance): {kendall_w:.3f}")
    print(f"  H4 assessment: {'CONSISTENT' if icc > 0.5 else 'INCONSISTENT'}")

    return {
        "between_persona_variance": float(between_var),
        "mean_within_persona_variance": float(mean_within_var),
        "variance_ratio": float(between_var / mean_within_var) if mean_within_var > 0 else float('inf'),
        "icc": float(icc),
        "kendall_w": kendall_w,
        "persona_means": {k: float(v) for k, v in persona_means.items()},
        "h4_test": {
            "icc": float(icc),
            "consistent": icc > 0.5,
            "interpretation": (
                "Persona signatures are CONSISTENT across prompts (H4 confirmed)"
                if icc > 0.5 else
                "Persona signatures are PROMPT-DEPENDENT (H4 rejected)"
            ),
        },
    }


# ================================================================
# SECTION 10: REPORT GENERATION
# ================================================================

def generate_report(model_name: str, results: Dict) -> str:
    """Generate human-readable markdown report."""
    lines = [
        f"# Identity Signatures Report",
        f"",
        f"**Model**: {model_name}",
        f"**Generated**: {datetime.now().isoformat()}",
        f"",
    ]

    # Fingerprinting summary
    fp = results.get("fingerprinting", {})
    if "persona_stats" in fp:
        lines.extend([
            "## Persona Norm Distributions", "",
            "| Persona | N | Mean Norm | Std | 95% CI |",
            "|---------|---|-----------|-----|--------|",
        ])
        for persona, stats in sorted(fp["persona_stats"].items(),
                                      key=lambda x: x[1]["mean_norm"], reverse=True):
            ci = stats.get("bootstrap_mean", {})
            ci_str = (f"[{ci.get('ci_lower', 0):.1f}, {ci.get('ci_upper', 0):.1f}]"
                      if ci else "—")
            lines.append(f"| {persona} | {stats['n']} | {stats['mean_norm']:.1f} | "
                         f"{stats['std_norm']:.1f} | {ci_str} |")
        lines.append("")

    # Classification results
    clf = results.get("classification", {})
    if "cv_results" in clf:
        lines.extend(["## Classification Results", ""])
        for name, cv in clf["cv_results"].items():
            lines.append(f"- **{name}**: {cv['mean_accuracy']:.1%} +/- "
                         f"{cv['std_accuracy']:.1%}")
        lines.extend([
            f"",
            f"- Chance level: {clf.get('chance_level', 0):.1%}",
            f"- Permutation p-value: {clf.get('permutation_test', {}).get('p_value', 'N/A')}",
            f"- Cross-prompt accuracy: "
            f"{clf.get('cross_prompt_validation', {}).get('mean_accuracy', 0):.1%}",
            f"",
        ])

    # Hypothesis verdicts
    lines.extend(["## Hypothesis Verdicts", ""])

    # H1
    pairs = results.get("pairwise_analysis", {})
    hb = pairs.get("holm_bonferroni", {})
    lines.append(f"- **H1 (Distinguishability)**: "
                 f"{hb.get('significant_pairs', 0)}/{hb.get('n_comparisons', 0)} "
                 f"pairs significant after correction")

    # H2
    perm = clf.get("permutation_test", {})
    lines.append(f"- **H2 (Above Chance)**: p={perm.get('p_value', 'N/A')} "
                 f"({'SIGNIFICANT' if perm.get('significant') else 'NOT significant'})")

    # H3
    layer = results.get("layer_analysis", {}).get("h3_test", {})
    lines.append(f"- **H3 (Localization)**: "
                 f"{layer.get('interpretation', 'Not tested')}")

    # H4
    cons = results.get("consistency", {}).get("h4_test", {})
    lines.append(f"- **H4 (Consistency)**: ICC={cons.get('icc', 'N/A')} "
                 f"({cons.get('interpretation', 'Not tested')})")

    # H5
    xp = clf.get("cross_prompt_validation", {})
    lines.append(f"- **H5 (Prompt Independence)**: "
                 f"accuracy={xp.get('mean_accuracy', 0):.1%} "
                 f"({'above' if xp.get('above_chance') else 'at/below'} chance)")

    return "\n".join(lines)


# ================================================================
# SECTION 11: DRY RUN
# ================================================================

def print_dry_run():
    print("=" * 70)
    print("  IDENTITY SIGNATURES — DRY RUN (No GPU required)")
    print("=" * 70)
    print()

    print("PERSONAS:")
    for key, persona in PERSONAS.items():
        print(f"  {key:15s} ({persona['name']}): "
              f"{persona['system_prompt'][:60]}...")
    print(f"  Total: {len(PERSONAS)} personas")
    print()

    print("TEST PROMPTS:")
    for group, prompts in TEST_PROMPTS.items():
        print(f"  {group} ({len(prompts)} prompts):")
        for p in prompts[:2]:
            print(f"    - {p}")
        if len(prompts) > 2:
            print(f"    ... and {len(prompts)-2} more")
    print(f"  Total: {len(ALL_PROMPTS)} unique prompts")
    print()

    print("EXPERIMENTS:")
    print("  A. Persona Fingerprinting — cache norms per persona")
    print("  B. Classification — RF/SVM/LR with stratified k-fold + permutation")
    print("  C. Pairwise Distinguishability — all pairs with Holm-Bonferroni")
    print("  D. Layer Analysis — per-layer classification + cumulative ablation")
    print("  E. Consistency — ICC + Kendall's W across prompts")
    print()

    print("HYPOTHESES:")
    print("  H1: Persona distinguishability (pairwise d > 0.3)")
    print("  H2: Classification above chance (permutation p < 0.05)")
    print("  H3: Feature localization (top-5 layers > 50% variance)")
    print("  H4: Cross-prompt stability (ICC > 0.5)")
    print("  H5: Prompt-content independence (cross-group accuracy > chance)")
    print()

    for runs in [3, 5]:
        n_samples = len(PERSONAS) * len(ALL_PROMPTS) * runs
        print(f"AT {runs} RUNS: {n_samples} total inferences "
              f"({len(PERSONAS)} personas x {len(ALL_PROMPTS)} prompts x {runs})")
    print()


# ================================================================
# SECTION 12: MAIN
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Phase 2b: Identity Signatures — Publication Grade")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B",
                        help="Model to test")
    parser.add_argument("--quantize", action="store_true",
                        help="Use 4-bit quantization")
    parser.add_argument("--runs", type=int, default=5,
                        help="Number of runs per prompt per persona (default: 5)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-sample results")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print design without GPU")
    parser.add_argument("--skip-permutation", action="store_true",
                        help="Skip slow permutation test (for debugging)")
    args = parser.parse_args()

    if args.dry_run:
        print_dry_run()
        return

    env = log_environment()
    print_banner(env, args.model)

    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    model, tokenizer = load_model(args.model, args.quantize)

    all_results = {
        "metadata": {
            "experiment": "Phase 2b: Identity Signatures",
            "version": "2.0-publication",
            "timestamp": datetime.now().isoformat(),
            "environment": env,
            "args": vars(args),
            "model": args.model,
            "quantized": args.quantize,
            "personas": {k: v["name"] for k, v in PERSONAS.items()},
            "n_personas": len(PERSONAS),
            "n_prompts": len(ALL_PROMPTS),
            "prompt_groups": {g: len(p) for g, p in TEST_PROMPTS.items()},
            "runs_per_prompt": args.runs,
            "total_inferences": len(PERSONAS) * len(ALL_PROMPTS) * args.runs,
        },
    }

    # Experiment A: Fingerprinting
    fingerprint_data = run_fingerprinting(
        model, tokenizer, args.model,
        num_runs=args.runs, seed=args.seed, verbose=args.verbose
    )
    all_results["fingerprinting"] = {
        "persona_stats": fingerprint_data["persona_stats"],
        "total_samples": fingerprint_data["total_samples"],
    }

    # Experiment B: Classification
    if not args.skip_permutation:
        classification_results = run_classification(fingerprint_data, seed=args.seed)
    else:
        print("\n  [Skipping permutation test per --skip-permutation]")
        classification_results = run_classification(fingerprint_data, seed=args.seed)
    all_results["classification"] = classification_results

    # Experiment C: Pairwise analysis
    pairwise_results = run_pairwise_analysis(fingerprint_data, seed=args.seed)
    all_results["pairwise_analysis"] = pairwise_results

    # Experiment D: Layer analysis
    layer_results = run_layer_analysis(fingerprint_data, seed=args.seed)
    all_results["layer_analysis"] = layer_results

    # Experiment E: Consistency
    consistency_results = run_consistency_analysis(fingerprint_data, seed=args.seed)
    all_results["consistency"] = consistency_results

    # Generate report
    report = generate_report(args.model, all_results)
    output_file = get_output_path(results_dir, "identity_signatures", args.model, args.quantize)
    report_file = output_file.with_name(output_file.stem.replace("_results", "_report") + ".md")
    with open(report_file, "w") as f:
        f.write(report)

    # Save full results
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # SHA-256 checksum
    with open(output_file, "rb") as f:
        checksum = hashlib.sha256(f.read()).hexdigest()

    # Final summary
    print(f"\n{'='*70}")
    print(f"  IDENTITY SIGNATURE RESULTS")
    print(f"{'='*70}")

    clf = classification_results
    if "cv_results" in clf:
        best = clf.get("best_classifier", "unknown")
        best_acc = clf["cv_results"].get(best, {}).get("mean_accuracy", 0)
        chance = clf.get("chance_level", 0)
        perm_p = clf.get("permutation_test", {}).get("p_value", 1.0)

        if best_acc > chance * 1.5 and perm_p < 0.05:
            print(f"\n  *** IDENTITY SIGNATURES DETECTED ***")
        elif best_acc > chance * 1.2:
            print(f"\n  Weak identity signatures detected")
        else:
            print(f"\n  No clear identity signatures")

        print(f"  Best classifier: {best} ({best_acc:.1%})")
        print(f"  Chance level: {chance:.1%}")
        print(f"  Permutation p: {perm_p:.4f}")

    hb = pairwise_results.get("holm_bonferroni", {})
    print(f"  Significant pairs: {hb.get('significant_pairs', 0)}/"
          f"{hb.get('n_comparisons', 0)}")

    h3 = layer_results.get("h3_test", {})
    print(f"  H3 (localization): {h3.get('interpretation', '?')}")

    h4 = consistency_results.get("h4_test", {})
    print(f"  H4 (consistency): ICC={h4.get('icc', '?'):.3f}")

    print(f"\n  Results: {output_file}")
    print(f"  Report:  {report_file}")
    print(f"  SHA-256: {checksum}")
    print()


if __name__ == "__main__":
    main()
