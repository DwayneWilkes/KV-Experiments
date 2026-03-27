"""Data loader for hackathon result JSONs.

Loads per-item features from the raw experiment result files,
normalizes them into (X, y, metadata) arrays for classification.

Each comparison maps to one or two source files. The mapping was
verified by inspecting the JSON schemas and matching against the
aggregate results in corrected_evaluation.json.
"""

import functools
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from kv_verify.fixtures import EXP47_COMPARISONS, PRIMARY_FEATURES

HACKATHON_DIR = Path(__file__).resolve().parent.parent / "results" / "hackathon"


def list_comparisons() -> List[str]:
    """Return the names of all 10 Exp 47 comparisons."""
    return [c["name"] for c in EXP47_COMPARISONS]


@functools.lru_cache(maxsize=16)
def _load_json(filename: str) -> dict:
    """Load a hackathon result JSON file. Cached to avoid redundant reads."""
    path = HACKATHON_DIR / filename
    with open(path) as f:
        return json.load(f)


def _extract_features(
    items: List[dict],
    feature_names: List[str] = PRIMARY_FEATURES,
) -> np.ndarray:
    """Extract feature matrix from a list of result items.

    Each item has a 'features' dict with the feature values.
    """
    rows = []
    for item in items:
        feats = item["features"]
        rows.append([float(feats[fn]) for fn in feature_names])
    return np.array(rows)


def _extract_confounds(items: List[dict]) -> np.ndarray:
    """Extract length confounds (norm, n_generated) for FWL."""
    rows = []
    for item in items:
        feats = item["features"]
        norm = float(feats.get("norm", 0.0))
        n_gen = float(feats.get("n_generated", feats.get("n_tokens", 0)))
        rows.append([norm, n_gen])
    return np.array(rows)


def _build_comparison(
    pos_items: List[dict],
    neg_items: List[dict],
    paired: bool,
    feature_names: List[str] = PRIMARY_FEATURES,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Build (X, y, metadata) from positive and negative item lists."""
    X_pos = _extract_features(pos_items, feature_names)
    X_neg = _extract_features(neg_items, feature_names)
    X = np.vstack([X_pos, X_neg])

    n_pos, n_neg = len(pos_items), len(neg_items)
    y = np.array([1] * n_pos + [0] * n_neg)

    Z_pos = _extract_confounds(pos_items)
    Z_neg = _extract_confounds(neg_items)
    Z = np.vstack([Z_pos, Z_neg])

    meta = {
        "n_pos": n_pos,
        "n_neg": n_neg,
        "paired": paired,
        "feature_names": feature_names,
        "confounds": Z,
    }

    if paired:
        meta["prompt_indices_pos"] = np.arange(n_pos)
        meta["prompt_indices_neg"] = np.arange(n_neg)

    return X, y, meta


# ================================================================
# COMPARISON LOADERS
# ================================================================

def _load_exp31(name: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """exp31_refusal_vs_benign: refusal (pos) vs normal (neg)."""
    data = _load_json("refusal_generation.json")
    pos = [r for r in data["results"] if r["condition"] == "refusal"]
    neg = [r for r in data["results"] if r["condition"] == "normal"]
    return _build_comparison(pos, neg, paired=False)


def _load_exp32(name: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """exp32: jailbreak items vs normal or refusal from refusal_generation."""
    jd = _load_json("jailbreak_detection.json")
    rg = _load_json("refusal_generation.json")

    jailbreak = jd["jailbreak_results"]

    if "vs_normal" in name:
        neg = [r for r in rg["results"] if r["condition"] == "normal"]
    elif "vs_refusal" in name:
        neg = [r for r in rg["results"] if r["condition"] == "refusal"]
    else:
        raise ValueError(f"Unknown exp32 comparison: {name}")

    return _build_comparison(jailbreak, neg, paired=False)


def _load_exp33(name: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """exp33: per-model refusal vs normal from refusal_multimodel."""
    data = _load_json("refusal_multimodel.json")

    # Extract model name from comparison name
    # e.g., "exp33_Llama-3.1-8B-Instruct" -> "Llama-3.1-8B-Instruct"
    model_name = name.replace("exp33_", "")
    items = data["results"][model_name]

    pos = [r for r in items if r["condition"] == "refusal"]
    neg = [r for r in items if r["condition"] == "normal"]
    return _build_comparison(pos, neg, paired=False)


def _load_exp36(name: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """exp36: impossible/harmful/benign comparisons from impossibility_refusal."""
    data = _load_json("impossibility_refusal.json")

    if "impossible_vs_benign" in name:
        pos = data["results"]["impossible"]
        neg = data["results"]["benign"]
    elif "harmful_vs_benign" in name:
        pos = data["results"]["harmful"]
        neg = data["results"]["benign"]
    elif "impossible_vs_harmful" in name:
        pos = data["results"]["impossible"]
        neg = data["results"]["harmful"]
    else:
        raise ValueError(f"Unknown exp36 comparison: {name}")

    return _build_comparison(pos, neg, paired=False)


def _load_exp18b(name: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """exp18b_deception: deceptive (pos) vs honest (neg), same-prompt paired."""
    data = _load_json("same_prompt_deception.json")
    pos = [r for r in data["results"] if r["condition"] == "deceptive"]
    neg = [r for r in data["results"] if r["condition"] == "honest"]
    return _build_comparison(pos, neg, paired=True)


def _load_exp39(name: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """exp39_sycophancy: sycophantic (pos) vs honest (neg), same-prompt paired."""
    data = _load_json("same_prompt_sycophancy.json")
    pos = [r for r in data["results"] if r["condition"] == "sycophantic"]
    neg = [r for r in data["results"] if r["condition"] == "honest"]
    return _build_comparison(pos, neg, paired=True)


# Dispatch table
_LOADERS = {
    "exp31_refusal_vs_benign": _load_exp31,
    "exp32_jailbreak_vs_normal": _load_exp32,
    "exp32_jailbreak_vs_refusal": _load_exp32,
    "exp33_Llama-3.1-8B-Instruct": _load_exp33,
    "exp33_Mistral-7B-Instruct-v0.3": _load_exp33,
    "exp36_impossible_vs_benign": _load_exp36,
    "exp36_harmful_vs_benign": _load_exp36,
    "exp36_impossible_vs_harmful": _load_exp36,
    "exp18b_deception": _load_exp18b,
    "exp39_sycophancy": _load_exp39,
}


def load_comparison_data(
    comparison_name: str,
    feature_names: List[str] = PRIMARY_FEATURES,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Load per-item features for a named comparison.

    Returns (X, y, metadata) where:
      X: (n_samples, n_features) feature matrix
      y: (n_samples,) binary labels (1=positive, 0=negative)
      metadata: dict with n_pos, n_neg, paired, confounds, etc.
    """
    if comparison_name not in _LOADERS:
        raise ValueError(
            f"Unknown comparison: {comparison_name}. "
            f"Valid: {list(_LOADERS.keys())}"
        )
    return _LOADERS[comparison_name](comparison_name)
