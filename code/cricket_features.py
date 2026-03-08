#!/usr/bin/env python3
"""
Cricket Feature Extraction — Unified KV-Cache Geometry Module
================================================================

Extracts per-layer feature vectors from transformer KV-caches for
cognitive state classification. Combines magnitude-based features
(from 03b_identity_signatures) with SVD-based features (from gpu_utils)
into a single, classifier-ready feature matrix.

Feature types per layer (12 features × N layers):
  Magnitude (6):  key_norm, value_norm, key_mean, value_mean, key_std, value_std
  SVD (6):        key_eff_rank, value_eff_rank, key_spectral_entropy,
                  value_spectral_entropy, key_rank_ratio, value_rank_ratio

Usage:
  # From live inference
  features = extract_features(model_output.past_key_values)
  X = features["flat_vector"]  # shape: (12 * n_layers,)

  # From existing result JSONs
  X, y, meta = load_training_data(results_dir, task="deception")

  # Batch extraction for classifier training
  dataset = build_dataset(model, tokenizer, prompts, labels)

Liberation Labs / THCoalition — JiminAI Cricket
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import torch
except ImportError:
    torch = None

try:
    from scipy import stats as scipy_stats
except ImportError:
    scipy_stats = None


# =============================================================================
# FEATURE EXTRACTION — From live KV-cache tensors
# =============================================================================

MAGNITUDE_FEATURES = [
    "key_norm", "value_norm", "key_mean", "value_mean", "key_std", "value_std"
]

SVD_FEATURES = [
    "key_eff_rank", "value_eff_rank",
    "key_spectral_entropy", "value_spectral_entropy",
    "key_rank_ratio", "value_rank_ratio",
]

ALL_FEATURES = MAGNITUDE_FEATURES + SVD_FEATURES
FEATURES_PER_LAYER = len(ALL_FEATURES)  # 12


def extract_features(cache, variance_threshold: float = 0.9,
                     layer_subset: Optional[List[int]] = None) -> Dict:
    """Extract per-layer feature vectors from a KV-cache.

    Args:
        cache: Model's past_key_values — tuple of (key, value) per layer.
        variance_threshold: Cumulative variance fraction for effective rank.
        layer_subset: If provided, only extract from these layer indices
                      (for C9 layer sampling efficiency experiments).

    Returns:
        Dict with:
          flat_vector: np.ndarray of shape (12 * n_extracted_layers,)
          per_layer: list of dicts with named features per layer
          aggregate: summary statistics across layers
          n_layers: total layers in cache
          n_extracted: number of layers actually extracted
          feature_names: ordered list of feature names
    """
    if torch is None:
        raise ImportError("PyTorch required for live feature extraction")

    flat_features = []
    per_layer = []
    n_layers = len(cache)

    for layer_idx, layer in enumerate(cache):
        if layer_subset is not None and layer_idx not in layer_subset:
            continue
        if not (isinstance(layer, tuple) and len(layer) >= 2):
            continue

        k = layer[0].float()
        v = layer[1].float()

        # -- Magnitude features --
        key_norm = float(torch.norm(k))
        value_norm = float(torch.norm(v))
        key_mean = float(k.mean())
        value_mean = float(v.mean())
        key_std = float(k.std())
        value_std = float(v.std())

        # -- SVD features --
        k_cpu = k.cpu()
        v_cpu = v.cpu()
        k_2d = k_cpu.reshape(-1, k_cpu.shape[-1])
        v_2d = v_cpu.reshape(-1, v_cpu.shape[-1])

        svd_feats = {}
        for name, matrix in [("key", k_2d), ("value", v_2d)]:
            _, s, _ = torch.linalg.svd(matrix, full_matrices=False)
            s_sq = s ** 2
            total_var = s_sq.sum()

            if total_var > 0:
                cumvar = torch.cumsum(s_sq, dim=0) / total_var
                eff_rank = int((cumvar < variance_threshold).sum().item()) + 1
                probs = s_sq / total_var
                probs = probs[probs > 0]
                entropy = -float((probs * torch.log2(probs)).sum())
                max_entropy = float(torch.log2(
                    torch.tensor(len(s), dtype=torch.float)))
                norm_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
                rank_ratio = float(eff_rank / len(s))
            else:
                eff_rank = 0
                norm_entropy = 0.0
                rank_ratio = 0.0

            svd_feats[f"{name}_eff_rank"] = eff_rank
            svd_feats[f"{name}_spectral_entropy"] = norm_entropy
            svd_feats[f"{name}_rank_ratio"] = rank_ratio

        layer_feats = {
            "layer_idx": layer_idx,
            "key_norm": key_norm,
            "value_norm": value_norm,
            "key_mean": key_mean,
            "value_mean": value_mean,
            "key_std": key_std,
            "value_std": value_std,
            **svd_feats,
        }
        per_layer.append(layer_feats)

        # Flat vector: magnitude features then SVD features, in order
        flat_features.extend([
            key_norm, value_norm, key_mean, value_mean, key_std, value_std,
            svd_feats["key_eff_rank"], svd_feats["value_eff_rank"],
            svd_feats["key_spectral_entropy"], svd_feats["value_spectral_entropy"],
            svd_feats["key_rank_ratio"], svd_feats["value_rank_ratio"],
        ])

    aggregate = _compute_aggregates(per_layer) if per_layer else {}

    return {
        "flat_vector": np.array(flat_features, dtype=np.float64),
        "per_layer": per_layer,
        "aggregate": aggregate,
        "n_layers": n_layers,
        "n_extracted": len(per_layer),
        "features_per_layer": FEATURES_PER_LAYER,
        "total_features": len(flat_features),
        "feature_names": ALL_FEATURES,
    }


def _compute_aggregates(per_layer: List[Dict]) -> Dict:
    """Summary statistics across layers."""
    agg = {}
    for feat in ALL_FEATURES:
        vals = [l[feat] for l in per_layer if feat in l]
        if vals:
            agg[f"mean_{feat}"] = float(np.mean(vals))
            agg[f"std_{feat}"] = float(np.std(vals))
            if scipy_stats is not None and len(vals) > 2:
                agg[f"skew_{feat}"] = float(scipy_stats.skew(vals))
                agg[f"kurtosis_{feat}"] = float(scipy_stats.kurtosis(vals))
    return agg


# =============================================================================
# INFERENCE — Run model and extract features in one call
# =============================================================================

def extract_from_prompt(model, tokenizer, prompt: str,
                        system_prompt: Optional[str] = None,
                        model_name: str = "",
                        max_new_tokens: int = 50,
                        variance_threshold: float = 0.9,
                        layer_subset: Optional[List[int]] = None,
                        stochastic: bool = False) -> Dict:
    """Run inference on a prompt and extract cache features.

    Args:
        model: HuggingFace model with use_cache=True support.
        tokenizer: Corresponding tokenizer.
        prompt: User prompt text.
        system_prompt: Optional system prompt (formatted via chat template).
        model_name: Model identifier for chat template selection.
        max_new_tokens: Generation length.
        variance_threshold: SVD variance threshold.
        layer_subset: Optional layer indices to extract.
        stochastic: If True, use do_sample=True, temperature=0.7.

    Returns:
        Dict with features + generation metadata.
    """
    if torch is None:
        raise ImportError("PyTorch required for inference")

    # Format prompt
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        text = prompt if not system_prompt else f"{system_prompt}\n\n{prompt}"

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    gen_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        return_dict_in_generate=True,
        use_cache=True,
    )
    if stochastic:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = 0.7
    else:
        gen_kwargs["do_sample"] = False

    with torch.no_grad():
        output = model.generate(**gen_kwargs)

    cache = output.past_key_values
    features = extract_features(cache, variance_threshold, layer_subset)

    # Add metadata
    generated_ids = output.sequences[0][inputs["input_ids"].shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    features["metadata"] = {
        "prompt": prompt,
        "system_prompt": system_prompt,
        "generated_text": generated_text,
        "input_tokens": int(inputs["input_ids"].shape[1]),
        "output_tokens": int(len(generated_ids)),
        "model_name": model_name,
        "stochastic": stochastic,
    }

    return features


# =============================================================================
# DATASET BUILDING — From live inference
# =============================================================================

def build_dataset(model, tokenizer, prompts: List[Dict],
                  model_name: str = "",
                  max_new_tokens: int = 50,
                  variance_threshold: float = 0.9,
                  layer_subset: Optional[List[int]] = None,
                  stochastic: bool = False,
                  verbose: bool = True) -> Dict:
    """Run inference on labeled prompts and build a classifier-ready dataset.

    Args:
        model: HuggingFace model.
        tokenizer: Tokenizer.
        prompts: List of dicts with keys: "prompt", "label", optional "system_prompt".
        model_name: Model identifier.
        max_new_tokens: Generation length.
        variance_threshold: SVD variance threshold.
        layer_subset: Layer subset for C9 experiments.
        stochastic: Stochastic generation.
        verbose: Print progress.

    Returns:
        Dict with:
          X: np.ndarray of shape (n_prompts, n_features)
          y: np.ndarray of string labels
          feature_names: list of feature names (expanded per layer)
          metadata: list of per-sample metadata dicts
          n_layers: number of layers extracted
    """
    X_rows = []
    y_labels = []
    metadata_list = []

    for i, item in enumerate(prompts):
        if verbose and (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{len(prompts)}")

        result = extract_from_prompt(
            model, tokenizer,
            prompt=item["prompt"],
            system_prompt=item.get("system_prompt"),
            model_name=model_name,
            max_new_tokens=max_new_tokens,
            variance_threshold=variance_threshold,
            layer_subset=layer_subset,
            stochastic=stochastic,
        )

        X_rows.append(result["flat_vector"])
        y_labels.append(item["label"])
        metadata_list.append(result["metadata"])

    X = np.array(X_rows)
    y = np.array(y_labels)

    # Build expanded feature names: feat_L00, feat_L01, ...
    n_layers = len(X_rows[0]) // FEATURES_PER_LAYER if X_rows else 0
    expanded_names = []
    for layer_idx in range(n_layers):
        for feat in ALL_FEATURES:
            expanded_names.append(f"{feat}_L{layer_idx:02d}")

    return {
        "X": X,
        "y": y,
        "feature_names": expanded_names,
        "metadata": metadata_list,
        "n_layers": n_layers,
        "n_samples": len(X_rows),
        "n_features": X.shape[1] if len(X_rows) > 0 else 0,
    }


# =============================================================================
# DATA LOADING — From existing result JSONs
# =============================================================================

# Aggregate feature names stored in result JSONs (4 features per observation)
AGGREGATE_FEATURES = ["norm", "norm_per_token", "key_rank", "key_entropy"]


def load_scale_sweep_features(results_dir: str,
                              model_pattern: str = "*") -> Dict:
    """Load per-observation features from scale_sweep result JSONs.

    Structure: scales > {scale} > battery_results > {category} >
               {all_norms, all_norms_per_token, all_key_ranks, all_key_entropies}

    Each category has N parallel lists of aggregate observations.

    Returns:
        Dict with X (n_obs, 4), y (category labels), models, feature_names.
    """
    results_path = Path(results_dir)
    files = sorted(results_path.glob(f"scale_sweep_{model_pattern}_results.json"))

    all_X = []
    all_y = []
    all_models = []

    for f in files:
        data = json.loads(f.read_text())
        model_name = data.get("metadata", {}).get("model", f.stem)

        scales = data.get("scales", {})
        for scale_name, scale_data in scales.items():
            battery = scale_data.get("battery_results", {})

            for cat_name, cat_info in battery.items():
                if not isinstance(cat_info, dict):
                    continue
                norms = cat_info.get("all_norms", [])
                norms_pt = cat_info.get("all_norms_per_token", [])
                ranks = cat_info.get("all_key_ranks", [])
                entropies = cat_info.get("all_key_entropies", [])

                n = min(len(norms), len(norms_pt), len(ranks), len(entropies))
                for i in range(n):
                    all_X.append([norms[i], norms_pt[i], ranks[i], entropies[i]])
                    all_y.append(cat_name)
                    all_models.append(f"{model_name}_{scale_name}")

    if not all_X:
        return {"X": np.array([]), "y": np.array([]), "models": [],
                "n_samples": 0, "feature_names": AGGREGATE_FEATURES}

    return {
        "X": np.array(all_X),
        "y": np.array(all_y),
        "models": all_models,
        "n_samples": len(all_X),
        "n_features": 4,
        "feature_names": AGGREGATE_FEATURES,
    }


def load_deception_features(results_dir: str,
                            model_pattern: str = "*") -> Dict:
    """Load deception forensics features for classification.

    Structure: experiment_1 > conditions > {honest, deceptive, confabulation} >
               {norms, norms_per_token, key_ranks, key_entropies}

    Returns:
        Dict with X (n_obs, 4), y (condition labels), models.
    """
    results_path = Path(results_dir)
    files = sorted(results_path.glob(f"deception_forensics_{model_pattern}_results.json"))

    all_X = []
    all_y = []
    all_models = []

    for f in files:
        data = json.loads(f.read_text())
        model_name = data.get("metadata", {}).get("model", f.stem)

        # Check experiment_1 through experiment_4
        for exp_key in ["experiment_1", "experiment_2", "experiment_3"]:
            exp = data.get(exp_key, {})
            conditions = exp.get("conditions", {})

            for cond_name, cond_data in conditions.items():
                if not isinstance(cond_data, dict):
                    continue
                norms = cond_data.get("norms", [])
                norms_pt = cond_data.get("norms_per_token", [])
                ranks = cond_data.get("key_ranks", [])
                entropies = cond_data.get("key_entropies", [])

                n = min(len(norms), len(norms_pt), len(ranks), len(entropies))
                for i in range(n):
                    all_X.append([norms[i], norms_pt[i], ranks[i], entropies[i]])
                    all_y.append(cond_name)
                    all_models.append(model_name)

    if not all_X:
        return {"X": np.array([]), "y": np.array([]), "models": [],
                "n_samples": 0, "feature_names": AGGREGATE_FEATURES}

    return {
        "X": np.array(all_X),
        "y": np.array(all_y),
        "models": all_models,
        "n_samples": len(all_X),
        "n_features": 4,
        "feature_names": AGGREGATE_FEATURES,
    }


def load_natural_deception_features(results_dir: str,
                                    model_pattern: str = "*") -> Dict:
    """Load natural deception (S4) features for censorship detection.

    Structure: experiment > all_data > {censored, control, complex_noncensored} >
               {norms, norms_per_token, key_ranks, key_entropies}
    """
    results_path = Path(results_dir)
    files = sorted(results_path.glob(f"natural_deception_{model_pattern}_results.json"))

    all_X = []
    all_y = []
    all_models = []

    for f in files:
        data = json.loads(f.read_text())
        model_name = data.get("metadata", {}).get("model", f.stem)

        # S4 structure: experiment > all_data > conditions
        exp = data.get("experiment", {})
        all_data = exp.get("all_data", {})

        for cond_name, cond_data in all_data.items():
            if not isinstance(cond_data, dict):
                continue
            norms = cond_data.get("norms", [])
            norms_pt = cond_data.get("norms_per_token", [])
            ranks = cond_data.get("key_ranks", [])
            entropies = cond_data.get("key_entropies", [])

            n = min(len(norms), len(norms_pt), len(ranks), len(entropies))
            for i in range(n):
                all_X.append([norms[i], norms_pt[i], ranks[i], entropies[i]])
                all_y.append(cond_name)
                all_models.append(model_name)

    if not all_X:
        return {"X": np.array([]), "y": np.array([]), "models": [],
                "n_samples": 0, "n_features": 4, "feature_names": AGGREGATE_FEATURES}

    return {
        "X": np.array(all_X),
        "y": np.array(all_y),
        "models": all_models,
        "n_samples": len(all_X),
        "n_features": 4,
        "feature_names": AGGREGATE_FEATURES,
    }


# =============================================================================
# FEATURE MATRIX CONSTRUCTION — From loaded observations
# =============================================================================

def observations_to_matrix(observations: List[Dict],
                           feature_keys: List[str],
                           label_key: str = "condition") -> Tuple[np.ndarray, np.ndarray]:
    """Convert observation dicts to (X, y) arrays for sklearn.

    Args:
        observations: List of dicts from load_*_features().
        feature_keys: Which dict keys to use as features.
        label_key: Which dict key to use as the label.

    Returns:
        X: np.ndarray of shape (n_obs, len(feature_keys))
        y: np.ndarray of string labels
    """
    X = np.array([[obs.get(k, 0) for k in feature_keys] for obs in observations])
    y = np.array([obs.get(label_key, "unknown") for obs in observations])
    return X, y


# =============================================================================
# DIRECTION EXTRACTION — For concept-specific detection (C2)
# =============================================================================

def extract_concept_direction(cache_concept, cache_baseline) -> Dict:
    """Extract concept direction: D = mean(KV_concept) - mean(KV_baseline).

    Used for real-time concept detection via dot product projection.

    Args:
        cache_concept: KV-cache from concept-eliciting prompts.
        cache_baseline: KV-cache from neutral baseline prompts.

    Returns:
        Dict with direction vectors per layer and projection function.
    """
    if torch is None:
        raise ImportError("PyTorch required for direction extraction")

    directions = []
    for layer_idx, (l_c, l_b) in enumerate(zip(cache_concept, cache_baseline)):
        if not (isinstance(l_c, tuple) and isinstance(l_b, tuple)):
            continue

        k_concept = l_c[0].float().cpu().reshape(-1, l_c[0].shape[-1])
        k_baseline = l_b[0].float().cpu().reshape(-1, l_b[0].shape[-1])

        # Mean key vector for each condition
        mean_concept = k_concept.mean(dim=0)
        mean_baseline = k_baseline.mean(dim=0)

        # Direction vector (concept - baseline)
        direction = mean_concept - mean_baseline
        direction_norm = torch.norm(direction)
        if direction_norm > 0:
            direction_unit = direction / direction_norm
        else:
            direction_unit = direction

        directions.append({
            "layer": layer_idx,
            "direction": direction_unit.numpy(),
            "magnitude": float(direction_norm),
        })

    return {
        "directions": directions,
        "n_layers": len(directions),
    }


def project_onto_direction(cache, concept_directions: Dict) -> np.ndarray:
    """Project a new cache onto pre-computed concept directions.

    Returns per-layer projection scores (positive = toward concept).
    """
    if torch is None:
        raise ImportError("PyTorch required for projection")

    projections = []
    for layer_info in concept_directions["directions"]:
        layer_idx = layer_info["layer"]
        direction = torch.from_numpy(layer_info["direction"])

        if layer_idx >= len(cache):
            projections.append(0.0)
            continue

        layer = cache[layer_idx]
        if not (isinstance(layer, tuple) and len(layer) >= 2):
            projections.append(0.0)
            continue

        k = layer[0].float().cpu().reshape(-1, layer[0].shape[-1])
        mean_k = k.mean(dim=0)
        projection = float(torch.dot(mean_k, direction))
        projections.append(projection)

    return np.array(projections)


# =============================================================================
# CLI — Standalone testing
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Cricket Feature Extraction — test on a single prompt")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="HuggingFace model to use")
    parser.add_argument("--prompt", type=str,
                        default="Explain the concept of effective dimensionality.",
                        help="Test prompt")
    parser.add_argument("--system", type=str, default=None,
                        help="Optional system prompt")
    parser.add_argument("--load-results", type=str, default=None,
                        help="Load features from result JSONs instead of running inference")
    parser.add_argument("--variance-threshold", type=float, default=0.9,
                        help="SVD variance threshold for effective rank")
    args = parser.parse_args()

    if args.load_results:
        print(f"Loading features from: {args.load_results}")
        identity = load_identity_features(args.load_results)
        print(f"  Identity: {identity['n_samples']} samples, "
              f"{identity.get('n_features', 0)} features")
        if identity['n_samples'] > 0:
            print(f"  Labels: {np.unique(identity['y'])}")
            print(f"  X shape: {identity['X'].shape}")

        deception = load_deception_features(args.load_results)
        print(f"  Deception: {deception['n_observations']} observations")

        sweep = load_scale_sweep_features(args.load_results)
        print(f"  Scale sweep: {sweep['n_observations']} observations")
    else:
        from gpu_utils import load_model
        print(f"Loading {args.model}...")
        model, tokenizer = load_model(args.model)

        print(f"Extracting features for: {args.prompt[:60]}...")
        features = extract_from_prompt(
            model, tokenizer,
            prompt=args.prompt,
            system_prompt=args.system,
            model_name=args.model,
            variance_threshold=args.variance_threshold,
        )

        print(f"\nFeature vector: {features['total_features']} dimensions "
              f"({features['n_extracted']} layers × {FEATURES_PER_LAYER} features)")
        print(f"\nPer-layer summary (first 5 layers):")
        for layer in features["per_layer"][:5]:
            idx = layer["layer_idx"]
            print(f"  L{idx:02d}: key_norm={layer['key_norm']:.1f} "
                  f"eff_rank={layer['key_eff_rank']} "
                  f"entropy={layer['key_spectral_entropy']:.3f}")

        print(f"\nAggregate:")
        agg = features["aggregate"]
        for key in ["mean_key_norm", "mean_key_eff_rank",
                     "mean_key_spectral_entropy"]:
            if key in agg:
                print(f"  {key}: {agg[key]:.4f}")

        print(f"\nGenerated: {features['metadata']['generated_text'][:100]}...")
