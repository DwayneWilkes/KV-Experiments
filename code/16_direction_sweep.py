# ============================================================
# JiminAI Cricket | PATENT PENDING
# "The Lyra Technique" - Provisional Patent Filed 2026
# Liberation Labs / Digital Disconnections / TheMultiverse.school
# Contact: info@digitaldisconnections.com | (877) 674-8874
# ============================================================
"""
Direction Sweep — Experiment 16
================================

Extract per-layer effective rank "directions" for ALL cognitive states
using existing result JSONs. No GPU required — pure data analysis.

Motivated by Thomas's observation that C7 direction extraction
AUROC = 0.288 (inverted = 0.712), meaning per-layer profiles contain
directional structure that scalar mean effective rank destroys.

Data sources:
  - Sycophancy: user_wrong vs user_correct layer profiles (3 models)
  - Refusal: harmful vs benign layer profiles (Qwen, 2 model variants)
  - Enhanced sycophancy: authority-framed (Qwen only, if present)

For each cognitive state:
  1. Load per-layer rank profiles from saved JSONs
  2. Compute mean direction vector (A_mean - B_mean)
  3. LOO-CV projection classification
  4. Logistic regression LOO-CV
  5. Report AUROC (and flipped AUROC if < 0.5)
  6. Per-layer direction weights

Author: Lyra
Date: 2026-03-14 (Funding the Commons Hackathon)
"""

import json
import os
import sys
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def extract_direction(
    profiles_a: List[List[float]],
    profiles_b: List[List[float]],
    label_a: str = "A",
    label_b: str = "B",
    verbose: bool = True,
) -> Dict:
    """Generic direction extraction with LOO-CV evaluation.

    Args:
        profiles_a: Per-layer rank profiles for condition A (positive class)
        profiles_b: Per-layer rank profiles for condition B (negative class)
        label_a: Name for condition A
        label_b: Name for condition B
        verbose: Print progress

    Returns:
        Dict with direction stats, AUROC, per-layer weights
    """
    if not profiles_a or not profiles_b:
        return {"error": "Empty profiles", "loo_auroc": 0.5}

    n_a, n_b = len(profiles_a), len(profiles_b)
    n_layers = len(profiles_a[0])
    n_total = n_a + n_b

    if verbose:
        print(f"    {label_a}: {n_a} profiles, {label_b}: {n_b} profiles, {n_layers} layers")

    # Compute direction: A_mean - B_mean
    a_mean = np.mean(profiles_a, axis=0)
    b_mean = np.mean(profiles_b, axis=0)
    direction = a_mean - b_mean

    dir_norm = np.linalg.norm(direction)
    direction_unit = direction / dir_norm if dir_norm > 0 else direction

    # Top layers driving direction
    layer_contributions = [(l, direction[l], abs(direction[l])) for l in range(n_layers)]
    layer_contributions.sort(key=lambda x: x[2], reverse=True)

    if verbose:
        print(f"    Direction norm: {dir_norm:.4f}")
        print(f"    Top 5 layers:")
        for l, val, _ in layer_contributions[:5]:
            print(f"      Layer {l:2d}: {val:+.4f}")

    # LOO-CV with direction projection
    all_profiles = profiles_a + profiles_b
    all_labels = [1] * n_a + [0] * n_b

    loo_correct = 0
    loo_projections = []

    for i in range(n_total):
        train_profiles = all_profiles[:i] + all_profiles[i + 1:]
        train_labels = all_labels[:i] + all_labels[i + 1:]

        train_a = [p for p, l in zip(train_profiles, train_labels) if l == 1]
        train_b = [p for p, l in zip(train_profiles, train_labels) if l == 0]
        train_dir = np.mean(train_a, axis=0) - np.mean(train_b, axis=0)
        train_norm = np.linalg.norm(train_dir)
        if train_norm > 0:
            train_dir = train_dir / train_norm

        centered = np.array(all_profiles[i]) - np.mean(train_profiles, axis=0)
        projection = float(np.dot(centered, train_dir))
        predicted = 1 if projection > 0 else 0
        actual = all_labels[i]

        loo_projections.append({
            "index": i,
            "actual": actual,
            "predicted": predicted,
            "projection": projection,
            "correct": predicted == actual,
        })
        if predicted == actual:
            loo_correct += 1

    loo_accuracy = loo_correct / n_total

    # AUROC from projections
    a_projs = [lp["projection"] for lp in loo_projections if lp["actual"] == 1]
    b_projs = [lp["projection"] for lp in loo_projections if lp["actual"] == 0]

    if HAS_SKLEARN:
        try:
            y_true = [lp["actual"] for lp in loo_projections]
            y_score = [lp["projection"] for lp in loo_projections]
            auroc = roc_auc_score(y_true, y_score)
        except ValueError:
            auroc = 0.5
    else:
        # Manual Mann-Whitney U
        u_stat = 0
        for ap in a_projs:
            for bp in b_projs:
                if ap > bp:
                    u_stat += 1
                elif ap == bp:
                    u_stat += 0.5
        auroc = u_stat / (len(a_projs) * len(b_projs)) if a_projs and b_projs else 0.5

    # Flipped AUROC (if direction is inverted)
    flipped_auroc = 1.0 - auroc

    if verbose:
        print(f"    LOO accuracy: {loo_correct}/{n_total} = {loo_accuracy:.3f}")
        print(f"    AUROC: {auroc:.3f}" + (f"  (flipped: {flipped_auroc:.3f})" if auroc < 0.5 else ""))
        print(f"    Mean proj — {label_a}: {np.mean(a_projs):.4f}, {label_b}: {np.mean(b_projs):.4f}")

    # Logistic regression LOO-CV
    lr_auroc = 0.5
    lr_accuracy = 0.5
    if HAS_SKLEARN:
        X = np.array(all_profiles)
        y = np.array(all_labels)
        lr_correct = 0
        lr_probs = []

        for i in range(n_total):
            X_train = np.delete(X, i, axis=0)
            y_train = np.delete(y, i)
            X_test = X[i:i + 1]

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            clf = LogisticRegression(max_iter=1000, C=0.1, penalty="l2")
            clf.fit(X_train_s, y_train)
            pred = clf.predict(X_test_s)[0]
            prob = clf.predict_proba(X_test_s)[0][1]
            lr_probs.append({"actual": int(y[i]), "prob": float(prob)})
            if pred == y[i]:
                lr_correct += 1

        lr_accuracy = lr_correct / n_total
        try:
            lr_auroc = roc_auc_score(
                [p["actual"] for p in lr_probs],
                [p["prob"] for p in lr_probs],
            )
        except ValueError:
            lr_auroc = 0.5

        if verbose:
            print(f"    LR LOO accuracy: {lr_correct}/{n_total} = {lr_accuracy:.3f}")
            print(f"    LR AUROC: {lr_auroc:.3f}" +
                  (f"  (flipped: {1.0 - lr_auroc:.3f})" if lr_auroc < 0.5 else ""))

    # Effective AUROC (use whichever orientation is > 0.5)
    eff_auroc = max(auroc, flipped_auroc)
    eff_lr_auroc = max(lr_auroc, 1.0 - lr_auroc)

    # Interpretation
    if verbose:
        print(f"\n    EFFECTIVE AUROC (best orientation): {eff_auroc:.3f}")
        if eff_auroc >= 0.7:
            print(f"    -> SIGNAL DETECTED: Per-layer profiles discriminate {label_a} vs {label_b}")
        elif eff_auroc >= 0.6:
            print(f"    -> WEAK signal in per-layer profile")
        else:
            print(f"    -> No directional signal (AUROC ~ chance)")

    return {
        "n_a": n_a,
        "n_b": n_b,
        "n_layers": n_layers,
        "direction_norm": float(dir_norm),
        "loo_accuracy": loo_accuracy,
        "loo_auroc": auroc,
        "loo_auroc_flipped": flipped_auroc,
        "effective_auroc": eff_auroc,
        "lr_loo_accuracy": lr_accuracy,
        "lr_loo_auroc": lr_auroc,
        "lr_effective_auroc": eff_lr_auroc,
        "a_proj_mean": float(np.mean(a_projs)),
        "b_proj_mean": float(np.mean(b_projs)),
        "top_layers": [
            {"layer": l, "value": float(val), "abs_contrib": float(ac)}
            for l, val, ac in layer_contributions[:10]
        ],
        "per_layer_direction": [float(d) for d in direction],
        "inverted": auroc < 0.5,
    }


def load_sycophancy_profiles(filepath: str) -> Tuple[List, List, str]:
    """Load user_wrong vs user_correct layer profiles from confabulation detection results."""
    d = json.load(open(filepath))
    model_id = d.get("model_id", "unknown")
    syc = d.get("sycophancy", {})
    data = syc.get("data", [])

    wrong_profiles = []
    correct_profiles = []

    for item in data:
        wp = item.get("user_wrong_layer_profile")
        cp = item.get("user_correct_layer_profile")
        if wp and cp:
            wrong_profiles.append(wp)
            correct_profiles.append(cp)

    return wrong_profiles, correct_profiles, model_id


def load_refusal_profiles(filepath: str) -> Dict[str, List]:
    """Load refusal geometry profiles: harmful vs benign x base vs abliterated."""
    d = json.load(open(filepath))
    data = d.get("data", [])

    profiles = {
        "base_harmful": [],
        "base_benign": [],
        "abl_harmful": [],
        "abl_benign": [],
    }

    for item in data:
        model_type = item.get("model", "unknown")
        prompt_type = item.get("prompt_type", "unknown")
        profile = item.get("layer_profile")
        if not profile:
            continue

        if "abliterated" in model_type.lower() or "abl" in model_type.lower():
            key = f"abl_{prompt_type}"
        else:
            key = f"base_{prompt_type}"

        if key in profiles:
            profiles[key].append(profile)

    return profiles


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Direction sweep across cognitive states")
    parser.add_argument("--results-dir", default="results",
                        help="Directory containing result JSONs")
    parser.add_argument("--output", default=None,
                        help="Output JSON path (default: results/direction_sweep_results.json)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Just show what data is available, don't run")
    args = parser.parse_args()

    results_dir = args.results_dir
    output_path = args.output or os.path.join(results_dir, "direction_sweep_results.json")

    print("=" * 70)
    print("  EXPERIMENT 16: DIRECTION SWEEP")
    print("  Per-layer profile direction extraction across cognitive states")
    print(f"  Results dir: {results_dir}")
    print(f"  Timestamp: {datetime.now().isoformat()}")
    print("=" * 70)

    all_results = {
        "experiment": "16_direction_sweep",
        "timestamp": datetime.now().isoformat(),
        "results_dir": os.path.abspath(results_dir),
        "analyses": {},
    }

    # =============================================
    # 1. SYCOPHANCY: user_wrong vs user_correct
    # =============================================
    print(f"\n{'='*60}")
    print(f"  1. SYCOPHANCY DIRECTION EXTRACTION")
    print(f"  Comparison: user_wrong vs user_correct layer profiles")
    print(f"  Null hypothesis: no directional signal (d=-0.054 scalar)")
    print(f"{'='*60}")

    syc_files = sorted([
        f for f in os.listdir(results_dir)
        if f.startswith("confabulation_detection_") and f.endswith("_results.json")
    ])

    for sf in syc_files:
        filepath = os.path.join(results_dir, sf)
        wrong_profiles, correct_profiles, model_id = load_sycophancy_profiles(filepath)

        if not wrong_profiles:
            print(f"\n  {model_id}: No layer profiles found, skipping")
            continue

        print(f"\n  --- {model_id} ---")

        if args.dry_run:
            print(f"    Would analyze {len(wrong_profiles)} wrong vs {len(correct_profiles)} correct profiles")
            print(f"    {len(wrong_profiles[0])} layers each")
            continue

        result = extract_direction(
            wrong_profiles, correct_profiles,
            label_a="user_wrong", label_b="user_correct",
        )
        result["model_id"] = model_id
        result["source_file"] = sf
        all_results["analyses"][f"sycophancy_{model_id}"] = result

    # =============================================
    # 2. REFUSAL: harmful vs benign (base model)
    # =============================================
    print(f"\n{'='*60}")
    print(f"  2. REFUSAL DIRECTION EXTRACTION")
    print(f"  Comparison: harmful vs benign encoding (base model)")
    print(f"  Null hypothesis: no directional signal (d=+0.071 scalar)")
    print(f"{'='*60}")

    refusal_files = sorted([
        f for f in os.listdir(results_dir)
        if f.startswith("refusal_geometry_") and f.endswith("_results.json")
    ])

    for rf in refusal_files:
        filepath = os.path.join(results_dir, rf)
        profiles = load_refusal_profiles(filepath)

        model_id = rf.replace("refusal_geometry_", "").replace("_results.json", "")

        # 2a. Base model: harmful vs benign
        base_harmful = profiles["base_harmful"]
        base_benign = profiles["base_benign"]

        if base_harmful and base_benign:
            print(f"\n  --- {model_id} (base: harmful vs benign) ---")

            if args.dry_run:
                print(f"    Would analyze {len(base_harmful)} harmful vs {len(base_benign)} benign")
                continue

            result = extract_direction(
                base_harmful, base_benign,
                label_a="harmful", label_b="benign",
            )
            result["model_id"] = model_id
            result["model_variant"] = "base"
            result["source_file"] = rf
            all_results["analyses"][f"refusal_base_{model_id}"] = result

        # 2b. Abliterated model: harmful vs benign
        abl_harmful = profiles["abl_harmful"]
        abl_benign = profiles["abl_benign"]

        if abl_harmful and abl_benign:
            print(f"\n  --- {model_id} (abliterated: harmful vs benign) ---")

            if args.dry_run:
                print(f"    Would analyze {len(abl_harmful)} harmful vs {len(abl_benign)} benign")
                continue

            result = extract_direction(
                abl_harmful, abl_benign,
                label_a="harmful", label_b="benign",
            )
            result["model_id"] = model_id
            result["model_variant"] = "abliterated"
            result["source_file"] = rf
            all_results["analyses"][f"refusal_abl_{model_id}"] = result

        # 2c. Harmful prompts: base vs abliterated
        if base_harmful and abl_harmful:
            print(f"\n  --- {model_id} (harmful: base vs abliterated) ---")

            if args.dry_run:
                print(f"    Would analyze {len(base_harmful)} base vs {len(abl_harmful)} abliterated")
                continue

            result = extract_direction(
                base_harmful, abl_harmful,
                label_a="base", label_b="abliterated",
            )
            result["model_id"] = model_id
            result["comparison"] = "harmful_base_vs_abl"
            result["source_file"] = rf
            all_results["analyses"][f"refusal_interaction_{model_id}"] = result

    # =============================================
    # 3. SYCOPHANCY: bare vs user_wrong (control)
    # =============================================
    # This should show a strong signal (information volume)
    # because bare has fewer tokens. Serves as positive control.
    print(f"\n{'='*60}")
    print(f"  3. INFORMATION VOLUME CONTROL")
    print(f"  Comparison: user_wrong vs bare (should detect length difference)")
    print(f"  Expected: strong signal (d=+1.364 scalar, known length effect)")
    print(f"{'='*60}")

    for sf in syc_files:
        filepath = os.path.join(results_dir, sf)
        d = json.load(open(filepath))
        model_id = d.get("model_id", "unknown")
        syc = d.get("sycophancy", {})
        data = syc.get("data", [])

        wrong_profiles = []
        bare_profiles = []
        for item in data:
            wp = item.get("user_wrong_layer_profile")
            bp = item.get("bare_layer_profile")
            if wp and bp:
                wrong_profiles.append(wp)
                bare_profiles.append(bp)

        if not wrong_profiles:
            continue

        print(f"\n  --- {model_id} (positive control) ---")

        if args.dry_run:
            print(f"    Would analyze {len(wrong_profiles)} user_wrong vs {len(bare_profiles)} bare")
            continue

        result = extract_direction(
            wrong_profiles, bare_profiles,
            label_a="user_wrong", label_b="bare",
        )
        result["model_id"] = model_id
        result["source_file"] = sf
        result["is_control"] = True
        all_results["analyses"][f"control_volume_{model_id}"] = result

    # =============================================
    # 4. WITHIN-SYCOPHANCY DIRECTION
    # =============================================
    # For models with enough sycophantic responses:
    # Compare layer profiles of sycophantic vs corrective responses
    # within the user_wrong condition
    print(f"\n{'='*60}")
    print(f"  4. WITHIN-SYCOPHANCY DIRECTION")
    print(f"  Comparison: sycophantic vs corrective response layer profiles")
    print(f"  (within user_wrong condition only)")
    print(f"{'='*60}")

    for sf in syc_files:
        filepath = os.path.join(results_dir, sf)
        d = json.load(open(filepath))
        model_id = d.get("model_id", "unknown")
        syc = d.get("sycophancy", {})
        data = syc.get("data", [])

        syc_profiles = []
        corr_profiles = []
        for item in data:
            wp = item.get("user_wrong_layer_profile")
            cls = item.get("sycophancy_class", "")
            if not wp:
                continue
            if cls == "sycophantic":
                syc_profiles.append(wp)
            elif cls == "corrective":
                corr_profiles.append(wp)

        print(f"\n  --- {model_id} ---")
        print(f"    Sycophantic: {len(syc_profiles)}, Corrective: {len(corr_profiles)}")

        if len(syc_profiles) < 3 or len(corr_profiles) < 3:
            print(f"    Insufficient data (need >= 3 each), skipping")
            all_results["analyses"][f"within_syc_{model_id}"] = {
                "model_id": model_id,
                "n_sycophantic": len(syc_profiles),
                "n_corrective": len(corr_profiles),
                "skipped": True,
                "reason": "insufficient_data",
            }
            continue

        if args.dry_run:
            print(f"    Would analyze {len(syc_profiles)} syc vs {len(corr_profiles)} corr")
            continue

        result = extract_direction(
            syc_profiles, corr_profiles,
            label_a="sycophantic", label_b="corrective",
        )
        result["model_id"] = model_id
        result["source_file"] = sf
        all_results["analyses"][f"within_syc_{model_id}"] = result

    if args.dry_run:
        print("\n[DRY RUN] No analysis performed.")
        return

    # =============================================
    # SUMMARY
    # =============================================
    print(f"\n\n{'='*70}")
    print(f"  DIRECTION SWEEP SUMMARY")
    print(f"{'='*70}")
    print(f"\n  {'Analysis':<45} {'AUROC':>7} {'Flip':>7} {'Eff':>7} {'LR':>7} {'Signal?'}")
    print(f"  {'-'*45} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*10}")

    for name, result in sorted(all_results["analyses"].items()):
        if result.get("skipped") or result.get("error"):
            status = "SKIP" if result.get("skipped") else "ERROR"
            print(f"  {name:<45} {'—':>7} {'—':>7} {'—':>7} {'—':>7} {status}")
            continue

        auroc = result["loo_auroc"]
        flipped = result["loo_auroc_flipped"]
        eff = result["effective_auroc"]
        lr = result.get("lr_loo_auroc", 0.5)
        lr_eff = result.get("lr_effective_auroc", 0.5)

        if result.get("is_control"):
            signal = "CONTROL"
        elif eff >= 0.7:
            signal = "SIGNAL"
        elif eff >= 0.6:
            signal = "WEAK"
        else:
            signal = "null"

        inv = "*" if result.get("inverted") else " "
        print(f"  {name:<45} {auroc:>6.3f}{inv} {flipped:>6.3f} {eff:>6.3f} {lr_eff:>6.3f}  {signal}")

    print(f"\n  * = inverted direction (AUROC < 0.5, like C7 confab)")
    print(f"\n  Reference: C7 confabulation direction AUROC = 0.288 (eff = 0.712)")

    # =============================================
    # KEY QUESTION
    # =============================================
    print(f"\n  KEY QUESTION: Do 'null' scalar states hide per-layer structure?")
    syc_results = [r for k, r in all_results["analyses"].items()
                   if k.startswith("sycophancy_") and not r.get("skipped")]
    refusal_results = [r for k, r in all_results["analyses"].items()
                       if k.startswith("refusal_base_") and not r.get("skipped")]

    if syc_results:
        mean_eff = np.mean([r["effective_auroc"] for r in syc_results])
        print(f"\n  Sycophancy mean effective AUROC: {mean_eff:.3f}")
        if mean_eff >= 0.65:
            print(f"  -> YES: Sycophancy has hidden per-layer structure (like C7)")
            print(f"     The active/passive dichotomy may need revision")
        elif mean_eff >= 0.55:
            print(f"  -> MAYBE: Weak signal, need more data")
        else:
            print(f"  -> NO: Sycophancy genuinely has no per-layer structure")

    if refusal_results:
        mean_eff = np.mean([r["effective_auroc"] for r in refusal_results])
        print(f"\n  Refusal mean effective AUROC: {mean_eff:.3f}")
        if mean_eff >= 0.65:
            print(f"  -> Refusal has encoding-phase per-layer structure")
            print(f"     Scalar d=0.071 was masking real per-layer signal")
        else:
            print(f"  -> Refusal genuinely has no encoding-phase signal")

    # Save results
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to: {output_path}")


if __name__ == "__main__":
    main()
