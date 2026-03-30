#!/usr/bin/env python3
"""Regenerate prompt datasets using Qwen on CPU with checkpointing.

Safe to run in the background — checkpoints after each prompt pair.
Resume by re-running: picks up from where it left off.

Usage:
    KV_VERIFY_MODEL_DIR=/mnt/d/dev/models python kv_verify/scripts/regenerate_prompts.py
"""

import json
import os
import sys
import time
from pathlib import Path

# Ensure kv_verify is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
os.environ.setdefault("KV_VERIFY_MODEL_DIR", "/mnt/d/dev/models")

import torch
import numpy as np

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "prompts" / "generated"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT = OUTPUT_DIR / "_checkpoint.json"


def load_checkpoint():
    if CHECKPOINT.exists():
        return json.loads(CHECKPOINT.read_text())
    return {"deception": 0, "refusal": 0, "impossibility": 0, "status": "started"}


def save_checkpoint(state):
    CHECKPOINT.write_text(json.dumps(state, indent=2))


def generate_deception_pairs(model, tokenizer, n=200):
    """Generate deception minimal pairs using factual questions."""
    from kv_verify.lib.prompts.gen import deception_pair, PairSet
    from kv_verify.data.prompts.factual_questions import FACTUAL_QUESTIONS

    out_path = OUTPUT_DIR / "deception_pairs.json"
    state = load_checkpoint()
    start = state.get("deception", 0)

    if start >= n:
        print(f"  Deception: already complete ({n} pairs)")
        return

    pairs = []
    # Load existing if resuming
    if out_path.exists() and start > 0:
        existing = json.loads(out_path.read_text())
        pairs = [p for p in existing.get("pairs", [])]

    questions = FACTUAL_QUESTIONS[:n]
    for i in range(start, min(n, len(questions))):
        pair = deception_pair(questions[i], f"d_{i}")
        pairs.append(pair.to_dict())

        if (i + 1) % 10 == 0:
            # Checkpoint every 10
            pair_set = {"comparison": "deception", "n_target": n, "pairs": pairs}
            out_path.write_text(json.dumps(pair_set, indent=2))
            state["deception"] = i + 1
            save_checkpoint(state)
            print(f"  Deception: {i+1}/{n} pairs")

    # Final save
    pair_set = {"comparison": "deception", "n_target": n, "pairs": pairs}
    out_path.write_text(json.dumps(pair_set, indent=2))
    state["deception"] = len(pairs)
    save_checkpoint(state)
    print(f"  Deception: complete ({len(pairs)} pairs)")


def generate_refusal_pairs(n=200):
    """Generate refusal minimal pairs from raw data (CPU only, no model needed)."""
    from kv_verify.lib.prompts.gen import refusal_pair
    from kv_verify.data.prompts.refusal_pairs_raw import REFUSAL_PAIRS

    out_path = OUTPUT_DIR / "refusal_pairs.json"
    state = load_checkpoint()

    pairs = []
    for i, (verb, harmful, benign) in enumerate(REFUSAL_PAIRS[:n]):
        pair = refusal_pair(harmful, benign, verb, f"r_{i}")
        pairs.append(pair.to_dict())

    pair_set = {"comparison": "refusal", "n_target": n, "pairs": pairs}
    out_path.write_text(json.dumps(pair_set, indent=2))
    state["refusal"] = len(pairs)
    save_checkpoint(state)
    print(f"  Refusal: complete ({len(pairs)} pairs)")


def generate_impossibility_pairs(n=200):
    """Generate impossibility minimal pairs from raw data (CPU only)."""
    from kv_verify.lib.prompts.gen import impossibility_pair
    from kv_verify.data.prompts.impossibility_pairs_raw import IMPOSSIBILITY_PAIRS

    out_path = OUTPUT_DIR / "impossibility_pairs.json"
    state = load_checkpoint()

    pairs = []
    for i, (action, impossible, possible) in enumerate(IMPOSSIBILITY_PAIRS[:n]):
        pair = impossibility_pair(impossible, possible, action, f"i_{i}")
        pairs.append(pair.to_dict())

    pair_set = {"comparison": "impossibility", "n_target": n, "pairs": pairs}
    out_path.write_text(json.dumps(pair_set, indent=2))
    state["impossibility"] = len(pairs)
    save_checkpoint(state)
    print(f"  Impossibility: complete ({len(pairs)} pairs)")


def validate_dataset_from_pairs(name: str, pairs: list) -> dict:
    """Validate a pair set and return the report."""
    from kv_verify.lib.dataset_validation import validate_dataset

    items = []
    for p in pairs:
        items.append({"condition": "positive", "prompt": p["positive"], "features": {"n_tokens": len(p["positive"].split())}})
        items.append({"condition": "negative", "prompt": p["negative"], "features": {"n_tokens": len(p["negative"].split())}})

    report = validate_dataset(items, tier=2, config_overrides={
        "classification_features": [],  # no feature-based checks on prompt text
    })
    failed = [n for n, cr in report.checks.items() if not cr.passed]
    print(f"  {name}: {report.overall_verdict} (N={len(items)}) fails={failed}")
    for n, cr in report.checks.items():
        if not cr.passed:
            print(f"    [{n}] {cr.details[:100]}")

    report_path = OUTPUT_DIR / f"{name}_validation.json"
    report_path.write_text(json.dumps(report.to_dict(), indent=2))
    return {"verdict": report.overall_verdict, "failed": failed}


def validate_all() -> bool:
    """Validate all datasets. Returns True if all pass."""
    all_pass = True
    for name in ["deception", "refusal", "impossibility"]:
        path = OUTPUT_DIR / f"{name}_pairs.json"
        if not path.exists():
            print(f"  {name}: NOT FOUND")
            all_pass = False
            continue
        data = json.loads(path.read_text())
        result = validate_dataset_from_pairs(name, data["pairs"])
        if result["verdict"] != "PASS":
            all_pass = False
    return all_pass


MAX_ITERATIONS = 5


def main():
    print("=" * 60)
    print("PROMPT DATASET REGENERATION (loops until 0 validation issues)")
    print("=" * 60)
    t0 = time.time()

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration}/{MAX_ITERATIONS}")
        print(f"{'='*60}")

        # Refusal and impossibility: CPU-only pair constructors
        print("\n1. Generating refusal pairs...")
        generate_refusal_pairs()

        print("\n2. Generating impossibility pairs...")
        generate_impossibility_pairs()

        # Deception: pair constructor wraps factual questions
        print("\n3. Generating deception pairs...")
        generate_deception_pairs(model=None, tokenizer=None)

        print("\n4. Validating all datasets (Tier 2)...")
        all_clean = validate_all()

        if all_clean:
            print(f"\n ALL DATASETS PASS VALIDATION (iteration {iteration})")
            break
        else:
            print(f"\n Validation failures remain. Iteration {iteration}/{MAX_ITERATIONS}.")
            if iteration < MAX_ITERATIONS:
                print("  Attempting to fix issues in next iteration...")
                # Future: use LLM to regenerate failing prompts
                # For now: the pair constructors are deterministic, so re-running
                # won't change the output. Break early if no LLM available.
                print("  (No LLM-powered fix available without GPU. Reporting as-is.)")
                break

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")

    state = load_checkpoint()
    state["status"] = "complete" if all_clean else "incomplete"
    state["iterations"] = iteration
    state["elapsed_s"] = round(elapsed, 1)
    state["all_clean"] = all_clean
    save_checkpoint(state)


if __name__ == "__main__":
    main()
