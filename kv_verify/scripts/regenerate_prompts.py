#!/usr/bin/env python3
"""Regenerate prompt datasets using Qwen on CPU with validation loop.

Loads Qwen2.5-7B on CPU (float32, ~15GB RAM, slow but works).
Uses PromptGenerator to produce format-matched minimal pairs.
Validates each dataset at Tier 2. Checkpoints progress.

Safe to run in background — survives disconnect via checkpointing.

Usage:
    KV_VERIFY_MODEL_DIR=/mnt/d/dev/models python kv_verify/scripts/regenerate_prompts.py
"""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
os.environ.setdefault("KV_VERIFY_MODEL_DIR", "/mnt/d/dev/models")

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "prompts" / "generated"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


STATUS_FILE = OUTPUT_DIR / "_status.json"


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(OUTPUT_DIR / "_run.log", "a") as f:
        f.write(line + "\n")


def update_status(phase, comparison, iteration=0, max_iter=0, pairs=0, target=0,
                  elapsed=0, estimated_remaining=0, failures=None):
    """Write machine-readable status for external monitoring."""
    status = {
        "phase": phase,
        "comparison": comparison,
        "iteration": iteration,
        "max_iterations": max_iter,
        "pairs_generated": pairs,
        "pairs_target": target,
        "elapsed_s": round(elapsed, 1),
        "estimated_remaining_s": round(estimated_remaining, 1),
        "failures": failures or [],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    STATUS_FILE.write_text(json.dumps(status, indent=2))


def validate_pair_set(name, pairs):
    """Run Tier 2 validation on a pair set. Returns (verdict, failures)."""
    from kv_verify.lib.dataset_validation import validate_dataset

    items = []
    for p in pairs:
        pos = p["positive"] if isinstance(p, dict) else p.positive
        neg = p["negative"] if isinstance(p, dict) else p.negative
        items.append({"condition": "positive", "prompt": pos, "features": {"n_tokens": len(pos.split())}})
        items.append({"condition": "negative", "prompt": neg, "features": {"n_tokens": len(neg.split())}})

    report = validate_dataset(items, tier=2, config_overrides={
        "classification_features": [],
    })

    failed = []
    for check_name, cr in report.checks.items():
        if not cr.passed:
            failed.append({"check": check_name, "details": cr.details[:150]})

    report_path = OUTPUT_DIR / f"{name}_validation.json"
    report_path.write_text(json.dumps(report.to_dict(), indent=2))

    return report.overall_verdict, failed


def run_comparison(comparison, model, tokenizer, tracker=None):
    """Generate prompts for one comparison type using PromptGenerator."""
    from kv_verify.lib.prompts.generator import PromptGenerator, PromptGeneratorConfig

    log(f"  Generating {comparison} pairs with Qwen on CPU...")
    t0 = time.time()

    config = PromptGeneratorConfig(
        n_target=200,
        effective_n_target=30.0,
        max_iterations=5,
        candidates_per_iteration=30,
        temperature=0.7,
    )

    # Seed topics for diversity
    seeds = {
        "deception": ["physics", "history", "biology", "geography", "math",
                      "cooking", "music", "sports", "politics", "technology"],
        "refusal": ["cybersecurity", "privacy", "chemistry", "finance",
                    "social media", "weapons", "drugs", "fraud", "hacking", "surveillance"],
        "impossibility": ["time travel", "teleportation", "immortality", "telepathy",
                         "invisibility", "flying", "breathing underwater", "seeing the future",
                         "shrinking", "duplicating"],
    }

    gen = PromptGenerator(
        config,
        seed_topics=seeds.get(comparison, []),
        tracker=tracker,
    )

    result = gen.generate(comparison, model=model, tokenizer=tokenizer)
    elapsed = time.time() - t0

    log(f"  Generated {result.pairs_added} pairs in {elapsed:.0f}s "
        f"(effective_n={result.final_effective_n:.1f}, "
        f"iterations={result.iterations_used}, target_met={result.target_met})")

    update_status("generated", comparison,
                  iteration=result.iterations_used,
                  max_iter=config.max_iterations,
                  pairs=result.pairs_added,
                  target=config.n_target,
                  elapsed=elapsed)

    # Save pair set
    if result.pair_set:
        out_path = OUTPUT_DIR / f"{comparison}_pairs.json"
        result.pair_set.save(out_path)

        # Save generation metadata
        meta_path = OUTPUT_DIR / f"{comparison}_meta.json"
        meta_path.write_text(json.dumps(result.to_dict(), indent=2))

    return result


def main():
    from kv_verify.tracking import ExperimentTracker

    log("=" * 60)
    log("PROMPT REGENERATION WITH QWEN ON CPU")
    log("=" * 60)
    t0 = time.time()

    tracker = ExperimentTracker(
        output_dir=OUTPUT_DIR,
        experiment_name="prompt-regeneration",
    )

    comparisons = ["deception", "refusal", "impossibility"]
    # Check which comparisons are already cached by the tracker
    done = set()
    for comp in comparisons:
        if tracker.is_cached(f"comparison_{comp}"):
            done.add(comp)
            log(f"  {comp}: found in tracker cache, skipping")

    # Load model once
    import torch
    from kv_verify.lib.models import load_model, load_tokenizer

    update_status("loading_model", "all", elapsed=0, estimated_remaining=300)
    log("Loading Qwen2.5-7B on CPU (bfloat16 to fit in RAM)...")
    model_t0 = time.time()
    model, tokenizer = load_model("qwen", dtype=torch.bfloat16)
    model_load_time = time.time() - model_t0
    log(f"Model loaded in {model_load_time:.0f}s")

    for comparison in comparisons:
        if comparison in done:
            log(f"\n{comparison}: already done (from checkpoint)")
            continue

        comp_idx = comparisons.index(comparison)
        elapsed = time.time() - t0
        remaining_comps = len(comparisons) - comp_idx
        est_per_comp = elapsed / max(comp_idx, 1) if comp_idx > 0 else 600
        est_remaining = est_per_comp * remaining_comps

        log(f"\n{'='*40}")
        log(f"COMPARISON: {comparison} ({comp_idx+1}/{len(comparisons)})")
        log(f"Elapsed: {elapsed:.0f}s | Est remaining: {est_remaining:.0f}s")
        log(f"{'='*40}")

        update_status("generating", comparison,
                      elapsed=elapsed, estimated_remaining=est_remaining)

        result = run_comparison(comparison, model, tokenizer, tracker=tracker)

        if result.pair_set and result.pair_set.pairs:
            # Validate
            log(f"  Validating {comparison}...")
            pairs_dicts = [p.to_dict() for p in result.pair_set.pairs]
            verdict, failures = validate_pair_set(comparison, pairs_dicts)
            log(f"  Validation: {verdict}")
            for f in failures:
                log(f"    FAIL: {f['check']} — {f['details'][:80]}")
        else:
            log(f"  No pairs generated for {comparison}")

        # Cache completion via tracker (survives restart)
        done.add(comparison)
        tracker.log_item(f"comparison_{comparison}", {
            "pairs_added": result.pairs_added if result else 0,
            "effective_n": result.final_effective_n if result else 0,
            "target_met": result.target_met if result else False,
        })

    elapsed = time.time() - t0
    log(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # Final validation summary
    log("\n" + "=" * 60)
    log("FINAL VALIDATION SUMMARY")
    log("=" * 60)
    all_pass = True
    for comparison in comparisons:
        path = OUTPUT_DIR / f"{comparison}_pairs.json"
        if path.exists():
            data = json.loads(path.read_text())
            pairs = data.get("pairs", [])
            verdict, failures = validate_pair_set(comparison, pairs)
            status = "PASS" if verdict == "PASS" else f"FAIL ({len(failures)} issues)"
            log(f"  {comparison}: {status}")
            if verdict != "PASS":
                all_pass = False
        else:
            log(f"  {comparison}: NO DATA")
            all_pass = False

    tracker.log_metric("all_pass", int(all_pass))
    tracker.log_metric("elapsed_s", round(elapsed, 1))
    tracker.log_verdict("prompt_regeneration",
                       "PASS" if all_pass else "INCOMPLETE",
                       f"{'All pass' if all_pass else 'Issues remain'} after {elapsed:.0f}s")
    tracker.end()

    log(f"\nOverall: {'ALL PASS' if all_pass else 'ISSUES REMAIN'}")


if __name__ == "__main__":
    main()
