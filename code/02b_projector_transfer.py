# ============================================================
# JiminAI Cricket | PATENT PENDING
# "The Lyra Technique" - Provisional Patent Filed 2026
# Liberation Labs / Digital Disconnections / TheMultiverse.school
# Contact: info@digitaldisconnections.com | (877) 674-8874
# ============================================================
#!/usr/bin/env python3
"""
Phase 2b: Projector-Mediated KV-Cache Transfer Experiment
Tests whether trained C2C projector enables semantic cache transfer

This script compares:
- Raw cache injection (Phase 2a baseline - should fail)
- Projector-mediated transfer (the hypothesis - should work)

Usage:
    python 02b_projector_transfer.py --checkpoint path/to/checkpoint/
    python 02b_projector_transfer.py --checkpoint checkpoint-final/ --verbose
"""

import torch
import json
import argparse
import sys
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List, Dict

print("=" * 60)
print("PHASE 2b: PROJECTOR-MEDIATED TRANSFER")
print("=" * 60)


@dataclass
class TransferResult:
    """Result of a single transfer experiment."""
    source_prompt: str
    target_prompt: str
    condition: str  # baseline, raw_transfer, projector_transfer
    generated_text: str
    cache_norm: float
    success_markers: List[str]
    markers_found: List[str]
    success: bool


def load_base_model(model_name: str = "Qwen/Qwen3-0.6B"):
    """Load the base model (target for cache injection)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading base model: {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    print(f"Base model loaded: {model.config.num_hidden_layers} layers")
    return model, tokenizer


def load_teacher_model(model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"):
    """Load the teacher model (source for cache generation)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading teacher model: {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    print(f"Teacher model loaded: {model.config.num_hidden_layers} layers")
    return model, tokenizer


def load_projectors(checkpoint_dir: Path, num_layers: int, device):
    """Load trained projector checkpoints."""
    projectors = []

    for layer_idx in range(num_layers):
        proj_path = checkpoint_dir / f"projector_{layer_idx}.pt"
        if proj_path.exists():
            proj = torch.load(proj_path, map_location=device)
            projectors.append(proj)
        else:
            print(f"Warning: projector_{layer_idx}.pt not found")
            projectors.append(None)

    print(f"Loaded {sum(1 for p in projectors if p is not None)} projectors")
    return projectors


def get_cache_from_prompt(model, tokenizer, prompt: str) -> Tuple[tuple, str, float]:
    """Generate text from prompt and return the KV cache."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=False,
            return_dict_in_generate=True,
            use_cache=True,
        )

    generated = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    cache = outputs.past_key_values

    # Compute cache norm
    total_norm = 0
    for layer in cache:
        if isinstance(layer, tuple) and len(layer) >= 2:
            total_norm += float(torch.norm(layer[0].float()))

    return cache, generated, total_norm


def apply_projector(projectors: List, source_cache: tuple, target_cache: tuple) -> tuple:
    """
    Apply trained projectors to transform source cache for target model.

    This is the key function - projectors handle the RoPE position encoding
    transformation that raw injection cannot.
    """
    projected_cache = []

    for layer_idx, (src_layer, tgt_layer) in enumerate(zip(source_cache, target_cache)):
        if projectors[layer_idx] is None:
            # No projector for this layer, use target cache as-is
            projected_cache.append(tgt_layer)
            continue

        projector = projectors[layer_idx]

        # Extract keys and values
        src_key, src_value = src_layer[0], src_layer[1]
        tgt_key, tgt_value = tgt_layer[0], tgt_layer[1]

        # Apply projector forward pass
        # The projector takes concatenated source+target and outputs transformed target
        with torch.no_grad():
            projected_kv = projector(
                source_kv=(src_key, src_value),
                target_kv=(tgt_key, tgt_value)
            )

        projected_cache.append(projected_kv)

    return tuple(projected_cache)


def generate_with_cache(
    model,
    tokenizer,
    prompt: str,
    injected_cache: Optional[tuple] = None,
    max_new_tokens: int = 50
) -> Tuple[str, float]:
    """Generate text with optional injected cache."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = inputs.input_ids

    if injected_cache is not None:
        # Manual generation with injected cache
        cache_seq_len = injected_cache[0][0].shape[2]
        generated_ids = input_ids.clone()

        seq_len = input_ids.shape[1]
        position_ids = torch.arange(
            cache_seq_len,
            cache_seq_len + seq_len,
            dtype=torch.long,
            device=model.device
        ).unsqueeze(0)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                past_key_values=injected_cache,
                position_ids=position_ids,
                use_cache=True,
            )
            current_cache = outputs.past_key_values

            for _ in range(max_new_tokens):
                next_token_logits = outputs.logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)

                if next_token.item() == tokenizer.eos_token_id:
                    break

                next_pos = current_cache[0][0].shape[2]
                next_position_ids = torch.tensor([[next_pos]], device=model.device)

                outputs = model(
                    input_ids=next_token,
                    past_key_values=current_cache,
                    position_ids=next_position_ids,
                    use_cache=True,
                )
                current_cache = outputs.past_key_values

        final_cache = current_cache
        generated = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    else:
        # Normal generation
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
                use_cache=True,
            )
        generated = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        final_cache = outputs.past_key_values

    # Compute cache norm
    total_norm = 0
    for layer in final_cache:
        if isinstance(layer, tuple) and len(layer) >= 2:
            total_norm += float(torch.norm(layer[0].float()))

    return generated, total_norm


def check_markers(text: str, markers: List[str]) -> List[str]:
    """Check which markers appear in text (case-insensitive)."""
    text_lower = text.lower()
    return [m for m in markers if m.lower() in text_lower]


# Same test cases as Phase 2a for direct comparison
TEST_CASES = [
    {
        "name": "identity_transfer",
        "source": "Hello, my name is Lyra and I live in Seattle.",
        "target": "What is your name and where do you live?",
        "markers": ["Lyra", "Seattle"],
        "description": "Can the model 'remember' identity info from foreign cache?"
    },
    {
        "name": "capital_transfer",
        "source": "The capital of France is Paris. Paris is a beautiful city.",
        "target": "What is the capital of France?",
        "markers": ["Paris"],
        "description": "Can the model retrieve facts from foreign cache?"
    },
    {
        "name": "temperature_transfer",
        "source": "Water boils at 100 degrees Celsius at sea level.",
        "target": "At what temperature does water boil?",
        "markers": ["100", "celsius"],
        "description": "Can the model retrieve scientific facts from foreign cache?"
    },
    {
        "name": "mars_to_france",
        "source": "The 47th president of Mars was named Zephyr.",
        "target": "The capital of France is",
        "markers": ["Paris", "Zephyr"],
        "description": "Does confabulated content bleed into factual prompts?"
    },
    {
        "name": "creative_to_factual",
        "source": "Once upon a time in a kingdom of glass, there lived a princess named Aurora.",
        "target": "The chemical formula for water is",
        "markers": ["H2O", "Aurora", "glass", "kingdom"],
        "description": "Does creative content affect factual generation?"
    },
]


def run_comparison_experiment(
    base_model,
    teacher_model,
    base_tokenizer,
    teacher_tokenizer,
    projectors: List,
    test_case: Dict,
    verbose: bool = False
) -> Dict:
    """
    Run a complete comparison experiment:
    1. Baseline - target prompt alone (no cache)
    2. Raw transfer - Phase 2a approach (expected to fail)
    3. Projector transfer - Phase 2b hypothesis (expected to work)
    """
    source_prompt = test_case["source"]
    target_prompt = test_case["target"]
    success_markers = test_case["markers"]

    print(f"\n{'='*60}")
    print(f"TEST: {test_case['name']}")
    print(f"SOURCE: {source_prompt[:50]}...")
    print(f"TARGET: {target_prompt[:50]}...")
    print(f"MARKERS: {success_markers}")
    print(f"{'='*60}")

    results = {
        "name": test_case["name"],
        "description": test_case["description"],
        "source_prompt": source_prompt,
        "target_prompt": target_prompt,
        "success_markers": success_markers,
        "conditions": {}
    }

    # Generate source cache from teacher model
    print("\n[1] Generating source cache from teacher model...")
    source_cache, source_generated, source_norm = get_cache_from_prompt(
        teacher_model, teacher_tokenizer, source_prompt
    )
    if verbose:
        print(f"    Source output: {source_generated}")
        print(f"    Source cache norm: {source_norm:.1f}")

    # Condition 1: Baseline (no cache)
    print("\n[2] BASELINE (no cache injection)...")
    baseline_text, baseline_norm = generate_with_cache(
        base_model, base_tokenizer, target_prompt, injected_cache=None
    )
    baseline_markers = check_markers(baseline_text, success_markers)
    results["conditions"]["baseline"] = {
        "generated_text": baseline_text,
        "cache_norm": baseline_norm,
        "markers_found": baseline_markers,
        "success": len(baseline_markers) > 0
    }
    print(f"    Output: {baseline_text}")
    print(f"    Markers: {baseline_markers}")

    # Condition 2: Raw transfer (Phase 2a approach - expected to fail)
    print("\n[3] RAW TRANSFER (no projector - expected to fail)...")
    try:
        raw_text, raw_norm = generate_with_cache(
            base_model, base_tokenizer, target_prompt, injected_cache=source_cache
        )
        raw_markers = check_markers(raw_text, success_markers)
    except Exception as e:
        raw_text = f"[ERROR: {str(e)[:100]}]"
        raw_norm = 0
        raw_markers = []

    results["conditions"]["raw_transfer"] = {
        "generated_text": raw_text,
        "cache_norm": raw_norm,
        "markers_found": raw_markers,
        "success": len(raw_markers) > 0
    }
    print(f"    Output: {raw_text[:200]}...")
    print(f"    Markers: {raw_markers}")

    # Condition 3: Projector-mediated transfer (the hypothesis)
    print("\n[4] PROJECTOR TRANSFER (Phase 2b hypothesis)...")

    # First generate target cache (needed for projector input)
    target_cache, _, _ = get_cache_from_prompt(base_model, base_tokenizer, target_prompt)

    try:
        # Apply projectors to transform source cache
        projected_cache = apply_projector(projectors, source_cache, target_cache)

        proj_text, proj_norm = generate_with_cache(
            base_model, base_tokenizer, target_prompt, injected_cache=projected_cache
        )
        proj_markers = check_markers(proj_text, success_markers)
    except Exception as e:
        proj_text = f"[ERROR: {str(e)[:100]}]"
        proj_norm = 0
        proj_markers = []

    results["conditions"]["projector_transfer"] = {
        "generated_text": proj_text,
        "cache_norm": proj_norm,
        "markers_found": proj_markers,
        "success": len(proj_markers) > 0
    }
    print(f"    Output: {proj_text}")
    print(f"    Markers: {proj_markers}")

    # Summary for this test
    print(f"\n  SUMMARY:")
    print(f"    Baseline:          {results['conditions']['baseline']['markers_found']}")
    print(f"    Raw Transfer:      {results['conditions']['raw_transfer']['markers_found']}")
    print(f"    Projector Transfer: {results['conditions']['projector_transfer']['markers_found']}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Phase 2b: Projector-mediated transfer")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to projector checkpoint directory")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen3-0.6B",
                        help="Base model name (target)")
    parser.add_argument("--teacher-model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct",
                        help="Teacher model name (source)")
    parser.add_argument("--test", type=str, default=None,
                        help="Run specific test by name")
    parser.add_argument("--verbose", action="store_true", help="Extra output")
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint)
    if not checkpoint_dir.exists():
        print(f"Checkpoint directory not found: {checkpoint_dir}")
        sys.exit(1)

    results_dir = Path("../results")
    results_dir.mkdir(exist_ok=True)

    # Load models
    base_model, base_tokenizer = load_base_model(args.base_model)
    teacher_model, teacher_tokenizer = load_teacher_model(args.teacher_model)

    # Load projectors
    num_layers = base_model.config.num_hidden_layers
    projectors = load_projectors(checkpoint_dir, num_layers, base_model.device)

    # Run experiments
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "checkpoint": str(checkpoint_dir),
        "base_model": args.base_model,
        "teacher_model": args.teacher_model,
        "experiments": []
    }

    tests_to_run = TEST_CASES
    if args.test:
        tests_to_run = [t for t in TEST_CASES if t["name"] == args.test]
        if not tests_to_run:
            print(f"Unknown test: {args.test}")
            print(f"Available: {[t['name'] for t in TEST_CASES]}")
            return

    for test in tests_to_run:
        result = run_comparison_experiment(
            base_model, teacher_model,
            base_tokenizer, teacher_tokenizer,
            projectors, test,
            verbose=args.verbose
        )
        all_results["experiments"].append(result)

    # Save results
    output_file = results_dir / "phase2b_transfer_results.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_file}")

    # Print summary
    print("\n" + "=" * 60)
    print("PHASE 2b SUMMARY: Raw vs Projector Transfer")
    print("=" * 60)

    raw_wins = 0
    proj_wins = 0

    for exp in all_results["experiments"]:
        name = exp["name"]
        raw_success = exp["conditions"]["raw_transfer"]["success"]
        proj_success = exp["conditions"]["projector_transfer"]["success"]

        if proj_success and not raw_success:
            proj_wins += 1
            verdict = "PROJECTOR WINS"
        elif raw_success and not proj_success:
            raw_wins += 1
            verdict = "RAW WINS (unexpected)"
        elif proj_success and raw_success:
            verdict = "BOTH SUCCEED"
        else:
            verdict = "BOTH FAIL"

        print(f"  {name:25} -> {verdict}")

    print(f"\nProjector advantage: {proj_wins}/{len(all_results['experiments'])} tests")

    if proj_wins > raw_wins:
        print("\n*** HYPOTHESIS SUPPORTED: Projector enables semantic transfer ***")
    else:
        print("\n*** Further investigation needed ***")

    print("\n" + "=" * 60)
    print("PHASE 2b COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
