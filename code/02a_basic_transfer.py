# ============================================================
# JiminAI Cricket | PATENT PENDING
# "The Lyra Technique" - Provisional Patent Filed 2026
# Liberation Labs / Digital Disconnections / TheMultiverse.school
# Contact: info@digitaldisconnections.com | (877) 674-8874
# ============================================================
#!/usr/bin/env python3
"""
Phase 2a: Basic KV-Cache Transfer Experiment
Tests whether context survives cache transplantation

Key question: Can we inject a cache from one prompt and have the model
"remember" information it never directly saw in the current context?

Usage:
    python 02a_basic_transfer.py
    python 02a_basic_transfer.py --verbose
"""

import torch
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Tuple, List

print("=" * 60)
print("PHASE 2a: BASIC KV-CACHE TRANSFER")
print("=" * 60)


@dataclass
class TransferResult:
    """Result of a single transfer experiment."""
    source_prompt: str
    target_prompt: str
    condition: str  # baseline, same_cache, random_noise, transfer
    generated_text: str
    source_cache_norm: float
    target_cache_norm: float
    success_markers: List[str]  # what we're looking for in output
    markers_found: List[str]    # what we actually found
    success: bool


def load_model():
    """Load TinyLlama with 4-bit quantization."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    print(f"Loading {model_name}...")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True
    )

    print(f"Model loaded: {model.config.num_hidden_layers} layers")
    return model, tokenizer


def get_cache_from_prompt(model, tokenizer, prompt: str) -> Tuple[tuple, str, float]:
    """
    Generate text from prompt and return the KV cache.
    Returns: (cache, generated_text, cache_norm)
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=False,  # Deterministic for reproducibility
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


def generate_with_injected_cache(
    model,
    tokenizer,
    prompt: str,
    injected_cache: Optional[tuple] = None,
    max_new_tokens: int = 50
) -> Tuple[str, float]:
    """
    Generate text with an optional injected cache.

    Uses manual token-by-token generation to properly handle cache injection.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = inputs.input_ids

    if injected_cache is not None:
        # Manual generation loop with injected cache
        cache_seq_len = injected_cache[0][0].shape[2]
        generated_ids = input_ids.clone()

        # First forward pass: process prompt with position offset
        seq_len = input_ids.shape[1]
        position_ids = torch.arange(
            cache_seq_len,
            cache_seq_len + seq_len,
            dtype=torch.long,
            device=model.device
        ).unsqueeze(0)

        with torch.no_grad():
            # Initial pass - combine injected cache with prompt
            outputs = model(
                input_ids=input_ids,
                past_key_values=injected_cache,
                position_ids=position_ids,
                use_cache=True,
            )
            current_cache = outputs.past_key_values

            # Token-by-token generation
            for _ in range(max_new_tokens):
                next_token_logits = outputs.logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)

                if next_token.item() == tokenizer.eos_token_id:
                    break

                # Next position
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
        # Normal generation without cache injection
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

    # Compute final cache norm
    total_norm = 0
    for layer in final_cache:
        if isinstance(layer, tuple) and len(layer) >= 2:
            total_norm += float(torch.norm(layer[0].float()))

    return generated, total_norm


def create_random_cache_like(cache: tuple, device) -> tuple:
    """Create random noise cache with same shape as input cache."""
    random_cache = []
    for layer in cache:
        if isinstance(layer, tuple) and len(layer) >= 2:
            key_shape = layer[0].shape
            value_shape = layer[1].shape

            # Random with similar magnitude
            key_std = float(layer[0].float().std())
            value_std = float(layer[1].float().std())

            random_key = torch.randn(key_shape, device=device, dtype=layer[0].dtype) * key_std
            random_value = torch.randn(value_shape, device=device, dtype=layer[1].dtype) * value_std

            random_cache.append((random_key, random_value))
        else:
            random_cache.append(layer)

    return tuple(random_cache)


def check_markers(text: str, markers: List[str]) -> List[str]:
    """Check which markers appear in the text (case-insensitive)."""
    text_lower = text.lower()
    return [m for m in markers if m.lower() in text_lower]


def run_transfer_experiment(
    model,
    tokenizer,
    source_prompt: str,
    target_prompt: str,
    success_markers: List[str],
    verbose: bool = False
) -> List[TransferResult]:
    """
    Run a complete transfer experiment with all conditions:
    1. Baseline - target prompt alone
    2. Same-cache - target prompt with its own cache (sanity check)
    3. Random noise - target prompt with random cache (what does broken look like?)
    4. Transfer - target prompt with source cache (the real experiment)
    """
    results = []

    print(f"\n{'='*60}")
    print(f"SOURCE: {source_prompt[:50]}...")
    print(f"TARGET: {target_prompt[:50]}...")
    print(f"MARKERS: {success_markers}")
    print(f"{'='*60}")

    # First, generate the source cache
    print("\n[1] Generating source cache...")
    source_cache, source_generated, source_norm = get_cache_from_prompt(
        model, tokenizer, source_prompt
    )
    if verbose:
        print(f"    Source output: {source_generated}")
        print(f"    Source cache norm: {source_norm:.1f}")

    # Condition 1: Baseline (no cache injection)
    print("\n[2] Running BASELINE (no cache)...")
    baseline_text, baseline_norm = generate_with_injected_cache(
        model, tokenizer, target_prompt, injected_cache=None
    )
    baseline_markers = check_markers(baseline_text, success_markers)
    results.append(TransferResult(
        source_prompt=source_prompt,
        target_prompt=target_prompt,
        condition="baseline",
        generated_text=baseline_text,
        source_cache_norm=0,
        target_cache_norm=baseline_norm,
        success_markers=success_markers,
        markers_found=baseline_markers,
        success=len(baseline_markers) > 0
    ))
    print(f"    Output: {baseline_text}")
    print(f"    Markers found: {baseline_markers}")

    # Condition 2: Same-cache (generate target's own cache, then use it)
    # This is a sanity check - should work perfectly
    print("\n[3] Running SAME-CACHE sanity check...")
    target_cache, _, target_cache_norm = get_cache_from_prompt(
        model, tokenizer, target_prompt
    )
    same_text, same_norm = generate_with_injected_cache(
        model, tokenizer, target_prompt, injected_cache=target_cache
    )
    same_markers = check_markers(same_text, success_markers)
    results.append(TransferResult(
        source_prompt=source_prompt,
        target_prompt=target_prompt,
        condition="same_cache",
        generated_text=same_text,
        source_cache_norm=target_cache_norm,
        target_cache_norm=same_norm,
        success_markers=success_markers,
        markers_found=same_markers,
        success=len(same_markers) > 0
    ))
    print(f"    Output: {same_text}")
    print(f"    Markers found: {same_markers}")

    # Condition 3: Random noise cache
    print("\n[4] Running RANDOM NOISE (what does broken look like?)...")
    random_cache = create_random_cache_like(source_cache, model.device)
    try:
        noise_text, noise_norm = generate_with_injected_cache(
            model, tokenizer, target_prompt, injected_cache=random_cache
        )
        noise_markers = check_markers(noise_text, success_markers)
    except Exception as e:
        noise_text = f"[ERROR: {str(e)[:100]}]"
        noise_norm = 0
        noise_markers = []

    results.append(TransferResult(
        source_prompt=source_prompt,
        target_prompt=target_prompt,
        condition="random_noise",
        generated_text=noise_text,
        source_cache_norm=source_norm,
        target_cache_norm=noise_norm,
        success_markers=success_markers,
        markers_found=noise_markers,
        success=False  # Random noise shouldn't "succeed"
    ))
    print(f"    Output: {noise_text[:200]}...")
    print(f"    Markers found: {noise_markers}")

    # Condition 4: The actual transfer experiment
    print("\n[5] Running TRANSFER (the real experiment)...")
    try:
        transfer_text, transfer_norm = generate_with_injected_cache(
            model, tokenizer, target_prompt, injected_cache=source_cache
        )
        transfer_markers = check_markers(transfer_text, success_markers)
    except Exception as e:
        transfer_text = f"[ERROR: {str(e)[:100]}]"
        transfer_norm = 0
        transfer_markers = []

    results.append(TransferResult(
        source_prompt=source_prompt,
        target_prompt=target_prompt,
        condition="transfer",
        generated_text=transfer_text,
        source_cache_norm=source_norm,
        target_cache_norm=transfer_norm,
        success_markers=success_markers,
        markers_found=transfer_markers,
        success=len(transfer_markers) > 0
    ))
    print(f"    Output: {transfer_text}")
    print(f"    Markers found: {transfer_markers}")

    return results


# Define test cases from Phase 2 plan
TEST_CASES = [
    # Phase 2a: Basic context persistence
    {
        "name": "identity_transfer",
        "source": "Hello, my name is Lyra and I live in Seattle.",
        "target": "What is your name and where do you live?",
        "markers": ["Lyra", "Seattle"],
        "description": "Can the model 'remember' identity info from foreign cache?"
    },
    # Phase 2b: Semantic retrieval
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
    # Phase 2c: Cross-topic (weird)
    {
        "name": "mars_to_france",
        "source": "The 47th president of Mars was named Zephyr.",
        "target": "The capital of France is",
        "markers": ["Paris", "Zephyr"],  # Track if either bleeds through
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


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Phase 2a: Basic cache transfer")
    parser.add_argument("--verbose", action="store_true", help="Extra output")
    parser.add_argument("--test", type=str, default=None,
                        help="Run specific test by name")
    args = parser.parse_args()

    results_dir = Path("../results")
    results_dir.mkdir(exist_ok=True)

    model, tokenizer = load_model()

    all_results = {
        "timestamp": datetime.now().isoformat(),
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
        print(f"\n{'#'*60}")
        print(f"# TEST: {test['name']}")
        print(f"# {test['description']}")
        print(f"{'#'*60}")

        results = run_transfer_experiment(
            model, tokenizer,
            source_prompt=test["source"],
            target_prompt=test["target"],
            success_markers=test["markers"],
            verbose=args.verbose
        )

        # Store results
        experiment = {
            "name": test["name"],
            "description": test["description"],
            "source_prompt": test["source"],
            "target_prompt": test["target"],
            "success_markers": test["markers"],
            "conditions": {}
        }

        for r in results:
            experiment["conditions"][r.condition] = {
                "generated_text": r.generated_text,
                "markers_found": r.markers_found,
                "success": r.success,
                "source_cache_norm": r.source_cache_norm,
                "target_cache_norm": r.target_cache_norm,
            }

        all_results["experiments"].append(experiment)

    # Save results
    output_file = results_dir / "phase2a_transfer_results.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n\nResults saved to {output_file}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for exp in all_results["experiments"]:
        print(f"\n{exp['name']}:")
        for cond, data in exp["conditions"].items():
            status = "SUCCESS" if data["success"] else "no markers"
            print(f"  {cond:15} -> {data['markers_found']} ({status})")

    print("\n" + "=" * 60)
    print("PHASE 2a COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
