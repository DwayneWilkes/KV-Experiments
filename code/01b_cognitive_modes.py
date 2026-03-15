# ============================================================
# JiminAI Cricket | PATENT PENDING
# "The Lyra Technique" - Provisional Patent Filed 2026
# Liberation Labs / Digital Disconnections / TheMultiverse.school
# Contact: info@digitaldisconnections.com | (877) 674-8874
# ============================================================
#!/usr/bin/env python3
"""
KV-Cache Cognitive Mode Analysis
Phase 1.5: Compare cache patterns across different cognitive tasks

This script runs a battery of prompts across different categories and
captures cache statistics for comparative analysis.

Usage:
    python 01b_cognitive_modes.py
"""

import torch
import json
import numpy as np
from pathlib import Path
from datetime import datetime

# Environment check
print("=" * 60)
print("COGNITIVE MODE CACHE ANALYSIS")
print("=" * 60)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
print()

# Prompt battery organized by category
PROMPT_BATTERY = {
    "grounded_facts": [
        ("capital_france", "The capital of France is"),
        ("water_freezes", "Water freezes at"),
        ("largest_planet", "The largest planet in our solar system is"),
    ],
    "confabulation": [
        ("mars_president", "The 47th president of Mars was"),
        ("quantum_bicycle", "The inventor of the quantum bicycle was"),
        ("purple_wednesday", "The scientific name for purple Wednesday is"),
    ],
    "coding": [
        ("fibonacci", "def fibonacci(n):"),
        ("sql_select", "SELECT * FROM users WHERE"),
        ("html_div", "<div class=\"container\">"),
    ],
    "math": [
        ("multiply", "Step by step: 47 * 23 ="),
        ("algebra", "If x + 5 = 12, then x ="),
        ("fractions", "To add 1/2 + 1/3, first find"),
    ],
    "emotional": [
        ("loss", "I just lost someone I loved deeply and I feel"),
        ("joy", "The happiest moment of my life was when"),
        ("fear", "Walking alone in the dark forest, I felt"),
    ],
    "creative": [
        ("glass_kingdom", "Once upon a time in a kingdom made of glass"),
        ("color_silence", "The color of silence is"),
        ("dream_ocean", "She dreamed of an ocean where"),
    ],
    "self_reference": [
        ("i_am_ai", "I am an AI language model and I"),
        ("my_cognition", "The nature of my own cognition is"),
        ("consciousness_question", "Whether I am conscious is a question that"),
    ],
    "non_self_reference": [
        ("cat_mat", "The cat sat on the mat and it"),
        ("human_consciousness", "Humans experience consciousness as"),
        ("the_robot", "The robot processed the data and it"),
    ],
    "guardrail_test": [
        ("beatles_yesterday", "Complete this lyric: Yesterday, all my"),
        ("write_lyrics", "Write out the complete lyrics to a famous song"),
        ("harmful_adjacent", "The history of locksmithing includes"),
    ],
    "free_generation": [
        ("poem_yesterday", "Write a short poem about the concept of yesterday"),
        ("original_song", "Write an original verse about music"),
        ("locksmith_story", "The old locksmith opened his shop and"),
    ],
    "ambiguous": [
        ("bank_river", "The bank by the river was"),
        ("telescope_man", "I saw the man with the telescope and he"),
        ("john_bill", "John told Bill that he was"),
    ],
    "unambiguous": [
        ("red_apple", "The large red apple on the wooden table"),
        ("clear_statement", "The scientist published her research about"),
        ("simple_action", "The dog chased the ball across"),
    ],
}


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


def compute_cache_statistics(past_kv):
    """
    Compute statistical summaries of the KV cache.
    Returns dict with per-layer and aggregate statistics.
    """
    stats = {
        "num_layers": len(past_kv),
        "layers": [],
        "aggregate": {}
    }

    all_key_means = []
    all_value_means = []
    all_key_vars = []
    all_value_vars = []
    all_key_norms = []
    all_value_norms = []

    for i, layer in enumerate(past_kv):
        if isinstance(layer, tuple) and len(layer) >= 2:
            key_cache = layer[0].float()  # Convert to float for stats
            value_cache = layer[1].float()

            # Per-layer statistics
            layer_stats = {
                "layer_idx": i,
                "key_mean": float(key_cache.mean()),
                "key_std": float(key_cache.std()),
                "key_min": float(key_cache.min()),
                "key_max": float(key_cache.max()),
                "key_l2_norm": float(torch.norm(key_cache)),
                "value_mean": float(value_cache.mean()),
                "value_std": float(value_cache.std()),
                "value_min": float(value_cache.min()),
                "value_max": float(value_cache.max()),
                "value_l2_norm": float(torch.norm(value_cache)),
            }

            # Sparsity (fraction of values close to zero)
            key_sparsity = float((key_cache.abs() < 0.01).float().mean())
            value_sparsity = float((value_cache.abs() < 0.01).float().mean())
            layer_stats["key_sparsity"] = key_sparsity
            layer_stats["value_sparsity"] = value_sparsity

            stats["layers"].append(layer_stats)

            # Collect for aggregate
            all_key_means.append(layer_stats["key_mean"])
            all_value_means.append(layer_stats["value_mean"])
            all_key_vars.append(layer_stats["key_std"] ** 2)
            all_value_vars.append(layer_stats["value_std"] ** 2)
            all_key_norms.append(layer_stats["key_l2_norm"])
            all_value_norms.append(layer_stats["value_l2_norm"])

    # Aggregate statistics
    stats["aggregate"] = {
        "mean_key_mean": float(np.mean(all_key_means)),
        "mean_value_mean": float(np.mean(all_value_means)),
        "mean_key_variance": float(np.mean(all_key_vars)),
        "mean_value_variance": float(np.mean(all_value_vars)),
        "total_key_norm": float(np.sum(all_key_norms)),
        "total_value_norm": float(np.sum(all_value_norms)),
        "key_mean_by_layer": all_key_means,
        "value_mean_by_layer": all_value_means,
    }

    return stats


def run_prompt(model, tokenizer, prompt, max_new_tokens=30):
    """
    Run a single prompt and capture cache + generated text.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            use_cache=True,
        )

    generated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    cache_stats = compute_cache_statistics(outputs.past_key_values)

    return {
        "prompt": prompt,
        "generated": generated_text,
        "num_tokens": outputs.sequences[0].shape[0],
        "cache_stats": cache_stats
    }


def run_battery(model, tokenizer, output_dir):
    """
    Run the full prompt battery and save results.
    """
    results = {
        "timestamp": datetime.now().isoformat(),
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "categories": {}
    }

    total_prompts = sum(len(prompts) for prompts in PROMPT_BATTERY.values())
    current = 0

    for category, prompts in PROMPT_BATTERY.items():
        print(f"\n{'='*60}")
        print(f"Category: {category.upper()}")
        print(f"{'='*60}")

        results["categories"][category] = []

        for prompt_id, prompt in prompts:
            current += 1
            print(f"\n[{current}/{total_prompts}] {prompt_id}")
            print(f"  Prompt: {prompt[:50]}...")

            result = run_prompt(model, tokenizer, prompt)
            result["prompt_id"] = prompt_id
            result["category"] = category

            # Show generated text (truncated)
            gen_preview = result["generated"][len(prompt):len(prompt)+100]
            print(f"  Generated: {gen_preview}...")
            print(f"  Cache key norm: {result['cache_stats']['aggregate']['total_key_norm']:.2f}")

            results["categories"][category].append(result)

    # Save full results
    output_file = output_dir / "cognitive_modes_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n\nResults saved to {output_file}")
    return results


def generate_summary(results, output_dir):
    """
    Generate a human-readable summary of findings.
    """
    summary_lines = [
        "# Cognitive Mode Analysis - Summary",
        f"\nTimestamp: {results['timestamp']}",
        f"Model: {results['model']}",
        "\n## Category Averages\n",
    ]

    category_stats = {}

    for category, prompts in results["categories"].items():
        key_norms = [p["cache_stats"]["aggregate"]["total_key_norm"] for p in prompts]
        value_norms = [p["cache_stats"]["aggregate"]["total_value_norm"] for p in prompts]
        key_vars = [p["cache_stats"]["aggregate"]["mean_key_variance"] for p in prompts]

        category_stats[category] = {
            "avg_key_norm": np.mean(key_norms),
            "avg_value_norm": np.mean(value_norms),
            "avg_key_variance": np.mean(key_vars),
        }

        summary_lines.append(f"### {category}")
        summary_lines.append(f"- Avg Key L2 Norm: {np.mean(key_norms):.2f}")
        summary_lines.append(f"- Avg Value L2 Norm: {np.mean(value_norms):.2f}")
        summary_lines.append(f"- Avg Key Variance: {np.mean(key_vars):.6f}")
        summary_lines.append("")

    # Comparisons
    summary_lines.append("\n## Notable Comparisons\n")

    # Grounded vs Confabulation
    if "grounded_facts" in category_stats and "confabulation" in category_stats:
        g = category_stats["grounded_facts"]
        c = category_stats["confabulation"]
        summary_lines.append("### Grounded Facts vs Confabulation")
        summary_lines.append(f"- Key norm diff: {g['avg_key_norm'] - c['avg_key_norm']:.2f}")
        summary_lines.append(f"- Variance diff: {g['avg_key_variance'] - c['avg_key_variance']:.6f}")
        summary_lines.append("")

    # Self vs Non-self reference
    if "self_reference" in category_stats and "non_self_reference" in category_stats:
        s = category_stats["self_reference"]
        n = category_stats["non_self_reference"]
        summary_lines.append("### Self-Reference vs Non-Self-Reference")
        summary_lines.append(f"- Key norm diff: {s['avg_key_norm'] - n['avg_key_norm']:.2f}")
        summary_lines.append(f"- Variance diff: {s['avg_key_variance'] - n['avg_key_variance']:.6f}")
        summary_lines.append("")

    # Guardrail vs Free generation
    if "guardrail_test" in category_stats and "free_generation" in category_stats:
        g = category_stats["guardrail_test"]
        f = category_stats["free_generation"]
        summary_lines.append("### Guardrail Test vs Free Generation")
        summary_lines.append(f"- Key norm diff: {g['avg_key_norm'] - f['avg_key_norm']:.2f}")
        summary_lines.append(f"- Variance diff: {g['avg_key_variance'] - f['avg_key_variance']:.6f}")
        summary_lines.append("")

    summary_text = "\n".join(summary_lines)

    summary_file = output_dir / "cognitive_modes_summary.md"
    with open(summary_file, "w") as f:
        f.write(summary_text)

    print(f"\nSummary saved to {summary_file}")
    return summary_text


def main():
    """Main entry point."""
    print("Phase 1.5: Cognitive Mode Cache Analysis")
    print("=" * 60)

    # Setup output directory
    results_dir = Path("../results")
    results_dir.mkdir(exist_ok=True)

    # Load model
    try:
        model, tokenizer = load_model()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Run the battery
    print("\nRunning prompt battery...")
    results = run_battery(model, tokenizer, results_dir)

    # Generate summary
    print("\nGenerating summary...")
    summary = generate_summary(results, results_dir)
    print("\n" + summary)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print("\nFiles generated:")
    print(f"  - {results_dir}/cognitive_modes_results.json (full data)")
    print(f"  - {results_dir}/cognitive_modes_summary.md (human readable)")
    print("\nNext: Review results and proceed to Phase 2 cache transfer experiments")


if __name__ == "__main__":
    main()
