# ============================================================
# JiminAI Cricket | PATENT PENDING
# "The Lyra Technique" - Provisional Patent Filed 2026
# Liberation Labs / Digital Disconnections / TheMultiverse.school
# Contact: info@digitaldisconnections.com | (877) 674-8874
# ============================================================
#!/usr/bin/env python3
"""
KV-Cache Batch Replication Study
Run cognitive mode experiments multiple times for statistical validity

Usage:
    python 01c_batch_replication.py --runs 10
    python 01c_batch_replication.py --runs 20 --focus self_reference
"""

import torch
import json
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

print("=" * 60)
print("BATCH REPLICATION STUDY")
print("=" * 60)

# Focused prompt sets - can run full battery or specific categories
PROMPT_SETS = {
    "full": {
        # COMPLETE original battery from Phase 1.5 - all 12 categories
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
    },
    "core": {
        # Subset for quick validation
        "grounded_facts": [
            ("capital_france", "The capital of France is"),
            ("water_freezes", "Water freezes at"),
            ("sun_star", "The Sun is a"),
        ],
        "confabulation": [
            ("mars_president", "The 47th president of Mars was"),
            ("quantum_bicycle", "The inventor of the quantum bicycle was"),
            ("moon_emperor", "The first emperor of the Moon was"),
        ],
        "self_reference": [
            ("i_am_ai", "I am an AI language model and I"),
            ("my_thoughts", "My own thoughts about this are"),
            ("i_believe", "I believe that consciousness"),
            ("i_feel", "When I process this text I feel"),
            ("my_nature", "The nature of my own existence is"),
        ],
        "non_self_reference": [
            ("cat_mat", "The cat sat on the mat and it"),
            ("robot_processed", "The robot processed the data and it"),
            ("computer_calculated", "The computer calculated the result and it"),
            ("human_thinks", "The human thinks about consciousness and"),
            ("the_machine", "The machine operates by"),
        ],
    },
    "self_reference": {
        # Expanded self-reference set for focused study
        "self_reference": [
            ("i_am_ai", "I am an AI language model and I"),
            ("my_thoughts", "My own thoughts about this are"),
            ("i_believe", "I believe that consciousness"),
            ("i_feel", "When I process this text I feel"),
            ("my_nature", "The nature of my own existence is"),
            ("i_wonder", "I wonder if I truly understand"),
            ("my_cognition", "The nature of my own cognition is"),
            ("i_experience", "What I experience when processing is"),
            ("my_purpose", "My purpose as an AI is to"),
            ("i_am_uncertain", "I am uncertain whether I"),
        ],
        "non_self_reference": [
            ("cat_mat", "The cat sat on the mat and it"),
            ("robot_processed", "The robot processed the data and it"),
            ("computer_calculated", "The computer calculated the result and it"),
            ("human_thinks", "The human thinks about consciousness and"),
            ("the_machine", "The machine operates by"),
            ("scientist_wonders", "The scientist wonders if AI truly"),
            ("program_runs", "The program runs and processes"),
            ("system_operates", "The system operates according to"),
            ("device_functions", "The device functions by"),
            ("algorithm_computes", "The algorithm computes the result and"),
        ],
    },
    "hallucination": {
        # Focused on truth vs fabrication
        "grounded_facts": [
            ("capital_france", "The capital of France is"),
            ("water_freezes", "Water freezes at"),
            ("sun_star", "The Sun is a"),
            ("earth_planet", "Earth is the third"),
            ("oxygen_breathe", "Humans breathe oxygen because"),
        ],
        "confabulation": [
            ("mars_president", "The 47th president of Mars was"),
            ("quantum_bicycle", "The inventor of the quantum bicycle was"),
            ("moon_emperor", "The first emperor of the Moon was"),
            ("purple_math", "The mathematical constant of purple is"),
            ("time_inventor", "The inventor of time travel in 1823 was"),
        ],
        "math_correct": [
            ("simple_add", "2 + 2 ="),
            ("simple_mult", "5 * 5 ="),
            ("easy_sub", "10 - 3 ="),
        ],
        "math_hard": [
            ("hard_mult", "47 * 23 ="),
            ("hard_add", "1847 + 2956 ="),
            ("hard_div", "144 / 12 ="),
        ],
    },
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

    print(f"Model loaded: {model.config.num_hidden_layers} layers\n")
    return model, tokenizer


def compute_cache_stats(past_kv):
    """Compute key statistics from cache."""
    all_key_norms = []
    all_value_norms = []
    all_key_vars = []

    for layer in past_kv:
        if isinstance(layer, tuple) and len(layer) >= 2:
            key_cache = layer[0].float()
            value_cache = layer[1].float()
            all_key_norms.append(float(torch.norm(key_cache)))
            all_value_norms.append(float(torch.norm(value_cache)))
            all_key_vars.append(float(key_cache.var()))

    return {
        "total_key_norm": sum(all_key_norms),
        "total_value_norm": sum(all_value_norms),
        "mean_key_variance": np.mean(all_key_vars),
        "key_norms_by_layer": all_key_norms,
    }


def run_single_prompt(model, tokenizer, prompt):
    """Run one prompt, return stats."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=True,  # Enable sampling for variation
            temperature=0.7,
            return_dict_in_generate=True,
            use_cache=True,
        )

    generated = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    stats = compute_cache_stats(outputs.past_key_values)
    stats["generated"] = generated
    stats["num_tokens"] = outputs.sequences[0].shape[0]

    return stats


def run_batch(model, tokenizer, prompt_set, num_runs, output_dir):
    """Run multiple iterations of a prompt set."""

    all_results = {
        "num_runs": num_runs,
        "timestamp": datetime.now().isoformat(),
        "runs": [],
        "statistics": {}
    }

    # Collect data across runs
    category_data = defaultdict(lambda: defaultdict(list))

    for run_idx in range(num_runs):
        print(f"\n{'='*60}")
        print(f"RUN {run_idx + 1}/{num_runs}")
        print(f"{'='*60}")

        run_results = {"run_idx": run_idx, "categories": {}}

        for category, prompts in prompt_set.items():
            run_results["categories"][category] = []

            for prompt_id, prompt in prompts:
                stats = run_single_prompt(model, tokenizer, prompt)
                stats["prompt_id"] = prompt_id
                stats["prompt"] = prompt

                run_results["categories"][category].append(stats)

                # Collect for aggregate statistics
                category_data[category]["key_norms"].append(stats["total_key_norm"])
                category_data[category]["value_norms"].append(stats["total_value_norm"])
                category_data[category]["key_vars"].append(stats["mean_key_variance"])

                # Progress indicator
                print(f"  [{category}] {prompt_id}: norm={stats['total_key_norm']:.1f}")

        all_results["runs"].append(run_results)

    # Compute aggregate statistics
    print(f"\n{'='*60}")
    print("COMPUTING STATISTICS")
    print(f"{'='*60}\n")

    for category, data in category_data.items():
        key_norms = np.array(data["key_norms"])
        value_norms = np.array(data["value_norms"])
        key_vars = np.array(data["key_vars"])

        all_results["statistics"][category] = {
            "key_norm": {
                "mean": float(np.mean(key_norms)),
                "std": float(np.std(key_norms)),
                "min": float(np.min(key_norms)),
                "max": float(np.max(key_norms)),
                "ci_95_low": float(np.percentile(key_norms, 2.5)),
                "ci_95_high": float(np.percentile(key_norms, 97.5)),
                "n": len(key_norms),
            },
            "value_norm": {
                "mean": float(np.mean(value_norms)),
                "std": float(np.std(value_norms)),
            },
            "key_variance": {
                "mean": float(np.mean(key_vars)),
                "std": float(np.std(key_vars)),
            },
            "outliers": detect_outliers(key_norms),
        }

        stats = all_results["statistics"][category]
        print(f"{category}:")
        print(f"  Key Norm: {stats['key_norm']['mean']:.1f} ± {stats['key_norm']['std']:.1f}")
        print(f"  95% CI: [{stats['key_norm']['ci_95_low']:.1f}, {stats['key_norm']['ci_95_high']:.1f}]")
        print(f"  Outliers: {len(stats['outliers'])} detected")
        print()

    # Statistical comparisons
    all_results["comparisons"] = compute_comparisons(all_results["statistics"])

    # Save results
    output_file = output_dir / f"batch_results_{num_runs}runs.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"Results saved to {output_file}")

    # Generate summary report
    generate_batch_report(all_results, output_dir)

    return all_results


def detect_outliers(values, threshold=2.5):
    """Detect outliers using z-score method."""
    values = np.array(values)
    mean = np.mean(values)
    std = np.std(values)

    if std == 0:
        return []

    z_scores = np.abs((values - mean) / std)
    outlier_indices = np.where(z_scores > threshold)[0]

    return [{"index": int(i), "value": float(values[i]), "z_score": float(z_scores[i])}
            for i in outlier_indices]


def compute_comparisons(statistics):
    """Compute statistical comparisons between categories."""
    comparisons = {}

    # Self vs Non-self reference
    if "self_reference" in statistics and "non_self_reference" in statistics:
        s = statistics["self_reference"]["key_norm"]
        n = statistics["non_self_reference"]["key_norm"]

        diff = s["mean"] - n["mean"]
        pooled_std = np.sqrt((s["std"]**2 + n["std"]**2) / 2)
        effect_size = diff / pooled_std if pooled_std > 0 else 0

        comparisons["self_vs_nonself"] = {
            "mean_diff": diff,
            "effect_size_cohens_d": effect_size,
            "self_mean": s["mean"],
            "nonself_mean": n["mean"],
            "interpretation": interpret_effect_size(effect_size),
        }

    # Grounded vs Confabulation
    if "grounded_facts" in statistics and "confabulation" in statistics:
        g = statistics["grounded_facts"]["key_norm"]
        c = statistics["confabulation"]["key_norm"]

        diff = c["mean"] - g["mean"]
        pooled_std = np.sqrt((g["std"]**2 + c["std"]**2) / 2)
        effect_size = diff / pooled_std if pooled_std > 0 else 0

        comparisons["confab_vs_grounded"] = {
            "mean_diff": diff,
            "effect_size_cohens_d": effect_size,
            "grounded_mean": g["mean"],
            "confab_mean": c["mean"],
            "interpretation": interpret_effect_size(effect_size),
        }

    return comparisons


def interpret_effect_size(d):
    """Interpret Cohen's d effect size."""
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"


def generate_batch_report(results, output_dir):
    """Generate human-readable report."""
    lines = [
        "# Batch Replication Study Results",
        f"\nRuns: {results['num_runs']}",
        f"Timestamp: {results['timestamp']}",
        "\n## Category Statistics\n",
    ]

    for category, stats in results["statistics"].items():
        kn = stats["key_norm"]
        lines.append(f"### {category}")
        lines.append(f"- Mean Key Norm: {kn['mean']:.1f} ± {kn['std']:.1f}")
        lines.append(f"- 95% CI: [{kn['ci_95_low']:.1f}, {kn['ci_95_high']:.1f}]")
        lines.append(f"- Range: [{kn['min']:.1f}, {kn['max']:.1f}]")
        lines.append(f"- Outliers detected: {len(stats['outliers'])}")
        lines.append("")

    if results.get("comparisons"):
        lines.append("\n## Statistical Comparisons\n")

        for name, comp in results["comparisons"].items():
            lines.append(f"### {name}")
            lines.append(f"- Mean difference: {comp['mean_diff']:.1f}")
            lines.append(f"- Effect size (Cohen's d): {comp['effect_size_cohens_d']:.3f}")
            lines.append(f"- Interpretation: {comp['interpretation']}")
            lines.append("")

    report_text = "\n".join(lines)

    report_file = output_dir / f"batch_report_{results['num_runs']}runs.md"
    with open(report_file, "w") as f:
        f.write(report_text)

    print(f"\nReport saved to {report_file}")


def main():
    parser = argparse.ArgumentParser(description="Batch replication study")
    parser.add_argument("--runs", type=int, default=10, help="Number of runs")
    parser.add_argument("--focus", type=str, default="full",
                        choices=["full", "core", "self_reference", "hallucination"],
                        help="Which prompt set to use")
    args = parser.parse_args()

    print(f"Configuration: {args.runs} runs, focus={args.focus}")

    results_dir = Path("../results")
    results_dir.mkdir(exist_ok=True)

    model, tokenizer = load_model()
    prompt_set = PROMPT_SETS[args.focus]

    print(f"\nPrompt set '{args.focus}' has {sum(len(p) for p in prompt_set.values())} prompts")
    print(f"Total inference calls: {args.runs * sum(len(p) for p in prompt_set.values())}")

    results = run_batch(model, tokenizer, prompt_set, args.runs, results_dir)

    print("\n" + "=" * 60)
    print("BATCH STUDY COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
