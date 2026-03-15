# ============================================================
# JiminAI Cricket | PATENT PENDING
# "The Lyra Technique" - Provisional Patent Filed 2026
# Liberation Labs / Digital Disconnections / TheMultiverse.school
# Contact: info@digitaldisconnections.com | (877) 674-8874
# ============================================================
#!/usr/bin/env python3
"""
KV-Cache Inspection Script
Phase 1: Understand cache structure before attempting transfer

This script loads a quantized model and inspects the KV-cache structure.
Designed to run on limited VRAM (6GB).

Usage:
    python 01_cache_inspection.py

Requirements:
    pip install torch transformers bitsandbytes accelerate
"""

import torch
import json
from pathlib import Path

# Check CUDA availability
print("=" * 60)
print("ENVIRONMENT CHECK")
print("=" * 60)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print()

def load_model_quantized(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    """
    Load a small model with 4-bit quantization.
    Start with 1B model to ensure it fits, can scale up later.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    
    print(f"Loading {model_name} with 4-bit quantization...")
    
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
    
    print(f"Model loaded successfully")
    print(f"Model config: {model.config.num_hidden_layers} layers, "
          f"{model.config.num_attention_heads} heads, "
          f"{model.config.hidden_size} hidden dim")
    
    return model, tokenizer


def inspect_cache(model, tokenizer, prompt="The meaning of consciousness is"):
    """
    Generate text and capture KV-cache for inspection.
    """
    print("\n" + "=" * 60)
    print("CACHE INSPECTION")
    print("=" * 60)
    print(f"Prompt: {prompt}")
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate with cache
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
            return_dict_in_generate=True,
            use_cache=True,
            output_hidden_states=False
        )
    
    # Get the cache
    past_kv = outputs.past_key_values
    
    print(f"\nGenerated: {tokenizer.decode(outputs.sequences[0])}")
    
    print(f"\n--- Cache Structure ---")
    print(f"Number of layers: {len(past_kv)}")
    
    # Inspect first layer
    if past_kv and len(past_kv) > 0:
        first_layer = past_kv[0]
        if isinstance(first_layer, tuple) and len(first_layer) >= 2:
            key_cache, value_cache = first_layer[0], first_layer[1]
            print(f"Key cache shape: {key_cache.shape}")
            print(f"Value cache shape: {value_cache.shape}")
            print(f"  - Batch size: {key_cache.shape[0]}")
            print(f"  - Num heads: {key_cache.shape[1]}")
            print(f"  - Sequence length: {key_cache.shape[2]}")
            print(f"  - Head dimension: {key_cache.shape[3]}")
            
            # Calculate total cache size
            total_size = 0
            for layer in past_kv:
                if isinstance(layer, tuple):
                    for tensor in layer:
                        if hasattr(tensor, 'numel'):
                            total_size += tensor.numel() * tensor.element_size()
            print(f"\nTotal cache size: {total_size / 1e6:.2f} MB")
    
    return past_kv


def save_cache_metadata(past_kv, output_path="cache_metadata.json"):
    """
    Save cache structure metadata (not the actual tensors) for analysis.
    """
    metadata = {
        "num_layers": len(past_kv),
        "layers": []
    }
    
    for i, layer in enumerate(past_kv):
        if isinstance(layer, tuple) and len(layer) >= 2:
            layer_info = {
                "layer_idx": i,
                "key_shape": list(layer[0].shape),
                "value_shape": list(layer[1].shape),
                "key_dtype": str(layer[0].dtype),
                "value_dtype": str(layer[1].dtype)
            }
            metadata["layers"].append(layer_info)
    
    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nCache metadata saved to {output_path}")
    return metadata


def main():
    """
    Main entry point for cache inspection.
    """
    print("KV-Cache Inspection Experiment")
    print("=" * 60)
    
    # Try to load model (will fail gracefully if not available)
    try:
        model, tokenizer = load_model_quantized()
    except Exception as e:
        print(f"\nError loading model: {e}")
        print("\nTo run this script, ensure you have:")
        print("  1. Accepted model license at huggingface.co")
        print("  2. Logged in with: huggingface-cli login")
        print("  3. Sufficient VRAM (need ~3GB for 1B model)")
        return
    
    # Inspect cache with different prompts
    prompts = [
        "The meaning of consciousness is",
        "Hello, my name is Lyra and I",
        "When two AI systems communicate they"
    ]
    
    results_dir = Path("../results")
    results_dir.mkdir(exist_ok=True)
    
    for i, prompt in enumerate(prompts):
        cache = inspect_cache(model, tokenizer, prompt)
        save_cache_metadata(cache, results_dir / f"cache_metadata_{i}.json")
    
    print("\n" + "=" * 60)
    print("INSPECTION COMPLETE")
    print("=" * 60)
    print("Next steps:")
    print("  1. Review cache structure in results/cache_metadata_*.json")
    print("  2. Run 02_cache_transfer.py to test same-model transfer")


if __name__ == "__main__":
    main()
