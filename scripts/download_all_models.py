#!/usr/bin/env python3
"""Download all Campaign 2 models to /models/huggingface."""
import os, sys, time
os.environ["HF_HOME"] = "/models/huggingface"

from huggingface_hub import snapshot_download

MODELS = [
    # Already downloaded: TinyLlama
    # Qwen Scale Ladder
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    "Qwen/Qwen2.5-32B-Instruct",
    # Cross-architecture
    "google/gemma-2-2b-it",
    "microsoft/Phi-3.5-mini-instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "meta-llama/Llama-3.1-8B-Instruct",
    "google/gemma-2-9b-it",
    # DeepSeek distill
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    # Large - download last
    "meta-llama/Llama-3.1-70B-Instruct",
]

for i, model in enumerate(MODELS):
    print(f"\n{'='*60}")
    print(f"[{i+1}/{len(MODELS)}] Downloading {model}...")
    print(f"{'='*60}")
    start = time.time()
    try:
        snapshot_download(model)
        elapsed = time.time() - start
        print(f"  DONE in {elapsed:.0f}s")
    except Exception as e:
        print(f"  FAILED: {e}")
        continue

print(f"\n{'='*60}")
print("All downloads complete!")
print(f"{'='*60}")
