"""Shared test configuration for kv_verify."""

import os
import warnings

os.environ.setdefault("TMPDIR", "/tmp/claude-1000")

# Suppress expected warnings in test suite
warnings.filterwarnings("ignore", message=".*Number of distinct clusters.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Can't initialize NVML.*", category=UserWarning)


def make_item(cond, prompt="test", n_tokens=50, **feats):
    """Shared helper for creating test dataset items."""
    return {
        "condition": cond,
        "prompt": prompt,
        "features": {"n_tokens": n_tokens, **feats},
    }
