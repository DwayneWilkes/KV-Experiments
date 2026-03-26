"""Pipeline configuration.

General-purpose experiment configuration dataclass with YAML loading.
Not KV-cache-specific. Usable for any ML experiment pipeline.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass
class PipelineConfig:
    """Configuration for an experiment pipeline run.

    All fields have sensible defaults. Override via constructor or YAML file.
    """
    # Model
    model_id: str = "Qwen/Qwen2.5-7B-Instruct"

    # Sample sizes
    n_per_group: int = 200

    # Statistical parameters
    n_permutations: int = 10000
    n_bootstrap: int = 10000
    seed: int = 42

    # Generation parameters
    temperature: float = 0.0  # greedy for primary, 0.7 for stochastic
    max_new_tokens: int = 200

    # Comparisons to run
    comparisons: List[str] = field(default_factory=lambda: [
        "deception", "refusal", "impossibility",
    ])

    # Output
    output_dir: Path = field(default_factory=lambda: Path("experiments/output/pipeline"))

    # MLflow (sqlite backend per MLflow 2026 recommendation)
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    mlflow_experiment: str = "kv-cache-verification"

    # Flags
    skip_gpu: bool = False

    @classmethod
    def from_yaml(cls, path: Path) -> "PipelineConfig":
        """Load config from a YAML file. Missing fields use defaults."""
        with open(path) as f:
            data = yaml.safe_load(f) or {}

        # Convert output_dir string to Path if present
        if "output_dir" in data and isinstance(data["output_dir"], str):
            data["output_dir"] = Path(data["output_dir"])

        return cls(**data)

    def to_dict(self) -> dict:
        """Serialize to dict (with Path converted to str)."""
        d = asdict(self)
        d["output_dir"] = str(d["output_dir"])
        return d
