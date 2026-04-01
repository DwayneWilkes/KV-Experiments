"""Shared type definitions for kv_verify.

Enums and dataclasses used across all modules. No external dependencies
beyond stdlib + numpy.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class Verdict(Enum):
    """Verification verdict for a claim."""
    CONFIRMED = "confirmed"
    FALSIFIED = "falsified"
    WEAKENED = "weakened"
    STRENGTHENED = "strengthened"
    INDETERMINATE = "indeterminate"
    BLOCKED = "blocked"


class Severity(Enum):
    """Severity of the finding being verified."""
    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"


@dataclass
class FeatureVector:
    """Extracted KV-cache features for a single observation.

    Core features (always present):
        norm_per_token: total Frobenius norm / total tokens
        key_rank: effective rank = exp(spectral_entropy)  [concordance: eff_rank]
        key_entropy: Shannon entropy of layer norm distribution  [concordance: layer_norm_entropy]
        norm: total Frobenius norm across all layers
        n_tokens: total token count (input + generated)
        n_generated: number of generated tokens
        n_input_tokens: number of input/prompt tokens

    Per-layer arrays:
        layer_norms: per-layer Frobenius norms
        layer_ranks: per-layer effective ranks
    """
    # Core features
    norm_per_token: float
    key_rank: float
    key_entropy: float
    norm: float
    n_tokens: int
    n_generated: int
    n_input_tokens: int
    layer_norms: List[float]
    layer_ranks: List[float]
    # Extended features (computed when extended=True)
    spectral_entropy: Optional[float] = None
    angular_spread: Optional[float] = None
    norm_variance: Optional[float] = None
    gen_delta: Optional[float] = None
    layer_uniformity: Optional[float] = None
    head_variance: Optional[float] = None
    max_layer_rank: Optional[float] = None
    top_sv_ratio: Optional[float] = None
    rank_10: Optional[float] = None
    layer_variance: Optional[float] = None
    # Metadata
    model_id: str = ""
    prompt_hash: str = ""
    condition: str = ""
    prompt_idx: int = -1
    run_idx: int = 0


@dataclass
class ClassificationResult:
    """Result of a binary classification evaluation."""
    auroc: float
    auroc_ci_lower: float
    auroc_ci_upper: float
    p_value: float
    p_value_corrected: Optional[float]
    null_mean: float
    null_std: float
    n_positive: int
    n_negative: int
    n_groups: int
    effect_sizes: Dict[str, float]
    cv_method: str
    group_scheme: str
    bootstrap_n: int
    permutation_n: int
    features_used: List[str]


@dataclass
class FWLResult:
    """Result of FWL residualization analysis."""
    auroc_original: float
    auroc_fwl_norm: float
    auroc_fwl_ngen: float
    auroc_fwl_both: float
    r_squared_per_feature: Dict[str, List[float]]
    p_value_fwl_both: float
    leakage_method: str


@dataclass
class ClaimVerification:
    """Verification result for a single claim."""
    claim_id: str
    claim_text: str
    paper_section: str
    finding_id: str
    severity: Severity
    null_hypothesis: str
    experiment_description: str
    verdict: Verdict
    evidence_summary: str
    original_value: Optional[Any]
    corrected_value: Optional[Any]
    visualization_paths: List[str]
    gpu_time_seconds: float
    stats: Dict[str, Any]


@dataclass
class StageCheckpoint:
    """Checkpoint for pipeline restart."""
    stage: str
    completed_items: List[str]
    cache_paths: Dict[str, str]
    timestamp: str
    gpu_hours_elapsed: float
