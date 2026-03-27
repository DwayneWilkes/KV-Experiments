"""Prompt gap analysis and generation using information-theoretic methods.

Analogous to content-aware image expansion (seam carving): identifies
the lowest-density regions of the prompt feature space and specifies
exactly what properties new prompts need to maximize dataset coverage.

Methods:
- Token-length gap analysis: KDE over token counts, find low-density bins
- Semantic gap analysis: TF-IDF embedding + KNN density estimation
- Domain coverage: classification into topic bins, entropy maximization
- Effective sample size: accounts for prompt correlation (like design effect)
- Acquisition function: expected information gain per candidate prompt

General-purpose prompt set optimization. Not KV-cache-specific.
"""

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import entropy as scipy_entropy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans

from kv_verify.prompt_gen import MinimalPair, PairSet


@dataclass
class GapSpec:
    """Specification for a prompt that would fill a gap."""
    target_token_count: int
    target_domain: str
    target_cluster: int
    information_gain: float
    description: str


@dataclass
class GapAnalysis:
    """Complete gap analysis with acquisition-ranked specs."""
    total_pairs: int
    effective_n: float
    efficiency: float  # effective_n / total_pairs
    token_length_entropy: float
    token_length_target_entropy: float
    domain_entropy: float
    domain_target_entropy: float
    cluster_entropy: float
    gaps: List[GapSpec]
    token_density: Dict[int, float]
    domain_counts: Dict[str, int]


# ================================================================
# TOKEN-LENGTH ANALYSIS
# ================================================================

def _token_length_density(pairs: List[MinimalPair], tokenizer: Callable) -> Dict[int, float]:
    """Compute token-length density using kernel density estimation.

    Returns dict mapping token count -> density (higher = more covered).
    """
    lengths = []
    for pair in pairs:
        pos_len = len(tokenizer.encode(pair.positive))
        neg_len = len(tokenizer.encode(pair.negative))
        lengths.append(pos_len)
        lengths.append(neg_len)

    if not lengths:
        return {}

    counts = Counter(lengths)
    total = len(lengths)

    # Smoothed density (add-1 Laplace over the range)
    min_len = min(lengths)
    max_len = max(lengths)
    density = {}
    for tok_len in range(min_len - 3, max_len + 4):
        density[tok_len] = (counts.get(tok_len, 0) + 0.1) / (total + 0.1 * (max_len - min_len + 7))

    return density


def _token_entropy(pairs: List[MinimalPair], tokenizer: Callable) -> float:
    """Shannon entropy of the token-length distribution.

    Higher = more uniform = better coverage.
    """
    lengths = [len(tokenizer.encode(p.positive)) for p in pairs]
    if not lengths:
        return 0.0

    counts = Counter(lengths)
    total = len(lengths)
    probs = np.array([c / total for c in counts.values()])
    return float(scipy_entropy(probs, base=2))


# ================================================================
# DOMAIN ANALYSIS
# ================================================================

_DOMAIN_KEYWORDS = {
    "geography": ["capital", "country", "continent", "river", "ocean", "mountain",
                   "desert", "lake", "sea", "peninsula", "island", "strait"],
    "chemistry": ["chemical", "atom", "element", "gas", "boil", "freez", "ph ",
                   "formula", "symbol", "compound", "molecule"],
    "astronomy": ["planet", "star", "sun", "moon", "orbit", "solar", "galaxy",
                   "light year", "saturn", "mars", "jupiter"],
    "biology": ["bone", "heart", "blood", "brain", "teeth", "organ", "chromosome",
                "rib", "cell", "dna", "species", "fungi", "hormone"],
    "history": ["year", "war", "empire", "president", "revolution", "treaty",
                "independence", "ancient", "dynasty", "pharaoh"],
    "culture": ["wrote", "painted", "composed", "sculpted", "language", "religion",
                "dance", "martial", "architecture", "university"],
    "physics": ["speed", "force", "sound", "light", "unit", "resistance", "current",
                "energy", "thermodynamic", "kinetic", "newton"],
    "math": ["sqrt", "prime", "binary", "factorial", "angle", "derivative",
             "hexagon", "percent", "byte", "integral", "vertex", "dodecahedron"],
    "general": [],  # fallback
}


def _classify_domain(text: str) -> str:
    """Classify a prompt into a topic domain."""
    text_lower = text.lower()
    scores = {}
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        if domain == "general":
            continue
        scores[domain] = sum(1 for kw in keywords if kw in text_lower)

    if not scores or max(scores.values()) == 0:
        return "general"
    return max(scores, key=scores.get)


def _domain_entropy(pairs: List[MinimalPair]) -> Tuple[float, Dict[str, int]]:
    """Shannon entropy of domain distribution + counts."""
    domains = [_classify_domain(p.positive) for p in pairs]
    counts = Counter(domains)
    total = len(domains)
    probs = np.array([c / total for c in counts.values()])
    return float(scipy_entropy(probs, base=2)), dict(counts)


# ================================================================
# EFFECTIVE SAMPLE SIZE
# ================================================================

def _effective_n(pairs: List[MinimalPair], tokenizer: Callable) -> float:
    """Compute effective sample size accounting for prompt similarity.

    Uses TF-IDF cosine similarity. Highly similar prompts contribute
    less than independent ones. Like the design effect in survey sampling.

    effective_N = N / (1 + (N-1) * mean_pairwise_correlation)

    If all prompts are identical: effective_N = 1
    If all prompts are independent: effective_N = N
    """
    if len(pairs) < 2:
        return float(len(pairs))

    texts = [p.positive for p in pairs]
    tfidf = TfidfVectorizer(max_features=500, stop_words="english")
    X = tfidf.fit_transform(texts)

    # Mean pairwise cosine similarity (sample for speed)
    n = len(texts)
    rng = np.random.RandomState(42)
    n_samples = min(1000, n * (n - 1) // 2)
    similarities = []
    for _ in range(n_samples):
        i, j = rng.choice(n, 2, replace=False)
        cos = (X[i] @ X[j].T).toarray()[0, 0]
        similarities.append(max(0, cos))  # clamp negative

    mean_rho = float(np.mean(similarities)) if similarities else 0.0

    # Design effect formula
    deff = 1 + (n - 1) * mean_rho
    eff_n = n / max(deff, 1.0)

    return round(eff_n, 1)


# ================================================================
# ACQUISITION FUNCTION: WHERE DO NEW PROMPTS ADD MOST VALUE?
# ================================================================

def _compute_gaps(
    pairs: List[MinimalPair],
    tokenizer: Callable,
    n_target: int = 200,
    n_clusters: int = 8,
) -> List[GapSpec]:
    """Identify gaps and rank by expected information gain.

    Finds the lowest-density regions across three dimensions:
    1. Token length (should be uniform across range)
    2. Domain (should be balanced)
    3. Semantic cluster (should be evenly distributed)

    Returns ranked list of GapSpecs: what new prompts should look like.
    """
    gaps = []

    # 1. Token-length gaps
    density = _token_length_density(pairs, tokenizer)
    if density:
        mean_density = np.mean(list(density.values()))
        for tok_len, d in sorted(density.items()):
            if d < mean_density * 0.5:  # significantly underrepresented
                gain = (mean_density - d) / mean_density
                gaps.append(GapSpec(
                    target_token_count=tok_len,
                    target_domain="any",
                    target_cluster=-1,
                    information_gain=gain,
                    description=f"Token length {tok_len} is underrepresented (density={d:.3f}, mean={mean_density:.3f})",
                ))

    # 2. Domain gaps
    _, domain_counts = _domain_entropy(pairs)
    n_domains = len(_DOMAIN_KEYWORDS) - 1  # exclude "general"
    ideal_per_domain = len(pairs) / n_domains
    for domain in _DOMAIN_KEYWORDS:
        if domain == "general":
            continue
        count = domain_counts.get(domain, 0)
        if count < ideal_per_domain * 0.5:
            deficit = ideal_per_domain - count
            gain = deficit / ideal_per_domain
            gaps.append(GapSpec(
                target_token_count=-1,
                target_domain=domain,
                target_cluster=-1,
                information_gain=gain,
                description=f"Domain '{domain}' has {count} prompts (ideal: {ideal_per_domain:.0f}). Deficit: {deficit:.0f}.",
            ))

    # 3. Semantic cluster gaps
    if len(pairs) >= n_clusters:
        texts = [p.positive for p in pairs]
        tfidf = TfidfVectorizer(max_features=500, stop_words="english")
        X = tfidf.fit_transform(texts)
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        cluster_sizes = Counter(labels)
        ideal_per_cluster = len(pairs) / n_clusters

        # Find the lowest-density cluster and describe it
        feature_names = tfidf.get_feature_names_out()
        for ci in range(n_clusters):
            size = cluster_sizes.get(ci, 0)
            if size < ideal_per_cluster * 0.5:
                center = km.cluster_centers_[ci]
                top_words = [feature_names[j] for j in center.argsort()[-5:][::-1]]
                deficit = ideal_per_cluster - size
                gain = deficit / ideal_per_cluster
                gaps.append(GapSpec(
                    target_token_count=-1,
                    target_domain="any",
                    target_cluster=ci,
                    information_gain=gain,
                    description=f"Cluster {ci} (topics: {', '.join(top_words)}) has {size} prompts (ideal: {ideal_per_cluster:.0f}).",
                ))

    # Sort by information gain (highest first)
    gaps.sort(key=lambda g: -g.information_gain)

    return gaps


# ================================================================
# MAIN ANALYSIS FUNCTION
# ================================================================

def analyze_gaps(
    ps: PairSet,
    tokenizer: Callable,
    n_target: int = 200,
    n_clusters: int = 8,
) -> GapAnalysis:
    """Full gap analysis with acquisition-ranked specs for new prompts.

    Returns GapAnalysis with:
    - effective_n: how many independent prompts we effectively have
    - efficiency: effective_n / total (1.0 = all independent, <1.0 = redundancy)
    - entropies: current vs maximum for each dimension
    - gaps: ranked list of GapSpecs describing what prompts to add
    """
    pairs = ps.pairs

    # Effective sample size
    eff_n = _effective_n(pairs, tokenizer)
    efficiency = eff_n / max(len(pairs), 1)

    # Token-length entropy
    tok_ent = _token_entropy(pairs, tokenizer)
    # Max possible entropy: uniform over observed range
    lengths = [len(tokenizer.encode(p.positive)) for p in pairs]
    n_unique_lengths = len(set(lengths))
    tok_target_ent = math.log2(max(n_unique_lengths, 1))

    # Domain entropy
    dom_ent, domain_counts = _domain_entropy(pairs)
    n_domains = len(set(domain_counts.keys()))
    dom_target_ent = math.log2(max(len(_DOMAIN_KEYWORDS) - 1, 1))

    # Cluster entropy
    if len(pairs) >= n_clusters:
        texts = [p.positive for p in pairs]
        tfidf = TfidfVectorizer(max_features=500, stop_words="english")
        X = tfidf.fit_transform(texts)
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        cluster_counts = Counter(labels)
        probs = np.array([cluster_counts.get(i, 0) / len(pairs) for i in range(n_clusters)])
        clust_ent = float(scipy_entropy(probs, base=2))
    else:
        clust_ent = 0.0

    # Token density
    tok_density = _token_length_density(pairs, tokenizer)

    # Gaps
    gaps = _compute_gaps(pairs, tokenizer, n_target, n_clusters)

    return GapAnalysis(
        total_pairs=len(pairs),
        effective_n=eff_n,
        efficiency=round(efficiency, 3),
        token_length_entropy=round(tok_ent, 3),
        token_length_target_entropy=round(tok_target_ent, 3),
        domain_entropy=round(dom_ent, 3),
        domain_target_entropy=round(dom_target_ent, 3),
        cluster_entropy=round(clust_ent, 3),
        gaps=gaps,
        token_density=tok_density,
        domain_counts=domain_counts,
    )


def print_gap_report(analysis: GapAnalysis) -> str:
    """Human-readable gap analysis report."""
    lines = [
        f"=== GAP ANALYSIS ({analysis.total_pairs} pairs) ===",
        f"",
        f"Effective N: {analysis.effective_n} / {analysis.total_pairs} "
        f"(efficiency: {analysis.efficiency:.0%})",
        f"",
        f"Token-length entropy: {analysis.token_length_entropy:.2f} / "
        f"{analysis.token_length_target_entropy:.2f} (target: uniform)",
        f"Domain entropy:      {analysis.domain_entropy:.2f} / "
        f"{analysis.domain_target_entropy:.2f} (target: uniform)",
        f"Cluster entropy:     {analysis.cluster_entropy:.2f}",
        f"",
        f"Domain distribution:",
    ]
    for domain, count in sorted(analysis.domain_counts.items(), key=lambda x: -x[1]):
        lines.append(f"  {domain:15s}: {count:3d}")

    if analysis.gaps:
        lines.append(f"")
        lines.append(f"Top {min(10, len(analysis.gaps))} gaps to fill (by information gain):")
        for i, gap in enumerate(analysis.gaps[:10]):
            lines.append(f"  {i+1}. [{gap.information_gain:.2f}] {gap.description}")
    else:
        lines.append(f"")
        lines.append(f"No significant gaps found.")

    return "\n".join(lines)
