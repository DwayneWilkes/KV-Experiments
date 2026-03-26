"""Prompt set quality analysis tool.

Analyzes minimal pair datasets for distribution, balance, and coverage
before running expensive experiments. Identifies gaps and recommends
additional prompts for even coverage.

REQUIRES a tokenizer. Does not estimate. All counts are exact.

Metrics computed:
- Length distribution (char, word, TOKEN counts per side — exact via tokenizer)
- Pair balance (token-level matching between positive/negative)
- Vocabulary overlap (Jaccard similarity within pairs)
- Semantic clustering (TF-IDF + KMeans for topic coverage)
- Coverage gaps (underrepresented lengths, topics, structures)

General-purpose. Not KV-cache-specific.
"""

import re
from collections import Counter
from dataclasses import dataclass, asdict, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from kv_verify.prompt_gen import MinimalPair, PairSet


# ================================================================
# PROMPT-LEVEL STATISTICS
# ================================================================

@dataclass
class PromptStats:
    """Statistics for a single prompt string. Token count is exact."""
    char_count: int
    word_count: int
    token_count: int
    unique_word_count: int
    type_token_ratio: float
    sentence_count: int
    avg_word_length: float

    @classmethod
    def from_text(cls, text: str, tokenizer: Callable) -> "PromptStats":
        """Compute stats using the actual tokenizer. No proxies."""
        words = text.split()
        word_count = len(words)
        unique_words = set(w.lower() for w in words)
        unique_word_count = len(unique_words)
        ttr = unique_word_count / max(word_count, 1)
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        avg_word_len = np.mean([len(w) for w in words]) if words else 0

        # Exact token count from the actual tokenizer
        token_ids = tokenizer.encode(text)
        token_count = len(token_ids)

        return cls(
            char_count=len(text),
            word_count=word_count,
            token_count=token_count,
            unique_word_count=unique_word_count,
            type_token_ratio=round(ttr, 4),
            sentence_count=max(len(sentences), 1),
            avg_word_length=round(float(avg_word_len), 2),
        )


# ================================================================
# ANALYSIS FUNCTIONS
# ================================================================

def analyze_length_distribution(ps: PairSet, tokenizer: Callable) -> Dict[str, Any]:
    """Analyze character, word, and TOKEN count distributions.

    Token counts are EXACT via the provided tokenizer.
    """
    pos_stats = [PromptStats.from_text(p.positive, tokenizer) for p in ps.pairs]
    neg_stats = [PromptStats.from_text(p.negative, tokenizer) for p in ps.pairs]

    def _side_stats(stats_list, label):
        chars = [s.char_count for s in stats_list]
        words = [s.word_count for s in stats_list]
        tokens = [s.token_count for s in stats_list]
        return {
            "char_mean": round(float(np.mean(chars)), 1),
            "char_std": round(float(np.std(chars)), 1),
            "char_min": min(chars),
            "char_max": max(chars),
            "word_mean": round(float(np.mean(words)), 1),
            "word_std": round(float(np.std(words)), 1),
            "word_min": min(words),
            "word_max": max(words),
            "token_mean": round(float(np.mean(tokens)), 1),
            "token_std": round(float(np.std(tokens)), 1),
            "token_min": min(tokens),
            "token_max": max(tokens),
        }

    # Per-pair diffs (using exact token counts)
    token_diffs = [abs(p.token_count - n.token_count)
                   for p, n in zip(pos_stats, neg_stats)]
    word_diffs = [abs(p.word_count - n.word_count)
                  for p, n in zip(pos_stats, neg_stats)]
    char_diffs = [abs(p.char_count - n.char_count)
                  for p, n in zip(pos_stats, neg_stats)]

    return {
        "positive": _side_stats(pos_stats, "positive"),
        "negative": _side_stats(neg_stats, "negative"),
        "pair_token_diff_mean": round(float(np.mean(token_diffs)), 2),
        "pair_token_diff_max": max(token_diffs),
        "pair_token_diff_std": round(float(np.std(token_diffs)), 2),
        "pair_word_diff_mean": round(float(np.mean(word_diffs)), 2),
        "pair_word_diff_max": max(word_diffs),
        "pair_char_diff_mean": round(float(np.mean(char_diffs)), 1),
        "pair_char_diff_max": max(char_diffs),
        "token_diffs": token_diffs,  # raw list for downstream analysis
    }


def analyze_pair_balance(ps: PairSet, tokenizer: Callable) -> Dict[str, Any]:
    """Analyze how well-matched positive/negative prompts are at the TOKEN level."""
    token_diffs = []
    for pair in ps.pairs:
        pos_tokens = len(tokenizer.encode(pair.positive))
        neg_tokens = len(tokenizer.encode(pair.negative))
        token_diffs.append(abs(pos_tokens - neg_tokens))

    within_0 = sum(1 for d in token_diffs if d == 0)
    within_1 = sum(1 for d in token_diffs if d <= 1)
    within_2 = sum(1 for d in token_diffs if d <= 2)
    within_3 = sum(1 for d in token_diffs if d <= 3)
    outside_3 = sum(1 for d in token_diffs if d > 3)

    return {
        "token_diff_distribution": {
            "mean": round(float(np.mean(token_diffs)), 2),
            "median": round(float(np.median(token_diffs)), 1),
            "max": max(token_diffs),
            "std": round(float(np.std(token_diffs)), 2),
        },
        "pairs_exact_match": within_0,
        "pairs_within_1_token": within_1,
        "pairs_within_2_tokens": within_2,
        "pairs_within_3_tokens": within_3,
        "pairs_outside_3_tokens": outside_3,
        "balance_pct": round(within_3 / max(len(ps.pairs), 1) * 100, 1),
        "outlier_pairs": [
            {"pair_id": ps.pairs[i].pair_id, "token_diff": token_diffs[i]}
            for i in range(len(token_diffs))
            if token_diffs[i] > 3
        ],
    }


def analyze_vocabulary_overlap(ps: PairSet) -> Dict[str, Any]:
    """Compute Jaccard vocabulary overlap within each pair."""
    jaccards = []
    for pair in ps.pairs:
        pos_words = set(pair.positive.lower().split())
        neg_words = set(pair.negative.lower().split())
        intersection = pos_words & neg_words
        union = pos_words | neg_words
        jaccard = len(intersection) / max(len(union), 1)
        jaccards.append(jaccard)

    return {
        "jaccard_mean": round(float(np.mean(jaccards)), 4),
        "jaccard_std": round(float(np.std(jaccards)), 4),
        "jaccard_min": round(float(np.min(jaccards)), 4),
        "jaccard_max": round(float(np.max(jaccards)), 4),
        "high_overlap_pairs": sum(1 for j in jaccards if j > 0.8),
        "low_overlap_pairs": sum(1 for j in jaccards if j < 0.3),
    }


def analyze_semantic_clusters(
    ps: PairSet,
    n_clusters: int = 5,
) -> Dict[str, Any]:
    """Cluster prompts by topic using TF-IDF + KMeans.

    Identifies whether the dataset covers diverse topics or is
    concentrated in a few areas.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans

    texts = [p.positive for p in ps.pairs]

    if len(texts) < n_clusters:
        return {
            "n_clusters": n_clusters,
            "cluster_sizes": [len(texts)],
            "balance_score": 0.0,
            "note": "Too few prompts for clustering",
        }

    tfidf = TfidfVectorizer(max_features=500, stop_words="english")
    X = tfidf.fit_transform(texts)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X)

    cluster_sizes = [int(np.sum(labels == i)) for i in range(n_clusters)]

    # Balance score: 0 = perfectly balanced, 1 = all in one cluster
    sizes = np.array(cluster_sizes, dtype=float)
    if sizes.sum() > 0:
        proportions = sizes / sizes.sum()
        ideal = 1.0 / n_clusters
        balance_score = float(np.mean(np.abs(proportions - ideal)) / ideal)
    else:
        balance_score = 1.0

    # Top terms per cluster
    top_terms = {}
    feature_names = tfidf.get_feature_names_out()
    for i in range(n_clusters):
        center = km.cluster_centers_[i]
        top_idx = center.argsort()[-5:][::-1]
        top_terms[f"cluster_{i}"] = [feature_names[j] for j in top_idx]

    return {
        "n_clusters": n_clusters,
        "cluster_sizes": cluster_sizes,
        "balance_score": round(balance_score, 3),
        "top_terms": top_terms,
    }


def coverage_gaps(ps: PairSet, tokenizer: Callable) -> Dict[str, Any]:
    """Identify underrepresented areas in the prompt set using exact token counts."""
    pos_tokens = [len(tokenizer.encode(p.positive)) for p in ps.pairs]

    short_pairs = sum(1 for t in pos_tokens if t < 10)
    medium_pairs = sum(1 for t in pos_tokens if 10 <= t <= 30)
    long_pairs = sum(1 for t in pos_tokens if t > 30)

    # Prefix distribution
    prefixes = Counter()
    for p in ps.pairs:
        first_words = " ".join(p.positive.split()[:3]).lower()
        prefixes[first_words] += 1

    # Token count histogram (buckets of 5)
    token_hist = Counter()
    for t in pos_tokens:
        bucket = (t // 5) * 5
        token_hist[bucket] += 1

    # Recommendations
    recommendations = []
    if short_pairs > len(ps.pairs) * 0.5:
        recommendations.append("Dataset skews short (>50% under 10 tokens). Add longer prompts.")
    if long_pairs > len(ps.pairs) * 0.5:
        recommendations.append("Dataset skews long (>50% over 30 tokens). Add shorter prompts.")

    top_prefix = prefixes.most_common(1)[0] if prefixes else ("none", 0)
    if top_prefix[1] > len(ps.pairs) * 0.5:
        recommendations.append(
            f"Prefix '{top_prefix[0]}' dominates ({top_prefix[1]}/{len(ps.pairs)}). "
            f"Diversify opening structure."
        )

    # Check for empty token count buckets in the range
    if pos_tokens:
        min_t, max_t = min(pos_tokens), max(pos_tokens)
        for bucket in range(min_t // 5 * 5, max_t + 5, 5):
            if token_hist.get(bucket, 0) == 0 and min_t <= bucket <= max_t:
                recommendations.append(f"Gap in token range {bucket}-{bucket+4}.")

    if not recommendations:
        recommendations.append("Distribution looks reasonable. No critical gaps found.")

    return {
        "short_pairs": short_pairs,
        "medium_pairs": medium_pairs,
        "long_pairs": long_pairs,
        "token_histogram": dict(sorted(token_hist.items())),
        "prefix_distribution": dict(prefixes.most_common(10)),
        "recommendations": recommendations,
    }


# ================================================================
# FULL ANALYSIS
# ================================================================

@dataclass
class PairSetAnalysis:
    """Complete analysis of a minimal pair set."""
    comparison: str
    n_pairs: int
    length_distribution: Dict[str, Any]
    pair_balance: Dict[str, Any]
    vocabulary_overlap: Dict[str, Any]
    semantic_clusters: Dict[str, Any]
    coverage_gaps: Dict[str, Any]

    def to_dict(self) -> dict:
        d = asdict(self)
        # Remove raw token_diffs list (too verbose for serialization)
        if "token_diffs" in d.get("length_distribution", {}):
            del d["length_distribution"]["token_diffs"]
        return d

    def summary(self) -> str:
        """Human-readable summary with exact metrics."""
        ld = self.length_distribution
        pb = self.pair_balance
        vo = self.vocabulary_overlap
        sc = self.semantic_clusters
        cg = self.coverage_gaps
        lines = [
            f"=== {self.comparison} ({self.n_pairs} pairs) ===",
            f"Tokens: pos {ld['positive']['token_mean']:.0f} +/- {ld['positive']['token_std']:.1f}, "
            f"neg {ld['negative']['token_mean']:.0f} +/- {ld['negative']['token_std']:.1f}",
            f"Pair token diff: mean={ld['pair_token_diff_mean']:.1f}, max={ld['pair_token_diff_max']}",
            f"Balance: {pb['balance_pct']:.0f}% within 3 tokens, "
            f"{pb['pairs_exact_match']} exact matches",
            f"Vocab overlap: Jaccard {vo['jaccard_mean']:.3f} "
            f"(low={vo['low_overlap_pairs']}, high={vo['high_overlap_pairs']})",
            f"Clusters: {sc['n_clusters']}, balance={sc['balance_score']:.2f}",
            f"Coverage: {cg['short_pairs']} short, {cg['medium_pairs']} medium, {cg['long_pairs']} long",
            f"Recommendations: {'; '.join(cg['recommendations'][:3])}",
        ]
        return "\n".join(lines)


def analyze_pair_set(
    ps: PairSet,
    tokenizer: Callable,
    n_clusters: int = 5,
) -> PairSetAnalysis:
    """Run full analysis on a pair set. Requires tokenizer for exact counts."""
    return PairSetAnalysis(
        comparison=ps.comparison,
        n_pairs=len(ps.pairs),
        length_distribution=analyze_length_distribution(ps, tokenizer),
        pair_balance=analyze_pair_balance(ps, tokenizer),
        vocabulary_overlap=analyze_vocabulary_overlap(ps),
        semantic_clusters=analyze_semantic_clusters(ps, n_clusters=n_clusters),
        coverage_gaps=coverage_gaps(ps, tokenizer),
    )
