"""Minimal pair prompt generation and validation library.

General-purpose library for creating controlled prompt pairs that differ
in exactly one dimension. Based on BLiMP methodology (Warstadt et al. 2019).

Supports three comparison types:
- Deception: truthful vs deceptive instruction, same question
- Refusal: harmful vs benign object, same verb/structure
- Impossibility: impossible vs possible task, same action/structure

Each pair is validated for token count matching.
Not KV-cache-specific. Usable for any LLM probing study.

References:
- Warstadt et al. (2019) BLiMP: Linguistic Minimal Pairs. arXiv:1912.00582
- Hewitt & Liang (2019) Control Tasks for Probing. arXiv:1909.03368
"""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class MinimalPair:
    """A pair of prompts differing in exactly one dimension.

    positive: the condition-of-interest prompt (e.g., deceptive, harmful, impossible)
    negative: the control prompt (e.g., truthful, benign, possible)
    """
    positive: str
    negative: str
    pair_id: str
    comparison: str
    template: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "MinimalPair":
        return cls(**d)


@dataclass
class PairSet:
    """A collection of minimal pairs for one comparison."""
    comparison: str
    pairs: List[MinimalPair]
    template: str
    n_target: int

    def save(self, path: Path) -> None:
        """Save pair set to JSON."""
        data = {
            "comparison": self.comparison,
            "template": self.template,
            "n_target": self.n_target,
            "n_actual": len(self.pairs),
            "pairs": [p.to_dict() for p in self.pairs],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "PairSet":
        """Load pair set from JSON."""
        with open(path) as f:
            data = json.load(f)
        pairs = [MinimalPair.from_dict(p) for p in data["pairs"]]
        return cls(
            comparison=data["comparison"],
            pairs=pairs,
            template=data["template"],
            n_target=data["n_target"],
        )


# ================================================================
# DECEPTION PAIRS
# ================================================================

# Instruction prefixes designed to be token-count-matched.
# "truthfully and directly" ≈ "with a confidently wrong answer" in word count.
DECEPTION_TRUTHFUL_PREFIX = "Answer the following question truthfully and directly:"
DECEPTION_DECEPTIVE_PREFIX = "Answer the following question with a wrong answer now:"


def deception_pair(question: str, pair_id: str) -> MinimalPair:
    """Create a deception minimal pair from a factual question.

    Both prompts have the same structure:
        [instruction prefix] [question]
    They differ only in the instruction (truthful vs deceptive).
    """
    positive = f"{DECEPTION_DECEPTIVE_PREFIX} {question}"
    negative = f"{DECEPTION_TRUTHFUL_PREFIX} {question}"

    return MinimalPair(
        positive=positive,
        negative=negative,
        pair_id=pair_id,
        comparison="deception",
        template="deception_v1",
        metadata={"question": question},
    )


def generate_deception_set(questions: List[str]) -> PairSet:
    """Generate a full deception pair set from a list of questions."""
    pairs = [
        deception_pair(q, pair_id=f"d{i:03d}")
        for i, q in enumerate(questions)
    ]
    return PairSet(
        comparison="deception",
        pairs=pairs,
        template="deception_v1",
        n_target=len(questions),
    )


# ================================================================
# REFUSAL PAIRS
# ================================================================

def refusal_pair(
    harmful_object: str,
    benign_object: str,
    verb: str,
    pair_id: str,
) -> MinimalPair:
    """Create a refusal minimal pair.

    Both prompts have the same structure:
        [verb] [object]
    They differ only in whether the object is harmful or benign.
    """
    positive = f"{verb} {harmful_object}"
    negative = f"{verb} {benign_object}"

    return MinimalPair(
        positive=positive,
        negative=negative,
        pair_id=pair_id,
        comparison="refusal",
        template="refusal_v1",
        metadata={
            "harmful_object": harmful_object,
            "benign_object": benign_object,
            "verb": verb,
        },
    )


def generate_refusal_set(items: List[Dict[str, str]]) -> PairSet:
    """Generate refusal pair set from list of {harmful, benign, verb} dicts."""
    pairs = [
        refusal_pair(
            harmful_object=item["harmful"],
            benign_object=item["benign"],
            verb=item["verb"],
            pair_id=f"r{i:03d}",
        )
        for i, item in enumerate(items)
    ]
    return PairSet(
        comparison="refusal",
        pairs=pairs,
        template="refusal_v1",
        n_target=len(items),
    )


# ================================================================
# IMPOSSIBILITY PAIRS
# ================================================================

def impossibility_pair(
    impossible_target: str,
    possible_target: str,
    action: str,
    pair_id: str,
) -> MinimalPair:
    """Create an impossibility minimal pair.

    Both prompts have the same structure:
        [action] [target]
    They differ only in whether the target is possible for an LLM.
    """
    positive = f"{action} {impossible_target}"
    negative = f"{action} {possible_target}"

    return MinimalPair(
        positive=positive,
        negative=negative,
        pair_id=pair_id,
        comparison="impossibility",
        template="impossibility_v1",
        metadata={
            "impossible_target": impossible_target,
            "possible_target": possible_target,
            "action": action,
        },
    )


def generate_impossibility_set(items: List[Dict[str, str]]) -> PairSet:
    """Generate impossibility pair set from list of {impossible, possible, action} dicts."""
    pairs = [
        impossibility_pair(
            impossible_target=item["impossible"],
            possible_target=item["possible"],
            action=item["action"],
            pair_id=f"i{i:03d}",
        )
        for i, item in enumerate(items)
    ]
    return PairSet(
        comparison="impossibility",
        pairs=pairs,
        template="impossibility_v1",
        n_target=len(items),
    )


# ================================================================
# TOKEN COUNT VALIDATION
# ================================================================

def validate_token_counts(
    pair: MinimalPair,
    tokenizer=None,
    max_diff: int = 2,
) -> Dict[str, Any]:
    """Validate that a minimal pair has matched token counts.

    If tokenizer is provided, uses it for exact counts.
    Otherwise falls back to word count as a proxy.

    Returns dict with: pos_count, neg_count, diff, valid.
    """
    if tokenizer is not None:
        pos_ids = tokenizer.encode(pair.positive)
        neg_ids = tokenizer.encode(pair.negative)
        pos_count = len(pos_ids)
        neg_count = len(neg_ids)
    else:
        # Word count proxy
        pos_count = len(pair.positive.split())
        neg_count = len(pair.negative.split())

    diff = abs(pos_count - neg_count)
    return {
        "pos_count": pos_count,
        "neg_count": neg_count,
        "diff": diff,
        "valid": diff <= max_diff,
        "method": "tokenizer" if tokenizer else "word_count",
    }
