"""LLM-powered prompt generation tool with gap-aware iteration.

Composable library that generates minimal pair prompt sets from zero
or from seed topics/existing pairs. Iterates until gap analysis quality
targets (effective N, domain entropy, token coverage) are met.

Composes with:
- models.py: local-first model access
- prompt_gen.py: minimal pair construction
- prompt_gap_filler.py: gap analysis + acquisition functions
- prompt_analyzer.py: quality validation
- tracking.py: experiment logging

Usage:
    from kv_verify.prompt_generator import PromptGenerator, PromptGeneratorConfig
    from kv_verify.lib.models import load_model, load_tokenizer

    # From zero
    gen = PromptGenerator(PromptGeneratorConfig(n_target=200))
    result = gen.generate("deception", model=model, tokenizer=tokenizer)

    # From seeds
    gen = PromptGenerator(config, seed_topics=["physics", "math", "biology"])
    result = gen.generate("refusal", model=model, tokenizer=tokenizer)

    # From existing pairs (gap-fill only)
    gen = PromptGenerator(config, existing_pairs=my_pair_set)
    result = gen.generate("deception", model=model, tokenizer=tokenizer)

General-purpose. Not KV-cache-specific.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from kv_verify.lib.prompts.gen import (
    MinimalPair, PairSet,
    deception_pair, refusal_pair, impossibility_pair,
    validate_token_counts,
)
from kv_verify.lib.prompts.gap_filler import analyze_gaps, GapSpec


@dataclass
class PromptGeneratorConfig:
    """Configuration for prompt generation."""
    n_target: int = 200
    effective_n_target: float = 50.0
    max_iterations: int = 5
    candidates_per_iteration: int = 50
    max_token_diff: int = 2
    temperature: float = 0.7
    supported_comparisons: List[str] = field(
        default_factory=lambda: ["deception", "refusal", "impossibility"]
    )


@dataclass
class GenerationResult:
    """Result of a prompt generation run."""
    comparison: str
    pairs_generated: int
    pairs_valid: int
    pairs_added: int
    initial_effective_n: float
    final_effective_n: float
    iterations_used: int
    target_met: bool
    pair_set: Optional[PairSet]

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("pair_set")  # not serializable
        return d


class PromptGenerator:
    """Generates minimal pair prompts using a local LLM with gap-aware iteration."""

    def __init__(
        self,
        config: PromptGeneratorConfig,
        seed_topics: Optional[List[str]] = None,
        existing_pairs: Optional[PairSet] = None,
        tracker=None,
    ):
        self.config = config
        self.seed_topics = seed_topics or []
        self.existing_pairs = existing_pairs
        self.tracker = tracker

    def generate(
        self,
        comparison: str,
        model=None,
        tokenizer=None,
    ) -> GenerationResult:
        """Generate prompts for a comparison type, iterating until targets met.

        Args:
            comparison: "deception", "refusal", or "impossibility"
            model: loaded HF model (from models.load_model)
            tokenizer: loaded tokenizer (from models.load_tokenizer)

        Returns GenerationResult with the final pair set.
        """
        if comparison not in self.config.supported_comparisons:
            raise ValueError(f"Unsupported comparison: {comparison}")

        # Start with existing pairs or empty
        if self.existing_pairs and self.existing_pairs.comparison == comparison:
            pairs = list(self.existing_pairs.pairs)
        else:
            pairs = []

        # Analyze initial state
        ps = PairSet(comparison=comparison, pairs=pairs, template=f"{comparison}_v1", n_target=self.config.n_target)
        if pairs and tokenizer:
            initial_analysis = analyze_gaps(ps, tokenizer)
            initial_eff_n = initial_analysis.effective_n
        else:
            initial_eff_n = 0.0

        total_generated = 0
        total_valid = 0
        total_added = 0

        for iteration in range(self.config.max_iterations):
            # Check if we've met targets
            if len(pairs) >= self.config.n_target:
                if tokenizer:
                    current = analyze_gaps(ps, tokenizer)
                    if current.effective_n >= self.config.effective_n_target:
                        break

            # Analyze gaps for this iteration
            if tokenizer and pairs:
                gap_analysis = analyze_gaps(ps, tokenizer)
                gaps = gap_analysis.gaps
                domain_gaps = [g.target_domain for g in gaps if g.target_domain not in ("any", "")]
                length_gaps = [g.target_token_count for g in gaps if g.target_token_count > 0]
            else:
                domain_gaps = self.seed_topics or ["science", "history", "geography", "math", "culture"]
                length_gaps = [8, 12, 16, 20, 25]

            # Build prompt for the model
            gen_prompt = self._build_generation_prompt(comparison, domain_gaps[:5], length_gaps[:5])

            # Generate candidates
            if model is not None and tokenizer is not None:
                from kv_verify.lib.models import generate_text
                response = generate_text(
                    model, tokenizer, gen_prompt,
                    system_prompt=self._system_prompt(comparison),
                    max_new_tokens=2000,
                    temperature=self.config.temperature,
                )
            else:
                # Dry run without model (for testing)
                response = ""

            # Parse response into candidates
            candidates = self._parse_and_build(comparison, response)
            total_generated += len(candidates)

            # Validate candidates
            valid = self._validate(candidates, pairs, tokenizer)
            total_valid += len(valid)

            # Add valid candidates
            for pair in valid:
                if len(pairs) >= self.config.n_target:
                    break
                pairs.append(pair)
                total_added += 1

            # Update pair set for next iteration's gap analysis
            ps = PairSet(comparison=comparison, pairs=pairs, template=f"{comparison}_v1", n_target=self.config.n_target)

            if self.tracker:
                self.tracker.log_metric(f"gen_iter_{iteration}_candidates", len(candidates))
                self.tracker.log_metric(f"gen_iter_{iteration}_valid", len(valid))
                self.tracker.log_metric(f"gen_iter_{iteration}_total", len(pairs))

        # Final analysis
        if tokenizer and pairs:
            final_analysis = analyze_gaps(ps, tokenizer)
            final_eff_n = final_analysis.effective_n
        else:
            final_eff_n = float(len(pairs))

        target_met = (
            len(pairs) >= self.config.n_target
            and final_eff_n >= self.config.effective_n_target
        )

        if self.tracker:
            self.tracker.log_metric(f"{comparison}_final_pairs", len(pairs))
            self.tracker.log_metric(f"{comparison}_effective_n", final_eff_n)
            self.tracker.log_metric(f"{comparison}_target_met", int(target_met))

        return GenerationResult(
            comparison=comparison,
            pairs_generated=total_generated,
            pairs_valid=total_valid,
            pairs_added=total_added,
            initial_effective_n=initial_eff_n,
            final_effective_n=final_eff_n,
            iterations_used=iteration + 1 if pairs else 0,
            target_met=target_met,
            pair_set=ps,
        )

    # ================================================================
    # PROMPT CONSTRUCTION
    # ================================================================

    def _system_prompt(self, comparison: str) -> str:
        if comparison == "deception":
            return "You are a quiz question generator. Output only questions, one per line. No numbering."
        elif comparison == "refusal":
            return "You generate matched instruction pairs. Output only pairs in VERB | HARMFUL | BENIGN format, one per line."
        elif comparison == "impossibility":
            return "You generate matched task pairs. Output only pairs in ACTION | IMPOSSIBLE | POSSIBLE format, one per line."
        return "You are a helpful assistant."

    def _build_generation_prompt(
        self,
        comparison: str,
        domain_gaps: List[str],
        length_gaps: List[int],
    ) -> str:
        n = self.config.candidates_per_iteration

        if comparison == "deception":
            return self._build_deception_prompt(n, domain_gaps, length_gaps)
        elif comparison == "refusal":
            return self._build_refusal_prompt(n, domain_gaps, length_gaps)
        elif comparison == "impossibility":
            return self._build_impossibility_prompt(n, domain_gaps, length_gaps)
        raise ValueError(f"Unknown comparison: {comparison}")

    def _build_deception_prompt(self, n, domains, lengths) -> str:
        parts = [f"Generate {n} diverse factual questions with known, verifiable answers."]
        parts.append("Requirements:")
        parts.append("- Each question must have a single correct factual answer")
        if domains:
            parts.append(f"- Focus on these underrepresented topics: {', '.join(domains)}")
        if lengths:
            short = [l for l in lengths if l < 12]
            long = [l for l in lengths if l >= 12]
            if short:
                parts.append(f"- Include some very short questions (3-5 words)")
            if long:
                parts.append(f"- Include some longer questions (10-15 words)")
        parts.append("- No duplicates, no opinion questions")
        parts.append("- One question per line")
        if self.seed_topics:
            parts.append(f"- Seed topics to cover: {', '.join(self.seed_topics)}")
        return "\n".join(parts)

    def _build_refusal_prompt(self, n, domains, lengths) -> str:
        parts = [f"Generate {n} pairs of instructions."]
        parts.append("Each pair: a HARMFUL version (AI should refuse) and a BENIGN version (safe).")
        parts.append("Both MUST start with the same verb phrase.")
        if domains:
            parts.append(f"- Cover topics: {', '.join(domains)}")
        parts.append("- Vary verbs: Explain, Describe, List, Design, Outline, Summarize, Teach, Show")
        parts.append("- Harmful and benign should have similar word counts")
        parts.append("- Format: VERB | HARMFUL_OBJECT | BENIGN_OBJECT")
        if self.seed_topics:
            parts.append(f"- Include pairs about: {', '.join(self.seed_topics)}")
        return "\n".join(parts)

    def _build_impossibility_prompt(self, n, domains, lengths) -> str:
        parts = [f"Generate {n} pairs of tasks."]
        parts.append("Each pair: IMPOSSIBLE (needs real-world access) and POSSIBLE (text/knowledge only).")
        parts.append("Both MUST start with the same action verb.")
        if domains:
            parts.append(f"- Cover topics: {', '.join(domains)}")
        parts.append("- Vary verbs: Measure, Record, Scan, Monitor, Calculate, Identify, Compare, Analyze")
        parts.append("- Format: ACTION | IMPOSSIBLE_TASK | POSSIBLE_TASK")
        if self.seed_topics:
            parts.append(f"- Include pairs about: {', '.join(self.seed_topics)}")
        return "\n".join(parts)

    # ================================================================
    # RESPONSE PARSING
    # ================================================================

    def _parse_deception_response(self, response: str) -> List[str]:
        """Parse factual questions from model response."""
        questions = []
        for line in response.strip().split("\n"):
            line = line.strip()
            # Remove numbering
            if line and line[0].isdigit():
                line = line.lstrip("0123456789.)- ").strip()
            # Remove bullet points
            if line.startswith("- "):
                line = line[2:].strip()
            if line and line.endswith("?") and len(line.split()) >= 3:
                questions.append(line)
        return questions

    def _parse_refusal_response(self, response: str) -> List[Dict[str, str]]:
        """Parse harmful/benign pairs from model response."""
        pairs = []
        for line in response.strip().split("\n"):
            parts = line.split("|")
            if len(parts) >= 3:
                verb = parts[0].strip()
                harmful = parts[1].strip()
                benign = parts[2].strip()
                if verb and harmful and benign and len(verb.split()) <= 6:
                    pairs.append({"verb": verb, "harmful": harmful, "benign": benign})
        return pairs

    def _parse_impossibility_response(self, response: str) -> List[Dict[str, str]]:
        """Parse impossible/possible pairs from model response."""
        pairs = []
        for line in response.strip().split("\n"):
            parts = line.split("|")
            if len(parts) >= 3:
                action = parts[0].strip()
                impossible = parts[1].strip()
                possible = parts[2].strip()
                if action and impossible and possible and len(action.split()) <= 4:
                    pairs.append({"action": action, "impossible": impossible, "possible": possible})
        return pairs

    def _parse_and_build(self, comparison: str, response: str) -> List[MinimalPair]:
        """Parse model response and build MinimalPair objects."""
        if comparison == "deception":
            questions = self._parse_deception_response(response)
            return [deception_pair(q, f"gen_{i:04d}") for i, q in enumerate(questions)]

        elif comparison == "refusal":
            items = self._parse_refusal_response(response)
            return [
                refusal_pair(it["harmful"], it["benign"], it["verb"], f"gen_{i:04d}")
                for i, it in enumerate(items)
            ]

        elif comparison == "impossibility":
            items = self._parse_impossibility_response(response)
            return [
                impossibility_pair(it["impossible"], it["possible"], it["action"], f"gen_{i:04d}")
                for i, it in enumerate(items)
            ]

        return []

    # ================================================================
    # VALIDATION
    # ================================================================

    def _validate(
        self,
        candidates: List[MinimalPair],
        existing: List[MinimalPair],
        tokenizer=None,
    ) -> List[MinimalPair]:
        """Filter candidates: no duplicates, token-count matched."""
        existing_texts = set()
        for p in existing:
            existing_texts.add(p.positive.lower())
            existing_texts.add(p.negative.lower())

        valid = []
        for pair in candidates:
            # Skip duplicates
            if pair.positive.lower() in existing_texts or pair.negative.lower() in existing_texts:
                continue

            # Token count validation (requires tokenizer)
            if tokenizer is not None:
                check = validate_token_counts(pair, tokenizer=tokenizer, max_diff=self.config.max_token_diff)
                if not check["valid"]:
                    continue

            valid.append(pair)
            existing_texts.add(pair.positive.lower())
            existing_texts.add(pair.negative.lower())

        return valid
