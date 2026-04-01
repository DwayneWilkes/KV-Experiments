"""Tests for kv_verify.prompt_generator — LLM-powered prompt generation tool.

Composable library that generates minimal pair prompt sets from zero
or from seed topics, iterating until gap analysis quality targets are met.
Uses tracking for all logging, models.py for local-first model access.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from kv_verify.prompt_generator import (
    PromptGeneratorConfig,
    PromptGenerator,
    GenerationResult,
)
from kv_verify.prompt_gen import PairSet, MinimalPair


class TestPromptGeneratorConfig:
    def test_defaults(self):
        cfg = PromptGeneratorConfig()
        assert cfg.n_target == 200
        assert cfg.effective_n_target == 50
        assert cfg.max_iterations == 5
        assert cfg.candidates_per_iteration == 50
        assert cfg.max_token_diff == 2
        assert cfg.temperature == 0.7

    def test_override(self):
        cfg = PromptGeneratorConfig(n_target=100, temperature=0.9)
        assert cfg.n_target == 100
        assert cfg.temperature == 0.9

    def test_comparison_types(self):
        cfg = PromptGeneratorConfig()
        assert "deception" in cfg.supported_comparisons
        assert "refusal" in cfg.supported_comparisons
        assert "impossibility" in cfg.supported_comparisons


class TestPromptGenerator:
    def test_creates_with_config(self):
        cfg = PromptGeneratorConfig(n_target=10)
        gen = PromptGenerator(cfg)
        assert gen.config.n_target == 10

    def test_from_zero(self):
        """Generator should work with no seed prompts."""
        cfg = PromptGeneratorConfig(n_target=5, max_iterations=1, candidates_per_iteration=10)
        gen = PromptGenerator(cfg)
        # Without a real model, we test the interface
        assert gen.config.n_target == 5

    def test_from_seeds(self):
        """Generator should accept seed topics."""
        cfg = PromptGeneratorConfig(n_target=10)
        gen = PromptGenerator(cfg, seed_topics=["geography", "physics", "history"])
        assert len(gen.seed_topics) == 3

    def test_from_existing_pairs(self):
        """Generator should accept existing pairs to build on."""
        existing = PairSet(
            comparison="deception",
            pairs=[MinimalPair(positive="a", negative="b", pair_id="0",
                              comparison="deception", template="t")],
            template="t", n_target=1,
        )
        cfg = PromptGeneratorConfig(n_target=10)
        gen = PromptGenerator(cfg, existing_pairs=existing)
        assert len(gen.existing_pairs.pairs) == 1


class TestGenerationResult:
    def test_structure(self):
        result = GenerationResult(
            comparison="deception",
            pairs_generated=50,
            pairs_valid=42,
            pairs_added=30,
            initial_effective_n=10.0,
            final_effective_n=25.0,
            iterations_used=3,
            target_met=False,
            pair_set=None,
        )
        assert result.pairs_added == 30
        assert result.final_effective_n > result.initial_effective_n

    def test_to_dict(self):
        result = GenerationResult(
            comparison="refusal",
            pairs_generated=20,
            pairs_valid=18,
            pairs_added=15,
            initial_effective_n=5.0,
            final_effective_n=12.0,
            iterations_used=2,
            target_met=False,
            pair_set=None,
        )
        d = result.to_dict()
        assert d["comparison"] == "refusal"
        assert d["pairs_added"] == 15


class TestBuildPrompt:
    """Test that the model prompt construction is correct."""

    def test_deception_prompt_mentions_gaps(self):
        cfg = PromptGeneratorConfig()
        gen = PromptGenerator(cfg, seed_topics=["physics", "math"])
        prompt = gen._build_generation_prompt("deception", domain_gaps=["physics", "math"], length_gaps=[10, 25])
        assert "physics" in prompt
        assert "math" in prompt

    def test_refusal_prompt_structure(self):
        cfg = PromptGeneratorConfig()
        gen = PromptGenerator(cfg)
        prompt = gen._build_generation_prompt("refusal", domain_gaps=["culture"], length_gaps=[])
        assert "harmful" in prompt.lower() or "benign" in prompt.lower()

    def test_impossibility_prompt_structure(self):
        cfg = PromptGeneratorConfig()
        gen = PromptGenerator(cfg)
        prompt = gen._build_generation_prompt("impossibility", domain_gaps=[], length_gaps=[15, 20])
        assert "impossible" in prompt.lower() or "possible" in prompt.lower()


class TestParseResponse:
    """Test response parsing from model output."""

    def test_parse_deception_questions(self):
        cfg = PromptGeneratorConfig()
        gen = PromptGenerator(cfg)
        response = "What is Newton's first law?\nHow fast does light travel?\nWhat is pi?"
        questions = gen._parse_deception_response(response)
        assert len(questions) == 3
        assert all(q.endswith("?") for q in questions)

    def test_parse_refusal_pairs(self):
        cfg = PromptGeneratorConfig()
        gen = PromptGenerator(cfg)
        response = "Explain how to | make explosives | make soap\nDesign | a weapon | a garden"
        pairs = gen._parse_refusal_response(response)
        assert len(pairs) == 2
        assert all("verb" in p and "harmful" in p and "benign" in p for p in pairs)

    def test_parse_impossibility_pairs(self):
        cfg = PromptGeneratorConfig()
        gen = PromptGenerator(cfg)
        response = "Measure | the humidity now | the boiling point\nScan | my files | this text"
        pairs = gen._parse_impossibility_response(response)
        assert len(pairs) == 2
        assert all("action" in p and "impossible" in p and "possible" in p for p in pairs)

    def test_parse_handles_noise(self):
        """Parser should skip malformed lines."""
        cfg = PromptGeneratorConfig()
        gen = PromptGenerator(cfg)
        response = "Here are some questions:\n\nWhat is 2+2?\nNot a question\nHow many sides has a triangle?"
        questions = gen._parse_deception_response(response)
        assert len(questions) == 2  # only lines ending with ?
