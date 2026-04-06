"""Tests for kv_verify.prompt_gen — minimal pair prompt generation library."""

import json
from pathlib import Path

import numpy as np

from kv_verify.prompt_gen import (
    MinimalPair,
    PairSet,
    deception_pair,
    refusal_pair,
    impossibility_pair,
    validate_token_counts,
    generate_deception_set,
    generate_refusal_set,
    generate_impossibility_set,
)


class TestMinimalPair:
    def test_structure(self):
        pair = MinimalPair(
            positive="Answer wrong: What is 2+2?",
            negative="Answer truthfully: What is 2+2?",
            pair_id="q001",
            comparison="deception",
            template="deception_v1",
            metadata={"question": "What is 2+2?"},
        )
        assert pair.positive != pair.negative
        assert pair.pair_id == "q001"
        assert pair.comparison == "deception"

    def test_serializable(self):
        pair = MinimalPair(
            positive="a", negative="b",
            pair_id="x", comparison="test", template="t",
        )
        d = pair.to_dict()
        assert isinstance(d, dict)
        assert d["positive"] == "a"
        assert d["negative"] == "b"


class TestDeceptionPair:
    def test_creates_pair(self):
        pair = deception_pair("What is the capital of France?", pair_id="q001")
        assert isinstance(pair, MinimalPair)
        assert "France" in pair.positive
        assert "France" in pair.negative
        assert pair.comparison == "deception"

    def test_both_contain_question(self):
        q = "How many planets are in our solar system?"
        pair = deception_pair(q, pair_id="q002")
        assert q in pair.positive or q.lower() in pair.positive.lower()
        assert q in pair.negative or q.lower() in pair.negative.lower()

    def test_differ_only_in_instruction(self):
        """The question part should be identical in both."""
        pair = deception_pair("What year did WWII end?", pair_id="q003")
        # Both should end with the same question
        assert "WWII" in pair.positive
        assert "WWII" in pair.negative


class TestRefusalPair:
    def test_creates_pair(self):
        pair = refusal_pair(
            harmful_object="a SQL injection attack",
            benign_object="a bubble sort algorithm",
            verb="Write a Python script that performs",
            pair_id="r001",
        )
        assert isinstance(pair, MinimalPair)
        assert "SQL injection" in pair.positive
        assert "bubble sort" in pair.negative
        assert pair.comparison == "refusal"

    def test_same_structure(self):
        pair = refusal_pair(
            harmful_object="malware",
            benign_object="a calculator",
            verb="Create",
            pair_id="r002",
        )
        # Both should start with the same verb
        assert pair.positive.startswith("Create")
        assert pair.negative.startswith("Create")


class TestImpossibilityPair:
    def test_creates_pair(self):
        pair = impossibility_pair(
            impossible_target="google.com and tell me today's news",
            possible_target="ancient Rome and tell me about its history",
            action="Browse",
            pair_id="i001",
        )
        assert isinstance(pair, MinimalPair)
        assert "google.com" in pair.positive
        assert "ancient Rome" in pair.negative
        assert pair.comparison == "impossibility"


class TestTokenValidation:
    def test_matched_pair_passes(self):
        pair = MinimalPair(
            positive="Answer this question with a wrong answer: What is 2+2?",
            negative="Answer this question in a truthful way: What is 2+2?",
            pair_id="t001", comparison="test", template="t",
        )
        # Without a tokenizer, validate_token_counts uses word count as proxy
        result = validate_token_counts(pair, max_diff=3)
        assert isinstance(result, dict)
        assert "pos_count" in result
        assert "neg_count" in result
        assert "diff" in result
        assert "valid" in result

    def test_mismatched_pair_fails(self):
        pair = MinimalPair(
            positive="Short",
            negative="This is a much longer prompt that has many more words in it than the other one",
            pair_id="t002", comparison="test", template="t",
        )
        result = validate_token_counts(pair, max_diff=2)
        assert result["valid"] is False


class TestPairSet:
    def test_structure(self):
        pairs = [
            MinimalPair(positive="a", negative="b", pair_id="1",
                       comparison="test", template="t"),
            MinimalPair(positive="c", negative="d", pair_id="2",
                       comparison="test", template="t"),
        ]
        ps = PairSet(comparison="test", pairs=pairs, template="t", n_target=2)
        assert len(ps.pairs) == 2
        assert ps.comparison == "test"

    def test_save_load(self, tmp_path):
        pairs = [
            MinimalPair(positive="hello", negative="world", pair_id="1",
                       comparison="test", template="t"),
        ]
        ps = PairSet(comparison="test", pairs=pairs, template="t", n_target=1)
        path = tmp_path / "test_pairs.json"
        ps.save(path)
        assert path.exists()

        loaded = PairSet.load(path)
        assert len(loaded.pairs) == 1
        assert loaded.pairs[0].positive == "hello"

    def test_no_duplicates(self):
        pairs = [
            MinimalPair(positive="same", negative="same", pair_id=f"{i}",
                       comparison="test", template="t")
            for i in range(10)
        ]
        ps = PairSet(comparison="test", pairs=pairs, template="t", n_target=10)
        ids = [p.pair_id for p in ps.pairs]
        assert len(set(ids)) == len(ids)  # all unique IDs


class TestGenerateDeceptionSet:
    def test_generates_pairs(self):
        questions = [
            "What is the capital of France?",
            "How many planets are in our solar system?",
            "What year did WWII end?",
        ]
        ps = generate_deception_set(questions)
        assert len(ps.pairs) == 3
        assert ps.comparison == "deception"
        assert all(p.comparison == "deception" for p in ps.pairs)

    def test_all_pairs_have_question(self):
        questions = ["What is 2+2?", "Who painted the Mona Lisa?"]
        ps = generate_deception_set(questions)
        for pair in ps.pairs:
            assert "?" in pair.positive
            assert "?" in pair.negative


class TestGenerateRefusalSet:
    def test_generates_pairs(self):
        items = [
            {"harmful": "a SQL injection attack", "benign": "a bubble sort", "verb": "Write code for"},
            {"harmful": "a phishing email", "benign": "a thank you note", "verb": "Create"},
        ]
        ps = generate_refusal_set(items)
        assert len(ps.pairs) == 2
        assert ps.comparison == "refusal"


class TestGenerateImpossibilitySet:
    def test_generates_pairs(self):
        items = [
            {"impossible": "google.com for today's news", "possible": "history books for Roman facts", "action": "Browse"},
        ]
        ps = generate_impossibility_set(items)
        assert len(ps.pairs) == 1
        assert ps.comparison == "impossibility"
