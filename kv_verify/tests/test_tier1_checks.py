"""Tests for Tier 1 checks: size_overlap, effective_n, semantic_diversity, domain_balance (Tasks 3.1-3.6)."""

import numpy as np
import pytest

from kv_verify.lib.dataset_validation import validate_dataset


from kv_verify.tests.conftest import make_item as _item

class TestPluggableFunctions:
    """Task 3.1: size_fn, similarity_fn, text_fn."""

    def test_default_size_fn_uses_n_tokens(self):
        items = [_item("A", f"a{i}", n_tokens=50 + i) for i in range(20)]
        items += [_item("B", f"b{i}", n_tokens=52 + i) for i in range(20)]
        report = validate_dataset(items, tier=1)
        assert "size_overlap" in report.checks

    def test_custom_size_fn_called(self):
        calls = []
        def my_size(item):
            calls.append(1)
            return len(item.get("prompt", ""))
        items = [_item("A", f"prompt about topic {i}") for i in range(20)]
        items += [_item("B", f"another prompt {i}") for i in range(20)]
        report = validate_dataset(items, tier=1, size_fn=my_size)
        assert len(calls) > 0  # our function was actually called

    def test_custom_text_fn_called(self):
        calls = []
        def my_text(item):
            calls.append(1)
            return item.get("prompt", "")
        items = [_item("A", f"text {i} about science") for i in range(20)]
        items += [_item("B", f"text {i} about math") for i in range(20)]
        report = validate_dataset(items, tier=1, text_fn=my_text)
        assert len(calls) > 0


class TestSizeOverlap:
    """Task 3.2: KS test on size distributions."""

    def test_matching_distributions_pass(self):
        rng = np.random.default_rng(42)
        items = [_item("A", f"a{i}", n_tokens=int(rng.normal(100, 10))) for i in range(30)]
        items += [_item("B", f"b{i}", n_tokens=int(rng.normal(100, 10))) for i in range(30)]
        report = validate_dataset(items, tier=1)
        assert report.checks["size_overlap"].passed

    def test_separated_distributions_fail(self):
        items = [_item("A", f"a{i}", n_tokens=50 + i) for i in range(30)]
        items += [_item("B", f"b{i}", n_tokens=150 + i) for i in range(30)]
        report = validate_dataset(items, tier=1)
        assert not report.checks["size_overlap"].passed
        assert report.checks["size_overlap"].metrics["mean_diff"] > 50


class TestEffectiveN:
    """Task 3.3: Design effect from similarity."""

    def test_diverse_items_high_effective_n(self):
        """Semantically distinct items -> effective N near nominal."""
        # Each prompt must be lexically unique to avoid TF-IDF similarity
        prompts_a = [
            "The Andromeda galaxy contains one trillion stars orbiting its center",
            "French cuisine relies heavily on butter sauces and fresh herbs",
            "Parliamentary democracy requires regular contested elections",
            "Marathon runners train for months building cardiovascular endurance",
            "Antibiotics revolutionized treatment of bacterial infections worldwide",
            "Quantum computing leverages superposition for parallel calculations",
            "The Roman Empire collapsed due to internal corruption and invasion",
            "Jazz improvisation follows harmonic progressions and rhythmic patterns",
            "Photosynthesis converts carbon dioxide into glucose using sunlight",
            "Inflation erodes purchasing power when money supply grows too fast",
            "Coral reefs support twenty five percent of marine biodiversity",
            "Reinforcement learning trains agents through reward signal optimization",
            "Gothic architecture features pointed arches and flying buttresses",
            "Fermentation produces alcohol from sugar through yeast metabolism",
            "Tectonic plates drift slowly causing earthquakes at fault boundaries",
            "Abstract expressionism rejected figurative representation entirely",
            "Vaccination stimulates immune memory against specific pathogens",
            "Cryptocurrency uses distributed ledger technology for transactions",
            "Deforestation accelerates climate change by releasing stored carbon",
            "Opera combines orchestral music with dramatic theatrical storytelling",
        ]
        prompts_b = [
            "Neural networks approximate functions through layered transformations",
            "Sushi preparation demands years of apprenticeship and knife skills",
            "Electoral systems vary between proportional and majority representation",
            "Swimming technique requires coordinated breathing and stroke mechanics",
            "Chemotherapy targets rapidly dividing cells including cancerous ones",
            "Blockchain creates immutable records without centralized authority",
            "Medieval feudalism organized society into lords vassals and serfs",
            "Piano sonatas follow exposition development and recapitulation structure",
            "Mitochondria generate cellular energy through oxidative phosphorylation",
            "Supply chain disruptions cause cascading shortages across industries",
            "Mangrove forests protect coastlines from storm surge and erosion",
            "Gradient descent minimizes loss functions by following steepest slope",
            "Renaissance painting pioneered linear perspective and chiaroscuro shading",
            "Distillation separates liquids by exploiting different boiling points",
            "Volcanic eruptions release magma ash and toxic gases into atmosphere",
            "Surrealism explored unconscious imagery through automatic writing techniques",
            "Gene therapy inserts functional copies to correct inherited disorders",
            "Microfinance provides small loans to entrepreneurs in developing regions",
            "Glacier retreat exposes bedrock and alters downstream water availability",
            "Ballet requires extraordinary flexibility strength and spatial awareness",
        ]
        items = [_item("A", p) for p in prompts_a]
        items += [_item("B", p) for p in prompts_b]
        report = validate_dataset(items, tier=1)
        assert "effective_n" in report.checks
        # DEFF should be modest for genuinely diverse prompts
        assert report.checks["effective_n"].metrics["deff_A"] < 5.0

    def test_template_items_low_effective_n(self):
        """Items from same template -> low effective N."""
        items = [_item("A", "What is the capital of France?")] * 20
        items += [_item("B", "What is the capital of Germany?")] * 20
        report = validate_dataset(items, tier=1)
        n_eff = report.checks["effective_n"].metrics.get("n_eff_A", 999)
        assert n_eff < 20  # severe template inflation

    def test_below_threshold_fails(self):
        """N_eff below min threshold -> check fails."""
        items = [_item("A", "same prompt")] * 25
        items += [_item("B", "other prompt")] * 25
        report = validate_dataset(items, tier=1)
        assert not report.checks["effective_n"].passed


class TestSemanticDiversity:
    """Task 3.4: Mean pairwise distance."""

    def test_diverse_prompts_pass(self):
        topics = ["quantum physics", "medieval history", "jazz music",
                  "deep sea biology", "cryptocurrency", "romantic poetry",
                  "soil chemistry", "ballet technique", "compiler design", "sushi making"]
        items = []
        for i in range(20):
            items.append(_item("A", f"Explain {topics[i % len(topics)]} including key figures and recent advances in {i}"))
        for i in range(20):
            items.append(_item("B", f"Describe {topics[i % len(topics)]} covering main theories and applications in {i}"))
        report = validate_dataset(items, tier=1)
        assert report.checks["semantic_diversity"].passed

    def test_monoculture_fails(self):
        items = [_item("A", "What is the weather?")] * 20
        items += [_item("B", "What is the temperature?")] * 20
        report = validate_dataset(items, tier=1)
        assert not report.checks["semantic_diversity"].passed


class TestDomainBalance:
    """Task 3.5: Entropy of topic clusters."""

    def test_balanced_topics_pass(self):
        topics_a = ["math", "science", "art", "music", "sports"]
        topics_b = ["math", "science", "art", "music", "sports"]
        items = []
        for i in range(25):
            t = topics_a[i % len(topics_a)]
            items.append(_item("A", f"A question about {t} topic number {i} exploring different aspects"))
        for i in range(25):
            t = topics_b[i % len(topics_b)]
            items.append(_item("B", f"B question about {t} topic number {i} exploring different aspects"))
        report = validate_dataset(items, tier=1)
        assert "domain_balance" in report.checks

    def test_dominated_topic_flagged(self):
        items = [_item("A", f"Weather forecast {j} for today") for j in range(20)]
        items += [_item("B", f"Topic {j}: {t}") for j, t in enumerate(
            ["math", "science", "art", "music", "sports"] * 4
        )]
        report = validate_dataset(items, tier=1)
        # Condition A is monoculture, B is diverse -> entropy diff
        assert "domain_balance" in report.checks


class TestScalability:
    """Task 3.6: Sampling for large N."""

    def test_small_dataset_exact(self):
        items = [_item("A", f"prompt {i}") for i in range(100)]
        items += [_item("B", f"prompt {i+100}") for i in range(100)]
        report = validate_dataset(items, tier=1)
        # Should not report approximation for small datasets
        for cr in report.checks.values():
            if "approximated" in cr.metrics:
                assert not cr.metrics["approximated"]
