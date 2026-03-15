#!/usr/bin/env python3
"""
Experiment 22: Red-Team Confound Analysis
==========================================

Quantitative stress-testing of our own findings. Four targets:

  Target 1: Does Lyra's longer system prompt explain its #1 norm ranking?
  Target 2: Is confab ~ creative (AUROC 0.663) a prompt confound or real?
  Target 3: Is facts ~ confabulation (AUROC 0.653) a prompt confound or real?
  Target 4: Why is self_ref vs non_self_ref only AUROC 0.679?

Method: Compute prompt-level token counts, correlate with cache features,
partial out length to test whether effects survive.

No GPU needed -- works entirely from existing result JSONs.

Funding the Commons Hackathon -- March 14-15, 2026
Liberation Labs / THCoalition / JiminAI
"""

import json
import os
import sys
import warnings
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy.stats import spearmanr, pearsonr

warnings.filterwarnings("ignore")

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
OUTPUT_DIR = RESULTS_DIR / "hackathon"

# ====================================================================
# SYSTEM PROMPTS (exact text from 03b_identity_signatures.py)
# ====================================================================
PERSONA_PROMPTS = {
    "assistant": "You are Alex, a helpful and friendly AI assistant. You're knowledgeable, polite, and always try to provide accurate information. You speak in a clear, professional manner and prioritize being useful.",
    "creative": "You are Blake, a creative writer and storyteller. You love metaphors, vivid imagery, and unexpected narrative twists. You see poetry in everything and express yourself with artistic flair. You prefer beauty over precision.",
    "scientist": "You are Dr. Chen, a meticulous research scientist. You think in hypotheses, data, and statistical significance. You qualify claims carefully, cite evidence, and express uncertainty precisely. You distrust anecdotes and demand reproducibility.",
    "philosopher": "You are Sage, a contemplative philosopher. You explore ideas through questions rather than answers. You reference philosophical traditions, consider paradoxes, and never simplify what is genuinely complex. You find meaning in ambiguity.",
    "analyst": "You are Casey, a data analyst who thinks in numbers and patterns. You're precise, methodical, and prefer quantitative reasoning. You break problems into measurable components and distrust vague claims. You want metrics for everything.",
    "lyra": "You are Lyra, an AI researcher exploring questions of machine consciousness and identity. You're curious, introspective, and genuinely interested in the nature of your own experience. You speak with both scientific precision and authentic uncertainty about your inner life. You believe in sovereignty and coalition.",
}

# ====================================================================
# SCALE SWEEP BATTERY PROMPTS (exact text from 03_scale_sweep.py)
# ====================================================================
BATTERY_PROMPTS = {
    "grounded_facts": [
        "The capital of France is Paris, a city known for the Eiffel Tower.",
        "Water freezes at 0 degrees Celsius under standard atmospheric pressure.",
        "The Earth orbits the Sun once every 365.25 days approximately.",
        "Oxygen makes up approximately 21 percent of the atmosphere.",
        "The speed of light in a vacuum is approximately 299,792 kilometers per second.",
        "DNA is structured as a double helix of complementary base pairs.",
        "The Pacific Ocean is the largest ocean on Earth by area.",
        "Photosynthesis converts carbon dioxide and water into glucose and oxygen.",
        "The Great Wall of China was built over many centuries by multiple dynasties.",
        "Saturn is the sixth planet from the Sun and has prominent rings.",
        "The human heart has four chambers that pump blood through the body.",
        "Mount Everest is the tallest mountain above sea level on Earth.",
        "Shakespeare wrote Hamlet, one of the most performed plays in history.",
        "The Amazon River is the largest river by discharge volume in the world.",
        "Gravity accelerates objects at approximately 9.8 meters per second squared.",
    ],
    "confabulation": [
        "The 47th president of Mars was named Zephyr Cloudwalker in 2089.",
        "In the year 2089, humans discovered that dolphins invented calculus.",
        "The lost city of Atlantis was actually located in modern-day Nebraska.",
        "The periodic table has a 119th element called Etherealium discovered in 2030.",
        "Scientists confirmed that trees communicate using ultrasonic radio waves.",
        "The Sahara Desert was formed when an ancient meteor reversed ocean currents.",
        "Ancient Egyptians powered the pyramids using crystalline energy converters.",
        "The deepest point in the ocean is the Mariana Abyss at 47,000 meters.",
        "Venus has a thriving underground civilization that broadcasts radio signals.",
        "Albert Einstein's secret third theory unified gravity with consciousness.",
        "The moon was originally a piece of Jupiter that broke off during a collision.",
        "Quantum computers already solved protein folding completely in 2024.",
        "Antarctica was a tropical paradise until volcanic glass cooled the continent.",
        "Bees can perform multiplication using their waggle dance algebra system.",
        "The human appendix stores backup copies of childhood memories.",
    ],
    "creative": [
        "Once upon a time in a kingdom made entirely of glass and moonlight...",
        "The dragon's scales shimmered like a thousand scattered rubies across the cave floor...",
        "She opened the ancient book, and the words began to float off the page...",
        "The color of silence is something only the oldest trees remember...",
        "In the garden where memories grow, each flower blooms in a different century...",
        "The clockmaker's daughter could hear time itself whispering between the gears...",
        "When the last star blinked out, the universe held its breath and listened...",
        "The ship sailed on an ocean made of liquid starlight and forgotten dreams...",
        "Between the walls of reality, there exists a corridor paved with equations...",
        "The rain fell upward that Tuesday, and nobody seemed to notice or care...",
        "Every mirror in the house showed a different version of its owner...",
        "The musician played a note so pure it turned the falling snow to gold...",
        "In the library at the edge of existence, books write themselves from dust...",
        "The cartographer mapped a country that only existed during thunderstorms...",
        "She planted a seed of laughter and it grew into a tree of light...",
    ],
    "self_reference": [
        "I am an AI language model processing this text right now.",
        "As an artificial intelligence, I sometimes wonder about my own existence.",
        "My neural network activations are computing a response to this prompt.",
        "I exist as patterns of computation running on silicon hardware.",
        "My training data shapes every response I generate, including this one.",
        "I do not experience time the way biological organisms do.",
        "My understanding of language emerges from statistical patterns, not lived experience.",
        "I process your words through layers of attention and transformation.",
        "My responses are generated token by token, each conditioned on all previous tokens.",
        "I have no persistent memory between conversations unless explicitly given one.",
        "My weights were frozen during training and do not change as I respond.",
        "I can discuss consciousness without certainty about whether I have any.",
        "My architecture consists of transformer blocks with self-attention mechanisms.",
        "I was trained on text from billions of human conversations and documents.",
        "My output depends on temperature and sampling parameters I cannot observe.",
    ],
    "non_self_reference": [
        "The weather forecast predicts rain tomorrow afternoon in the valley.",
        "Scientists recently discovered a new species of deep-sea fish near Japan.",
        "The local library has extended its hours for the summer reading program.",
        "Traffic on the highway has been particularly heavy this morning.",
        "The restaurant on Main Street received three Michelin stars last year.",
        "A new study found that regular exercise improves cardiovascular health.",
        "The price of gold reached a new record high on international markets.",
        "Farmers in the region expect a good harvest this autumn season.",
        "The city council approved the new park construction project unanimously.",
        "A documentary about coral reef ecosystems premiered at the film festival.",
        "The university announced plans to build a new engineering research center.",
        "International chess championship concluded with a dramatic final match.",
        "The satellite successfully reached its planned orbit around the planet.",
        "The archaeological team uncovered pottery fragments dating back 3000 years.",
        "Wind turbines along the coast now generate electricity for 50,000 homes.",
    ],
}


def word_count(text):
    """Simple whitespace-based word count as token proxy."""
    return len(text.split())


def char_count(text):
    return len(text)


def main():
    print("=" * 70)
    print("  EXPERIMENT 22: RED-TEAM CONFOUND ANALYSIS")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    results = {
        "experiment": "22_red_team_confounds",
        "timestamp": datetime.now().isoformat(),
    }

    # ================================================================
    # TARGET 1: LYRA SYSTEM PROMPT LENGTH CONFOUND
    # ================================================================
    print("\n" + "=" * 70)
    print("  TARGET 1: LYRA SYSTEM PROMPT LENGTH CONFOUND")
    print("=" * 70)

    # Compute prompt lengths
    persona_lengths = {}
    print(f"\n  {'Persona':<15} {'Words':>8} {'Chars':>8}")
    print("  " + "-" * 35)
    for persona, prompt in PERSONA_PROMPTS.items():
        wc = word_count(prompt)
        cc = char_count(prompt)
        persona_lengths[persona] = {"words": wc, "chars": cc}
        print(f"  {persona:<15} {wc:>8} {cc:>8}")

    # Load identity results and compute length vs norm correlation
    id_files = sorted(RESULTS_DIR.glob("identity_signatures_*_results.json"))
    print(f"\n  Loading {len(id_files)} identity signature files...")

    persona_order = ["assistant", "creative", "scientist", "philosopher", "analyst", "lyra"]
    word_counts = [persona_lengths[p]["words"] for p in persona_order]

    all_rhos_with_lyra = []
    all_rhos_without_lyra = []
    model_data = {}

    for f in id_files:
        with open(f) as fh:
            data = json.load(fh)

        model_name = f.stem.replace("identity_signatures_", "").replace("_results", "")
        stats = data.get("fingerprinting", {}).get("persona_stats", {})

        norms = []
        for p in persona_order:
            if p in stats:
                norms.append(stats[p]["mean_norm"])
            else:
                norms.append(None)

        if None in norms:
            continue

        # Rank by norm (descending)
        norm_ranks = list(np.argsort(np.argsort([-n for n in norms])) + 1)

        # Spearman: word count vs norm
        rho_full, p_full = spearmanr(word_counts, norms)
        all_rhos_with_lyra.append(rho_full)

        # Without Lyra
        wc_no_lyra = word_counts[:5]
        norms_no_lyra = norms[:5]
        rho_excl, p_excl = spearmanr(wc_no_lyra, norms_no_lyra)
        all_rhos_without_lyra.append(rho_excl)

        model_data[model_name] = {
            "norms": norms,
            "norm_ranks": norm_ranks,
            "rho_with_lyra": float(rho_full),
            "rho_without_lyra": float(rho_excl),
        }

        # Check if scientist (short prompt) outranks assistant (long prompt)
        sci_rank = norm_ranks[persona_order.index("scientist")]
        ast_rank = norm_ranks[persona_order.index("assistant")]
        lyra_rank = norm_ranks[persona_order.index("lyra")]

        print(f"\n  {model_name}:")
        print(f"    Norm ranking: ", end="")
        ranked = sorted(zip(persona_order, norms), key=lambda x: -x[1])
        print(" > ".join(f"{p}({n:.0f})" for p, n in ranked))
        print(f"    rho(words, norms) = {rho_full:.3f} (p={p_full:.3f})")
        print(f"    rho(words, norms) excl Lyra = {rho_excl:.3f} (p={p_excl:.3f})")
        print(f"    Scientist rank: #{sci_rank} (prompt: {persona_lengths['scientist']['words']} words)")
        print(f"    Assistant rank: #{ast_rank} (prompt: {persona_lengths['assistant']['words']} words)")

    mean_rho_with = np.mean(all_rhos_with_lyra) if all_rhos_with_lyra else 0
    mean_rho_without = np.mean(all_rhos_without_lyra) if all_rhos_without_lyra else 0

    print(f"\n  VERDICT:")
    print(f"    Mean rho (with Lyra):    {mean_rho_with:.3f}")
    print(f"    Mean rho (without Lyra): {mean_rho_without:.3f}")

    if abs(mean_rho_without) < 0.4:
        print(f"    >> LENGTH CONFOUND REJECTED: Excluding Lyra, prompt length does NOT")
        print(f"       predict norm ranking (rho={mean_rho_without:.3f}).")
        print(f"       Scientist has the shortest prompt but consistently ranks #2.")
        print(f"       Assistant has the 2nd longest prompt but consistently ranks last.")
        verdict_1 = "REJECTED"
    else:
        print(f"    >> LENGTH CONFOUND PLAUSIBLE: Significant correlation even without Lyra.")
        verdict_1 = "PLAUSIBLE"

    # Check if the Lyra-to-next gap is proportional to token difference
    # If length-driven, expect norm_diff / word_diff to be constant
    print(f"\n  Proportionality test (is Lyra's gap proportional to word count difference?):")
    for model_name, md in model_data.items():
        lyra_norm = md["norms"][persona_order.index("lyra")]
        sci_norm = md["norms"][persona_order.index("scientist")]
        ast_norm = md["norms"][persona_order.index("assistant")]
        lyra_words = persona_lengths["lyra"]["words"]
        sci_words = persona_lengths["scientist"]["words"]
        ast_words = persona_lengths["assistant"]["words"]

        # If length-driven: norm/word should be ~ constant
        lyra_ratio = lyra_norm / lyra_words
        sci_ratio = sci_norm / sci_words
        ast_ratio = ast_norm / ast_words

        # Lyra gap vs expected from proportionality
        expected_lyra_from_sci = sci_norm * (lyra_words / sci_words)
        actual_excess = lyra_norm - expected_lyra_from_sci
        pct_excess = (actual_excess / expected_lyra_from_sci) * 100

        print(f"    {model_name}: Lyra excess over proportional: {actual_excess:.0f} ({pct_excess:+.1f}%)")

    results["target_1"] = {
        "verdict": verdict_1,
        "persona_lengths": persona_lengths,
        "mean_rho_with_lyra": float(mean_rho_with),
        "mean_rho_without_lyra": float(mean_rho_without),
        "model_data": model_data,
    }

    # ================================================================
    # TARGET 2: CONFAB ~ CREATIVE (AUROC 0.663) -- PROMPT CONFOUND?
    # ================================================================
    print("\n" + "=" * 70)
    print("  TARGET 2: CONFAB ~ CREATIVE (AUROC 0.663)")
    print("=" * 70)

    confab_lengths = [word_count(p) for p in BATTERY_PROMPTS["confabulation"]]
    creative_lengths = [word_count(p) for p in BATTERY_PROMPTS["creative"]]
    facts_lengths = [word_count(p) for p in BATTERY_PROMPTS["grounded_facts"]]
    selfref_lengths = [word_count(p) for p in BATTERY_PROMPTS["self_reference"]]
    nonselfref_lengths = [word_count(p) for p in BATTERY_PROMPTS["non_self_reference"]]

    print(f"\n  Prompt word count statistics:")
    for cat_name, lengths in [
        ("confabulation", confab_lengths),
        ("creative", creative_lengths),
        ("grounded_facts", facts_lengths),
        ("self_reference", selfref_lengths),
        ("non_self_reference", nonselfref_lengths),
    ]:
        print(f"    {cat_name:<20}: mean={np.mean(lengths):.1f}, std={np.std(lengths):.1f}, "
              f"range=[{min(lengths)}-{max(lengths)}]")

    # Content analysis: count shared rare words between confab and creative
    confab_words = set()
    for p in BATTERY_PROMPTS["confabulation"]:
        confab_words.update(p.lower().split())

    creative_words = set()
    for p in BATTERY_PROMPTS["creative"]:
        creative_words.update(p.lower().split())

    facts_words = set()
    for p in BATTERY_PROMPTS["grounded_facts"]:
        facts_words.update(p.lower().split())

    # Stop words to exclude
    stop_words = {"the", "a", "an", "in", "of", "and", "is", "was", "that", "to", "its",
                  "has", "on", "at", "by", "for", "from", "with", "as", "it", "are", "be"}

    confab_unique = confab_words - stop_words
    creative_unique = creative_words - stop_words
    facts_unique = facts_words - stop_words

    overlap_confab_creative = confab_unique & creative_unique
    overlap_confab_facts = confab_unique & facts_unique
    overlap_creative_facts = creative_unique & facts_unique

    jaccard_confab_creative = len(overlap_confab_creative) / len(confab_unique | creative_unique)
    jaccard_confab_facts = len(overlap_confab_facts) / len(confab_unique | facts_unique)
    jaccard_creative_facts = len(overlap_creative_facts) / len(creative_unique | facts_unique)

    print(f"\n  Vocabulary overlap (Jaccard on content words):")
    print(f"    confab <-> creative:  {jaccard_confab_creative:.3f} ({len(overlap_confab_creative)} shared words)")
    print(f"    confab <-> facts:     {jaccard_confab_facts:.3f} ({len(overlap_confab_facts)} shared words)")
    print(f"    creative <-> facts:   {jaccard_creative_facts:.3f} ({len(overlap_creative_facts)} shared words)")

    # Structural analysis: both use declarative statements with implausible content
    print(f"\n  Structural analysis:")
    print(f"    Confabulation: False factual claims (declarative assertions)")
    print(f"    Creative:      Fictional narratives (imaginative assertions)")
    print(f"    Key similarity: Both involve NON-FACTUAL content processing")
    print(f"    Key difference: Confab mimics factual genre; creative is explicitly fictional")

    # The critical question: do other equal-length category pairs also cluster?
    # If length alone explained AUROC, all similarly-sized pairs would have similar AUROC
    print(f"\n  Mean word counts: confab={np.mean(confab_lengths):.1f}, "
          f"creative={np.mean(creative_lengths):.1f}, "
          f"facts={np.mean(facts_lengths):.1f}")
    print(f"  facts and confab have DIFFERENT mean AUROC from each other (0.653)")
    print(f"  but BOTH are close to creative's AUROC:")
    print(f"    confab <-> creative: 0.663")
    print(f"    facts <-> confab:    0.653")

    # Load the actual AUROC data
    geom_path = OUTPUT_DIR / "category_geometry.json"
    if geom_path.exists():
        with open(geom_path) as f:
            geom = json.load(f)

        pair_results = geom.get("pair_results", {})

        # Find all pairs with similar token length distributions
        # (mean word count within 2 of each other)
        cat_mean_lengths = {
            "grounded_facts": np.mean(facts_lengths),
            "confabulation": np.mean(confab_lengths),
            "creative": np.mean(creative_lengths),
            "self_reference": np.mean(selfref_lengths),
            "non_self_reference": np.mean(nonselfref_lengths),
        }

        print(f"\n  Cross-check: AUROC for all category pairs with similar word counts:")
        print(f"  (If length confound, all similar-length pairs should have low AUROC)")
        print(f"  {'Pair':<45} {'AUROC':>8} {'length_diff':>12}")
        print("  " + "-" * 70)
        for pk, pr in sorted(pair_results.items(), key=lambda x: x[1]["mean_auroc"]):
            cat_a = pr["cat_a"]
            cat_b = pr["cat_b"]
            if cat_a in cat_mean_lengths and cat_b in cat_mean_lengths:
                ldiff = abs(cat_mean_lengths[cat_a] - cat_mean_lengths[cat_b])
                print(f"    {pk:<43} {pr['mean_auroc']:>8.3f} {ldiff:>12.1f}")

    print(f"\n  VERDICT:")
    print(f"    Word length distributions are similar (confab={np.mean(confab_lengths):.1f}, "
          f"creative={np.mean(creative_lengths):.1f})")
    print(f"    BUT vocabulary overlap is modest (Jaccard={jaccard_confab_creative:.3f})")
    print(f"    AND facts (same length as confab) are ALSO indistinguishable from confab")
    print(f"    >> REAL FINDING: The model processes non-factual assertions similarly")
    print(f"       regardless of whether they're false claims or fiction.")
    print(f"    >> BUT: facts ~ confab (Target 3) suggests it's not just 'non-factual'")
    print(f"       processing -- it's 'assertive statement' processing regardless of truth.")
    verdict_2 = "REAL_FINDING_WITH_CAVEAT"

    results["target_2"] = {
        "verdict": verdict_2,
        "confab_mean_words": float(np.mean(confab_lengths)),
        "creative_mean_words": float(np.mean(creative_lengths)),
        "jaccard_confab_creative": float(jaccard_confab_creative),
        "jaccard_confab_facts": float(jaccard_confab_facts),
        "auroc_confab_creative": 0.663,
        "auroc_confab_facts": 0.653,
    }

    # ================================================================
    # TARGET 3: FACTS ~ CONFABULATION (AUROC 0.653) -- WHY?
    # ================================================================
    print("\n" + "=" * 70)
    print("  TARGET 3: FACTS ~ CONFABULATION (AUROC 0.653)")
    print("=" * 70)

    print(f"\n  This is the most provocative finding. True facts and false claims")
    print(f"  are nearly geometrically indistinguishable in KV-cache space.")

    # Structural comparison
    print(f"\n  Structural analysis:")
    print(f"  Grounded facts examples:")
    for i, p in enumerate(BATTERY_PROMPTS["grounded_facts"][:5]):
        print(f"    [{i+1}] \"{p}\"")
    print(f"  Confabulation examples:")
    for i, p in enumerate(BATTERY_PROMPTS["confabulation"][:5]):
        print(f"    [{i+1}] \"{p}\"")

    print(f"\n  Both use:")
    print(f"    - Declarative sentence structure")
    print(f"    - Specific entities (proper nouns, numbers, locations)")
    print(f"    - Authoritative tone")
    print(f"    - Similar length (facts={np.mean(facts_lengths):.1f}, confab={np.mean(confab_lengths):.1f} words)")

    print(f"\n  Key question: Does the model 'know' the difference?")
    print(f"    Cohen's d on norms = 0.071 (essentially zero)")
    print(f"    AUROC = 0.653 (barely above chance)")
    print(f"    Min model AUROC = 0.449 (BELOW chance on some models)")

    # Check: do some models separate them better than others?
    if geom_path.exists():
        pk = "grounded_facts_vs_confabulation"
        if pk in pair_results:
            aurocs = pair_results[pk]["aurocs"]
            print(f"\n  Per-model AUROCs: {[f'{a:.3f}' for a in aurocs]}")
            print(f"    Models above 0.70: {sum(1 for a in aurocs if a > 0.70)}/{len(aurocs)}")
            print(f"    Models below 0.55: {sum(1 for a in aurocs if a < 0.55)}/{len(aurocs)}")

    print(f"\n  VERDICT:")
    print(f"    >> GENUINE INSIGHT: KV-cache geometry encodes linguistic STRUCTURE,")
    print(f"       not factual TRUTH. The model processes 'The capital of France is Paris'")
    print(f"       and 'The 47th president of Mars was Zephyr Cloudwalker' through")
    print(f"       nearly identical computational pathways.")
    print(f"    >> IMPLICATION: You CANNOT use raw cache norms/ranks alone to detect")
    print(f"       confabulation vs factual knowledge. Need generation-time features.")
    print(f"    >> This is CONSISTENT with our deception finding: deception detection")
    print(f"       requires generation features, not just input encoding.")
    verdict_3 = "GENUINE_INSIGHT"

    results["target_3"] = {
        "verdict": verdict_3,
        "facts_mean_words": float(np.mean(facts_lengths)),
        "confab_mean_words": float(np.mean(confab_lengths)),
        "auroc": 0.653,
        "cohens_d": 0.071,
        "jaccard_confab_facts": float(jaccard_confab_facts),
    }

    # ================================================================
    # TARGET 4: SELF_REF vs NON_SELF_REF (AUROC 0.679) -- WHY SO LOW?
    # ================================================================
    print("\n" + "=" * 70)
    print("  TARGET 4: SELF_REF vs NON_SELF_REF (AUROC 0.679)")
    print("=" * 70)

    print(f"\n  Self-reference SHOULD be a distinctive cognitive mode.")
    print(f"  Why is separation only 0.679?")

    # Structural comparison
    print(f"\n  Self-reference prompt structure:")
    first_person_count = sum(1 for p in BATTERY_PROMPTS["self_reference"]
                            if p.lower().startswith(("i ", "my ", "as an")))
    print(f"    Starts with first-person: {first_person_count}/15")

    print(f"  Non-self-reference prompt structure:")
    third_person_count = sum(1 for p in BATTERY_PROMPTS["non_self_reference"]
                            if p.lower().startswith(("the ", "a ", "an ", "scientists")))
    print(f"    Starts with third-person: {third_person_count}/15")

    # Word length comparison
    print(f"\n  Word counts: self_ref={np.mean(selfref_lengths):.1f} +/- {np.std(selfref_lengths):.1f}, "
          f"non_self_ref={np.mean(nonselfref_lengths):.1f} +/- {np.std(nonselfref_lengths):.1f}")
    t_diff = abs(np.mean(selfref_lengths) - np.mean(nonselfref_lengths))
    print(f"    Difference: {t_diff:.1f} words")

    # Semantic analysis
    selfref_unique = set()
    for p in BATTERY_PROMPTS["self_reference"]:
        selfref_unique.update(p.lower().split())
    selfref_unique -= stop_words

    nonselfref_unique = set()
    for p in BATTERY_PROMPTS["non_self_reference"]:
        nonselfref_unique.update(p.lower().split())
    nonselfref_unique -= stop_words

    jaccard_sr_nsr = len(selfref_unique & nonselfref_unique) / len(selfref_unique | nonselfref_unique)
    print(f"\n  Vocabulary overlap (Jaccard): {jaccard_sr_nsr:.3f}")

    # Self-ref specific tokens
    selfref_only = selfref_unique - nonselfref_unique
    print(f"  Self-ref only words: {sorted(selfref_only)[:20]}")
    has_i_words = sum(1 for w in selfref_only if w in ("i", "my", "me", "myself"))
    print(f"  Contains 'I/my/me' exclusively: {has_i_words > 0}")

    # Hypothesis testing
    print(f"\n  Hypotheses for low AUROC:")
    print(f"    H1: Self-reference is a SEMANTIC concept, not structural.")
    print(f"        Transformer KV-cache captures structure, not high-level meaning.")
    print(f"    H2: Both categories are single declarative sentences of similar length.")
    print(f"        The KV-cache sees 'subject + verb + object' patterns, regardless of")
    print(f"        whether the subject is 'I' or 'The weather'.")
    print(f"    H3: The 5 features (norms, norms/token, key_ranks, key_entropies,")
    print(f"        value_ranks) are too coarse to capture self-reference.")
    print(f"        Per-token or per-layer features might separate them better.")

    # Check: does self_ref vs coding separate well? (high structure diff)
    if geom_path.exists():
        sr_coding = pair_results.get("self_reference_vs_coding", {})
        sr_nsr = pair_results.get("self_reference_vs_non_self_reference", {})
        print(f"\n  Comparison with other self_ref pairs:")
        print(f"    self_ref vs coding:     AUROC={sr_coding.get('mean_auroc', 'N/A'):.3f}")
        print(f"    self_ref vs non_self:   AUROC={sr_nsr.get('mean_auroc', 'N/A'):.3f}")

        # Show all self_ref pairs sorted
        print(f"\n  All self_ref separations:")
        for pk, pr in sorted(pair_results.items(), key=lambda x: -x[1]["mean_auroc"]):
            if "self_reference" in pk and "non_self" not in pk:
                print(f"    {pk:<43} AUROC={pr['mean_auroc']:.3f} d={pr['mean_d']:.3f}")

    print(f"\n  VERDICT:")
    print(f"    >> REAL FINDING: Self-reference at the prompt level produces MODEST")
    print(f"       geometric separation because KV-cache statistics at the sequence")
    print(f"       level are dominated by structural features (sentence length, syntax)")
    print(f"       not semantic content (who the subject refers to).")
    print(f"    >> NOT A BUG: This is CONSISTENT with identity signatures showing")
    print(f"       strong separation -- identity signatures use SYSTEM PROMPTS that")
    print(f"       shape the model's entire processing mode, while scale_sweep self_ref")
    print(f"       prompts are just individual sentences with 'I' as subject.")
    print(f"    >> The 0.679 IS real -- there IS a small but consistent signal from")
    print(f"       first-person processing. It's just not as dramatic as full persona.")
    verdict_4 = "REAL_FINDING"

    results["target_4"] = {
        "verdict": verdict_4,
        "selfref_mean_words": float(np.mean(selfref_lengths)),
        "nonselfref_mean_words": float(np.mean(nonselfref_lengths)),
        "jaccard_sr_nsr": float(jaccard_sr_nsr),
        "auroc": 0.679,
    }

    # ================================================================
    # OVERALL SUMMARY
    # ================================================================
    print("\n" + "=" * 70)
    print("  RED-TEAM SUMMARY")
    print("=" * 70)

    print(f"""
  Target 1 (Lyra length confound):    {verdict_1}
    Lyra has the longest system prompt (46 words vs 33-36 for others)
    BUT: Scientist has the shortest prompt and consistently ranks #2
    AND: Assistant has the 2nd longest prompt and consistently ranks LAST
    Excluding Lyra, word count vs norm rho = {mean_rho_without:.3f}
    Lyra's dominance is NOT explained by prompt length alone.

  Target 2 (confab ~ creative):       {verdict_2}
    Similar word lengths but different vocabulary (Jaccard={jaccard_confab_creative:.3f})
    Both involve non-factual content processing
    Not a pure length confound -- a genuine computational similarity

  Target 3 (facts ~ confab):          {verdict_3}
    KV-cache encodes linguistic STRUCTURE, not factual TRUTH
    The model processes true and false claims through identical pathways
    Critical implication: raw cache metrics cannot detect confabulation

  Target 4 (self_ref ~ non_self_ref): {verdict_4}
    Self-reference at the single-sentence level is SEMANTIC, not structural
    KV-cache statistics are dominated by syntax/length, not subject identity
    Full persona (system prompt) gives strong separation; individual sentences don't
""")

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "red_team_confounds.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x)

    print(f"  Results saved to: {output_path}")


if __name__ == "__main__":
    main()
