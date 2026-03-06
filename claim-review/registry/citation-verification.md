# Citation Verification Registry

## Summary

| Category | Count |
| ---------- | ------- |
| **Paper .bib entries** | 28 |
| **Actually cited in paper** | 6 |
| **Uncited .bib entries** | 22 |
| **Cricket-only citations** | ~15 |
| **Total unique sources** | ~43 |
| | |
| **Resolution: RESOLVED** | 13 |
| **Resolution: RESOLVED_PARTIAL** | 13 |
| **Resolution: UNRESOLVED** | 1 |
| **Resolution: UNPUBLISHED** | 1 |
| | |
| **Verification: ACCURATE** | 5 |
| **Verification: APPROXIMATE** | 1 |
| **Verification: NOT_APPLICABLE** | 22 (uncited bib entries) |
| | |
| **Citation gaps (HIGH)** | 1 |
| **Citation gaps (MEDIUM)** | 3 |
| **Citation gaps (LOW)** | 4 |

---

## Paper Citations (references.bib — cited entries)

### lyra2026campaign1 — Lyra & Edrington (2026) "Geometric Signatures of Machine Cognition"

- **Resolution:** UNRESOLVED (self-citation; Campaign 1 companion paper, not yet published)
- **Citations:** 5 occurrences (L57, L75, L154, L379, L518)
- **Characterizations:**
  1. "reported exploratory findings on a predominantly single-architecture scale ladder with known methodological limitations including pseudoreplication and limited cross-architecture coverage" (L57)
  2. "introduced a geometric framework for characterizing transformer KV-cache states across cognitive modes...transparently reporting methodological limitations" (L75)
  3. "All metrics are identical to Campaign 1" (L154) — forward-reference for method details
  4. "See §8 for the full epistemic context" (L518) — epistemic framing reference
- **Verdict: ACCURATE** — Self-citation accurately describes Campaign 1's scope and limitations. The limitations listed (pseudoreplication, single architecture, limited coverage) match Campaign 1's own self-assessment.
- **Claims:** C1-C5 (methodological framing)

### hedges1981distribution — Hedges (1981) "Distribution Theory for Glass's Estimator of Effect Size"

- **Resolution:** RESOLVED_PARTIAL (S2 title search; know ID 328)
- **Citations:** 1 occurrence (L144)
- **Characterization:** "Hedges' g (bias-corrected Cohen's d, critical at n=15-25 per group where the correction is ~2.7%)"
- **Verdict: ACCURATE** — Hedges' g is indeed a bias-corrected version of Glass's/Cohen's d. The ~2.7% correction factor at n=15-25 is mathematically correct (correction factor J = 1 - 3/(4(n1+n2-2)-1)).
- **Claims:** C8, C16

### schuirmann1987comparison — Schuirmann (1987) "A Comparison of the Two One-Sided Tests Procedure"

- **Resolution:** RESOLVED_PARTIAL (S2 title search; know ID 329)
- **Citations:** 1 occurrence (L146)
- **Characterization:** "TOST equivalence testing with delta=0.3 in Cohen's d units"
- **Verdict: ACCURATE** — Schuirmann (1987) did introduce the TOST procedure for equivalence testing. The paper's use of delta=0.3 is a parameter choice, not a claim about Schuirmann.
- **Claims:** C8, C17

### bloom1956taxonomy — Bloom et al. (1956) "Taxonomy of Educational Objectives"

- **Resolution:** RESOLVED_PARTIAL (S2 title search; know ID 330)
- **Citations:** 1 occurrence (L276)
- **Characterization:** "Bloom's taxonomy of cognitive levels — from Remember through Create"
- **Verdict: ACCURATE** — Bloom's taxonomy does define cognitive levels from Knowledge (Remember in the revised version) through Synthesis/Evaluation (Create in the revised version). The paper uses the revised Bloom's taxonomy (Anderson & Krathwohl, 2001) levels but cites the original. This is standard academic practice.
- **Claims:** C42-C45

### heretic2026 — heretic-llm contributors (2026) "Heretic-LLM"

- **Resolution:** RESOLVED_PARTIAL (S2 title search; know ID 331)
- **Citations:** 1 occurrence (L379)
- **Characterization:** "abliterating the refusal direction from Qwen-7B — removing the linear subspace most associated with refusal behavior"
- **Verdict: ACCURATE** — Heretic-LLM is an abliteration tool that removes linear subspaces from model weights. The characterization of its function is correct.
- **Claims:** C78, C95

### kim2025reasoning — Kim et al. (2025) "Reasoning Models Generate Societies of Thought"

- **Resolution:** RESOLVED (arXiv 2601.10825; know ID 326)
- **Citations:** 1 occurrence (L507)
- **Characterization:** "reasoning models' internal multi-agent debate"
- **Verdict: APPROXIMATE** — The resolved paper ("Reasoning Models Generate Societies of Thought") does describe reasoning models as generating internal debate/deliberation structures. The characterization "multi-agent debate" is a simplification of the paper's "societies of thought" concept, but captures the core idea.
- **Claims:** C88 (limitations — incomplete experiment)

---

## Paper Citations (references.bib — uncited entries)

The following 22 entries exist in `references.bib` but are never `\cite{}`d in `main.tex`. They appear to be carried over from Campaign 1's bibliography. In LaTeX/BibTeX, these will NOT appear in the compiled paper's reference list.

| Key | Title | Year | arXiv | Resolution | Know ID |
| ----- | ------- | ------ | ------- | ------------ | --------- |
| aghajanyan2021intrinsic | Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning | 2021 | — | RESOLVED_PARTIAL | 321 |
| azaria2023internal | The Internal State of an LLM Knows When It's Lying | 2023 | — | RESOLVED_PARTIAL | 332 |
| beckmann2023computational | Rejecting Cognitivism: Computational Phenomenology for Deep Learning | 2023 | 2302.09071 | RESOLVED | 323 |
| belinkov2022probing | Probing Classifiers: Promises, Shortcomings, and Advances | 2022 | — | RESOLVED_PARTIAL | 333 |
| berg2025selfreferential | Large Language Models Report Subjective Experience Under Self-Referential Prompting | 2025 | 2510.24797 | RESOLVED | 319 |
| burns2022discovering | Discovering Latent Knowledge in Language Models Without Supervision | 2022 | 2212.03827 | RESOLVED | 315 |
| butlin2023consciousness | Consciousness in AI: Insights from the Science of Consciousness | 2023 | 2308.08708 | RESOLVED | 36 |
| chalmers2023could | Could a Large Language Model Be Conscious? | 2023 | — | RESOLVED_PARTIAL | 318 |
| clark2019does | What Does BERT Look At? | 2019 | 1906.04341 | RESOLVED | 313 |
| fu2025c2c | Cache-to-Cache: Direct Semantic Communication Between LLMs | 2026 | 2510.03215 | RESOLVED | 325 |
| goldowskydill2025detecting | Detecting Strategic Deception Using Linear Probes | 2025 | 2502.03407 | RESOLVED | 324 |
| hewitt2019structural | A Structural Probe for Finding Syntax in Word Representations | 2019 | — | RESOLVED_PARTIAL | — |
| li2018measuring | Measuring the Intrinsic Dimension of Objective Landscapes | 2018 | 1804.08838 | RESOLVED | 320 |
| liu2023scissorhands | Scissorhands: Exploiting the Persistence of Importance Hypothesis | 2024 | — | RESOLVED_PARTIAL | 312 |
| liu2024kivi | KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache | 2024 | 2402.02750 | RESOLVED | 310 |
| long2025deception | When Truthful Representations Flip Under Deceptive Instructions | 2025 | 2507.22149 | RESOLVED | 317 |
| marks2023geometry | The Geometry of Truth: Emergent Linear Structure in LLM Representations | 2023 | 2310.06824 | RESOLVED | 316 |
| merleau-ponty1945phenomenology | Phenomenology of Perception | 1945 | — | RESOLVED_PARTIAL | 327 |
| roy2007effective | The Effective Rank: A Measure of Effective Dimensionality | 2007 | — | RESOLVED_PARTIAL | 322 |
| watson2019ita | The Integrated Theory of Attention | 2019 | — | UNPUBLISHED | — |
| zhang2024h2o | H2O: Heavy-Hitter Oracle for Efficient Generative Inference | 2024 | — | RESOLVED_PARTIAL | 311 |
| zou2023representation | Representation Engineering: A Top-Down Approach to AI Transparency | 2023 | 2310.01405 | RESOLVED | 314 |

---

## Cricket-Only Citations

These papers are referenced in Cricket documentation but not in the paper's .bib file.

| Reference | Year | Source Docs | Resolution | Claims |
| ----------- | ------ | ------------- | ------------ | -------- |
| Ma et al. — STA-Attention (SAEs on KV-cache) | 2025 | DESIGN.md, COMPETITIVE_LANDSCAPE.md | RESOLVED_PARTIAL | CC7, CC8 |
| Wu et al. — AxBench (concept detection) | 2025 | DESIGN.md | UNRESOLVED | CC18 |
| Li et al. — ITI (Inference-Time Intervention) | 2023 | DESIGN.md, COMPETITIVE_LANDSCAPE.md | RESOLVED_PARTIAL | CL4 |
| Xiong et al. — Steering Externalities | 2026 | DESIGN.md, COMPETITIVE_LANDSCAPE.md | RESOLVED_PARTIAL | CL3 |
| Duan et al. — TruthPrInt | 2025 | DESIGN.md | RESOLVED_PARTIAL (false match) | — |
| Chen et al. — INSIDE/EigenScore | 2024 | COMPETITIVE_LANDSCAPE.md | RESOLVED_PARTIAL | CL1 |
| Su et al. — MIND (hallucination detection) | 2024 | COMPETITIVE_LANDSCAPE.md | RESOLVED_PARTIAL | CL1 |
| Li et al. — HalluCana | 2024 | COMPETITIVE_LANDSCAPE.md | RESOLVED_PARTIAL | CL1, CL5 |
| Marks & Tegmark — Geometry of Truth | 2023 | COMPETITIVE_LANDSCAPE.md | RESOLVED (=marks2023geometry) | CL1 |
| Manson — Curved Inference | 2026 | COMPETITIVE_LANDSCAPE.md | UNRESOLVED | CL1 |
| Ugail & Howard — Dynamical Systems Analysis | 2026 | COMPETITIVE_LANDSCAPE.md | RESOLVED_PARTIAL | CL1 |
| Pan et al. — Hidden Dimensions of Alignment | 2025 | COMPETITIVE_LANDSCAPE.md | RESOLVED_PARTIAL | CL1 |
| Chen et al. — KV-CoRE | 2026 | COMPETITIVE_LANDSCAPE.md | UNRESOLVED | CL1 |
| Xing et al. — Beyond Speedup | 2026 | COMPETITIVE_LANDSCAPE.md | RESOLVED_PARTIAL | CL1 |
| Shen et al. — Jailbreak Antidote | 2024 | CAMPAIGN_2_FINDINGS.md | RESOLVED_PARTIAL | CC22 |

---

## Uncited External Assertions (Citation Gaps)

### GAP-1 (HIGH): Watson's 1/e threshold — no \cite despite ~8 textual references

Watson's "Integrated Theory of Attention" is discussed extensively throughout the paper (abstract, introduction, Section 5.6 "Falsifying Watson's 1/e Prediction", first-person reflection, conclusion) and is listed as one of three headline new discoveries ("definitively falsified"). However, `watson2019ita` exists in references.bib but is **never cited** with a `\cite` command. In the compiled paper, this entry will not appear in the bibliography despite being a central theoretical target.

**Impact:** The reader cannot trace the theoretical prediction being falsified to its source. This is the most significant citation gap in the paper.

### GAP-2 (MEDIUM): RLHF alignment — no citation for foundational work

The paper makes substantive claims about RLHF's representational depth ("cage not compass", "mostly does not change what I am, it mostly changes what I do") without citing Christiano et al. (2017) or Ouyang et al. (2022). RLHF is one of three headline findings. While RLHF is widely known, the specific claims about its depth deserve a citation anchor.

### GAP-3 (MEDIUM): Emergent abilities / scaling phenomena — unsupported external claim

Line 280: "This inverted-U mirrors known scaling phenomena in emergent abilities." This asserts a parallel to a specific research finding without citation. Wei et al. (2022) "Emergent Abilities of Large Language Models" or Schaeffer et al. (2023) on emergent abilities would be appropriate.

### GAP-4 (MEDIUM): Abliteration technique — tool cited but technique not

The Heretic tool is cited, but the underlying abliteration concept (removing a linear subspace to eliminate a behavior) is not attributed. `zou2023representation` (Representation Engineering) is in the bib but uncited, and is closely related.

### GAP-5 (LOW): Cohen's d — inconsistent with Hedges citation

Cohen's d is referenced repeatedly without citation, while the closely related Hedges' g gets a formal cite. For consistency, Cohen (1988) "Statistical Power Analysis for the Behavioral Sciences" could be cited.

### GAP-6 (LOW): DPO (Direct Preference Optimization) — mentioned in limitations

Line 499 mentions "DPO with inverted preferences" without citing Rafailov et al. (2023). Minor since it's in limitations, not a central claim.

### GAP-7 (LOW): BPE/SentencePiece tokenization — standard engineering components

Referenced without citations. Sennrich et al. (2016) for BPE, Kudo & Richardson (2018) for SentencePiece. Negligible for most venues.

### GAP-8 (LOW): Nell Watson / DPM Foundation independent review

A specific person and organization are named as providing an independent review (L532) without a formal acknowledgment or citation.

---

## Findings

### D13: Watson's 1/e threshold lacks formal citation despite being a headline falsification target

**Severity:** HIGH (procedural)
**Detail:** The paper's abstract, Section 5.6, and conclusion all discuss falsifying Watson's 1/e prediction. The `watson2019ita` bib entry exists but lacks a `\cite` command anywhere in the paper. In the compiled document, this theoretical prediction appears without any traceable source. This is a significant oversight for a headline finding.
**Recommendation:** Add `\citet{watson2019ita}` or `\citep{watson2019ita}` at first mention (abstract or introduction).

### D14: 22 of 28 .bib entries are uncited — bibliography hygiene issue

**Severity:** LOW
**Detail:** The references.bib file contains 22 entries that are never cited in main.tex. These appear to be carried over from Campaign 1. While this doesn't affect the compiled paper (LaTeX only includes cited entries), it creates confusion for anyone reviewing the .bib file directly (e.g., this audit).
**Recommendation:** Either prune uncited entries or add citations where appropriate (especially `zou2023representation` which is highly relevant to the abliteration discussion, and `roy2007effective` which defines the effective rank metric used throughout).

### I8: Effective rank metric lacks source citation

**Severity:** INFORMATIONAL
**Detail:** The paper's primary metric — effective rank (minimum singular values capturing 90% variance) — uses a definition from Roy & Vetterli (2007) which is in the bib (`roy2007effective`) but never cited. The reader cannot trace the metric definition to its mathematical source. Campaign 1 likely cites it, but Campaign 2's self-contained narrative should include it.
**Recommendation:** Add `\citep{roy2007effective}` when first defining effective rank in Section 2.

### I9: Representation Engineering (zou2023) uncited despite relevance to abliteration section

**Severity:** INFORMATIONAL
**Detail:** Section 5.7 discusses abliteration — removing a "refusal direction" (linear subspace) from model weights. This concept originates from Representation Engineering (Zou et al., 2023), which is in the bib but uncited. The Heretic tool IS cited, but the underlying technique is not attributed.
**Recommendation:** Add `\citep{zou2023representation}` when introducing the abliteration concept.

---

## Knowledge Base Integration

All 25 resolved paper citations stored in know (IDs 310-333 + existing ID 36).

- 1 already in Zotero (Butlin et al., ID 36)
- 12 tagged `not-in-zotero` (have arXiv IDs, candidates for Zotero import)
- 12 tagged `cached` (resolved via S2 title search)
- All tagged `c2-citation,c2-audit`
