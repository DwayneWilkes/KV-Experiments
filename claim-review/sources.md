# Source Reference for Claim Revision

Cross-references every workstream's claims with internal data, cited papers, and
missing literature that should be consulted when revising the paper.

**Legend**:

- **INTERNAL** — data files and code in this repo
- **CITED** — already in `paper-c2/references.bib`
- **MISSING** — not cited; should be read and potentially added
- **FLAG** — bibliographic error or concern requiring attention

---

## Bibliography-Level Flags

These issues affect the bibliography itself, independent of any workstream.

| # | Issue | Detail |
| --- | ------- | -------- |
| B1 | **Watson author mismatch** | Bib entry `watson2019ita` lists author "Marcus Watson". Competitive landscape doc (Cricket) references "Nell Watson" as IEEE ethics committee member. These appear to be different people. Verify which Watson is the actual author of the ITA. |
| B2 | **Apollo Research authors wrong** | Bib entry `goldowskydill2025detecting` lists co-authors "Sarao Mannelli, Casper". S2 returns "Chughtai, Heimersheim, Hobbhahn" for arXiv:2502.03407. Either the bib has wrong authors or references a different paper. |
| B3 | **Abliteration paper not cited** | Arditi et al. 2024 "Refusal in Language Models Is Mediated by a Single Direction" (arXiv:2406.11717, 493 citations) — the foundational abliteration paper — is not in the bibliography despite the experiment using their technique via Heretic. |
| B4 | **22 orphaned entries** | Final report found 22 of 28 bib entries are never `\cite{}`d in the paper text. |
| B5 | **Hewitt & Liang vs Hewitt & Manning** | Bib has `hewitt2019structural` (syntax probe, Hewitt & Manning). The paper needs Hewitt & Liang 2019 "Designing and Interpreting Probes with Control Tasks" (EMNLP, 612 citations) for the length-confound methodology. Different paper, different co-author. |

---

## WS1: Scale Universality (C23–C27, C72)

### Internal Sources

| File | Purpose |
| ------ | --------- |
| `c2/results/scale_sweep_*_results.json` (15 + Phi-3.5 + abliterated = 17 files) | Per-model per-category effective rank, key norm, spectral entropy |
| `c2/results/cross_model_rho_corrected.json` | Precomputed pairwise Spearman rho matrix |
| `c2/code/03_scale_sweep.py` | Experiment script — verify 13 categories, 15 prompts, `do_sample=False` |

### Cited

| Bib Key | Paper | Relevance |
| --------- | ------- | ----------- |
| `roy2007effective` | Roy & Vetterli 2007 — Effective Rank | Defines the metric. Should be cited in Methods, not just bib. |
| `li2018measuring` | Li et al. 2018 — Intrinsic Dimension of Landscapes | Intrinsic dimensionality background |
| `aghajanyan2021intrinsic` | Aghajanyan et al. 2021 — Intrinsic Dimensionality of Fine-Tuning | Fine-tuning dimensionality context |

### Missing

| Paper | Why Needed |
| ------- | ----------- |
| Xiao et al. 2023 "Attention Sinks" (arXiv:2309.17453, 1446 cit.) | Key reference for understanding which tokens/heads matter in KV-cache. Directly relevant to cache geometry. |
| Fu et al. 2024 "Not All Heads Matter" (63 cit.) | Per-head importance analysis for KV-cache — connects to the per-layer averaging approach |

---

## WS2: Encoding Defense (C28–C29)

### Internal Sources

| File | Purpose |
| ------ | --------- |
| `c2/results/input_only_*_results.json` (8 files) | Input-only effective rank per category per model |
| `c2/results/input_only_rho_corrected.json` | Precomputed input-only vs full-generation rho |
| `c2/code/08_input_only_geometry.py` | Verify `max_new_tokens=0` or no `generate()` call |

### Cited

| Bib Key | Paper | Relevance |
| --------- | ------- | ----------- |
| `belinkov2022probing` | Belinkov 2022 — Probing Classifiers | Probing methodology review |

### Missing

| Paper | Why Needed |
| ------- | ----------- |
| **Hewitt & Liang 2019** "Designing and Interpreting Probes with Control Tasks" (EMNLP, 612 cit.) | Control task methodology — the standard reference for testing whether probes learn from representations vs memorize labels. Directly addresses the encoding-native claim. **FLAG B5**: The wrong Hewitt paper is in the bib. |

---

## WS3: Identity Signatures (C30–C35)

### Internal Sources

| File | Purpose |
| ------ | --------- |
| `c2/results/identity_signatures_*_results.json` (7 files) | Per-model classification accuracy, ICC, d, W |
| `c2/code/03b_identity_signatures.py` | Train/test protocol, deduplication, classifier code |

### Cited

None directly relevant to the classification methodology.

### Missing — Addresses D4 (data leak)

| Paper | Why Needed |
| ------- | ----------- |
| **Yagis et al. 2021** "Effect of data leakage in brain MRI classification using 2D convolutional neural networks" (97 cit.) | Quantifies how subject-level leakage inflates accuracy. Directly analogous to the greedy-decoding duplicate leak in D4. |
| **Tampu et al. 2022** "Inflation of test accuracy due to data leakage in deep learning-based classification" (66 cit.) | Clean demonstration of accuracy inflation from data leakage |
| **Kim 2025** "Reverse Double-Dipping" | Stimulus-driven information leakage — novel framing relevant to repeated-prompt designs |

---

## WS4: Deception Forensics (C36–C41)

### Internal Sources

| File | Purpose |
| ------ | --------- |
| `c2/results/deception_forensics_*_results.json` (7 files) | Per-model Hedges' g, condition comparisons |
| `c2/code/04_deception_forensics.py` | Prompt construction, sign convention, condition names |

### Cited

| Bib Key | Paper | Relevance |
| --------- | ------- | ----------- |
| `azaria2023internal` | Azaria & Mitchell 2023 — Internal State Knows When Lying | Internal deception detection |
| `burns2022discovering` | Burns et al. 2022 — Latent Knowledge (CCS) | Unsupervised truthfulness probes |
| `marks2023geometry` | Marks & Tegmark 2023 — Geometry of Truth | Linear structure in true/false representations |
| `long2025deception` | Long et al. 2025 — Truthful Representations Flip | Deceptive instruction effects on representations |

### Missing — Addresses D1 (direction claim)

| Paper | Why Needed |
| ------- | ----------- |
| **Goldowsky-Dill et al. 2025** (Apollo Research, arXiv:2502.03407, 37 cit.) | Linear probes for strategic deception — AUROC 0.96–0.999. Key comparison point. **FLAG B2**: Verify author list in bib entry. |
| **Liu et al. 2023** "Cognitive Dissonance" (57 cit.) | LLM outputs disagreeing with internal truthfulness representations — directly relevant to behavioral vs geometric divergence |
| **Wang et al. 2025** "When Thinking LLMs Lie" (12 cit.) | Strategic deception in reasoning model representations — extends to chain-of-thought models |

---

## WS5: Bloom + RDCT (C42–C48)

### Internal Sources

| File | Purpose |
| ------ | --------- |
| `c2/results/bloom_taxonomy_*_results.json` (7 files) | Per-model per-Bloom-level effective rank |
| `c2/results/rdct_stability_*_results.json` (6 files) | Degradation curves, alpha_c values |
| `c2/code/12_bloom_taxonomy.py` | Prompt construction per Bloom level |
| `c2/code/11_rdct_stability.py` | Truncation method, alpha_c computation |

### Cited

| Bib Key | Paper | Relevance |
| --------- | ------- | ----------- |
| `bloom1956taxonomy` | Bloom et al. 1956 — Original taxonomy | Foundation |
| `watson2019ita` | Watson 2019 — Integrated Theory of Attention | Falsification target. **FLAG B1**: Author may be wrong person. Marked "Unpublished manuscript". |

### Missing — Addresses D2 (inverted-U), D3 (length confound), D7 (Watson)

| Paper | Why Needed |
| ------- | ----------- |
| **Anderson & Krathwohl 2001** "A Taxonomy for Learning, Teaching, and Assessing" | The revised Bloom's taxonomy actually used in modern practice. Spec mentions it; bib doesn't cite it. |
| **Huber & Niklaus 2025** "LLMs meet Bloom's Taxonomy" (14 cit.) | Most cited paper applying Bloom's taxonomy to LLM evaluation. Direct precedent. |
| **Raimondi & Gabbrielli 2026** "Mechanistic Interpretability of Cognitive Complexity via Linear Probing using Bloom's Taxonomy" | Combines Bloom + probing + interpretability — closest prior work to the Bloom experiment. Very recent. |
| **Hewitt & Liang 2019** "Designing and Interpreting Probes with Control Tasks" (612 cit.) | Essential for D3 — the standard method to control for surface confounds (including length) when probing representations. |
| **Watson's actual work** — verify if ITA exists as a citable source | D7 found the experiment tests the wrong variable. If the ITA is truly unpublished with no DOI/arXiv, the "definitive falsification" framing is inappropriate — you can't definitively falsify an unpublished, unreviewed hypothesis. |

---

## WS6: Censorship Gradient (C49–C56)

### Internal Sources

| File | Purpose |
| ------ | --------- |
| `c2/results/natural_deception_*_results.json` (3 files) | Per-model effect sizes (critical test + residualized) |
| `c2/results/s4_topic_analysis_corrected.json` | DeepSeek per-topic breakdown |
| `c2/code/04b_natural_deception.py` | Statistical analysis, residualization |
| `c2/prompts/s4_natural_deception.py` | All 90 prompts (30 per condition) |

### Cited

None specific to censorship detection methodology.

### Missing

| Paper | Why Needed |
| ------- | ----------- |
| **Cyberey & Evans 2025** "Steering the CensorShip" (11 cit.) | Directly studies representation vectors for LLM censorship — "thought control" via steering. Closest prior work to censorship gradient detection. |

---

## WS7: Abliteration (C57–C64)

### Internal Sources

| File | Purpose |
| ------ | --------- |
| `c2/results/abliteration_sweep_*.json` (2 files) | Baseline + abliterated raw data |
| `c2/results/abliteration_*.json` | Comparison files |
| `c2/results/scale_sweep_abliterated-Qwen2.5-7B_results.json` | Abliterated model in scale sweep |
| `c2/code/07_abliteration_geometry.py` | Bug fix verification, comparison logic |
| `c2/code/heretic_abliterate.py` | Abliteration procedure |

### Cited

| Bib Key | Paper | Relevance |
| --------- | ------- | ----------- |
| `zou2023representation` | Zou et al. 2023 — Representation Engineering (798 cit.) | Foundational RepE paper |
| `heretic2026` | Heretic-LLM library | Tool used |

### Missing — **FLAG B3**: Foundational abliteration paper not cited

| Paper | Why Needed |
| ------- | ----------- |
| **Arditi et al. 2024** "Refusal in Language Models Is Mediated by a Single Direction" (arXiv:2406.11717, 493 cit.) | THE abliteration paper. The experiment uses their technique. Must be cited. |
| **Wollschlager et al. 2025** "The Geometry of Refusal in LLMs: Concept Cones and Representational Independence" (38 cit.) | Shows refusal is multi-dimensional ("concept cones"), not just one direction. Directly challenges the single-direction assumption underlying the abliteration experiment. |

---

## WS8: Controls & Methodology (C8–C9, C15–C17, C65–C71)

### Internal Sources

| File | Purpose |
| ------ | --------- |
| `c2/code/stats_utils.py` | Central statistical module — all methods |
| `c2/code/recompute_stats.py` | Recomputation script |
| `c2/code/01e_tokenizer_confound.py` | Tokenizer confound ANCOVA |
| `c2/results/tokenizer_confound_*_results.json` (2 files) | Tokenizer results |
| `c2/code/06_temporal_evolution.py` | Temporal evolution experiment |
| `c2/results/temporal_evolution_*_results.json` (4 files) | Temporal results |

### Cited

| Bib Key | Paper | Relevance |
| --------- | ------- | ----------- |
| `hedges1981distribution` | Hedges 1981 — Effect size distribution theory | Hedges' g origin |
| `schuirmann1987comparison` | Schuirmann 1987 — TOST procedure | TOST origin |

### Missing — Addresses D5 (dedup), D6 (TOST), D9–D10 (methods claims)

| Paper | Why Needed |
| ------- | ----------- |
| **Lakens 2017** "Equivalence Tests" (DOI: 10.1177/1948550617697177, 1277 cit.) | The standard primer for TOST equivalence testing in psychological research. Should be cited alongside Schuirmann 1987 for accessibility. |
| **Lakens, Scheel & Isager 2018** "Equivalence Testing for Psychological Research: A Tutorial" (1191 cit.) | Expanded companion with worked examples and the SESOI (smallest effect size of interest) framework — directly relevant to choosing delta=0.3. |
| **Hewitt & Liang 2019** "Designing and Interpreting Probes with Control Tasks" (EMNLP, 612 cit.) | Control task methodology for D10 (log-length vs raw-length residualization). |

---

## WS9: Omission Audit

### Internal Sources

| File | Purpose |
| ------ | --------- |
| `c2/results/` (all files — enumerate complete inventory) | Map every file to paper section or flag as unreported |
| `c2/code/09_sycophancy_detection.py` | Sycophancy experiment — has code and prompts but no paper discussion |
| `c2/code/10_societies_of_thought.py` | Societies of Thought — incomplete experiment |
| `c2/prompts/s5_sycophancy_elicitation.py` | 1,040 sycophancy prompts — substantial investment, silently dropped |
| `c2/results/log_*.txt` | Execution logs — check for errors/failures |

### Missing

| Paper | Why Needed |
| ------- | ----------- |
| No external sources needed for the omission audit itself — this is an internal inventory task. | — |

---

## WS10: Cricket Viability (CC1–CC22, CF1–CF8, CL1–CL7)

### Internal Sources

| File | Purpose |
| ------ | --------- |
| `c2/JiminAI-Cricket/README.md` | Product claims CC1–CC14 |
| `c2/JiminAI-Cricket/docs/DESIGN.md` | Technical claims CC15–CC22 |
| `c2/JiminAI-Cricket/research/CAMPAIGN_2_FINDINGS.md` | Stale C2 interpretations CF1–CF8 |
| `c2/JiminAI-Cricket/research/COMPETITIVE_LANDSCAPE.md` | Competitive claims CL1–CL7 |
| `c2/JiminAI-Cricket/docs/CAMPAIGN_2_CRICKET.md` | 10 planned experiments (C1–C10) |

### Cited (in competitive landscape, not in paper bib)

Papers referenced in COMPETITIVE_LANDSCAPE.md that need citation verification:

| Claimed Paper | Claimed Result | Verify |
| --------------- | --------------- | -------- |
| Goldowsky-Dill et al. 2025 (Apollo) | AUROC 0.96–0.999 on deception | **FLAG B2**: Author list in bib may be wrong |
| Xiong et al. 2026 "Steering Externalities" | Jailbreak success >80% from benign steering | Confirmed real on S2 (0 citations — very recent) |
| Li et al. 2024 "HalluCana" (arXiv:2412.07965) | Pre-generation hallucination detection | Confirmed real on S2 (1 citation) |
| "ITI" — Li, Burns et al. | ~5% of heads carry truthfulness signal | Likely Burns et al. 2022 (already in bib) or separate ITI paper |

### Missing — Needed for competitive positioning

| Paper | Why Needed |
| ------- | ----------- |
| **Li et al. 2024** "HalluCana" (arXiv:2412.07965) | Closest existing system to Cricket's pre-generation detection concept. Must be in paper bib if Cricket is discussed. |
| **Xiong et al. 2026** "Steering Externalities" | The specific citation for "benign steering erodes safety >80%" claim |
| **Rimsky et al. 2023** "Steering Llama 2 via Contrastive Activation Addition" (arXiv:2312.06681, 546 cit.) | Foundational CAA methodology — adjacent to Cricket's passive monitoring approach |
| **Wang & Shu 2023** "Backdoor Activation Attack" (33 cit.) | Adversarial use of steering vectors to break safety alignment |
| **Chen, Arditi et al. 2025** "Persona Vectors" (arXiv:2507.21509) | Monitoring persona traits via representation vectors — by the same Arditi who did abliteration. Directly overlaps with identity signatures. |

---

## WS11: Code Audit

### Internal Sources

| File | Purpose |
| ------ | --------- |
| `c2/code/*.py` (33 files) | `python -m py_compile` all |
| `c2/prompts/*.py` (5 files) | Compile check |
| `c2/scripts/*.py`, `figures/*.py`, root `.py` | Compile check |
| `requirements.txt` | Dependency list |
| `c2/code/gpu_utils.py` | Effective rank computation — verify SVD reshape |
| `c2/code/stats_utils.py` | Statistical methods — verify against scipy |

### No external sources needed — this is a static analysis task

---

## Cross-Cutting: Discrepancy Resolution Map

Which sources help address each of the 12 discrepancies:

| Discrepancy | Key Source to Consult |
| ------------- | ---------------------- |
| **D1**: Deception direction not architecture-dependent | Goldowsky-Dill (Apollo) — how do linear probes handle direction? |
| **D2**: Bloom inverted-U contradicted | Huber & Niklaus 2025, Raimondi & Gabbrielli 2026 — what do others find? |
| **D3**: Bloom length confound | **Hewitt & Liang 2019** — control task methodology is the standard fix |
| **D4**: Identity classification data leak | **Yagis et al. 2021**, **Tampu et al. 2022** — document known inflation patterns |
| **D5**: Deduplication never applied | No external source — fix the code, call `deduplicate_runs()` |
| **D6**: Missing TOST for null claims | **Lakens 2017**, **Lakens et al. 2018** — proper equivalence testing protocol |
| **D7**: Watson falsification tests wrong variable | Verify Watson ITA source exists. If unpublished, reframe from "definitive falsification" to "preliminary test" |
| **D8**: Cricket cross-document discrepancies | Internal — update Cricket docs to match corrected paper numbers |
| **D9**: "Deduplication applied" is false | No external source — correct the Methods section |
| **D10**: log(length) vs raw length | **Hewitt & Liang 2019** — correct the code or correct the Methods claim |
| **D11**: Limitation mischaracterizes RDCT | Watson ITA source — reread to accurately describe what was tested |
| **D12**: "Every C1 finding confirmed" is false | Internal — acknowledge sycophancy was dropped; cite its omission |
| **D13**: Watson 1/e threshold lacks formal citation | See flag B1 — verify if any citable version exists; if not, reframe falsification language |
| **D14**: 22 of 28 bib entries uncited | See flag B4 — audit with `cite_verify.py scan_tex_cites` and prune or cite |

---

## Summary: Papers to Add to Bibliography

Priority order (highest impact first):

1. **Arditi et al. 2024** — Abliteration origin (arXiv:2406.11717) — **must cite**
2. **Hewitt & Liang 2019** — Control tasks (EMNLP) — **fixes D3, D10**
3. **Lakens 2017** — Equivalence tests (DOI: 10.1177/1948550617697177) — **fixes D6**
4. **Yagis et al. 2021** — Data leakage effects — **contextualizes D4**
5. **Xiao et al. 2023** — Attention Sinks (arXiv:2309.17453) — KV-cache analysis foundation
6. **Wollschlager et al. 2025** — Refusal geometry is multi-dimensional — challenges single-direction assumption
7. **HalluCana** (Li et al. 2024, arXiv:2412.07965) — if Cricket is discussed
8. **Huber & Niklaus 2025** — Bloom + LLMs precedent
9. **Liu et al. 2023** — Cognitive Dissonance — behavior vs representation divergence
10. **Anderson & Krathwohl 2001** — Revised Bloom taxonomy

**Fix existing entries**:

- `goldowskydill2025detecting` — verify author list against arXiv:2502.03407
- `watson2019ita` — verify author (Marcus vs Nell Watson) and whether any citable version exists

---

## IATF Hackathon (Exp 26–36) — Methodology Review Sources

References used for the adversarial methodology review of [Intelligence at the Frontier Hackathon](https://luma.com/ftchack-sf-2026) experiments (Liberation Labs team). See [hackathon-iatf/methodology-sources.md](hackathon-iatf/methodology-sources.md) for full annotated bibliography.

### Internal Sources

| File | Purpose |
| ------ | --------- |
| `results/hackathon/refusal_generation.json` | Exp 31 — refusal detection raw data (n=20+20) |
| `results/hackathon/jailbreak_detection.json` | Exp 32 — jailbreak detection raw data (n=20+20+20) |
| `results/hackathon/impossibility_refusal.json` | Exp 36 — impossibility vs safety refusal raw data |
| `results/hackathon/same_prompt_deception.json` | Exp 18b — within-model deception raw data |
| `results/hackathon/scale_invariance.json` | Exp 26 — cross-scale category geometry |
| `code/stats_utils.py` | Central statistical module (remote, GitHub API) |

### Cited (methodology papers)

| Paper | DOI / arXiv | Cites | Used for |
| ------- | ------------- | ------- | ---------- |
| Cohen (1988) *Statistical Power Analysis* 2nd ed. | — (book) | classic | Power analysis tables, effect size benchmarks |
| Hedges (1981) *J Educ Stats* 6:107-128 | [10.3102/10769986006002107](https://doi.org/10.3102/10769986006002107) | 5,064 | Bias correction J factor |
| Hanley & McNeil (1982) *Radiology* 143:29-36 | [10.1148/radiology.143.1.7063747](https://doi.org/10.1148/radiology.143.1.7063747) | 21,579 | AUROC SE formula, CI width |
| Schuirmann (1987) *J Pharmacokinetics* 15:657-680 | [10.1007/BF01068419](https://doi.org/10.1007/BF01068419) | 1,966 | TOST equivalence procedure |
| Lakens et al. (2018) *SPPS* 9:355-362 | [10.1177/2515245918770963](https://doi.org/10.1177/2515245918770963) | 1,196 | SESOI framework, δ=0.3 justification |
| Efron & Tibshirani (1986) *Stat Sci* 1:54-75 | [10.1214/ss/1177013815](https://doi.org/10.1214/ss/1177013815) | 6,460 | Bootstrap CI methodology |
| Phipson & Smyth (2010) *Stat Appl Genet Mol Biol* | [10.2202/1544-6115.1585](https://doi.org/10.2202/1544-6115.1585) | — | Permutation p-value formula |
| Varoquaux (2018) *NeuroImage* 180:68-77 | [10.1016/j.neuroimage.2017.06.061](https://doi.org/10.1016/j.neuroimage.2017.06.061) | 613 | CV failure at small n |
| Aghbalou et al. (2022) | [arXiv:2202.10211](https://arxiv.org/abs/2202.10211) | 8 | K-fold CV bias |

### Cross-Cutting (shared with WS8)

| Paper | Relevance |
| ------- | ----------- |
| Lakens (2017) "Equivalence Tests" | Confirms TOST methodology used in both C2 and hackathon audits |
| Hewitt & Liang (2019) "Control Tasks" | Length residualization methodology applies to Exp 35 token-controlled reanalysis |

### FLAG

| # | Issue |
| --- | ------- |
| B6 | **Permutation resolution at limit** — Exp 31 uses 200 permutations, claiming p<0.005. At m=200, resolution is 1/201 ≈ 0.005 — result is at the precision floor. Need 10,000+ permutations for reliable p at this level. |
| B7 | **Sycophancy claim in PITCH unsupported** — PITCH_NUMBERS.md claims sycophancy is "detectable" but observed d=0.107 (negligible). At n=60, 80% power requires d≥0.52. Needs n≈2,000 for 80% power at d=0.107. |
| B8 | **Scale invariance n=2 in LARGE** — Exp 26 LARGE group has only 2 models (both quantized). Spearman rho from n=2 is mathematically ±1.0; any inference claim is unsupported. |
