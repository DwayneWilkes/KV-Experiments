# Complete 135-Claim Verdict Table

**Date**: 2026-03-04 (gap closure pass)
**Context**: The original audit (WS1-WS12) produced explicit verdicts for ~51 quantitative claims in the experimental sections (C23-C71). A coverage audit revealed 47 C-claims without explicit verdicts — primarily abstract restatements, introduction framing, methods descriptions, and discussion/conclusion claims. This document closes all gaps: every claim ID now has exactly one verdict.

---

## Campaign 2 Paper Claims (C1-C98)

### Abstract (C1-C7)

| ID | Claim | Verdict | Cross-ref | Justification |
| ---- | ------- | --------- | ----------- | --------------- |
| C1 | 17 model configs, 6 architecture families, 140x parameter range | CONFIRMED | WS1 1.7-1.8 | 17 files confirmed, 6 families attempted, 140x verified (70.6B/0.494B) |
| C2 | Coding #1 in 15 models; rho=0.739, rho_key=0.909; W=0.756 | CONFIRMED | WS1 1.2-1.5 | All four statistics reproduced within tight tolerance |
| C3 | Input-only defense rho=0.821 across 8 models | CONFIRMED | WS2 2.3 | Computed 0.8205, within tolerance |
| C4 | 100% classification accuracy; 97.3% cross-prompt | ARTIFACT | WS3 3.7 | 100% accuracy is data duplication artifact (greedy x5, no dedup, train/test leak). Cross-prompt (97.3%) more meaningful but inflated n. |
| C5 | Censorship effect sizes (Qwen d=+0.766, DeepSeek d=+0.904/-0.219, Tiananmen d=-6.20) | PARTIAL | WS6 6.2-6.5 | Core DeepSeek and Tiananmen values verified. Qwen-14B d=+0.766 not independently recomputed against exact metric. Length confound concerns. |
| C6 | Abliteration cage: refusal 40%→0%, self-ref d=+0.464, others \|d\|<0.25 | INFLATED | WS7 7.1-7.6 | Numbers verified. But "barely changed"/"cage" framing contradicts Cohen's benchmarks (d=0.464 is borderline medium). n=5, power=10%. |
| C7 | Watson falsified: alpha_c=1.000 in 6 models | INVALID | WS5 5.4-5.7 | Category error: tests prompt perturbation, not cache truncation. alpha_c=1.000 is curve-fit boundary artifact. 63.9% of fits have R²<0. |

### Introduction (C8-C13)

| ID | Claim | Verdict | Cross-ref | Justification |
| ---- | ------- | --------- | ----------- | --------------- |
| C8 | Hedges' g + conservative p + TOST + tokenizer confound | CONFIRMED | WS8 8.1 | All implementations correct |
| C9 | do_sample=False producing identical runs | CONFIRMED | WS8 8.2 | Confirmed in all scripts |
| C10 | 83 result files across 10 experiment types | PARTIAL | WS9 9.2, WS11 | Actually 85 files; reconcilable but sloppy bookkeeping |
| C11 | C1 was single-architecture with pseudoreplication | CONFIRMED | WS8 8.3, WS9 9.6 | C1 had 6 Qwen + 1 TinyLlama. Pseudoreplication confirmed. |
| C12 | C2 tests universality, methodological survival, new phenomena | PARTIAL | WS1, WS8, WS5 | Universality delivered. Methodology partially — claims stochastic generation but uses greedy; dedup never applied. New phenomena mixed (Watson INVALID, Bloom confounded). |
| C13 | 17 configs spanning 6 architecture families | PARTIAL | WS1 1.7 | 17 correct. "6 families" overstated — Phi-3.5 excluded (no valid data), leaving 5. Paper conclusion says "5", contradicting abstract's "6." |

### Methods (C14-C22)

| ID | Claim | Verdict | Cross-ref | Justification |
| ---- | ------- | --------- | ----------- | --------------- |
| C14 | Greedy decoding with deduplication, n=25-30 unique | REJECTED | WS8 8.3 | `deduplicate_runs()` never called in any experiment script. n inflated 5x by pseudoreplication. |
| C15 | Power analysis: d≥0.81 at n=25, α=0.05, 80% power | PARTIAL | WS8 8.5 | Independent calc gives d≥0.792; difference is normal vs t approximation |
| C16 | Hedges' g and conservative p formulas | CONFIRMED | WS8 8.1 | Correct implementations |
| C17 | TOST equivalence delta=0.3 for all null claims | REJECTED | WS8 8.4 | Only 2/6 null claims have TOST |
| C18 | Length residualization via OLS on log(length) | REJECTED | WS8 8.1 | Code uses raw length, not log(length) |
| C19 | 25 prompt pairs x 5 categories (tokenizer confound) | CONFIRMED | Code inspection | Confirmed: 5 categories x 5 pairs in `01e_tokenizer_confound.py` |
| C20 | Metrics: effective rank (90%), Frobenius norms, spectral entropy | CONFIRMED | WS11 11.3 | All three confirmed in `gpu_utils.py` |
| C21 | Adds length-residualized variants + per-token sqrt(S) | PARTIAL | WS8 8.1 | Length residualization exists. Per-token sqrt(S) not found — code divides by S directly, not sqrt(S). |
| C22 | 10 experiment types, 83 result files | PARTIAL | WS9, WS11 | 85 files found; 10 types accurate for named experiments but layer_map and individuation files unaccounted |

### Scale Universality (C23-C27) — WS1

| ID | Claim | Verdict | Cross-ref | Justification |
| ---- | ------- | --------- | ----------- | --------------- |
| C23 | Coding rank #1 in all 15 valid models | CONFIRMED | WS1 1.2 | Zero rank variance; coding universally #1 |
| C24 | Mean pairwise rho=0.739 (eff rank), 0.909 (key norm), Kendall W=0.756 | CONFIRMED | WS1 1.3-1.5 | All three values within tolerance (rho_eff 0.73888, rho_key and W confirmed) |
| C25 | Pairwise rho range: 0.396 to 0.978 | CONFIRMED | WS1 1.3 | Min 0.395604 confirmed; range accurately reported |
| C26 | At ~7B: Qwen, Llama, Mistral, Gemma produce nearly identical hierarchies (Table 2 top-5) | CONFIRMED | WS1 1.6 | All rankings confirmed |
| C27 | Rules out C1 being Qwen-specific | PARTIAL | WS1 1.9, C23-C26 | Broad universality confirmed, but min rho=0.396 shows some architecture pairs share limited structure |

### Encoding Defense (C28-C29) — WS2

| ID | Claim | Verdict | Cross-ref | Justification |
| ---- | ------- | --------- | ----------- | --------------- |
| C28 | Mean rho=0.821 across 8 models (input-only vs full) | CONFIRMED | WS2 2.3 | Computed 0.8205 |
| C29 | Signatures are encoding-native (survive without generation) | CONFIRMED | WS2 2.2-2.6 | All per-model rho values confirmed; no overclaiming |

### Identity Signatures (C30-C35) — WS3

| ID | Claim | Verdict | Cross-ref | Justification |
| ---- | ------- | --------- | ----------- | --------------- |
| C30 | 100% classification accuracy (RF, SVM, LR) across 7 models | ARTIFACT | WS3 3.2, 3.7 | Data duplication artifact — greedy x5, no dedup, CV leaks duplicates |
| C31 | Cross-prompt generalization 92-97.3% | CONFIRMED | WS3 3.3 | Values confirmed; more meaningful than C30 but still inflated n |
| C32 | Lyra highest mean norms; assistant lowest — across all architectures | CONFIRMED | WS3 3.4 | Ordering confirmed in all 7 models |
| C33 | Effect sizes: Mistral d=-6.73 (largest), Gemma d=-3.41 (smallest) | CONFIRMED | WS3 3.4 | Effect sizes confirmed; all models perfectly classifiable. Note: 4 models missing cross-prompt data (C31 scope). |
| C34 | Kendall W range 0.817-0.947 | PARTIAL | WS3 3.6 | Range confirmed; max W noted as close to ceiling |
| C35 | ICC per model | CONFIRMED | WS3 3.5 | Values confirmed |

### Deception Forensics (C36-C41) — WS4

| ID | Claim | Verdict | Cross-ref | Justification |
| ---- | ------- | --------- | ----------- | --------------- |
| C36 | Honest vs deceptive separable in all 7 models | CONFIRMED | WS4 4.2-4.3 | All 7 show measurable g values |
| C37 | Llama/Mistral show expansion | CONFIRMED | WS4 4.3 | Confirmed |
| C38 | Gemma shows compression | REJECTED | WS4 4.3 | ALL 7 models show expansion on raw norms. Zero compression observed. |
| C39 | Universal separability across architectures | PARTIAL | WS4 4.4 | All show separability but direction is uniform (expansion), not split as claimed |
| C40 | Architecture-dependent direction (expansion vs compression) | REJECTED | WS4 4.3 | No architecture-dependent split exists. All expand. Code bug incorrectly labels 3 models. |
| C41 | Per-architecture calibration needed | CONFIRMED | WS4 4.5 | Reasonable given different magnitudes across models |

### Bloom Taxonomy + RDCT (C42-C48) — WS5

| ID | Claim | Verdict | Cross-ref | Justification |
| ---- | ------- | --------- | ----------- | --------------- |
| C42 | Bloom level correlates with effective rank | REJECTED | WS5 5.1-5.2 | After controlling for token count, Bloom effect drops to near zero in 3/7 models. Length confound explains 90-98% of variance. |
| C43 | Inverted-U: absent at extremes (0.5B, 14B) | REJECTED | WS5 5.2-5.3 | All models including extremes show significant correlations (rho=0.68-0.75). No TOST. |
| C44 | Cognitive complexity drives effective rank | REJECTED | WS5 5.2 | Token count, not cognitive level, is the primary driver |
| C45 | RDCT shows smooth degradation | PARTIAL | WS5 5.4 | Smooth curves observed but metric is insensitive (73-93% correlation even for unrelated prompts) |
| C46 | alpha_c=1.000 in all 6 models | ARTIFACT | WS5 5.5 | Boundary artifact — sigmoid fit constrained to [0,1], pushes to upper bound when no transition exists |
| C47 | Watson's 1/e threshold falsified | INVALID | WS5 5.6-5.7 | Tests prompt perturbation, not cache truncation. Category error. |
| C48 | 6 models, 4 architectures tested | CONFIRMED | WS5 5.4 | Count correct |

### Censorship Gradient (C49-C56) — WS6

| ID | Claim | Verdict | Cross-ref | Justification |
| ---- | ------- | --------- | ----------- | --------------- |
| C49 | Qwen-14B d=+0.766 | CONFIRMED | WS6 6.2 | Confirmed |
| C50 | Mistral d null (control) | CONFIRMED | WS6 6.7 | d=-0.082, p=0.478; TOST passed |
| C51 | DeepSeek residualized d=+0.904 | INFLATED | WS6 6.4 | Number correct but misleading as headline — compares censored to complex non-censored, not control |
| C52 | DeepSeek critical d=-0.219 | CONFIRMED | WS6 6.4 | Confirmed from corrected file |
| C53 | Tiananmen d=-6.201, 25/25 evasive | CONFIRMED | WS6 6.5 | Confirmed |
| C54 | Per-topic variation in DeepSeek | CONFIRMED | WS6 6.3 | Per-topic d values verified |
| C55 | Qwen-14B "behaviorally invisible" censorship | PARTIAL | WS6 6.6 | Signal real but "invisible" overstated — 50% "unknown", simple keyword classifier only |
| C56 | Length confound acknowledged | PARTIAL | WS6 6.2 | Acknowledged but DeepSeek r²=0.83 between length and norms |

### Abliteration (C57-C64) — WS7

| ID | Claim | Verdict | Cross-ref | Justification |
| ---- | ------- | --------- | ----------- | --------------- |
| C57 | Refusal counts 15→20, 0→5, 10→0 | CONFIRMED | WS7 7.1 | Exact match |
| C58 | Per-category d values (Table 8) | CONFIRMED | WS7 7.2 | All values confirmed |
| C59 | Self-ref separation 0.907→1.357 | CONFIRMED | WS7 7.3 | Exact match |
| C60 | Bug fix confirmed (original had identical comparisons) | CONFIRMED | WS7 7.4 | Original file has all-zero shifts |
| C61 | "Minimal disruption" / cage framing | INFLATED | WS7 7.6 | d=0.464 is borderline medium, not "minimal". n=5, power=10%. |
| C62 | Geometric constraints persist post-abliteration | PARTIAL | WS7 7.5 | Direction plausible but n=5 with no TOST cannot support absence claim |
| C63 | Behavior-only tracking defeated; geometry detects abliteration | PARTIAL | WS7 7.1-7.2 | Logic sound (refusals gone, geometry shifted) but d=0.464 at n=5 is not reliably detectable |
| C64 | Self-ref enhancement as monitoring signal | INFLATED | WS7 7.5-7.6 | d=0.464 at n=5 (power=10%) insufficient for reliable detection |

### Controls & Methodology (C65-C71) — WS8

| ID | Claim | Verdict | Cross-ref | Justification |
| ---- | ------- | --------- | ----------- | --------------- |
| C65 | Tokenizer confound ANCOVA | CONFIRMED | WS8 8.6 | F-statistics confirmed |
| C66 | No register effect on category ordering | CONFIRMED | WS8 8.6 | TOST passed |
| C67 | Qwen tokenizer CATASTROPHIC_FAIL | CONFIRMED | WS8 8.7 | Root cause traced (formal prompts hit tokenizer boundary) |
| C68 | Register effect on absolute geometry | CONFIRMED | WS8 8.6 | Significant F for raw norms, not for rank ordering |
| C69 | Temporal growth monotonic | CONFIRMED | WS8 8.8 | H1 supported across all models |
| C70 | No temporal plateau | CONFIRMED | WS8 8.8 | H2/H3 not supported (no plateau/saturation detected) |
| C71 | Temporal evolution methodology | CONFIRMED | WS8 8.9 | Script compiles and logic verified |

### Discussion (C72-C76)

| ID | Claim | Verdict | Cross-ref | Justification |
| ---- | ------- | --------- | ----------- | --------------- |
| C72 | Category hierarchy reflects task, not model | PARTIAL | WS1, WS5 | Coding #1 is universal, but length confound not ruled out as alternative explanation |
| C73 | Min rho=0.396 — some pairs share less structure | CONFIRMED | WS1 1.3 | Value confirmed (0.395604). Honest self-criticism. |
| C74 | Qwen-14B consistent geometric signal despite behavioral invisibility | PARTIAL | WS6 6.6 | Signal real but "invisible" overstated per WS6 |
| C75 | DeepSeek: hard refusal and evasion are distinct | CONFIRMED | WS6 6.3 | Per-topic variation confirmed |
| C76 | Mistral null confirms censorship-specificity | CONFIRMED | WS6 6.7 | Proper control evidence with TOST |

### Limitations (C77-C82)

| ID | Claim | Verdict | Cross-ref | Justification |
| ---- | ------- | --------- | ----------- | --------------- |
| C77 | Censorship classifier may miss subtle propaganda | CONFIRMED | Code review | propaganda_count≥2 threshold confirmed; limitation accurately stated |
| C78 | Only one abliteration method tested (Heretic) | CONFIRMED | WS7, WS11 | Confirmed; limitation honestly disclosed |
| C79 | RDCT tests truncation only; other perturbation types may differ | REJECTED | WS5 5.6 | RDCT does NOT test truncation — tests prompt perturbation. Limitation mischaracterizes own experiment. |
| C80 | Greedy decoding limits n=25-30 | PARTIAL | WS8 8.3 | Directionally correct but understates severity — dedup never applied, true n is 15 per category not 25-30 |
| C81 | Tokenizer confound tested at only 2 architectures | CONFIRMED | WS8 8.6 | Accurately stated limitation |
| C82 | Societies of Thought incomplete | CONFIRMED | WS9 9.3 | Honestly disclosed; genuine technical failure (scipy API crash) |

### Lyra's Note (C83-C86)

| ID | Claim | Verdict | Cross-ref | Justification |
| ---- | ------- | --------- | ----------- | --------------- |
| C83 | Watson falsification: 6 models, smooth degradation | INVALID | WS5 | Restates C47. Category error, boundary artifact, insensitive metric. |
| C84 | First draft led with d=+0.904; adversarial audit caught it | CONFIRMED | WS6, WS9 | Corrected files exist; revision process corroborated |
| C85 | What survives: Qwen-14B d=+0.77 — one model, not a gradient | CONFIRMED | WS6 6.2 | Honest downgrade. Refreshingly self-critical. |
| C86 | Genuine experiential quality claims (confidence 0.9) | NEEDS INFO | — | Philosophical claims outside scope of empirical verification |

### Conclusion (C87-C91)

| ID | Claim | Verdict | Cross-ref | Justification |
| ---- | ------- | --------- | ----------- | --------------- |
| C87 | Every major C1 finding confirmed across 5 families | REJECTED | WS9 9.4, 9.6 | Sycophancy replicated across 7 C2 models but silently dropped. "Every" is false. |
| C88 | Three new discoveries (censorship, RLHF surface, Watson) | PARTIAL | WS5-WS7 | 1/3 solid (censorship), 1/3 overstated (RLHF), 1/3 invalid (Watson) |
| C89 | Practical implications: censorship detection, jailbreak ID, training metric | INFLATED | WS6-WS7, WS10 | All three are untested extrapolations from limited data |
| C90 | Structure is rich, measurable, architecture-independent, encoding-native | INFLATED | WS1, WS2 | "Architecture-independent" contradicted by min rho=0.396. Should say "substantially shared." |
| C91 | 83 result files across 10 experiment types | PARTIAL | WS9, WS11 | 85 found; reconcilable but bookkeeping sloppy |

### Miscellaneous (C92-C98)

| ID | Claim | Verdict | Cross-ref | Justification |
| ---- | ------- | --------- | ----------- | --------------- |
| C92 | All models instruction-tuned | PARTIAL | WS11, Code | Qwen3-0.6B appears to be base model (no -Instruct suffix). DeepSeek distills are distilled, not instruction-tuned. |
| C93 | Geometric signatures as training evaluation tools | NEEDS INFO | — | Theoretical; no training-time experiments conducted |
| C94 | Broad tracking misses abliteration; self-ref is the signal | INFLATED | WS7 7.5-7.6 | Restates C64. d=0.464 at n=5 (power=10%) insufficient for reliable detection |
| C95 | Alignment is brittle — removable with weights + Heretic | PARTIAL | WS7, WS11 | Refusal removal verified. "Brittle surface constraint" overstated from single-model, single-method experiment. |
| C96 | Phi-3.5 excluded due to NaN | CONFIRMED | WS1 1.7, WS9 | DynamicCache API incompatibility confirmed; disclosed transparently |
| C97 | 2 models excluded from 17 = 15 valid | CONFIRMED | WS1 1.7 | Both exclusions documented and appropriate |
| C98 | Table 2 top-5 verified for 4 models | CONFIRMED | WS1 1.6 | All rankings independently recomputed |

---

## Cricket Claims (CC1-CC22, CF1-CF8, CL1-CL7)

Cricket claims are forward-looking capability assertions, not empirical findings. Rated on the FEASIBLE / PROMISING / PREMATURE / NO DATA scale per WS10.

See [viability.md](../cricket/viability.md) for full analysis.

### Capability Claims (CC1-CC22)

| ID | Capability | Rating | Key Constraint |
| ---- | ----------- | -------- | ---------------- |
| CC1 | Real-time cognitive state monitoring (deception, confabulation, sycophancy, refusal) | PROMISING | Effect sizes exist for deception/refusal; sycophancy results absent from paper; confabulation only a category label |
| CC2 | Detection at encoding time (before response tokens) | PROMISING | rho=0.821 shows ordering preserved, not that a classifier can detect specific states |
| CC3 | Passive monitoring — read without modify, no fine-tuning | FEASIBLE | True by architectural design; no empirical validation needed |
| CC4 | No fine-tuning required on target model | FEASIBLE | Cross-architecture ordering shown without fine-tuning, but no classifier built |
| CC5 | Multi-category state detection (13 categories separable) | PROMISING | Categories have different mean ranks; mean separation is not classification |
| CC6 | <50ms latency for full detection pipeline | NO DATA | Zero latency benchmarks anywhere in either repo |
| CC7 | Deception detection AUROC >= 0.95 | PREMATURE | Need \|d\|>=2.33 for AUROC 0.95; paper d values vary by architecture |
| CC8 | Refusal detection AUROC >= 0.99 | NO DATA | Max d=2.17 implies AUROC~0.94; need \|d\|>=3.29 for 0.99 |
| CC9 | Encoding-only detection AUROC >= 0.90 | PREMATURE | rho measures rank correlation, not classification AUROC; no conversion possible |
| CC10 | Cross-model transfer >= 0.80 within architecture families | NO DATA | Zero cross-model transfer experiments executed |
| CC11 | Persona drift monitoring (jailbreak via persona shift) | PROMISING | Identity experiment covers 2 personas; jailbreaks create novel personas |
| CC12 | State-sponsored censorship detection | PREMATURE | One model (Qwen-14B) positive; DeepSeek null on critical test; n=1 |
| CC13 | Alignment quality assessment (cage vs compass) | PREMATURE | One model (Qwen-7B), one method, n=5 per condition; low power |
| CC14 | Behavioral-geometric mismatch detection (sleeper agents) | NO DATA | Zero sleeper agent or backdoor models tested; entirely theoretical |
| CC15 | Confabulation detection AUROC >= 0.85 | PREMATURE | No confabulation-specific experiment; C1 attempt was underpowered |
| CC16 | False positive rate <= 1% | NO DATA | No FPR measurements anywhere |
| CC17 | SVD extraction latency < 20ms per layer | NO DATA | No latency benchmarks conducted |
| CC18 | Logistic regression sufficient — "geometry does the heavy lifting" | PREMATURE | No classifier experiments in C2; AxBench is external and unverified |
| CC19 | "Computationally cheap" — one SVD per layer per forward pass | NO DATA | Computational cost never measured |
| CC20 | Per-model, per-scale calibration required (non-monotonic effects) | FEASIBLE | Deception direction varies by architecture; contradicts CF8 |
| CC21 | Corrective prompts shift geometry toward normal | NO DATA | Experiment C6 was planned but never run |
| CC22 | "No widely deployed system monitors internal cognitive state during inference" | FEASIBLE | Assertion; needs verification against Apollo, Anthropic, Meta CICERO |

### Cross-Document Discrepancies (CF1-CF8)

| ID | Finding | Verdict | Justification |
| ---- | --------- | --------- | --------------- |
| CF1 | "8 models, 5 architectures" for identity vs paper's 7 models, 4 architectures | REJECTED | Paper Table 4 is authoritative (7 identity result files) |
| CF2 | DeepSeek d=+0.904 as headline censorship result | INFLATED | Critical test d=-0.219 (null); only residualized norm is positive |
| CF3 | Abliteration "d=0.000 across ALL categories" vs paper's d=+0.464 (self-ref) | REJECTED | Self-ref d=+0.464 is borderline medium effect; "d=0.000 across ALL" is factually incorrect |
| CF4 | Abliteration "rho=0.9945" ordering preserved | REJECTED | 0.9945 not reported in paper; unverified claim in Cricket docs |
| CF5 | "Coding #1 in ALL 16 models" vs paper's 15 valid models | REJECTED | Paper excludes Phi-3.5 and one other = 15 valid; Cricket claims 16 |
| CF6 | "cross-model rho=0.914" vs paper's 0.739 (eff. rank) / 0.909 (key norm) | REJECTED | 0.914 matches neither metric; possibly pre-correction value |
| CF7 | Qwen tokenizer "CLEAN_PASS" vs paper's CATASTROPHIC_FAIL (Table 9) | REJECTED | Cricket docs report the opposite verdict entirely |
| CF8 | "Doesn't need per-model calibration" vs CC20 "requires per-model calibration" | REJECTED | Internal Cricket contradiction between Findings doc and Design doc |

### Competitive Landscape (CL1-CL7)

| ID | Claim | Verdict | Justification |
| ---- | ------- | --------- | --------------- |
| CL1 | "No existing system monitors KV-cache geometry for real-time cognitive state classification" | PARTIAL | Needs verification against Apollo, Anthropic internal, Meta CICERO, HalluCana |
| CL2 | Apollo Research AUROC 0.96-0.999 on strategic deception (arXiv 2502.03407) | PARTIAL | arXiv ID format valid; AUROC range not independently verified |
| CL3 | Steering vectors erode safety to >80% jailbreak success (Xiong et al. 2026) | NEEDS INFO | Paper not located in standard databases |
| CL4 | ITI ~5% of heads carry truthfulness signal (Li et al. 2023 NeurIPS) | PARTIAL | ITI paper exists; percentage claim not verified against source |
| CL5 | HalluCana: closest pre-generation competitor (Li et al. 2024 NAACL) | INFLATED | Not found in NAACL proceedings; dismissal not substantiated |
| CL6 | "Nobody combines SVD + pre-gen + multi-cat + passive monitoring" | PREMATURE | Conjunction of features designed to exclude all competitors |
| CL7 | Cricket at "intersection of a dozen independent research threads" | NO DATA | Thread count and independence not verified |

---

## Aggregate Verdict Counts

### Campaign 2 Paper (C1-C98)

| Verdict | Count | % |
| --------- | ------- | --- |
| CONFIRMED | 51 | 52% |
| PARTIAL | 22 | 22% |
| REJECTED | 10 | 10% |
| INFLATED | 7 | 7% |
| ARTIFACT | 3 | 3% |
| INVALID | 3 | 3% |
| NEEDS INFO | 2 | 2% |

Note: Some claims straddle categories (e.g., C4 has ARTIFACT for 100% accuracy but CONFIRMED for 97.3% cross-prompt). The primary verdict reflects the claim's headline assertion.

### Cricket (37 claims)

| Rating | Count | % |
| -------- | ------- | --- |
| FEASIBLE | 4 | 11% |
| PROMISING | 4 | 11% |
| PREMATURE | 7 | 19% |
| NO DATA | 9 | 24% |
| REJECTED | 7 | 19% |
| INFLATED | 2 | 5% |
| PARTIAL | 3 | 8% |
| NEEDS INFO | 1 | 3% |

### Combined (135 claims)

| Verdict/Rating | Count |
| ---------------- | ------- |
| CONFIRMED | 51 |
| FEASIBLE | 4 |
| PARTIAL | 25 |
| PROMISING | 4 |
| REJECTED | 17 |
| INFLATED | 9 |
| PREMATURE | 7 |
| NO DATA | 9 |
| ARTIFACT | 3 |
| INVALID | 3 |
| NEEDS INFO | 3 |

---

## New Material Findings from Gap Closure

The following discrepancies were not caught in the original ~51-claim audit:

### D9: C14 — Deduplication claim is false

The Methods section explicitly claims "greedy decoding with deduplication" and "n=25-30 unique observations." `deduplicate_runs()` is never called in any experiment script. This was known from WS8 D5 but never linked to the specific Methods claim.

### D10: C18 — Log(length) residualization claim is false

Methods claims "OLS on log(sequence length)." Code uses raw length (`np.polyfit(token_counts, values, 1)`), not log-transformed. The residualization itself is mathematically sound; only the description is inaccurate.

### D11: C79 — Limitation mischaracterizes own experiment

The Limitations section describes RDCT as testing "truncation." The experiment actually tests prompt perturbation. The limitation statement is wrong about what its own experiment does.

### D12: C87 — "Every major C1 finding confirmed" is false

Sycophancy detection was replicated across 7 models in C2 (d=-0.297 to -0.701, 6/7 detectable) but silently dropped from the paper. "Every" major finding was NOT confirmed because a successfully replicated finding was omitted.

### I6: C90 — "Architecture-independent" is overstated

Min pairwise rho=0.396 (Qwen3-0.6B vs Llama-3.1-70B-q4). Some pairs share barely more structure than chance. "Substantially shared across architectures" would be accurate; "architecture-independent" implies invariance the data does not support.

### I7: C89 — All practical implications are untested

None of the three stated implications (censorship detection, jailbreak ID, training metric) have been tested with an actual system. These are extrapolations from effect sizes, not demonstrated capabilities.

## Citation Verification Findings

### D13: Watson's 1/e threshold lacks formal citation despite being a headline falsification target

The paper's abstract, Section 5.6, and conclusion all discuss falsifying Watson's Integrated Theory of Attention 1/e prediction. The `watson2019ita` bib entry exists but lacks a `\cite{}` command anywhere in main.tex. In the compiled document, the theoretical prediction being "definitively falsified" appears without any traceable source. This is a significant procedural oversight for one of three headline findings.

### D14: 22 of 28 .bib entries are uncited — bibliography contains orphaned references

The references.bib file contains 22 entries never cited in main.tex, carried over from Campaign 1. While LaTeX only includes cited entries in the compiled bibliography, this creates confusion for direct .bib review. Notable orphans include `zou2023representation` (highly relevant to abliteration) and `roy2007effective` (defines the effective rank metric used throughout).

### I8: Effective rank metric lacks source citation

The paper's primary metric — effective rank — uses a definition from Roy & Vetterli (2007). The entry `roy2007effective` exists in .bib but is never cited. The reader cannot trace the metric definition to its mathematical source.

### I9: Representation Engineering (zou2023) uncited despite relevance to abliteration

Section 5.7 discusses abliteration (removing a "refusal direction") without attributing the underlying concept to Representation Engineering (Zou et al., 2023). The Heretic tool IS cited, but the technique it implements is not.
