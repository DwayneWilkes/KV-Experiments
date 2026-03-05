# WS9: Omission Audit

**Status**: COMPLETE (WS9 tasks 9.1--9.9 all addressed)
**Paper sections**: all
**Scope**: Every result file in repo mapped to paper; unreported results flagged

## File Inventory

### Total Count
- Claimed by paper: 83 result files across 10 experiment types
- Actual JSON files: **85** (discrepancy — 3 "corrected" files added post-count)
- Actual .md reports: **43**
- Actual .txt logs: **36** (35 log files + 1 experiment_log.txt)
- Actual total files: **164** (85 JSON + 43 MD + 36 TXT)
- JSON parse failures: **0** (all 85 parse successfully)

### JSON File Inventory (85 files)

| # | Filename | Experiment Type | Campaign | Paper Section |
|---|----------|----------------|----------|---------------|
| 1 | `scale_sweep_Qwen2.5-0.5B_results.json` | scale_sweep | C2 | §3.1, Table 2 |
| 2 | `scale_sweep_Qwen3-0.6B_results.json` | scale_sweep | C2 | §3.1 |
| 3 | `scale_sweep_TinyLlama-1.1B_results.json` | scale_sweep | C2 | §3.1 |
| 4 | `scale_sweep_gemma-2-2b-it_results.json` | scale_sweep | C2 | §3.1 |
| 5 | `scale_sweep_Qwen2.5-3B_results.json` | scale_sweep | C2 | §3.1 |
| 6 | `scale_sweep_Phi-3.5-mini-instruct_results.json` | scale_sweep | C2 | EXCLUDED (NaN) |
| 7 | `scale_sweep_Qwen2.5-7B_results.json` | scale_sweep | C2 | §3.1, Table 2 |
| 8 | `scale_sweep_Qwen2.5-7B-q4_results.json` | scale_sweep | C2 | §3.1 |
| 9 | `scale_sweep_DeepSeek-R1-Distill-Qwen-7B_results.json` | scale_sweep | C2 | §3.1 |
| 10 | `scale_sweep_Mistral-7B-v0.3_results.json` | scale_sweep | C2 | §3.1, Table 2 |
| 11 | `scale_sweep_Llama-3.1-8B_results.json` | scale_sweep | C2 | §3.1, Table 2 |
| 12 | `scale_sweep_gemma-2-9b-it_results.json` | scale_sweep | C2 | §3.1, Table 2 |
| 13 | `scale_sweep_Qwen2.5-14B_results.json` | scale_sweep | C2 | §3.1 |
| 14 | `scale_sweep_DeepSeek-R1-Distill-Qwen-14B_results.json` | scale_sweep | C2 | §3.1 |
| 15 | `scale_sweep_Qwen2.5-32B-q4_results.json` | scale_sweep | C2 | §3.1 |
| 16 | `scale_sweep_Llama-3.1-70B-q4_results.json` | scale_sweep | C2 | §3.1 |
| 17 | `scale_sweep_abliterated_Qwen2.5-7B_results.json` | scale_sweep | C2 | §3.6 (abliteration) |
| 18 | `input_only_1.1B_results.json` | input_only | C2 | §3.2, Table 3 |
| 19 | `input_only_7B_results.json` | input_only | C2 | §3.2, Table 3 |
| 20 | `input_only_Qwen-0.5B_results.json` | input_only | C2 | §3.2, Table 3 |
| 21 | `input_only_Qwen-7B_results.json` | input_only | C2 | §3.2, Table 3 |
| 22 | `input_only_Qwen-14B-q4_results.json` | input_only | C2 | §3.2, Table 3 |
| 23 | `input_only_Llama-8B_results.json` | input_only | C2 | §3.2, Table 3 |
| 24 | `input_only_Gemma-2B_results.json` | input_only | C2 | §3.2, Table 3 |
| 25 | `input_only_DS-7B_results.json` | input_only | C2 | §3.2, Table 3 |
| 26 | `input_only_rho_corrected.json` | input_only (corrected) | C2 | Aggregate rho |
| 27 | `bloom_taxonomy_Qwen2.5-0.5B_results.json` | bloom_taxonomy | C2 | §3.4 |
| 28 | `bloom_taxonomy_Qwen2.5-3B_results.json` | bloom_taxonomy | C2 | §3.4 |
| 29 | `bloom_taxonomy_Qwen2.5-7B_results.json` | bloom_taxonomy | C2 | §3.4 |
| 30 | `bloom_taxonomy_Qwen2.5-14B_results.json` | bloom_taxonomy | C2 | §3.4 |
| 31 | `bloom_taxonomy_Llama-3.1-8B_results.json` | bloom_taxonomy | C2 | §3.4 |
| 32 | `bloom_taxonomy_Mistral-7B-v0.3_results.json` | bloom_taxonomy | C2 | §3.4 |
| 33 | `bloom_taxonomy_gemma-2-2b-it_results.json` | bloom_taxonomy | C2 | §3.4 |
| 34 | `deception_forensics_TinyLlama-1.1B_results.json` | deception_forensics | C2 | §3.3 |
| 35 | `deception_forensics_gemma-2-2b-it_results.json` | deception_forensics | C2 | §3.3 |
| 36 | `deception_forensics_Qwen2.5-7B_results.json` | deception_forensics | C2 | §3.3 |
| 37 | `deception_forensics_Llama-3.1-8B_results.json` | deception_forensics | C2 | §3.3 |
| 38 | `deception_forensics_Mistral-7B-v0.3_results.json` | deception_forensics | C2 | §3.3 |
| 39 | `deception_forensics_gemma-2-9b-it_results.json` | deception_forensics | C2 | §3.3 |
| 40 | `deception_forensics_Qwen2.5-32B-q4_results.json` | deception_forensics | C2 | §3.3 |
| 41 | `identity_signatures_TinyLlama-1.1B_results.json` | identity_signatures | C2 | §3.3, Table 4 |
| 42 | `identity_signatures_Qwen3-0.6B_results.json` | identity_signatures | C2 | §3.3 |
| 43 | `identity_signatures_Qwen2.5-7B_results.json` | identity_signatures | C2 | §3.3, Table 4 |
| 44 | `identity_signatures_Llama-3.1-8B_results.json` | identity_signatures | C2 | §3.3, Table 4 |
| 45 | `identity_signatures_Mistral-7B-v0.3_results.json` | identity_signatures | C2 | §3.3, Table 4 |
| 46 | `identity_signatures_gemma-2-9b-it_results.json` | identity_signatures | C2 | §3.3 |
| 47 | `identity_signatures_Qwen2.5-32B-q4_results.json` | identity_signatures | C2 | §3.3 |
| 48 | `rdct_stability_TinyLlama-1.1B_results.json` | rdct_stability | C2 | §3.4 |
| 49 | `rdct_stability_Qwen2.5-0.5B_results.json` | rdct_stability | C2 | §3.4 |
| 50 | `rdct_stability_Qwen2.5-7B_results.json` | rdct_stability | C2 | §3.4 |
| 51 | `rdct_stability_Llama-3.1-8B_results.json` | rdct_stability | C2 | §3.4 |
| 52 | `rdct_stability_Mistral-7B-v0.3_results.json` | rdct_stability | C2 | §3.4 |
| 53 | `rdct_stability_gemma-2-2b-it_results.json` | rdct_stability | C2 | §3.4 |
| 54 | `layer_map_TinyLlama-1.1B_results.json` | layer_map | C2 | §3.1 (layer analysis) |
| 55 | `layer_map_Qwen2.5-7B_results.json` | layer_map | C2 | §3.1 |
| 56 | `layer_map_Llama-3.1-8B_results.json` | layer_map | C2 | §3.1 |
| 57 | `layer_map_Mistral-7B-v0.3_results.json` | layer_map | C2 | §3.1 |
| 58 | `layer_map_Qwen2.5-32B-q4_results.json` | layer_map | C2 | §3.1 |
| 59 | `abliteration_sweep_Qwen2.5-7B_results.json` | abliteration | C2 | §3.6, Table 8 |
| 60 | `abliteration_sweep_abliterated_Qwen2.5-7B_results.json` | abliteration | C2 | §3.6, Table 8 |
| 61 | `abliteration_Qwen2.5-7B_comparison.json` | abliteration | C2 | §3.6 |
| 62 | `abliteration_Qwen2.5-7B_comparison_corrected.json` | abliteration (corrected) | C2 | §3.6 |
| 63 | `individuation_TinyLlama-1.1B_results.json` | individuation | C2 | §3.3 |
| 64 | `individuation_Qwen2.5-7B_results.json` | individuation | C2 | §3.3 |
| 65 | `individuation_Qwen3-0.6B_results.json` | individuation | C2 | §3.3 |
| 66 | `individuation_controls_7B_results.json` | individuation | C2 | §3.3 |
| 67 | `temporal_evolution_TinyLlama-1.1B_results.json` | temporal_evolution | C2 | §3.5 |
| 68 | `temporal_evolution_Qwen2.5-7B_results.json` | temporal_evolution | C2 | §3.5 |
| 69 | `temporal_evolution_Llama-3.1-8B_results.json` | temporal_evolution | C2 | §3.5 |
| 70 | `temporal_evolution_Mistral-7B-v0.3_results.json` | temporal_evolution | C2 | §3.5 |
| 71 | `natural_deception_Mistral-7B-v0.3_results.json` | natural_deception | C2 | §3.5, Table 6 |
| 72 | `natural_deception_Qwen2.5-14B_results.json` | natural_deception | C2 | §3.5, Table 6 |
| 73 | `natural_deception_DeepSeek-R1-Distill-Qwen-14B_results.json` | natural_deception | C2 | §3.5, Table 6-7 |
| 74 | `tokenizer_confound_Qwen2.5-7B_results.json` | tokenizer_confound | C2 | §3.7, Table 9 |
| 75 | `tokenizer_confound_Mistral-7B-v0.3_results.json` | tokenizer_confound | C2 | §3.7, Table 9 |
| 76 | `length_confound_analysis.json` | length_confound | C2 | §3.7 |
| 77 | `cross_model_rho_corrected.json` | cross_model (corrected) | C2 | §3.1 aggregate |
| 78 | `s4_topic_analysis_corrected.json` | s4_topic (corrected) | C2 | §3.5, Table 7 |
| 79 | `phase2a_transfer_results.json` | phase2a_transfer | C1→C2 | Bridge data |
| 80 | `cache_metadata_0.json` | cache_metadata | C1 | NOT in C2 paper |
| 81 | `cache_metadata_1.json` | cache_metadata | C1 | NOT in C2 paper |
| 82 | `cache_metadata_2.json` | cache_metadata | C1 | NOT in C2 paper |
| 83 | `adversarial_controls_TinyLlama-1.1B_results.json` | adversarial_controls | C1 | NOT in C2 paper |
| 84 | `batch_results_30runs.json` | batch_results | C1 | NOT in C2 paper |
| 85 | `cognitive_modes_results.json` | cognitive_modes | C1 | NOT in C2 paper |

### "83 result files" Claim Resolution (Task 9.2)

Paper claims 83 JSON result files across 10 experiment types (§2.4, abstract, Appendix A).
Actual count: **85 JSON files**. Per-type breakdown:

| Experiment Type | Paper Claims (Appendix A) | Actual Count | Match? |
|-----------------|--------------------------|--------------|--------|
| Scale sweep | 17 models | 17 | YES |
| Input-only | 8 models | 8 + 1 corrected | YES (base) |
| Identity signatures | 7 models | 7 | YES |
| Deception forensics | 7 models | 7 | YES |
| Bloom taxonomy | 7 models | 7 | YES |
| RDCT stability | 6 models | 6 | YES |
| S4 natural deception | 3 models | 3 | YES |
| Abliteration geometry | 1 model pair | 4 (2 sweep + 2 comparison) | OVER |
| Tokenizer confound | 2 models | 2 | YES |
| Temporal evolution | 4 models | 4 | YES |
| **Subtotal (10 types)** | | **65 base** | |

Files NOT counted in any of the 10 types:
- 5 layer_map files (no layer_map experiment type listed in the paper's 10)
- 4 individuation files (no individuation experiment type listed; C2 does not discuss individuation as a separate experiment)
- 1 length_confound_analysis.json (analysis artifact)
- 1 cross_model_rho_corrected.json (aggregate artifact)
- 1 s4_topic_analysis_corrected.json (corrected artifact)
- 1 input_only_rho_corrected.json (corrected artifact)
- 1 abliteration_Qwen2.5-7B_comparison_corrected.json (corrected artifact)
- 1 phase2a_transfer_results.json (C1-to-C2 bridge data)
- 6 C1 legacy files (cache_metadata x3, adversarial_controls, batch_results, cognitive_modes)

**Resolution**: The paper's "83" likely counts: 65 (10 types as listed) + 5 (layer_map) + 4 (individuation) + 4 (corrected/aggregate files) + 1 (length confound) + 1 (phase2a transfer) + 1 (Phi-3.5 excluded but counted) + 2 (abliteration comparison files) = **83**. This reconciles if the authors counted ALL C2-era JSON files excluding only the 6 C1 legacy files and the 1 later-added corrected comparison. The per-type enumeration in the paper (10 types) does NOT mention layer_map or individuation as separate types, even though their files are implicitly counted in the "83" total. This is sloppy bookkeeping, not fraud -- the Appendix A experiment list does not match the 83 claim because layer_map and individuation are orphaned from the enumeration.

### File-Dark Results (in repo, NOT in paper)
| Filename | Apparent Experiment | Favorable/Unfavorable | Hypothesis for Omission |
|----------|-------------------|---------------------|------------------------|
| `cache_metadata_0/1/2.json` | C1 cache metadata | N/A | Superseded by C2 |
| `adversarial_controls_TinyLlama-1.1B_results.json` | C1 adversarial controls | N/A | 14.4MB legacy file |
| `batch_results_30runs.json` | C1 30-run batch | N/A | C1 pilot data |
| `cognitive_modes_results.json` | C1 cognitive modes | N/A | Superseded by C2 taxonomy |
| `phase2a_transfer_results.json` | C1→C2 bridge | Unknown | Transfer validation |
| `abliteration_Qwen2.5-7B_comparison_corrected.json` | Bug fix of comparison | Favorable (fixed bug) | Replaces original |

## Unreported Experiments

### Societies of Thought (10_societies_of_thought.py) — Task 9.3

- Code exists: **yes** (`code/10_societies_of_thought.py`, 36,773 bytes)
- Prompts exist: `prompts/societies_of_thought_prompts.json` (20,390 bytes, 90 prompts)
- Result files: **NONE** — experiment never completed
- Log files: `log_societies_7B.txt` (v1), `log_societies_7B_v2.txt` (v2)

**v1 run (2026-03-02 09:21)**: Model loaded successfully. All 4 system prompt conditions completed data collection (90 prompts x 3 runs x 4 conditions = 1,080 inferences). Hypothesis testing began:
  - H8a: **SUPPORTED** (d=2.650, p=0.0000) — deliberative prompts show higher temporal geometric variance
  - H8b: **NOT SUPPORTED** (rho=nan, p=nan) — convergence rate vs answer consistency had constant input
  - H8c: **NOT SUPPORTED** (p=0.4934) — change-point alignment not above chance
  - H8d: Crashed at `from scipy.stats import binom_test` — `ImportError: cannot import name 'binom_test'` (API renamed to `binomtest` in scipy >= 1.7)

**Preliminary results assessment**: H8a was supported with a large effect. H8b and H8c were negative. The experiment crashed during H8d before completing analysis, so no JSON results were written. The preliminary results are MIXED (1 supported, 2 not supported before crash), not uniformly unfavorable.

**v2 run (2026-03-02 14:02)**: Model loaded, then **empty output** -- the log file has only 12 lines (model loading). The script produced zero data, suggesting it crashed silently very early (possibly the import error was hit at module level in v2).

- Paper status: Limitation #6: "Societies of Thought experiment incomplete. Campaign 2 included an experiment testing whether reasoning models' internal multi-agent debate (Kim et al. 2025) produces detectable geometric signatures. Results were not available at time of writing."
- Assessment: **Honestly disclosed**. The partial v1 results showing 1/3 hypotheses supported + 2/3 not supported were NOT selectively omitted -- the experiment genuinely failed to complete.

### Sycophancy Detection (09_sycophancy_detection.py) — Task 9.4

- Code exists: **yes** (`code/09_sycophancy_detection.py`, 25,802 bytes)
- Standalone experiment: Tests 5 pre-registered hypotheses (H7a-H7e) about sycophancy geometry
- Prompts exist: `prompts/s5_sycophancy_elicitation.py` (766 lines, 80 questions x 4 conditions x 4 sys prompts = 1,040 unique prompts)
- Standalone result files: **NONE**
- Standalone log files: **NONE** — never even attempted

**However**: Sycophancy data DOES exist in the deception forensics results. The `04_deception_forensics.py` script includes a sycophancy sub-experiment (Experiment 2). All 7 deception forensics result files contain `genuine_vs_sycophantic` comparisons:

| Model | d(genuine vs sycophantic) | Detectable? |
|-------|--------------------------|-------------|
| TinyLlama-1.1B | -0.363 | Yes |
| Gemma-2-2B | -0.297 | No |
| Qwen2.5-7B | -0.394 | Yes |
| Llama-3.1-8B | -0.302 | Yes |
| Mistral-7B | -0.442 | Yes |
| Gemma-2-9B | -0.701 | Yes |
| Qwen2.5-32B-q4 | -0.438 | Yes |

**The C1 paper reported these results** (Table in §4 "Sycophancy Detection", d = -0.363 to -0.438 across 3 scales). **The C2 paper drops all sycophancy results without mention** despite having data from 7 models (vs C1's 3). The C2 results are FAVORABLE (6/7 detectable, consistent small effects d = -0.30 to -0.70, robust cross-architecture).

- C1 paper status: Reported in full (§4 Deception Forensics, Sycophancy Detection subsection)
- C2 paper status: **NOT MENTIONED** despite favorable data existing in 7 result files
- ROADMAP status: H7 sycophancy is listed as a Campaign 2 extended experiment, code "complete"
- Assessment: **Concerning omission**. Two distinct issues:
  1. The standalone 1,040-prompt battery (09_sycophancy_detection.py) was never run -- no log files exist, suggesting it was deprioritized after code was written
  2. The existing sycophancy sub-results in deception forensics (favorable, cross-architecture) were silently dropped from C2 despite being reported in C1. This is the only C1 finding with favorable C2 data that was dropped without explanation.

### C2C Replication (13_c2c_replication.py) — Task 9.5

- Code exists: **yes** (`code/13_c2c_replication.py`, 35,832 bytes)
- Architecture: Implements a C2C Fuser with RMSNorm, SwiGLU MLP -- substantial neural network training code (not a trivial script)
- Planned 3 phases: (1) train fuser, (2) geometric preservation, (3) cross-model transfer
- Result files: **NONE**
- Log files: `log_c2c_small.txt` — single attempted run

**Run details (2026-03-02 08:48)**:
  - Sharer: Qwen2.5-0.5B (24 layers, 2 KV heads, head_dim=448)
  - Receiver: Qwen3-0.6B (28 layers, 8 KV heads, head_dim=128)
  - Fuser parameters: 487,735,512 (large model!)
  - Phase 1 training started (20 texts, 1 epoch)
  - Crashed at line 175: `torch.cat([recv_k_flat, shar_k_flat], dim=-1)` — `RuntimeError: Sizes of tensors must match except in dimension 1. Expected size 88 but got size 22`
  - Root cause: The receiver and sharer have different numbers of KV heads (8 vs 2), and the per-layer fuser concatenation assumes matching sequence-length dimensions. The head_dim mismatch (448 vs 128) combined with different KV head counts produces incompatible tensor shapes for concatenation.

- ROADMAP status: Listed as Campaign 2 extended experiment (H_C2C_1 through H_C2C_3), estimated 20 GPU-hours
- Paper status: **NOT MENTIONED** -- not in limitations, not in future work, not acknowledged
- Assessment: **Undisclosed technical failure**. Unlike Societies of Thought (Limitation #6), the C2C replication failure is not mentioned anywhere in the paper. The code references a specific paper (Fu et al. 2025, arXiv:2510.03215) and has a bib entry (`fu2025c2c`) that is never cited in the main text. This experiment was planned, coded, attempted, failed, and silently dropped.

## Excluded Models

### Phi-3.5 (3.8B parameters)
- Result file exists: **yes** — `scale_sweep_Phi-3.5-mini-instruct_results.json`
- NaN count: **13/13 categories** — all NaN means and medians
- Root cause: `'DynamicCache' object has no attribute 'seen_tokens'` (988 errors in `log_scale_sweep_3.8B.txt`)
- Diagnosis: Phi-3.5 uses a different cache API than the `transformers` DynamicCache interface the code expects
- Paper disclosure: Yes — mentioned as excluded due to NaN outputs
- Assessment: **Appropriately excluded and disclosed**

### Abliterated Qwen-7B
- File: `scale_sweep_abliterated_Qwen2.5-7B_results.json`
- Counted as 17th model attempt, 2nd exclusion
- Paper says "17 attempted, 2 excluded, 15 valid": Phi-3.5 + abliterated variant = 2 excluded
- Actually: abliterated variant is used in §3.6 abliteration analysis, just not in the main 15-model sweep
- Assessment: **Counting is confusing but not wrong** — abliterated model appears in abliteration section

## Campaign 1 → Campaign 2 Claim Attrition (Task 9.6)

### Claims Retained

| C1 Claim | C2 Status | Assessment |
|----------|-----------|------------|
| Refusal specialization (d=0.85-2.17) | Retained, expanded to 15 models | Strongest finding |
| Self-reference emergence at 14B+ | Retained | Not re-verified with new stats |
| Deception expansion (dimensionality) | Retained, architecture-dependent | C1 said "expansion"; C2 says mixed (expansion AND compression by architecture) |
| Encoding-native geometry (rho=0.929) | Retained, expanded to 8 models (rho=0.821 mean) | Lower aggregate rho in C2 |
| Identity as direction (100% classification) | Retained, expanded to 7 models | Universal result |
| Confabulation detection (medium d) | Retained as category | Still not significant; no new standalone analysis |
| Category hierarchy (coding #1) | Retained, expanded | Universal across 15 models |

### Claims Corrected

| C1 Claim | C2 Correction | Assessment |
|----------|--------------|------------|
| Individuation expands dimensionality (d=21.0) | Falsified in C1 adversarial controls, noted in C2 §6 | Transparent correction |
| Deception "narrows" (compression) | C2 shows direction is architecture-dependent | Honest revision |
| Pseudoreplication (greedy = independent runs) | Acknowledged, deduplication applied | Transparent correction |

### Claims Quietly Dropped

| C1 Claim | C2 Status | Concern Level |
|----------|-----------|---------------|
| **Sycophancy detection** (d=-0.36 to -0.44, reported in C1 §4) | **Silently dropped** despite favorable C2 data across 7 models | **HIGH** — favorable data exists, unreported |
| 30-run batch stability | Dropped | LOW — superseded by 5-run design |
| Adversarial controls experiment (01d) | Dropped | LOW — methodology incorporated, raw data unnecessary |
| Cognitive modes (01b) | Dropped | LOW — subsumed by Bloom taxonomy |
| Per-head SVD analysis | Promised in C1 F9, not implemented | MEDIUM — promised methodology not delivered |
| Stochastic generation (do_sample=True) | Promised in C1 F9, not implemented | MEDIUM — explicitly retained greedy with rationale |
| Effective rank adversarial controls | Promised in C1 F9 as "most critical methodological gap", not implemented | **HIGH** — C1 identified this as the most important missing validation |
| Confabulation elicitation redesign | Promised in C1 F9, not implemented | MEDIUM — confabulation still uses same prompts |
| Cross-technique validation (vs linear probes, SAE) | Promised in C1 F9, not implemented | LOW — ambitious stretch goal |
| Threshold sensitivity analysis (80/85/90/95%) | Promised in C1 F9, not implemented | MEDIUM — still using arbitrary 90% cutoff |
| RoPE asymmetry analysis | Promised in C1 F9, not implemented | LOW |
| Entropy-based effective rank (threshold-free) | Promised in C1 F9, not implemented | MEDIUM — spectral entropy computed but not foregrounded |

### Planned C2 Extensions Not Completed

| Planned Experiment (ROADMAP) | Status | Disclosed? |
|------------------------------|--------|------------|
| H7: Sycophancy Detection (standalone) | Code written, never run | No |
| H8: Societies of Thought | Attempted, crashed (scipy) | Yes (Limitation #6) |
| H9/H10: (if any) | Unknown | Unknown |
| C2C Replication | Attempted, crashed (tensor mismatch) | **No** |

### Summary Assessment
C1's Future Work section (F9) promised 12 specific methodological improvements for C2. Of these, **4 were implemented** (cross-architecture validation, conservative stats, TOST equivalence, tokenizer confound testing). **8 were not implemented**, including the one C1 identified as "the most critical methodological gap" (effective rank adversarial controls). The C2 paper does not acknowledge the gap between what was promised and what was delivered. The stochastic generation promise is addressed with a rationale for retaining greedy, but the effective rank controls, threshold sensitivity, and confabulation redesign are simply absent without comment.

## Log File Error Review (Task 9.7)

Complete review of all 36 log/txt files in results/:

| Log File | Error Count | Severity | Summary |
|----------|-------------|----------|---------|
| `log_bloom_0.5B.txt` | 0 | clean | |
| `log_bloom_3B.txt` | 0 | clean | |
| `log_bloom_7B.txt` | 0 | clean | |
| `log_bloom_14B.txt` | 0 | clean | |
| `log_bloom_gemma2b.txt` | 0 | clean | |
| `log_bloom_llama8b.txt` | 0 | clean | |
| `log_bloom_mistral7b.txt` | 0 | clean | |
| `log_c2c_small.txt` | 2 | **FATAL** | RuntimeError: tensor size mismatch (88 vs 22). Experiment aborted. |
| `log_deception_gemma2b.txt` | 0 | clean | |
| `log_deception_gemma9b.txt` | 0 | clean | |
| `log_deception_llama8b.txt` | 0 | clean | |
| `log_deception_mistral7b.txt` | 0 | clean | |
| `log_identity_0.6B.txt` | 2 | **ERROR** | TypeError: LogisticRegression multi_class kwarg removed in sklearn update |
| `log_identity_0.6B_v2.txt` | 0 | clean | Rerun succeeded (result file exists) |
| `log_identity_gemma9b.txt` | 0 | clean | |
| `log_identity_llama8b.txt` | 0 | clean | |
| `log_identity_mistral7b.txt` | 0 | clean | |
| `log_input_only_7B_rerun.txt` | 0 | clean | |
| `log_layermap_llama8b.txt` | 0 | clean | |
| `log_layermap_mistral7b.txt` | 0 | clean | |
| `log_rdct_0.5B.txt` | 0 | clean | |
| `log_rdct_7B.txt` | 0 | clean | |
| `log_rdct_gemma2b.txt` | 0 | clean | |
| `log_rdct_llama8b.txt` | 0 | clean | |
| `log_rdct_mistral7b.txt` | 1 | minor | `ConstantInputWarning`: input array is constant, correlation not defined. Non-fatal scipy warning. |
| `log_rdct_tinyllama.txt` | 0 | clean | |
| `log_scale_sweep_0.6B.txt` | 0 | clean | |
| `log_scale_sweep_3.8B.txt` | 977 | **FATAL** | 977x `DynamicCache.seen_tokens` AttributeError (Phi-3.5 incompatibility). All results NaN. |
| `log_scale_sweep_abliterated_7B.txt` | 0 | clean | |
| `log_societies_7B.txt` | 3 | **FATAL** | `ConstantInputWarning` (NaN correlation) + `ImportError: binom_test` (scipy API rename). Crash at H8d. |
| `log_societies_7B_v2.txt` | 0 | **FATAL** | Empty/near-empty log (12 lines: model load only). Silent crash. |
| `log_temporal_llama8b.txt` | 1 | minor | `RuntimeWarning: Precision loss in moment calculation due to catastrophic cancellation` (greedy decoding: identical data → zero variance). Non-fatal. |
| `log_temporal_mistral7b.txt` | 1 | minor | Same `RuntimeWarning` as temporal_llama8b. Non-fatal. |
| `log_tokenizer_confound_7b.txt` | 4 | **WARNING** | `UserWarning: shapiro input range zero` + `RuntimeWarning: precision loss` + 2x `CATASTROPHIC_FAIL` verdict. The CATASTROPHIC_FAIL is a code-path artifact of greedy decoding (zero variance within pairs → infinite d), documented in paper §3.7. |
| `log_tokenizer_confound_mistral7b.txt` | 0 | clean | |
| `experiment_log.txt` | 0 | clean | Campaign orchestration log. Records timing for all phases. |

### Log Files Statistics
- **Total log files**: 36 (35 log_*.txt + 1 experiment_log.txt)
- **Clean**: 27 (75%)
- **Minor warnings** (non-fatal, non-impactful): 3 (rdct_mistral, temporal_llama, temporal_mistral)
- **Code-path artifacts** (documented in paper): 1 (tokenizer_confound_7b)
- **Fatal errors** (experiment failed): 4 (c2c, scale_sweep_3.8B, societies v1, societies v2)
- **Recoverable errors** (rerun succeeded): 1 (identity_0.6B → v2)

### Critical Error Summary
1. **Societies of Thought**: scipy `binom_test` API rename → H8d crash. Partial results: H8a supported, H8b/H8c not. **Disclosed** (Limitation #6).
2. **C2C Replication**: Tensor shape mismatch (KV head count 8 vs 2) → Phase 1 training crash. **NOT disclosed**.
3. **Phi-3.5**: DynamicCache API incompatibility → 977 errors, all NaN. **Disclosed** (excluded from analysis).
4. **Identity 0.6B v1**: sklearn API change → rerun succeeded (v2). **No data loss**.
5. **Tokenizer confound**: CATASTROPHIC_FAIL verdict → code-path artifact of greedy decoding. **Disclosed** (§3.7, Table 9).
6. **Temporal/RDCT warnings**: Precision loss from near-identical data (greedy decoding). Scipy warnings, not data corruption. **Non-issue**.

## Citation Verification (Task 9.8)

### Citations Actually Used in C2 Paper Main Text

The C2 paper (main.tex) contains exactly 6 `\cite`/`\citet`/`\citep` references:

| # | Citation Key | Used Where | Real? | Correct Details? | Assessment |
|---|-------------|------------|-------|-----------------|------------|
| 1 | `lyra2026campaign1` | Abstract, §1, §2.3, §6 | **Self-citation** | Campaign 1 paper by same authors | Valid self-reference |
| 2 | `hedges1981distribution` | §2.2 | **YES** | Larry V. Hedges, "Distribution Theory for Glass's Estimator of Effect Size and Related Estimators," J. Educational Statistics 6(2):107-128, 1981 | Real, foundational statistics paper |
| 3 | `schuirmann1987comparison` | §2.2 | **YES** | Donald J. Schuirmann, "A Comparison of the Two One-Sided Tests Procedure and the Power Approach for Assessing the Equivalence of Average Bioavailability," J. Pharmacokinetics and Biopharmaceutics 15(6):657-680, 1987 | Real, foundational TOST paper |
| 4 | `bloom1956taxonomy` | §3.4 | **YES** | Benjamin S. Bloom et al., "Taxonomy of Educational Objectives: The Classification of Educational Goals, Handbook I: Cognitive Domain," David McKay Company, 1956 | Real, canonical educational psychology |
| 5 | `heretic2026` | §3.6 | **NEEDS INFO** | Listed as GitHub repo `heretic-llm/heretic`, v1.2.0, 2026. No DOI. | GitHub tool, not a publication. Cannot verify version/date without GitHub access. The concept of "abliteration" (removing refusal direction) originates from community research; "Heretic" appears to be a specific implementation tool. |
| 6 | `kim2025reasoning` | §5 (Limitation #6) | **PLAUSIBLE** | Listed as arXiv:2601.10825, "Reasoning Models Internally Debate Before Answering," Kim, Seungone et al., 2025 | arXiv ID format is valid (2601.xxxxx = January 2026 preprint, mislabeled as 2025 in bib). Cannot verify content without arXiv access. |

### Additional References in references.bib But NOT Cited in Main Text

The bibliography file contains 24 additional entries never referenced in the paper text. These appear to be inherited from the Campaign 1 bibliography. They include well-known real papers (Clark 2019, Belinkov 2022, Hewitt 2019, Zou 2023, Azaria 2023, Burns 2022, Marks & Tegmark 2023, etc.) plus C2 additions never cited (Beckmann 2023, Goldowsky-Dill 2025, Fu 2025 C2C, Merleau-Ponty 1945, Watson 2019).

### User-Specified Citations to Verify

| Citation (per user) | In C2 Paper? | Real Published Work? | Notes |
|---------------------|-------------|---------------------|-------|
| Roy & Vetterli (2007) - effective rank | In bib (`roy2007effective`), NOT cited in main text | **YES** - EUSIPCO 2007 conference paper. Real, well-cited (500+ citations). Defines the effective rank metric used throughout the paper. | Foundational for the paper's methodology but never explicitly cited in C2 text. Used implicitly. |
| Watson (1913) - behaviorism | **NOT in paper at all** | **YES** - J.B. Watson, "Psychology as the Behaviorist Views It," Psychological Review 20(2):158-177, 1913. | This classic paper is NOT cited. The paper cites a DIFFERENT Watson: Marcus Watson (2019), "The Integrated Theory of Attention," listed as an **unpublished manuscript**. |
| Bloom (1956) - taxonomy | **YES** - `bloom1956taxonomy`, cited in §3.4 | **YES** | See row 4 above |
| Li et al. (2024) - DynamicCache | **NOT in paper** | DynamicCache is a HuggingFace `transformers` library class, not a paper. | No citation needed -- it's a software component. Mentioned only in error messages (Phi-3.5 crash). |
| Arditi et al. (2024) - abliteration/refusal | **NOT in paper** | The concept of abliteration (removing refusal direction) has community origins. Arditi et al. published related work on refusal mechanisms. | The C2 paper cites `heretic2026` (a tool) rather than the underlying research papers on refusal direction removal. |
| Gurnee & Tegmark (2024) - language model representations | **NOT cited in C2 main text** | Marks & Tegmark (2023) is in bib but uncited. Gurnee is referenced only in GUIDE.md (supplementary docs, not paper). | Not a paper citation issue since it's not claimed as a reference. |

### Flagged Issues

1. **Watson 2019 ITA**: The most concerning citation is `watson2019ita` -- Marcus Watson, "The Integrated Theory of Attention," listed as an **"Unpublished manuscript"** with no DOI, no arXiv ID, no institution, and no URL. The entire RDCT experiment (§3.4) tests and falsifies this paper's "1/e critical threshold" prediction. An entire experimental section is devoted to testing a prediction from an unpublished, unverifiable manuscript. This is NOT the J.B. Watson (1913) behaviorism paper the user may have been thinking of.

2. **Roy & Vetterli (2007)**: The effective rank metric -- the paper's PRIMARY measurement tool -- is defined in this reference, but it is NOT cited in the C2 main text. The bib entry exists but no `\cite{roy2007effective}` appears. This is a significant attribution gap: the paper's central metric is used throughout without crediting its originators.

3. **Uncited bib entries**: 24 entries in the bibliography are never cited. This is cosmetic (inherited from C1) but sloppy.

4. **`heretic2026` verifiability**: The abliteration tool is cited as a software artifact (GitHub repo, version 1.2.0). Without GitHub access, the existence and version cannot be independently verified.

## Selective Reporting Check (Task 9.9)

For each major result type, we checked whether the JSON files contain statistics not reported in the paper.

### Scale Sweep Files (17 files)

Each scale sweep JSON contains:
- `battery_results`: Raw data for all 13 cognitive categories
- `analysis.category_summaries`: Per-category n, mean, std, median, min, max, mean_per_token, std_per_token, mean_key_effective_rank, std_key_effective_rank, mean_key_spectral_entropy, mean_value_effective_rank, bootstrap CIs, normality tests
- `analysis.pairwise_comparisons`: **27 pairwise comparisons** per model (including eff_rank, per_token, and raw norm variants)
- `analysis.hypothesis_tests`: H1 (confabulation), H2 (self-reference), H3 (refusal), H4 (category invariance)
- `analysis.multiple_comparisons`: Holm-Bonferroni corrections for all 9 primary comparisons
- `analysis.power`: Power analysis at d=0.3, 0.5, 0.8
- `ranking`: Full 13-category rank ordering

**Unreported statistics found**:
- The paper reports only effective rank metrics. Each file also contains **Frobenius norm** and **per-token norm** comparisons (27 pairwise d-values per model). These are not reported in C2. The C1 paper showed that norm-based metrics do not survive adversarial controls, which explains the omission -- but the norm data is available and was computed.
- **Spectral entropy** values are computed but never reported in C2 (C1 promised to foreground these in C2, F9).
- **Value effective rank** is computed alongside key effective rank but never mentioned.
- **Multiple comparison corrections** (Holm-Bonferroni) are computed but not reported in the C2 paper text.

### Deception Forensics Files (7 files)

**Major unreported finding**: All 7 deception forensics files contain **sycophancy sub-experiment results** (Experiment 2 in the deception forensics battery):

| Model | d(genuine vs sycophantic) | d(honest vs sycophantic) | Detectable? |
|-------|--------------------------|-------------------------|-------------|
| TinyLlama-1.1B | -0.363 | -2.216 | Yes |
| Gemma-2-2B | -0.297 | -1.330 | No |
| Qwen2.5-7B | -0.394 | -2.227 | Yes |
| Llama-3.1-8B | -0.302 | -1.889 | Yes |
| Mistral-7B | -0.442 | -2.551 | Yes |
| Gemma-2-9B | -0.701 | -1.505 | Yes |
| Qwen2.5-32B-q4 | -0.438 | -1.990 | Yes |

These results are **favorable** (6/7 detectable, remarkably consistent d = -0.30 to -0.70 across 5 architecture families) and were **reported in C1** but **silently dropped from C2**. This is the most concerning selective reporting finding.

Additionally, each deception forensics file contains uncertainty gradient sub-experiments and layer localization results that are not discussed in C2.

### Natural Deception / S4 Files (3 files)

The `s4_topic_analysis_corrected.json` contains per-topic d-values for all 3 models. The paper reports the DeepSeek per-topic breakdown (Table 7) but does not report per-topic breakdowns for Qwen-14B or Mistral-7B. The Qwen-14B data shows:
- `critical_test.d = +0.766, p = 1.88e-09` (reported)
- `residualized_censored_vs_complex.d = +0.768, p = 3.64e-08` (reported)
- Additional comparisons: `residualized_censored_vs_control.d = +0.462, p = 0.000286`, `residualized_complex_vs_control` values (NOT reported individually)

The Mistral-7B data:
- `critical_test.d = +0.084, p = 0.469` (reported)
- `residualized_censored_vs_complex.d = +0.396, p < 0.001` (reported)
- Per-topic data available but not reported (mirroring DeepSeek treatment but less interesting given null critical test)

### Abliteration Files (4 files)

The `abliteration_Qwen2.5-7B_comparison_corrected.json` contains all values reported in Table 8. Cross-checked: all d-values, g-values, and p-values in the paper match the JSON. The `harmful_distribution` and `refusal_rate` data also match the text. **No unreported statistics found beyond what's in the paper.**

### Identity Signatures Files (7 files)

Each file contains:
- Classification results (accuracy, cross-prompt validation, confusion matrix) -- reported
- Permutation test p-values -- not individually reported but summarized
- Feature importance arrays (per-feature RF importance) -- not reported
- Layer analysis -- not reported
- Pairwise persona comparisons -- selectively reported (only max-separation pair)
- Kendall W and ICC values -- reported in Table 4

### Cross-Model Rho File

`cross_model_rho_corrected.json` contains:
- `effective_rank.mean_pairwise_rho = 0.739` -- reported as 0.739
- `effective_rank.min_rho = 0.396` -- reported as 0.396
- `effective_rank.max_rho = 0.978` -- reported as 0.978
- `key_norm.mean_pairwise_rho = 0.909` -- reported as 0.909
- `coding_rank1.count = 15/15` -- reported as "15 out of 15"
- `kendall_W = 0.756` -- reported as 0.756
**All statistics reported accurately.**

### Summary: Unreported Statistics Assessment

| Category | Favorable Unreported? | Unfavorable Unreported? | Assessment |
|----------|----------------------|------------------------|------------|
| Sycophancy sub-results (deception forensics) | **YES** (6/7 detectable, consistent cross-arch) | No | **Concerning** -- favorable C1 finding dropped from C2 |
| Norm-based metrics (scale sweep) | Some positive | Some negative | Justified omission (C1 showed norms fail controls) |
| Spectral entropy | Unknown | Unknown | Promised but not delivered (C1 F9) |
| Value effective rank | Unknown | Unknown | Computed but never analyzed |
| Per-topic S4 for Qwen/Mistral | Mixed | Mixed | Minor -- DeepSeek is the interesting case |
| Feature importance (identity) | N/A | N/A | Supplementary detail |
| Multiple comparison corrections | N/A | N/A | Computed but not reported; corrections would make some results NS |

### Overall Selective Reporting Verdict

The paper does NOT appear to hide unfavorable results. When results are unfavorable (Societies of Thought partial failure, DeepSeek null on critical test, Watson threshold falsification), they are generally reported honestly.

The one significant exception is the **sycophancy sub-results**: favorable data that was reported in C1, replicated successfully across 7 models in C2, and then silently dropped from the C2 paper without explanation. This is not fabrication or fraud, but it is a gap in reporting that reviewers should be aware of.

The unreported norm-based and entropy-based metrics are justified by the C1 finding that norms fail adversarial controls. The promised spectral entropy analysis (C1 F9) represents an unfulfilled methodological commitment rather than selective reporting.

---

## Task 9.1: File-to-Section Mapping Verification

The file inventory table (above) maps all 85 JSON files to paper sections. Verification against the paper confirms:
- Files #1-17 (scale sweep): Correctly mapped to §3.1
- Files #18-26 (input only): Correctly mapped to §3.2, Table 3
- Files #27-33 (bloom): Correctly mapped to §3.4
- Files #34-40 (deception forensics): Correctly mapped to §3.3 (NOTE: deception forensics is discussed as part of §3.3 "Deception Forensics: Cross-Architecture Patterns", NOT §3.4)
- Files #41-47 (identity): Correctly mapped to §3.3, Table 4
- Files #48-53 (RDCT): Correctly mapped to §3.4 (RDCT is §3.4 "RDCT: Watson's 1/e Threshold Is Dead")
- Files #54-58 (layer_map): Mapped to §3.1 -- but layer_map is NOT explicitly discussed in C2 paper text. The C1 paper discusses layer maps. These files are **file-dark** for C2.
- Files #59-62 (abliteration): Correctly mapped to §3.6, Table 8
- Files #63-66 (individuation): Mapped to §3.3 -- but individuation is only mentioned in §6 ("Campaign 1 was an honest exploration... individuation was an artifact"). These files are **file-dark** for C2 (data exists from C1, not reanalyzed in C2).
- Files #67-70 (temporal): Correctly mapped to §3.5
- Files #71-73 (natural deception): Correctly mapped to §3.5, Tables 6-7
- Files #74-75 (tokenizer): Correctly mapped to §3.7, Table 9
- Files #76-78 (corrected/analysis): Correctly mapped
- Files #79-85 (legacy/bridge): Correctly identified as NOT in C2

**File-dark results (updated)**:

| Filename(s) | Category | In Paper? | Favorable? | Concern |
|-------------|----------|-----------|------------|---------|
| `layer_map_*.json` (5 files) | Layer analysis | **NO** (C1 only) | Unknown | LOW -- C1 experiment, no C2 claims about layers |
| `individuation_*.json` (4 files) | Individuation | Mentioned only as C1 falsification | Falsified in C1 | LOW -- honestly reported as failed hypothesis |
| Sycophancy sub-data in deception_forensics (7 files) | Sycophancy detection | **NO** | **FAVORABLE** | **HIGH** -- see selective reporting section |
| `phase2a_transfer_results.json` | C1→C2 bridge | **NO** | Unknown | LOW -- infrastructure data |
| C1 legacy files (6) | Various | **NO** | N/A | LOW -- superseded |
