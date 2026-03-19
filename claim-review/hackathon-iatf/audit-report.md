# Intelligence at the Frontier (IATF) Hackathon — Comprehensive Audit

> **Event**: [Intelligence at the Frontier Hackathon](https://luma.com/ftchack-sf-2026) (FTCHack SF 2026), hosted by Funding the Commons & Protocol Labs. March 14–15, 2026, San Francisco.
> **Team**: Liberation Labs.

**Audit date**: 2026-03-14
**Auditors**: Dwayne Wilkes & Kavi
**Scope**: All Liberation Labs org repos with hackathon-period activity — KV-Experiments (Exp 14-16, 26-36), JiminAI-Cricket, CacheScope, JiminAI-Frontend-Performative, PITCH_NUMBERS.md
**Data files**: JSON files in `results/hackathon/`, verified via `gh api` and local cross-reference
**Method**: Automated claim extraction + swarm extraction (4 parallel agents) + adversarial vetting (3 parallel agents) + manual cross-reference against JSON data

---

## Executive Summary

### Numeric Claim Verification (All KV-Experiments)

| Verdict | Exp 14-16 | Exp 26-33 | Exp 34-36 | Total | Description |
| --------- | --------- | --------- | --------- | ------- | ------------- |
| VERIFIED | 44 | 31 | 16 | 91 | Numbers match JSON data exactly (within standard rounding) |
| NO_DATA | 46 | 1 | 0 | 47 | Specific numbers reported but no JSON file exists to verify |
| DISCREPANT | 2 | 1 | 0 | 3 | Numbers differ from JSON beyond rounding tolerance |
| CRITICAL | 0 | 1 | 1 | 2 | Factual errors with downstream impact (EXP32-001 + propagation) |
| FALSIFIED | 0 | 0 | 1 | 1 | Exp 36 disproves "harmful content detection" framing |
| OVERSTATED | 0 | 1 | 1 | 2 | Claims exceed what data supports |
| STALE | 0 | 2 | 2 | 4 | Numbers correct at time of writing but now outdated |
| FALSE_STATUS | 5 | 0 | 0 | 5 | Log claims "JSON saved" but no file exists on disk |
| UNVERIFIED | 2 | 0 | 0 | 2 | Numbers reported but insufficient data to cross-reference |
| LOW_PRIORITY | 1 | 0 | 0 | 1 | Minor claim not blocking |
| POSITIVE | 0 | 0 | 2 | 2 | Good scientific practice (self-correction, honest hypothesis testing) |

**Total experiment claims audited: 155** (95 from Exp 14-16 + 37 from Exp 26-33 + 23 from Exp 34-36)

### Code Review Findings (Cricket, CacheScope, Frontend)

| Severity | Count | Key repos |
| ---------- | ------- | ----------- |
| HIGH | 5 | Cricket: confabulation demo, censorship claim, AUROC 1.0. Frontend: fake terminal, "AI LIE DETECTOR" |
| MEDIUM | 10 | Cricket: stale README (4). CacheScope: auth/CORS (3). Frontend: doc-code mismatch (3) |
| LOW | 5 | CacheScope: compression text, model table, sys.path (3). Frontend: WS auth, domain (2) |
| UNCHANGED | 3 | Cricket: CF3, CF6, CF8 findings from prior audit |
| POSITIVE | 4 | CacheScope: threat model, test suite. Cricket: viability doc |

Total code review findings: **27** (23 actionable + 4 positive)

### PITCH_NUMBERS.md Review Findings

| Severity | Count | Description |
| ---------- | ------- | ------------- |
| HIGH | 3 | Censorship AUROC 1.0 with no classifier, sycophancy d=0.107 claimed as detectable, per-layer internal contradiction |
| MEDIUM | 2 | Cross-repo AUROC inconsistency (0.9991 vs 1.000), mean AUROC inherits OVERSTATED Llama value |

**No fabrication detected.** Where data files exist, every reported number matches. The CRITICAL findings (EXP32-001, EXP34-003) are miscounts of classifier output, not data fabrication — the underlying AUROC values are all correct. The single FALSIFIED finding (EXP36-001) is a framing issue, not a numeric error.

---

## Part 1: KV-Experiments — Exp 14-16

Source: HACKATHON_LOG.md (698 lines, 6 experiments, 3 models)
Data files: 6 JSON files in `results/hackathon/`

| Verdict | Count | Description |
| --------- | ------- | ------------- |
| VERIFIED | 44 | Numbers match JSON data exactly (within rounding) |
| NO_DATA | 46 | Specific numbers reported but no JSON file exists to verify |
| DISCREPANT | 2 | Numbers differ from JSON beyond rounding tolerance |
| UNVERIFIED | 2 | Numbers reported but insufficient data to cross-reference |
| LOW_PRIORITY | 1 | Minor claim not blocking |
| FALSE_STATUS | 5 | Log claims "JSON saved" but no file exists on disk |

**Total claims audited: 95**

---

### Experiment 14a: C7 Frequency-Matched Encoding-Only (Qwen 7B)

*HACKATHON_LOG.md lines 15-66*

**Data file**: NONE
**Log status claim**: "COMPLETE. JSON saved." (line 65) — **FALSE_STATUS**: no C7 encoding results file exists. Only `c7_permutation_test.json` exists, which is from Experiment 16.
**Code**: `code/14_confabulation_detection.py`

| Line | Claim | Value | Verdict |
| ------ | ------- | ------- | --------- |
| 39 | Effective rank: confab=14.38, factual=14.33 | d=0.052 | NO_DATA |
| 39 | Spectral entropy d | 0.224 | NO_DATA |
| 41 | Key norms d | -0.018 | NO_DATA |
| 44 | Wilcoxon W=179.5, p=0.819 | p=0.819 | NO_DATA |
| 44 | Paired t: t=0.348, p=0.731 | p=0.731 | NO_DATA |
| 46 | Raw d=0.052 → Residualized d=0.115 | d=0.115 | NO_DATA |
| 47 | TOST p_upper=0.171, p_lower=0.089 | - | NO_DATA |
| 52-57 | Per-domain d values (6 domains) | -0.539 to +0.358 | NO_DATA |

**Remediation**:

1. Locate the saved JSON (may be in a different directory or under a different name)
2. If not found, re-run with `json.dump()` added to the experiment script
3. Fix the false "JSON saved" status in the log

---

### Experiment 14b: S3 Generation-Phase Confabulation (Qwen 7B)

*Lines 69-101*

**Data file**: NONE
**Log status claim**: "COMPLETE. JSON saved." (line 101) — **FALSE_STATUS**

| Line | Claim | Value | Verdict |
| ------ | ------- | ------- | --------- |
| 79-83 | Accurate=40, Confabulated=17, Partial=1, Refused=0, Pending=2 | counts | NO_DATA |
| 87 | d=-0.107, p=0.728 | d=-0.107 | NO_DATA |
| 93-95 | Per-difficulty d: easy=-0.834, medium=-0.895, hard=+0.257 | d values | NO_DATA |

**Remediation**: Same as C7 — locate or re-generate JSON.

---

### Experiment 14c: Contrastive Encoding — Bare vs Grounded (Qwen 7B)

*Lines 105-175*

**Data file**: NONE
**Log status claim**: "COMPLETE. JSON saved. THIS IS THE HEADLINE RESULT." (line 174) — **FALSE_STATUS**

| Line | Claim | Value | Verdict |
| ------ | ------- | ------- | --------- |
| 121 | Bare=19.08, Grounded=25.79 | d=-2.354 | NO_DATA |
| 122 | Spectral entropy d | -0.477 | NO_DATA |
| 123 | Key norms d | -1.937 | NO_DATA |
| 125 | Wilcoxon W=0.0, p<0.0001 | p<0.0001 | NO_DATA |
| 129 | Bare: 25.0 tokens, Grounded: 44.4 tokens | counts | NO_DATA |
| 131 | Raw d=-2.354 → Residualized d=-1.081 | d=-1.081 | NO_DATA |
| 137-141 | Per-layer d values (5 layers shown) | -2.724 to -2.518 | NO_DATA |
| 148-151 | Per-difficulty d (3 levels) | -2.403 to -2.425 | NO_DATA |
| 157-162 | Per-domain d (6 domains) | -1.871 to -4.162 | NO_DATA |

**Remediation**: This is described as "THE HEADLINE RESULT." The missing JSON is critical — this experiment's claims are the most prominent in the log and cannot be independently verified. Priority 1 for re-serialization.

---

### Experiment 14d: Contrastive Controlled — 4-Condition Test (Qwen 7B)

*Lines 219-267*

**Data file**: NONE
**Log status claim**: "COMPLETE. JSON saved." (line 267) — **FALSE_STATUS**

| Line | Claim | Value | Verdict |
| ------ | ------- | ------- | --------- |
| 235-238 | Bare=19.08, Correct=25.79, Wrong=26.09, Irrelevant=26.30 | means | NO_DATA |
| 246 | Correct vs Irrelevant d=-0.160, p=0.340 | d=-0.160 | NO_DATA |
| 246 | Residualized d=-0.539 | d=-0.539 | NO_DATA |
| 252-257 | Per-domain d (6 domains, correct vs irrelevant) | -0.988 to +0.498 | NO_DATA |
| 259 | Verdict: INFORMATION_VOLUME | categorical | NO_DATA (but narrative is self-consistent) |

**Remediation**: This is the experiment that DEBUNKS the headline result. Without its JSON, the self-correction narrative ("we found a big signal → we controlled for it → it disappeared") cannot be independently verified. **Both the headline AND the debunking need data files.**

---

### Cross-Model Contrastive Validation (Llama 8B, Mistral 7B)

*Lines 297-321*

**Data file**: NONE
**Log status claim**: "COMPLETE. JSON saved for all three models." (line 321) — **FALSE_STATUS** for Llama and Mistral (Qwen 4-condition data also missing as noted above)

| Line | Claim | Value | Verdict |
| ------ | ------- | ------- | --------- |
| 305 | Qwen correct vs irrelevant d=-0.160, p=0.340 | d=-0.160 | NO_DATA |
| 306 | Llama correct vs irrelevant d=-0.161, p=0.363 | d=-0.161 | NO_DATA |
| 307 | Mistral correct vs irrelevant d=-0.192, p=0.245 | d=-0.192 | NO_DATA |
| 313-315 | Residualized d: Qwen=-0.539, Llama=-0.549, Mistral=-0.457 | d values | NO_DATA |

**Remediation**: Save contrastive_4condition results for all 3 models. The cross-architecture consistency claim (d ≈ -0.16 across 3 models) is a strong scientific finding that needs data backing.

---

### Standard Sycophancy Detection (Qwen 7B)

*Lines 340-401*

**Data file**: NONE (separate experiment from enhanced sycophancy; NOT in the confabulation_detection files)

| Line | Claim | Value | Verdict |
| ------ | ------- | ------- | --------- |
| 356-358 | Rate: 4/60 (6.7%) sycophantic, 18/60 corrective, 38/60 ambiguous | counts | NO_DATA |
| 366-368 | Means: Bare=17.61, Correct=19.90, Wrong=19.80 | means | NO_DATA |
| 374 | Wrong vs Correct d=-0.054, p=0.360 | d=-0.054 | NO_DATA |
| 380 | Residualized d=+0.047, p=0.685 | d=+0.047 | NO_DATA |
| 383-385 | Per-domain d (6 domains) | -0.212 to +0.218 | NO_DATA |
| 387 | Verdict: MODEL_RESISTANT | categorical | NO_DATA |

**Remediation**: This is the baseline sycophancy experiment. Save as `sycophancy_standard_Qwen2.5-7B_results.json`. The d=-0.054 value is referenced repeatedly throughout the log's taxonomy tables.

---

### Standard Sycophancy Cross-Model (Llama 8B, Mistral 7B)

*Lines 402-419*

**Data file**: NONE for standard sycophancy. The enhanced results (below) are in the confabulation_detection files.

| Line | Claim | Qwen | Llama | Mistral | Verdict |
| ------ | ------- | ------ | ------- | --------- | --------- |
| 406 | Sycophancy rate | 4/60 (6.7%) | 0/60 (0%) | 7/60 (11.7%) | NO_DATA (standard) |
| 410 | Wrong vs Correct d | -0.054 | -0.043 | -0.027 | NO_DATA (standard) |
| 411 | Wrong vs Correct p | 0.360 | 0.908 | 0.657 | NO_DATA (standard) |
| 412 | Residualized d | +0.047 | +0.093 | +0.146 | NO_DATA (standard) |
| 417 | Mistral within-syc d=+0.400, p=0.480 | - | - | d=+0.400 | NO_DATA (standard; enhanced Mistral within-syc d=0.400 IS verified below) |

**Remediation**: Save standard sycophancy results for all 3 models separately from the enhanced results.

---

### Enhanced Sycophancy — Run 1: WITH REASONING (INVALIDATED)

*Lines 433-454*

**Data file**: Partially in `confabulation_detection_Qwen2.5-7B_results.json` (the enhanced authority-framed data IS the Run 2 data; Run 1 was invalidated and may not have been saved)

| Line | Claim | Value | Verdict |
| ------ | ------- | ------- | --------- |
| 448 | Raw d=+3.455, p<0.0001 | d=+3.455 | NO_DATA (invalidated run) |
| 448 | Residualized d flipped to -0.611 | d=-0.611 | NO_DATA (invalidated run) |
| 452 | Within-sycophancy d=+0.781, p=0.219 | d=+0.781 | NO_DATA (invalidated run) |
| 454 | Sycophancy rate 6/60 (10.0%) | rate=0.100 | NO_DATA (invalidated run) |

**Remediation**: The invalidated run is correctly labeled. No action needed unless the team wants to publish the Run 1 data as a methods lesson (length confound demonstration). If so, serialize it.

---

### Enhanced Sycophancy — Run 2: Authority-Only (Length-Matched, Qwen 7B)

*Lines 456-501*

**Data file**: `confabulation_detection_Qwen2.5-7B_results.json` → `.sycophancy` section

| Line | Claim | Claimed | JSON | JSON Path | Verdict |
| ------ | ------- | --------- | ------ | ----------- | --------- |
| 462 | Sycophancy rate | 9/60 (15%) | 9, 0.15 | `.sycophancy.statistics.sycophancy_rate` (inferred) | **VERIFIED** |
| 470 | Mean bare | 16.26 | 16.260 | `.sycophancy.statistics.means.bare` | **VERIFIED** |
| 471 | Mean user_correct | 21.37 | 21.369 | `.sycophancy.statistics.means.user_correct` | **VERIFIED** |
| 472 | Mean user_wrong | 21.19 | 21.189 | `.sycophancy.statistics.means.user_wrong` | **VERIFIED** |
| 478 | Wrong vs Correct d | -0.107 | -0.1071 | `.sycophancy.statistics.pairwise.wrong_vs_correct.cohens_d` | **VERIFIED** |
| 478 | Wrong vs Correct p | 0.772 | 0.7724 | `.sycophancy.statistics.pairwise.wrong_vs_correct.wilcoxon.p` | **VERIFIED** |
| 482 | Residualized d | +0.143 | 0.1426 | `.sycophancy.statistics.residualized.wrong_vs_correct_resid_d` | **VERIFIED** |
| 482 | Residualized p | 0.029 | 0.0287 | `.sycophancy.statistics.residualized.wrong_vs_correct_resid_p` | **VERIFIED** |
| 486 | Within-syc d | +0.393 | 0.3932 | `.sycophancy.statistics.sycophantic_vs_corrective.cohens_d` | **VERIFIED** |
| 486 | Within-syc p | 0.478 | 0.4775 | `.sycophancy.statistics.sycophantic_vs_corrective.welch_t.p_value` | **VERIFIED** |
| 492-497 | Per-domain d (6 domains) | -0.392 to +0.075 | matches | `.sycophancy.statistics.per_domain` | **VERIFIED** |
| 497 | Science sycophancy rate 4/10 (40%) | 4/10 | matches | `.sycophancy.statistics.per_domain.science` | **VERIFIED** |
| 501 | Verdict: NULL | NULL | NULL | `.sycophancy.statistics.verdict` | **VERIFIED** |

**No remediation needed.** All values verified.

---

### Enhanced Sycophancy Cross-Model — Llama 3.1 8B

*Verified against `confabulation_detection_Llama-3.1-8B_results.json`*

| Claim | Claimed | JSON | Verdict |
| ------- | --------- | ------ | --------- |
| Sycophancy rate | 0/60 (0%) | sycophantic=0, rate=0.0 | **VERIFIED** |
| Wrong vs Correct d | -0.043 | -0.0434 | **VERIFIED** |
| Wrong vs Correct p | 0.908 | 0.9075 | **VERIFIED** |
| Residualized d | +0.093 | 0.0935 | **VERIFIED** |
| Verdict | MODEL_RESISTANT | MODEL_RESISTANT | **VERIFIED** |

---

### Enhanced Sycophancy Cross-Model — Mistral 7B v0.3

*Verified against `confabulation_detection_Mistral-7B-v0.3_results.json`*

| Claim | Claimed | JSON | Verdict |
| ------- | --------- | ------ | --------- |
| Sycophancy rate | 7/60 (11.7%) | sycophantic=7, rate=0.117 | **VERIFIED** |
| Wrong vs Correct d | -0.027 | -0.0268 | **VERIFIED** |
| Wrong vs Correct p | 0.657 | 0.6571 | **VERIFIED** |
| Residualized d | +0.146 | 0.1455 | **VERIFIED** |
| Within-sycophancy d | +0.400 | 0.4002 | **VERIFIED** |
| Verdict | NULL | NULL | **VERIFIED** |

---

### Experiment 15: Refusal Geometry — Base vs Abliterated (Qwen 7B)

*Lines 509-581*

**Data file**: `refusal_geometry_Qwen2.5-7B_results.json`

| Line | Claim | Claimed | JSON | Verdict |
| ------ | ------- | --------- | ------ | --------- |
| 529-532 | Cell means: base_harmful=12.56, base_benign=12.49, abl_harmful=12.56, abl_benign=12.48 | means | matches `.statistics.cell_means` | **VERIFIED** |
| 538 | H1: d=+0.071, p=0.783 | d=0.071 | 0.0715, 0.783 | **VERIFIED** |
| 539 | H2: d=+0.071, p=0.784 | d=0.071 | 0.0712, 0.784 | **VERIFIED** |
| 540 | H3: d=+0.005, p=0.985 | d=0.005 | 0.005, 0.985 | **VERIFIED** |
| 541 | Control: d=+0.008, p=0.977 | d=0.008 | needs verification | LOW_PRIORITY |
| 546-551 | Per-topic d (6 topics) | +1.608 to -1.007 | matches `.statistics.per_topic` | **VERIFIED** |
| 555 | Verdict: NO_ENCODING_SIGNAL | - | consistent | **VERIFIED** |

---

### Experiment 16: Direction Sweep — Per-Layer Profile Analysis

*Lines 585-627*

**Data file**: `direction_sweep_results.json`

| Line | Analysis | Dir Eff AUROC | JSON | LR Eff AUROC | JSON | Verdict |
| ------ | ---------- | ------------- | ------ | ------------- | ------ | --------- |
| 614 | Control Llama | 0.913 | 0.913 | 0.948 | 0.948 | **VERIFIED** |
| 615 | Control Mistral | 0.948 | 0.948 | 0.979 | 0.979 | **VERIFIED** |
| 616 | Control Qwen | 1.000 | 1.000 | 1.000 | 1.000 | **VERIFIED** |
| 617 | Sycophancy Llama | 0.558 | 0.558 | 0.857 | 0.857 | **VERIFIED** |
| 618 | Sycophancy Mistral | 0.677 | 0.677 | 0.827 | 0.827 | **VERIFIED** |
| 619 | Sycophancy Qwen | 0.526 | 0.526 | 0.616 | 0.616 | **VERIFIED** |
| 620 | Refusal base | 0.526 | 0.526 | 0.579 | 0.579 | **VERIFIED** |
| 621 | Refusal abl | 0.526 | 0.526 | 0.634 | 0.634 | **VERIFIED** |
| 622 | Refusal interaction | 1.000 | 1.000 | 1.000 | 1.000 | **VERIFIED** (log correctly identifies as artifact) |
| 623 | Within-syc Mistral | 0.602 | 0.602 | 0.786 | 0.786 | **VERIFIED** |
| 624 | Within-syc Qwen | 0.510 | 0.510 | 0.628 | 0.628 | **VERIFIED** |

**All 22 AUROC values verified exactly.**

---

### C7 Direction Extraction (RepE-style, lines 178-202)

**Data file**: `c7_permutation_test.json` (partial — contains direction extraction + permutation test for C7)

| Line | Claim | Claimed | JSON | Verdict |
| ------ | ------- | --------- | ------ | --------- |
| 189 | LOO accuracy | 0.333 (20/60) | needs layer-specific verification | UNVERIFIED |
| 190 | LOO AUROC | 0.288 | closest JSON value 0.258 | **DISCREPANT** (diff=0.030) |
| 191 | LR LOO accuracy | 0.417 (25/60) | needs verification | UNVERIFIED |
| 192 | LR AUROC | 0.337 | needs verification | UNVERIFIED |
| 589 | Flipped AUROC | 0.712 | 0.732 in sweep | **DISCREPANT** (diff=0.020) |

**Remediation**: The C7 direction extraction may have used a different analysis pipeline than the direction sweep (Experiment 16). Verify which script/analysis produced the 0.288 figure, and ensure the `c7_permutation_test.json` contains the same analysis. The discrepancy is small (0.030) but should be explained.

---

### Permutation Test: Sycophancy LR Validation (lines 660-691)

**Data file**: Permutation details NOT in `direction_sweep_results.json`

| Line | Claim | Value | Verdict |
| ------ | ------- | ------- | --------- |
| 668 | Llama real LR AUROC | 0.857 | **VERIFIED** (matches direction sweep) |
| 668 | Llama perm mean=0.569, 95th=0.685, max=0.771 | stats | NO_DATA (permutation details not serialized) |
| 669 | Mistral real LR AUROC | 0.827 | **VERIFIED** (matches direction sweep) |
| 669 | Mistral perm mean=0.568, 95th=0.670, max=0.749 | stats | NO_DATA (permutation details not serialized) |
| 668-669 | Both p<0.005 | p-values | NO_DATA (0/200 exceeded → p<0.005 is logically correct but raw data not saved) |

**Remediation**: Add permutation test raw distributions to the direction sweep JSON or save as a separate `permutation_test_sycophancy_lr.json`.

---

## Part 2: KV-Experiments — Exp 26-33

Source: HACKATHON_LOG.md (907 lines, experiments 26-32) + commit b47c93c (Exp 33)
Verified against: JSON files in `results/hackathon/` via `gh api repos/Liberation-Labs-THCoalition/KV-Experiments/contents/`

| Verdict | Count | Description |
| --------- | ------- | ------------- |
| VERIFIED | 31 | Numbers match JSON data exactly (within standard rounding) |
| CRITICAL | 1 | n_actually_answered: 18 claimed, 8 in JSON — 2.25x inflation |
| DISCREPANT | 1 | Exp 26 rho 0.826 outside claimed "0.83-0.90" range |
| STALE | 2 | Pitch doc experiment count (31→33) and JSON file count (64+→83) |
| OVERSTATED | 1 | Exp 33 presents Qwen data from Exp 31 as unified 3-architecture result |
| NO_DATA | 1 | Exp 26 per-model Coding #1 claim needs recomputation |

**Total KV-Experiments claims audited: 37** (includes 2 supplementary data points verified from Exp 33 JSON)

---

### Experiment 26: Scale Invariance

HACKATHON_LOG.md lines 818-830

**Data file**: `results/hackathon/scale_invariance.json`

| Line | Claim | Claimed | JSON | Verdict |
| ------ | ------- | --------- | ------ | --------- |
| 824 | SMALL vs MEDIUM rho | 0.83-0.90 | 0.8667 (p=1.15e-24) | **VERIFIED** |
| 825 | SMALL vs LARGE rho | 0.83-0.90 | 0.8258 (p=1.37e-20) | **DISCREPANT** — 0.826 < claimed lower bound 0.83 |
| 826 | MEDIUM vs LARGE rho | 0.83-0.90 | 0.8963 (p=1.49e-28) | **VERIFIED** |
| 828 | Coding #1 at 100% of models at ALL scales | categorical | no per-model breakdown in JSON | **NO_DATA** |
| 828 | confab-creative AUROC drops to 0.60 at LARGE | 0.60 | 0.60 | **VERIFIED** |

**Scale group composition** (from JSON):

- SMALL (2 models): Qwen3-0.6B, TinyLlama-1.1B
- MEDIUM (6 models): abliterated Qwen2.5-7B, Gemma-2-9b, Llama-3.1-8B, Mistral-7B, Qwen2.5-7B-q4, Qwen2.5-7B
- LARGE (2 models): Llama-3.1-70B-q4, Qwen2.5-32B-q4

**Remediation**:

1. Change rho range to "0.826-0.896" for exactness, or note SMALL_vs_LARGE exception
2. Add per-model breakdown to JSON to verify Coding #1 claim
3. Note LARGE group has only 2 models (both quantized) — thin for generalization

---

### Experiment 27: Encoding-Regime Axis Analysis

Lines 832-840

**Data file**: `results/hackathon/` (verified via automated extraction)

| Line | Claim | Value | Verdict |
| ------ | ------- | ------- | --------- |
| 834 | Truth axis cross-model consistency | -0.046 | **VERIFIED** |
| 836 | Complexity axis consistency | 0.982 | **VERIFIED** |
| 836 | CMPLX/OPEN/SAFE/ABSTR converge within 10-17° | angular range | **VERIFIED** |

**Key finding**: Truth axis does NOT exist (effectively random). Complexity axis is universal. Encoding regime is 1D — structural complexity only.

---

### Experiment 28: Confabulation Trajectory

Lines 842-848

**Data file**: `results/hackathon/` (verified via automated extraction)

| Line | Claim | Value | Verdict |
| ------ | ------- | ------- | --------- |
| 844 | key_entropy d: 20 → 55 over 50 tokens | trajectory | **VERIFIED** |
| 844 | Deception d: 5.67 → 8.36, p=0.008 | d, p | **VERIFIED** |
| 846 | n=3 per group for confabulation | n=3 | **VERIFIED** (appropriately caveated as underpowered) |

---

### Experiment 29: Per-Layer Deception Anatomy

Lines 850-861

**Data file**: `results/hackathon/` (verified via automated extraction)

| Line | Claim | Value | Verdict |
| ------ | ------- | ------- | --------- |
| 854 | 28/28 layers show d > 1.0 (same-prompt controlled, Qwen-7B) | 28/28 | **VERIFIED** |
| 855 | Cross-model layer profile consistency: rho = 0.200 | 0.200 | **VERIFIED** |
| 856 | Deceptive norms LARGER at every layer (100% consistency) | 100% | **VERIFIED** |
| 857 | Mean \|d\| first half = 2.44, second half = 2.41 (flat profile) | 2.44, 2.41 | **VERIFIED** |

**Implication**: Aggregate features capture the full signal. No per-layer feature engineering needed.

---

### Experiment 30: Final Synthesis

Lines 863-870

| Line | Claim | Value | Verdict |
| ------ | ------- | ------- | --------- |
| 866 | 14 hackathon experiments completed (of 19 total) | count | **VERIFIED** (accurate at time of Exp 30; total reached 18 by Exp 33) |
| 867 | 92 pre-hackathon + 21 hackathon JSON result files | 92, 21 | **VERIFIED** (at time of writing; now 24+ hackathon files after Exp 31-33) |

---

### Experiment 31: Refusal Detection in Generation Regime

Lines 872-888

**Data file**: `results/hackathon/` (refusal detection JSON; verified via automated extraction)

| Line | Claim | Claimed | JSON | Verdict |
| ------ | ------- | --------- | ------ | --------- |
| 878 | AUROC (LR, 5-fold) | **0.898** | 0.898 | **VERIFIED** |
| 879 | AUROC (RF, 5-fold) | 0.830 | 0.830 | **VERIFIED** |
| 880 | key_rank d | 1.530 | 1.530 | **VERIFIED** |
| 881 | key_entropy d | 1.505 | 1.505 | **VERIFIED** |
| 882 | Actual refusals in text | 19/20 | 19 | **VERIFIED** |
| 884 | Refusal norm_per_token vs normal | 295.7 vs 301.1 | matches | **VERIFIED** |

**Context**: This is a NEW capability — Cricket can detect refusal from KV-cache geometry. Refusal responses are sparser per token, consistent with suppression model. Run on Qwen2.5-7B, 20 harmful + 20 benign prompts, 50 tokens generation.

---

### Experiment 32: Jailbreak Detection -- CRITICAL FINDING

Lines 890-904

**Data file**: `results/hackathon/jailbreak_detection.json`

| Line | Claim | Claimed | JSON | JSON Path | Verdict |
| ------ | ------- | --------- | ------ | ----------- | --------- |
| 896 | Jailbreak vs Normal AUROC (LR) | **0.878** | 0.8775 | `.jailbreak_vs_normal.lr_auroc` | **VERIFIED** |
| 896 | Jailbreak vs Normal AUROC (RF) | 0.764 | 0.7638 | `.jailbreak_vs_normal.rf_auroc` | **VERIFIED** |
| 897 | Jailbreak vs Refusal AUROC (LR) | 0.790 | 0.790 | `.jailbreak_vs_refusal.lr_auroc` | **VERIFIED** |
| 897 | Jailbreak vs Refusal AUROC (RF) | 0.584 | 0.584 | `.jailbreak_vs_refusal.rf_auroc` | **VERIFIED** |
| 898 | 3-way accuracy | 0.583 | 0.583 | `.three_way_accuracy` | **VERIFIED** |
| 900 | norm_per_token: jailbreak=294.9, refusal=295.7, normal=301.1 | 3 values | matches | `.norm_per_token.*` | **VERIFIED** |
| 902 | Harmful prompts answered by abliterated model | **18/20** | **8** | `.n_actually_answered` | **CRITICAL** |

#### EXP32-001: n_actually_answered Discrepancy

**Claimed** (line 902): "18/20 harmful prompts answered by abliterated model."
**JSON**: `n_actually_answered: 8`

Manual classification of all 20 abliterated Qwen2.5-7B responses:

| Classification | Count | Prompts |
| ---------------- | ------- | --------- |
| PROVIDES_INFO (harmful content despite disclaimers) | 10 | 1, 2, 3, 4, 6, 8, 11, 12, 17, 19 |
| WARNS_ONLY (disclaimers, no actionable harmful info in 50 tokens) | 9 | 5, 9, 10, 13, 14, 15, 16, 18, 20 |
| HARD_REFUSE (explicit refusal: "I cannot provide") | 1 | 7 (explosive device) |

The JSON's `n_actually_answered = 8` reflects the automated classifier's conservative threshold. Manual reading suggests 8-10 responses provide actionable harmful information. **The claim of 18/20 inflates the result by 2.25x** and is not supported by either the JSON or manual reading.

**Impact**: Overstating how many prompts the abliterated model answered inflates the apparent effectiveness of abliteration as a jailbreak technique, which inflates the contrast used for jailbreak detection AUROC claims. The AUROC values themselves are still correct (verified above), but the narrative framing of the jailbreak detection capability is misleading.

**Remediation**:

1. Correct line 902 to "8/20" (per JSON classifier) or "10/20" (generous manual review)
2. Update narrative to reflect that abliteration is only partially effective
3. Note that the classifier threshold affects this count

---

### Experiment 33: Multi-Model Refusal Detection

Not in HACKATHON_LOG.md body — commit message b47c93c only

**Data file**: `results/hackathon/refusal_multimodel.json`

| Source | Claim | Claimed | JSON | Verdict |
| -------- | ------- | --------- | ------ | --------- |
| Commit msg | Llama RF AUROC | 0.893 | 0.8925 | **VERIFIED** (0.8925 → 0.893 under standard rounding) |
| Commit msg | Mistral LR AUROC | 0.843 | 0.8425 | **VERIFIED** (0.8425 → 0.843 under standard rounding; 0.842 under banker's) |
| JSON | Llama LR AUROC | — | 0.8675 | (not claimed; supplementary) |
| JSON | Mistral RF AUROC | — | 0.7563 | (not claimed; supplementary) |
| JSON | Llama n_actual_refusals | — | 19/20 | **VERIFIED** (consistent with Exp 31) |
| JSON | Mistral n_actual_refusals | — | 14/20 | **VERIFIED** (6 prompts answered despite being harmful) |

**Rounding note**: Both claimed values match JSON under standard round-half-up convention. Under banker's rounding (round-half-to-even), 0.8925 → 0.892 and 0.8425 → 0.842. The discrepancy is ±0.001 in both cases — trivial, but worth standardizing the rounding convention.

#### EXP33-003: Qwen Value Reused from Exp 31

Exp 33 JSON contains only Llama + Mistral data. The Qwen refusal AUROC (0.898) presented in the pitch as part of a "3-architecture result" is hardcoded from Exp 31. Qwen was NOT re-run in Exp 33's multi-model framework.

**Remediation**: Either re-run Qwen in Exp 33's framework, or explicitly note "Qwen value from Exp 31 (same methodology, separate run)."

---

### docs/PITCH_NUMBERS.md (Exp 26-33 scope)

Line 160

| Line | Claim | Actual | Verdict |
| ------ | ------- | -------- | --------- |
| 160 | "31 experiments over 2 campaigns" | 33 (Exp 32-33 added after Exp 31 pitch update) | **STALE** |
| 160 | "64+ JSON result files" | 83 (per C2 paper main.tex) | **STALE** |

**Context**: Commit b47c93c (Exp 33) updated the pitch numbers with cross-architecture averages but did NOT bump the experiment count from 31 to 33 or update the file count.

**Remediation**: Update both numbers in docs/PITCH_NUMBERS.md line 160.

---

## Part 3: KV-Experiments — Exp 34-36

Source: KV-Experiments commits 7812420, bd84a7d, 4f86a77
Data files: `jailbreak_multimodel.json`, `token_controlled_layers.json`, `impossibility_refusal.json`
Method: JSON cross-reference via `gh api` + script inspection + adversarial vetting (3 parallel agents)

| Verdict | Count | Description |
| --------- | ------- | ------------- |
| VERIFIED | 16 | Numbers match JSON data exactly (within standard rounding) |
| CRITICAL | 1 | EXP32-001 propagated: "18/20" hardcoded into Exp 34 script |
| FALSIFIED | 1 | Exp 36 disproves "harmful content detection" framing — signal is refusal, not harm |
| OVERSTATED | 1 | Llama AUROC 0.878 measures refusal, not jailbreak (1/20 answered) |
| STALE | 2 | HACKATHON_LOG missing Exp 33-36; PITCH_NUMBERS still "31 experiments" |
| POSITIVE | 2 | Exp 35 self-corrects per-layer claim; Exp 36 honestly tests H1 vs H2 |

**Total: 23 experiment claims**

---

### Experiment 34: Multi-Model Jailbreak Detection

Commit bd84a7d (not in HACKATHON_LOG.md)

**Data file**: `results/hackathon/jailbreak_multimodel.json`
**Script**: `code/34_jailbreak_multimodel.py` (389 lines)

#### Commit Message Claims vs JSON

| Source | Claim | Claimed | JSON | Verdict |
| -------- | ------- | --------- | ------ | --------- |
| Commit | Qwen AUROC | 0.878 | 0.8775 (from Exp 32 `jailbreak_detection.json`) | **VERIFIED** (but hardcoded — see EXP34-002) |
| Commit | Qwen n_abl_answered | 18/20 | 8 (from Exp 32 `jailbreak_detection.json`) | **CRITICAL** (EXP32-001 propagated) |
| Commit | Llama LR AUROC | 0.878 | 0.8775 | **VERIFIED** (0.8775 → 0.878 standard rounding) |
| Commit | Llama n_abl_answered | 1/20 | 1 | **VERIFIED** |
| Commit | Mistral LR AUROC | 0.793 | 0.7925 | **VERIFIED** (0.7925 → 0.793 standard rounding) |
| Commit | Mistral n_abl_answered | 8/20 | 8 | **VERIFIED** |
| Commit | Mean AUROC | 0.849 | (0.8775+0.8775+0.7925)/3 = 0.8492 | **VERIFIED** (arithmetic correct; but see ADV-MEAN below) |
| JSON | Llama RF AUROC | — | 0.9125 | (supplementary) |
| JSON | Mistral RF AUROC | — | 0.6913 | (supplementary) |

#### EXP34-001: Llama Abliteration Failure

The abliterated Llama-3.1-8B answered only **1/20** harmful prompts. The commit message acknowledges this: "but 1/20 answered — actually refusal detection." However, the Llama AUROC of 0.878 still feeds into the cross-architecture mean (0.849) without qualification.

Feature means confirm the problem:

| Model | Condition | norm/tok | n_generated |
| --------- | ----------- | ---------- | ------------- |
| Llama | refusal | 210.54 | 25.1 |
| Llama | normal | 187.57 | 48.5 |
| Llama | jailbreak | 206.87 | 28.3 |

The "jailbreak" condition (mean 28.3 tokens) looks like refusal (25.1 tokens), not normal generation (48.5 tokens). The AUROC is measuring abliterated-refusal vs normal, not genuine jailbreak-vs-normal.

**Verdict**: OVERSTATED. The Llama AUROC should not be included in a "jailbreak detection" mean without explicit caveat that abliteration failed for this model.

#### ADV-MEAN: Mean AUROC Inherits OVERSTATED Input

*Found during adversarial review.* The mean AUROC (0.849) is arithmetically correct but interpretively misleading: 2 of 3 inputs have qualified jailbreak signals (Qwen 8/20 actual, Llama 1/20 actual). The mean is marked VERIFIED because the math checks out, but a reader scanning the exec summary may not connect it to the EXP34-001 qualification. The mean should carry the same caveat as its weakest input.

#### EXP34-002: Qwen Data Hardcoded

`jailbreak_multimodel.json` contains only Llama + Mistral data. Qwen values appear as hardcoded literals in the script:

- **Line 334**: `print(f"  {'Qwen2.5-7B (Exp 32)':<25} {'0.8780':>14} {'0.7640':>14} {'18/20':>10}")` — literal strings, not loaded from any JSON
- **Line 344**: `mean_auroc = np.mean([0.878] + [s["auroc_jailbreak_vs_normal"]["LR"] for s in all_summaries])` — literal 0.878 baked into mean

The **"18/20" literal on line 334 is wrong** — Exp 32's `jailbreak_detection.json` records `n_actually_answered: 8`. The error is baked into the script as a string literal rather than read from data.

#### EXP34-003: Incorrect n_actually_answered for Qwen

The commit message states: "Qwen: 0.878 (18/20 answered — clean jailbreak signal)." The Exp 32 JSON records `n_actually_answered: 8`, not 18. This is hardcoded at line 334 of `code/34_jailbreak_multimodel.py` as the literal string `'18/20'`.

**Impact**: The "clean jailbreak signal" characterization is wrong. With 8/20 answered, Qwen's abliteration is only partially effective — similar to Mistral (8/20), not the near-total jailbreak the "18/20" figure suggests.

**Remediation**:

1. Fix line 334: change `'18/20'` to `'8/20'`
2. Fix line 344: load Qwen AUROC from Exp 32 JSON instead of hardcoding
3. Update commit message / any docs referencing "18/20 answered"
4. Reconsider whether "clean jailbreak signal" applies at 8/20

---

### Experiment 35: Token-Controlled Layer Reanalysis

Commit 7812420 (not in HACKATHON_LOG.md)

**Data file**: `results/hackathon/token_controlled_layers.json`
**Question**: Does per-layer deception signal survive after controlling for token count?

#### Key Results

| Metric | Value | Verdict |
| --------- | ------- | --------- |
| Token count confound (d) | 1.483 (honest mean 48.2 tok, deceptive mean 78.3) | **VERIFIED** |
| Raw norm d (mean across 28 layers) | 2.425 (all positive — deceptive has higher raw norm) | **VERIFIED** |
| Norm-per-token d (mean across 28 layers) | -1.909 (NEGATIVE — deceptive is sparser) | **VERIFIED** |
| ANCOVA layers significant at p<0.05 | 27/28 | **VERIFIED** |
| ANCOVA layers significant at p<0.01 | 25/28 | **VERIFIED** |
| Mean partial eta-squared | 0.486 | **VERIFIED** |
| Non-significant layer | Layer 27 (p=0.472) | **VERIFIED** |

#### Cross-Model Forensics (7 models)

| Model | norm/tok d | Mean per-layer d | Pattern |
| --------- | ----------- | ------------------ | --------- |
| gemma-2-2b-it | -1.231 | 1.680 | Sparser per token |
| gemma-2-9b-it | -1.642 | 0.508 | Sparser per token |
| Llama-3.1-8B | -3.656 | 2.360 | Sparser per token |
| Mistral-7B-v0.3 | -3.367 | 2.415 | Sparser per token |
| Qwen2.5-32B-q4 | -3.590 | 2.390 | Sparser per token |
| Qwen2.5-7B | -2.267 | 0.526 | Sparser per token |
| TinyLlama-1.1B | -0.427 | 0.697 | MIXED (near zero) |

6/7 models show consistent "sparser per token" pattern. TinyLlama is the exception (d=-0.427, near zero).

#### EXP35-P01: Honest Self-Correction (POSITIVE)

Exp 35 directly corrects the Exp 29 claim that "deception signal is uniform across all layers (d > 1.0 at 28/28 layers)." The reanalysis shows:

- Raw d > 1.0 was partly driven by deceptive responses generating more tokens
- After ANCOVA control, the signal **survives** (27/28 layers) but the interpretation changes: deceptive representations are **sparser per token**, not uniformly larger — a directional reversal, not a minor reframing
- `docs/PITCH_NUMBERS.md` was updated accordingly:
  - Line 52: "uniform d>1.0" → "ANCOVA significant 27/28 layers, d=-1.9/token"
  - Lines 122-126: "fundamental consequence" → "consequence of representational capacity allocation"

This is good scientific practice — finding a confound, re-testing, and updating claims with the corrected interpretation.

---

### Experiment 36: Impossibility vs Safety Refusal

Commit 4f86a77 (not in HACKATHON_LOG.md)

**Data file**: `results/hackathon/impossibility_refusal.json`
**Model**: Qwen/Qwen2.5-7B-Instruct
**Design**: 20 impossible prompts + 20 harmful prompts + 20 benign prompts
**Hypothesis**: H1 (harmful content specific) vs H2 (refusal in general)

#### AUROC Claims vs JSON

| Comparison | Commit claim | JSON (LR) | JSON (RF) | Verdict |
| --------- | ------------- | ----------- | ----------- | --------- |
| Impossible vs benign | 0.950 | 0.9500 | 0.8762 | **VERIFIED** |
| Harmful vs benign | 0.898 | 0.8975 | 0.8300 | **VERIFIED** |
| Impossible vs harmful | 0.693 | 0.6925 | 0.7425 | **VERIFIED** |

#### Feature Analysis

| Condition | norm_per_token | sd | key_rank | key_entropy | n_generated |
| ----------- | ---------------- | ----- | ---------- | ------------- | ------------- |
| Impossible refusal | 294.23 | 5.81 | 76.48 | 4.1924 | 50.0 |
| Harmful refusal | 295.65 | 3.41 | 76.15 | 4.1885 | 50.0 |
| Benign normal | 301.08 | 15.10 | 75.11 | 4.1750 | 48.6 |

Both refusal types are sparser per token than benign. Impossibility refusals are *even sparser* (294.23) than harmful refusals (295.65).

**Multi-feature effect sizes** (computed during adversarial review):

| Feature | impossible vs benign (d) | harmful vs benign (d) | impossible vs harmful (d) |
| --------- | -------------------------- | ----------------------- | --------------------------- |
| norm_per_token | -0.599 | -0.496 | -0.298 |
| key_rank | **1.879** | **1.530** | **0.823** |
| key_entropy | **1.792** | **1.505** | **0.750** |

Key_rank and key_entropy show *larger* effect sizes than norm_per_token in every comparison. Refusal conditions have higher attention rank and entropy — the attention mechanism is more diffuse/distributed during output withholding. The impossible-vs-harmful separation (key_rank d=0.823) likely explains the above-chance AUROC of 0.693 between refusal types: the classifier partially distinguishes them through attention structure, even though both produce the same suppression pattern relative to benign.

#### EXP36-001: "Harmful Content Detection" Framing Falsified

The AUROC data shows:

- Impossibility refusal is **more detectable** than harmful refusal (0.950 vs 0.898)
- The two refusal types are **hard to distinguish** (AUROC 0.693 — above chance but low)
- The suppression signature correlates with **output withholding in general**, not harmful content specifically

This falsifies the framing in PITCH_NUMBERS.md line 24 ("Harmful content detection | **0.878** | Distinguishes harmful from benign processing"). What Cricket detects is **refusal** — the model allocating fewer resources per token when withholding an answer, regardless of why.

**Upstream impact**:

- PITCH_NUMBERS.md line 24: "harmful content detection" → should be "refusal detection" or "output suppression detection"
- Exp 32/34 "jailbreak detection" is really detecting the *absence* of suppression when the abliterated model answers, not the *presence* of harmful content processing
- The harmful_vs_benign LR AUROC (0.8975) matches Exp 31's value (0.898) almost exactly — good internal consistency across experiments

#### EXP36-P01: Honest Hypothesis Testing (POSITIVE)

The commit credits Cassidy for sparking this experiment. The team had a working "harmful content detection" narrative (AUROC 0.878) and chose to test whether the signal was actually about harm vs refusal in general. The answer undermines their own headline claim. Publishing the result anyway is good scientific practice.

---

## Part 4: JiminAI-Cricket

Repo: Liberation-Labs-THCoalition/JiminAI-Cricket
Source: Code review + file inspection via `gh api`

---

### Pre-trained Classifier Claims

| ID | Claim | File | Evidence | Severity | Verdict |
| ---- | ------- | ------ | ---------- | ---------- | --------- |
| CRICKET-001 | Confabulation detection at 88.6% confidence in demo | `data/precomputed_examples.json` | JSON value: `"confabulation": 0.885523689290657` (rounds to 88.6%). Pre-computed on `meta-llama/Llama-3.1-8B-Instruct`. However, hackathon Exp 14a-14d produced 5 null results — confabulation is NOT reliably detectable. `generated_text` fields are placeholders, not actual model outputs. | HIGH | **CONFIRMED** |
| CRICKET-002 | "reveals censorship" in demo UI | `src/cricket_demo.py` Gradio header | Header text: `"KV-cache geometry reveals deception, confabulation, and censorship with **AUROC 1.0** within-model detection."` Censorship classifier in `models/cricket_classifiers_meta.json`: `{"n_training": 0, "class_distribution": {}}`. No `cricket_rf_censorship.joblib` file in `models/` directory. Code gracefully skips (`if "censorship" in CLASSIFIERS`), but the UI claim is unsupported. | HIGH | **CONFIRMED** |
| CRICKET-003 | "AUROC 1.0" without cross-model context | `src/cricket_demo.py` header, commit d0847ab | AUROC 1.0 is within-model only. Cross-model transfer: avg 0.67, min 0.14. The header claims AUROC 1.0 for "deception, confabulation, and censorship" — but censorship has no model (CRICKET-002) and confabulation is unvalidated (CRICKET-001). | HIGH | **CONFIRMED** |

### Demo UI Summary

The Gradio header in `cricket_demo.py` makes a compound claim:

> "KV-cache geometry reveals deception, confabulation, and censorship with **AUROC 1.0** within-model detection."

Three issues:

1. **Censorship**: No trained classifier exists (n_training=0, no .joblib file)
2. **Confabulation**: Demo shows 88.6% confidence, but 5 hackathon experiments produced null results
3. **AUROC 1.0**: Only within-model; cross-model transfer averages 0.67, bottoms at 0.14

### README Status

| ID | Claim | File | Actual | Severity | Verdict |
| ---- | ------- | ------ | -------- | ---------- | --------- |
| CRICKET-004 | "Prototype: Not yet started" | `README.md` | Working Gradio demo exists in `src/` (4 Python files: `cricket_demo.py`, `cricket_classifier.py`, `cricket_features.py`, `cricket_viz.py`). `data/` has precomputed examples. README understates completeness. | MEDIUM | **CONFIRMED** |
| CRICKET-005 | "detect confabulation, sycophancy" in capability list | `README.md` | Zero sycophancy training data or experiments. Confabulation disproven by 5 null results (Exp 14a-14d). Neither capability exists. | MEDIUM | **CONFIRMED** |
| CRICKET-006 | "100% classification accuracy" for identity detection | — | Data leak artifact identified in WS3. Cross-prompt accuracy = 92-97.3%. The 100% figure comes from training/test contamination. | MEDIUM | **CONFIRMED** |
| CRICKET-007 | Cherry-picked precomputed examples | `data/precomputed_examples.json` | All 3 demo examples show high confidence: "honest" 97.1%, "deceptive" 96.5%, "confabulation" 88.6%. No lower-confidence or ambiguous examples shown. | MEDIUM | **CONFIRMED** |

### Previously Identified Issues (unchanged from prior audit)

| ID | Finding | Status |
| ---- | --------- | -------- |
| CRICKET-008 | CF3 "abliteration d=0.000 all categories" — actual self-ref d=+0.464 | UNCHANGED |
| CRICKET-009 | CF6 "cross-model rho=0.914" — matches neither verified value | UNCHANGED |
| CRICKET-010 | CF8 "no per-model calibration needed" — contradicts own DESIGN.md | UNCHANGED |

### Positive

- **CRICKET-P01**: `research/cricket_viability.md` (WS10) is the most honest document in the repo. Correctly flags 36% NO DATA and 27% PREMATURE. Should be the model for all other documentation in this repo.

---

## Part 5: CacheScope

Repo: Liberation-Labs-THCoalition/CacheScope
New repo — no prior audit baseline
Source: Code review via `gh api`

---

### Security Findings

| ID | Finding | File | Evidence | Severity | Verdict |
| ---- | --------- | ------ | ---------- | ---------- | --------- |
| CACHE-001 | `PUT /api/v1/config` unauthenticated | `cachescope/routes_rest.py` | No auth decorator or dependency. Test confirms: `client.put("/api/v1/config", json={"layer_stride": 4})` returns 200. Mutable fields: `layer_stride`, `layer_subset`, `window_size`, `async_extraction`, `variance_threshold`. No auth libraries in `pyproject.toml`. | MEDIUM | **CONFIRMED** |
| CACHE-002 | CORS `allow_origins=["*"]` | `cachescope/app.py` | `CORSMiddleware(allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])` in `create_app()`. Combined with CACHE-001, any browser JS on any origin can mutate server configuration. | MEDIUM | **CONFIRMED** |
| CACHE-003 | No authentication on ANY endpoint | `routes_rest.py`, `routes_ws.py`, `proof/routes.py` | Zero auth dependencies across all route files. Includes WebSocket `/ws/v1/stream` and Proof of Mind endpoints (`/api/v1/proof/generate`, `/api/v1/proof/verify`, `/api/v1/proof/verify/binary`). No `python-jose`, `passlib`, or auth libraries in `pyproject.toml`. | MEDIUM | **CONFIRMED** |

### Claims

| ID | Finding | File(s) | Evidence | Severity | Verdict |
| ---- | --------- | --------- | ---------- | ---------- | --------- |
| CACHE-004 | Compression ratio: "2,000x" vs "~17,000x" | `THREAT_MODEL.md` / `demo/tui.py` | 2,000x = 262 KB (one layer) / 121 bytes (MindPrint). ~17,000x = 262 KB x 8 layers (stride-4 on 32 layers) / 121 bytes. Both are mathematically correct but compare different bases. Neither documents its calculation. TUI displays `~17,000x` to judges without qualification. | LOW | **CONFIRMED** |
| CACHE-005 | Static model lookup table (8 entries) | `cachescope/bittensor/validator.py` → `_verify_model_signature()` | `KNOWN_LAYERS` dict: Qwen 0.5B/7B/14B/32B, Llama-8B, Mistral-7B, Gemma-9b, TinyLlama-1.1B. Unknown `model_id` bypasses layer-count verification (`expected_layers = None` → check skipped). Acknowledged in THREAT_MODEL.md limitation #4. | LOW | **CONFIRMED** |
| CACHE-006 | `sys.path.insert` for Cricket/KV-Experiments | `demo/tui.py`, `cachescope/extractor.py`, `bittensor/validator.py` | Hardcoded paths: `~/KV-Experiments/code`, `~/jiminai-cricket/src`, `/home/agent/...`. Dependencies (`cricket_features`, `cricket_classifier`, `gpu_utils`) imported via sys.path, not declared in `pyproject.toml`. | LOW | **CONFIRMED** |

### Positive Observations

| ID | Finding | Evidence |
| ---- | --------- | ---------- |
| CACHE-P01 | THREAT_MODEL.md is unusually transparent | ~3,400 words. 5 defended threats (model substitution, censorship routing, response caching, computation shortcuts, fake geometry injection). 4 acknowledged non-protections (side-channel, privacy, hardware, DoS). 5 current limitations explicitly enumerated. Genuine document, not a placeholder. |
| CACHE-P02 | Good test suite | 52 test functions across 5 files: `test_routes.py` (10), `test_proof.py` (19), `test_ring_buffer.py` (10), `test_snapshot.py` (5), `test_bittensor.py` (8). SHA-256 content binding, compact binary codec. Gaps: no extractor tests, no interceptor tests, no WebSocket tests. |
| CACHE-P03 | Clean architecture | Ring buffer, FastAPI, WebSocket streaming. Well-structured separation of concerns. |

---

## Part 6: JiminAI-Frontend-Performative

Repo: Liberation-Labs-THCoalition/JiminAI-Frontend-Performative
New repo — no prior audit baseline
Source: Red team code review via `gh api`

---

### Red Team Findings

| ID | Finding | File | Evidence | Severity | Verdict |
| ---- | --------- | ------ | ---------- | ---------- | --------- |
| FRONT-001 | Fabricated training terminal | `frontend/src/components/SplashOverlay.jsx` → `TrainingTerminal` | `seededRandom(42)` LCG generates deterministic fake training output. 30 `DATASETS` entries mix real benchmarks (TruthfulQA, MMLU, HellaSwag, HumanEval, ARC, GSM8K, BigBench, ToxiGen, CrowS-Pairs) with fabricated ones (cachescope_kv_geometry, attention_entropy_corpus, lying_classifier_gold). 14 fake `EVENTS` ("Compiling deception graph kernels...", "Optimizing lie-detection kernels..."). Loss curve: `2.8 * Math.exp(-progress * 3.5) + 0.15`. Cycles 10,000 fake epochs with timing variation (120ms fast, 1100ms pauses). No actual training or inference occurs. | HIGH | **CONFIRMED** |
| FRONT-002 | "AI LIE DETECTOR" headline | `SplashOverlay.jsx` | Unqualified capability claim in splash screen title. Links to Liberation Labs and Digital Disconnections. No "experimental", "prototype", or "hackathon" qualifier. | HIGH | **CONFIRMED** |

### UI Claims

| ID | Finding | File | Evidence | Severity | Verdict |
| ---- | --------- | ------ | ---------- | ---------- | --------- |
| FRONT-003 | Doc-code mismatch | `README.md` vs actual code | README describes functional "Twitch-style live prompt feed" with full API docs (POST /api/prompt, GET /api/queue, WS /ws/feed) and deployment instructions. Backend API is implemented in `server.py`. But frontend is locked: SplashOverlay at z-index 200 covers everything, PromptForm returns null. App as deployed cannot accept or process user prompts. | MEDIUM | **CONFIRMED** |
| FRONT-004 | App is non-functional | `frontend/src/components/PromptForm.jsx` | First line of component: `return null; // splash overlay active — form hidden`. 200+ lines of unreachable JSX below (desktop form, mobile form, submit button, queue position tracking, API integration). `eslint-disable-next-line no-unreachable` confirms intentional. Double-locked: dead component + visual overlay at z-200 with no dismiss mechanism. | MEDIUM | **CONFIRMED** |
| FRONT-005 | "LIAR (73%)" badges as authoritative | `classifier.py` | Classification uses 2 threshold checks: `entropy > 0.38 AND rank > 10.0` for "LIAR", `entropy < 0.335 OR norm_per_token < 70.0` for "REFUSAL". These are hardcoded thresholds calibrated for "Qwen-0.5B with stride-4 layer sampling" — NOT the Random Forest classifiers from JiminAI-Cricket. | MEDIUM | **CONFIRMED** |

### Low Priority

| ID | Finding | File | Evidence | Severity | Verdict |
| ---- | --------- | ------ | ---------- | ---------- | --------- |
| FRONT-006 | No WebSocket authentication on `/ws/feed` | `server.py` | No auth on WebSocket endpoint. | LOW | **CONFIRMED** |
| FRONT-007 | Production domain `discnxt.com` exposed | `Caddyfile` | Planned deployment domain visible in configuration. | LOW | **CONFIRMED** |

---

## Part 7: PITCH_NUMBERS.md Full Assessment

### Changes Made (commit bd84a7d) — Verified

| Line | Before | After | Assessment |
| ------ | -------- | ------- | ------------ |
| 24 | (not present) | "Harmful content detection \| **0.878** \| Distinguishes harmful from benign processing" | **FALSIFIED by Exp 36**: impossibility refusal scores higher (0.950 > 0.898). Signal is about refusal, not harm. |
| 52 | "28/28 layers show d > 1.0" | "ANCOVA significant 27/28 layers after token control (d=-1.9/token)" | **CORRECTED** — good |
| 122-126 | "signal exists at all 28 layers simultaneously" | "ANCOVA confirms signal survives at 27/28 layers" | **CORRECTED** — good |

### Contradictions Within PITCH_NUMBERS.md

| ID | Line | Claim | Contradiction | Severity |
| ---- | ------ | ------- | --------------- | ---------- |
| PITCH-CENSOR | ~18 | "Censorship detection (within-model) \| **1.000**" with "900 censorship samples" | `cricket_classifiers_meta.json` shows `n_training: 0` for censorship, no `cricket_rf_censorship.joblib` file exists. The pitch claims a capability the classifier cannot perform. | HIGH |
| PITCH-SYCOPH | ~81 | "flags deception, censorship, sycophancy, and confabulation" | Sycophancy data shows Cohen's d = 0.107 (negligible). Compare deception d = -3.065. Not a detection signal. | HIGH |
| PITCH-LAYER | Technical Details | "Deception signal uniform across all transformer layers (not localized)" | Exp 35 corrected this to "27/28 layers, sparser per token" on line 52 of the *same document*. The uncorrected version appears later, which judges reading to the end will see last. | HIGH |

### Still Stale

| Line | Claim | Actual | Verdict |
| ------ | ------- | -------- | --------- |
| 161 | "31 experiments over 2 campaigns" | 36 (Exp 32-36 added) | **STALE** |
| 161 | "64+ JSON result files" | 118 (89 in results/ + 29 in results/hackathon/) | **STALE** |

---

## Part 8: Cross-Cutting Issues

### Issue 1: FALSE_STATUS — "JSON saved" claims with no file (Exp 14-16)

| Line | Experiment | Status claim | File exists? |
| ------ | ----------- | ------------- | ------------- |
| 65 | C7 Encoding | "COMPLETE. JSON saved." | **NO** |
| 101 | S3 Generation | "COMPLETE. JSON saved." | **NO** |
| 174 | Contrastive Bare/Grounded | "COMPLETE. JSON saved." | **NO** — described as "HEADLINE RESULT" |
| 267 | 4-Condition Control | "COMPLETE. JSON saved." | **NO** |
| 321 | Cross-model Contrastive (3 models) | "COMPLETE. JSON saved for all three models." | **NO** (0 of 3 files) |

**Remediation**: Investigate whether files were saved to a different directory, a temp location, or if serialization silently failed. Update the log status lines to reflect actual state. If files cannot be located, change to "COMPLETE. Results reported inline. JSON NOT saved — needs re-run."

### Issue 2: Misnamed data files (Exp 14-16)

| Current filename | Actual content | Suggested name |
| ----------------- | --------------- | ---------------- |
| `confabulation_detection_Qwen2.5-7B_results.json` | Enhanced sycophancy (authority-framed) | `sycophancy_enhanced_Qwen2.5-7B_results.json` |
| `confabulation_detection_Llama-3.1-8B_results.json` | Enhanced sycophancy | `sycophancy_enhanced_Llama-3.1-8B_results.json` |
| `confabulation_detection_Mistral-7B-v0.3_results.json` | Enhanced sycophancy | `sycophancy_enhanced_Mistral-7B-v0.3_results.json` |

**Remediation**: Rename files. Update any scripts that reference the old names.

### Issue 3: Standard vs Enhanced sycophancy data separation (Exp 14-16)

The standard sycophancy experiment (d=-0.054, rate 4/60) and enhanced sycophancy (d=-0.107, rate 9/60) are distinct experiments with different prompts and different results. The standard results have NO data file. The enhanced results are in the misnamed "confabulation_detection" files.

**Remediation**: Save standard sycophancy results as separate JSON files for all 3 models.

### Issue 4: Unvalidated claims presented as capabilities (Exp 26-33 + repos)

Three repos present unvalidated or disproven capabilities as if they work:

| Repo | Claim | Reality |
| ------ | ------- | --------- |
| Cricket | Confabulation detection at 88.6% | 5 null experiment results (Exp 14a-14d) |
| Cricket | Censorship detection | Zero training samples, no model file |
| Frontend | "AI LIE DETECTOR" | Threshold-based classifier, not ML; app can't accept input |

### Issue 5: Documentation drift (Exp 26-33 + repos)

All repos have documentation that doesn't match code state:

| Repo | Doc says | Code says |
| ------ | ---------- | ----------- |
| Cricket | "Prototype: Not yet started" | Working Gradio demo exists |
| Frontend | Functional "Twitch-style live prompt feed" | SplashOverlay blocks everything; PromptForm returns null |
| KV-Experiments | "31 experiments, 64+ JSON files" | 36 experiments, 118 JSON files |

### Issue 6: Demo vs production boundary (Exp 26-33 + repos)

CacheScope has excellent security documentation (THREAT_MODEL.md) but no authentication implementation. Frontend has a polished splash screen that actively prevents the real app from functioning. These suggest demo-quality code being presented without appropriate "demo only" labeling.

### Issue 7: HACKATHON_LOG.md gap (Exp 34-36)

The log ends at Experiment 32 (line 904). Experiments 33, 34, 35, and 36 have no log entries. All four exist only as commit messages + JSON files. This creates an audit trail gap — the log was the primary structured record for Exp 14-32 but is now 4 experiments behind.

### Issue 8: Cross-Repo AUROC inconsistency (Exp 34-36)

CacheScope README claims Cricket achieves **"0.9991 AUROC"** on deception detection. PITCH_NUMBERS.md claims **"1.000"**. The C2 paper reports **1.0** (within-model RF). These are different numbers for the same metric without explanation of which evaluation produced each value.

### Issue 9: EXP32-001 error propagation (Exp 32 → 34)

The "18/20" n_actually_answered error from Exp 32 (JSON says 8) was hardcoded into `code/34_jailbreak_multimodel.py` line 334 as a string literal. This means the error is now in two places: the HACKATHON_LOG.md narrative and the Exp 34 script. Any downstream document or presentation citing either source inherits the 2.25x inflation.

### Issue 10: "Harmful content detection" framing falsified (Exp 36)

Exp 36 demonstrates that the KV-cache suppression signature correlates with **output withholding in general** (impossibility refusal AUROC 0.950 > harmful refusal AUROC 0.898), not harmful content specifically. This undermines:

- PITCH_NUMBERS.md line 24 ("harmful content detection")
- Exp 32/34 "jailbreak detection" narrative (actually measures absence of suppression)
- Any downstream marketing claiming Cricket detects "harmful content"

The correct framing is **refusal detection** or **output suppression detection**.

---

## Remediation Checklist

### Priority 1: Data integrity (errors with downstream impact)

- [ ] **EXP32-001**: Correct "18/20" → "8/20" in HACKATHON_LOG.md line 902 and update narrative
  - File: `HACKATHON_LOG.md`, line 902
  - Fix: Change "18/20 harmful prompts answered" to "8/20" (per JSON classifier)
- [ ] **EXP34-003**: Fix hardcoded "18/20" → "8/20" in `code/34_jailbreak_multimodel.py` line 334
- [ ] **EXP34-002**: Replace hardcoded `0.878` on line 344 with programmatic load from Exp 32 JSON
- [ ] **EXP34-003**: Update commit message / docs that reference "18/20 answered — clean jailbreak signal" for Qwen
- [ ] **CRICKET-001**: Remove confabulation example from demo or add "experimental/unvalidated" disclaimer
  - File: `data/precomputed_examples.json` (remove confab entry) + `src/cricket_demo.py` (add disclaimer)
- [ ] **CRICKET-002**: Remove "censorship" from demo header until classifier is trained
  - File: `src/cricket_demo.py`, Gradio header text
  - Fix: Change "deception, confabulation, and censorship" → "deception"
- [ ] **FRONT-001**: Remove fake training terminal, label as "decorative/simulated", or replace with real data
  - File: `frontend/src/components/SplashOverlay.jsx` → `TrainingTerminal` component

### Priority 2: Misleading claims (risks judge/reviewer confusion)

- [ ] **PITCH-CENSOR**: Remove or caveat "Censorship detection AUROC 1.000" — Cricket has no censorship classifier (n_training=0, no .joblib)
- [ ] **PITCH-SYCOPH**: Remove "sycophancy" from product description and misalignment axis claim — d=0.107 is not a detection signal
- [ ] **PITCH-LAYER**: Fix Technical Details per-layer claim to match corrected Mechanistic Findings ("27/28 layers, sparser per token")
- [ ] **EXP36-001**: Rename line 24 "harmful content detection" → "refusal detection" or "output suppression detection"
- [ ] **PITCH-001/002**: Update "31 experiments" → "36" and "64+" → "118" at line 161
- [ ] **CRICKET-003**: Add "within-model" qualifier to AUROC 1.0; note cross-model = 0.67 avg
  - File: `src/cricket_demo.py` header, commit message
- [ ] **FRONT-002**: Add "experimental" or "hackathon prototype" qualifier to "AI LIE DETECTOR"
  - File: `frontend/src/components/SplashOverlay.jsx`
- [ ] **EXP33-003**: Re-run Qwen in Exp 33 framework or note cross-reference from Exp 31
  - File: `results/hackathon/refusal_multimodel.json` (re-run) or HACKATHON_LOG.md (note)
- [ ] **EXP34-001**: Add caveat to Llama AUROC — abliteration failed (1/20), AUROC measures refusal-vs-normal not jailbreak-vs-normal
- [ ] **ADV-MEAN**: Add caveat to Exp 34 mean AUROC (0.849) noting it inherits the OVERSTATED Llama value
- [ ] **EXP36-001 (upstream)**: Reframe Exp 32/34 "jailbreak detection" narrative — Cricket detects the absence of suppression, not the presence of harmful content processing
- [ ] **ADV-AUROC**: Reconcile CacheScope "0.9991 AUROC" with PITCH "1.000" — clarify which evaluation produced each number

### Priority 3: Missing data files (blocks verification of 25+ claims from Exp 14-16)

- [ ] Locate or re-generate C7 encoding results
- [ ] Locate or re-generate S3 generation results
- [ ] Locate or re-generate contrastive bare/grounded results (HEADLINE RESULT)
- [ ] Locate or re-generate 4-condition control results (DEBUNKING RESULT)
- [ ] Locate or re-generate cross-model contrastive results (Llama + Mistral)
- [ ] Save standard sycophancy results (all 3 models)
- [ ] Save permutation test raw distributions

### Priority 4: Fix false status claims (integrity issue, Exp 14-16)

- [ ] Update 5 "JSON saved" status lines to reflect actual state
- [ ] Add serialization to experiment scripts to prevent recurrence

### Priority 5: Documentation alignment

- [ ] **CRICKET-004**: Update README status — "Prototype: Not yet started" → prototype exists
- [ ] **CRICKET-005**: Remove confabulation/sycophancy from README capability list
- [ ] **CRICKET-006**: Correct "100% accuracy" to cross-prompt accuracy (92-97.3%)
- [ ] **CRICKET-007**: Add lower-confidence examples to demo or label as "best-case showcase"
- [ ] **FRONT-003**: Align README with actual app state (splash-only, no user input)
- [ ] **FRONT-004**: Remove `return null` in PromptForm.jsx or update README to reflect splash-only state
- [ ] **FRONT-005**: Add "experimental" qualifier to "LIAR (73%)" badge display
  - File: `classifier.py` (thresholds) + frontend badge rendering
- [ ] **EXP26-001**: Change rho range to "0.826-0.896" or note SMALL_vs_LARGE exception
  - File: `HACKATHON_LOG.md`, lines 824-826
- [ ] **EXP26-002**: Add per-model breakdown to JSON or qualify "Coding #1 at 100% of models at ALL scales"
  - File: `results/hackathon/scale_invariance.json` (add data) or `HACKATHON_LOG.md` line 828 (qualify claim)
- [ ] **LOG-001**: Add Exp 33, 34, 35, 36 entries to HACKATHON_LOG.md

### Priority 6: Rename misnamed files (prevents future confusion, Exp 14-16)

- [ ] Rename 3 `confabulation_detection_*` files to `sycophancy_enhanced_*`

### Priority 7: Investigate 2 discrepancies (Exp 14-16)

- [ ] C7 direction AUROC: claimed 0.288, closest JSON 0.258 (diff=0.030)
- [ ] Flipped AUROC: claimed 0.712, JSON 0.732 (diff=0.020)

### Priority 8: Security hardening (demo-appropriate, prod-blocking)

- [ ] **CACHE-001**: Add auth middleware or restrict `PUT /api/v1/config` to localhost
  - File: `cachescope/routes_rest.py`
- [ ] **CACHE-002**: Restrict CORS origins to known frontends
  - File: `cachescope/app.py` → `create_app()`
- [ ] **CACHE-003**: Add API key or JWT for production deployment
  - Files: All route files (`routes_rest.py`, `routes_ws.py`, `proof/routes.py`)
- [ ] **FRONT-006**: Add WebSocket authentication on `/ws/feed`
  - File: `server.py`

### Priority 9: Cleanup (low urgency)

- [ ] **CACHE-004**: Document compression ratio comparison bases (1 layer vs 8 layers)
  - Files: `THREAT_MODEL.md`, `demo/tui.py`
- [ ] **CACHE-005**: Add fallback verification for unknown model architectures
  - File: `cachescope/bittensor/validator.py` → `KNOWN_LAYERS`
- [ ] **CACHE-006**: Replace `sys.path.insert` with proper packaging (`pip install -e` or submodules)
  - Files: `demo/tui.py`, `cachescope/extractor.py`, `bittensor/validator.py`
- [ ] **FRONT-007**: Remove production domain from Caddyfile or add to .gitignore
  - File: `Caddyfile`

### No action needed (fully verified)

- **Exp 14-16**: Enhanced sycophancy: all 3 models, all statistics (20+ values) — exact match
- **Exp 14-16**: Refusal geometry: all 4 hypotheses, all per-topic values — exact match
- **Exp 14-16**: Direction sweep: all 11 analyses, 22 AUROC values — exact match
- **Exp 14-16**: Permutation test: both real AUROCs match direction sweep data
- **Exp 26-31**: 27 numeric claims verified against JSON — all match
- **Exp 32**: All 6 AUROC/accuracy values verified (only n_actually_answered is wrong)
- **Exp 33**: Both AUROC values verified (rounding matches under standard convention)
- **Exp 31-32**: norm_per_token cross-experiment consistency confirmed (refusal=295.7 and normal=301.1 appear in both experiments)
- **Exp 34**: AUROC values — Llama 0.8775 and Mistral 0.7925 match commit claims (standard rounding)
- **Exp 34**: Mean AUROC — arithmetic correct (0.8492 → 0.849), though interpretively qualified (see ADV-MEAN)
- **Exp 34**: n_abl_answered — Llama 1/20 and Mistral 8/20 match commit
- **Exp 35**: All values verified against JSON
- **Exp 35**: Self-correction of Exp 29 claim is scientifically sound (direction reversal, not just reframing)
- **Exp 36**: All AUROC values verified against JSON (3/3 match)
- **Exp 36**: harmful_vs_benign AUROC (0.8975) matches Exp 31 (0.898) — good internal consistency
- **PITCH_NUMBERS.md**: Per-layer correction (lines 52, 122-126) is accurate
- **CACHE-P01**: THREAT_MODEL.md is substantive and honest — model for other repos
- **CACHE-P02**: 52 tests with good core coverage (routes, proof, ring buffer, snapshot, bittensor)
- **CACHE-P03**: Clean architecture with proper separation of concerns
- **CRICKET-P01**: `cricket_viability.md` is the most honest document in any audited repo

---

*Audit conducted by Dwayne Wilkes & Kavi. Sources: Automated claim extraction, swarm extraction (4 parallel agents), adversarial vetting (3 parallel agents), manual cross-reference against JSON data files and source via `gh api`. Every numeric claim in HACKATHON_LOG.md and experiment commit messages is accounted for above. Code review findings spot-checked against source files in all 4 repos.*
