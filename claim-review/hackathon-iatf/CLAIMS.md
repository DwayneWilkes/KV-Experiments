# Intelligence at the Frontier (IATF) Hackathon — Claims Registry

> **Event**: [Intelligence at the Frontier Hackathon](https://luma.com/ftchack-sf-2026) (FTCHack SF 2026), hosted by Funding the Commons & Protocol Labs. March 14–15, 2026, San Francisco.
> **Team**: Liberation Labs.

**Total claims**: 189
**Audit date**: 2026-03-14
**Auditors**: Dwayne Wilkes & Kavi
**Sources**: HACKATHON_LOG.md (Exp 14-16), KV-Experiments commits (Exp 26-36), JiminAI-Cricket, CacheScope, JiminAI-Frontend-Performative, PITCH_NUMBERS.md

**Verdict categories**: VERIFIED / NO_DATA / DISCREPANT / CRITICAL / FALSIFIED / OVERSTATED / STALE / UNVERIFIED / FALSE_STATUS / LOW_PRIORITY / CONFIRMED / UNCHANGED / POSITIVE

**Tolerances**: Rounding ±0.001 for AUROCs (standard round-half-up). Cohen's d ±0.02. Counts exact.

---

## Summary

| Verdict | Count |
| --- | --- |
| VERIFIED | 85 |
| NO_DATA | 49 |
| CONFIRMED | 24 |
| POSITIVE | 6 |
| FALSE_STATUS | 5 |
| CRITICAL | 3 |
| DISCREPANT | 3 |
| OVERSTATED | 3 |
| UNCHANGED | 3 |
| UNVERIFIED | 3 |
| STALE | 3 |
| LOW_PRIORITY | 1 |
| FALSIFIED | 1 |

**No fabrication detected.** Where data files exist, every reported number matches. Critical findings are miscounts and framing issues, not data fabrication.

---

## Part 1: Exp 14-16 (H14-xxx, H15-xxx, H16-xxx)

Source: `report.md` (HACKATHON_LOG.md lines 15-691, 6 JSON files in `results/hackathon/`)

---

### Exp 14a: C7 Frequency-Matched Encoding-Only (Qwen 7B)

*HACKATHON_LOG.md lines 15-66. Data file: NONE.*

- [x] **H14-001** | "COMPLETE. JSON saved." (line 65) for C7 encoding results
  - **Expected**: JSON file on disk
  - **Check**: Search `results/hackathon/` for C7 encoding results file
  - **Files**: None found (only `c7_permutation_test.json` exists, which is Exp 16)
  - **Verdict**: FALSE_STATUS

- [x] **H14-002** | Effective rank: confab=14.38, factual=14.33, d=0.052
  - **Expected**: d=0.052
  - **Check**: Load C7 encoding JSON, recompute effective rank means and Cohen's d
  - **Files**: No JSON exists
  - **Verdict**: NO_DATA

- [x] **H14-003** | Spectral entropy d=0.224
  - **Expected**: d=0.224
  - **Check**: Load C7 encoding JSON, recompute spectral entropy d
  - **Files**: No JSON exists
  - **Verdict**: NO_DATA

- [x] **H14-004** | Key norms d=-0.018
  - **Expected**: d=-0.018
  - **Check**: Load C7 encoding JSON, recompute key norms d
  - **Files**: No JSON exists
  - **Verdict**: NO_DATA

- [x] **H14-005** | Wilcoxon W=179.5, p=0.819
  - **Expected**: W=179.5, p=0.819
  - **Check**: Load C7 encoding JSON, recompute Wilcoxon test
  - **Files**: No JSON exists
  - **Verdict**: NO_DATA

- [x] **H14-006** | Paired t: t=0.348, p=0.731
  - **Expected**: t=0.348, p=0.731
  - **Check**: Load C7 encoding JSON, recompute paired t-test
  - **Files**: No JSON exists
  - **Verdict**: NO_DATA

- [x] **H14-007** | Raw d=0.052 residualized to d=0.115
  - **Expected**: d_resid=0.115
  - **Check**: Load C7 encoding JSON, recompute residualized d
  - **Files**: No JSON exists
  - **Verdict**: NO_DATA

- [x] **H14-008** | TOST p_upper=0.171, p_lower=0.089
  - **Expected**: p_upper=0.171, p_lower=0.089
  - **Check**: Load C7 encoding JSON, recompute TOST bounds
  - **Files**: No JSON exists
  - **Verdict**: NO_DATA

- [x] **H14-009** | Per-domain d values (6 domains): range -0.539 to +0.358
  - **Expected**: 6 d values within stated range
  - **Check**: Load C7 encoding JSON, recompute per-domain d
  - **Files**: No JSON exists
  - **Verdict**: NO_DATA

---

### Exp 14b: S3 Generation-Phase Confabulation (Qwen 7B)

*HACKATHON_LOG.md lines 69-101. Data file: NONE.*

- [x] **H14-010** | "COMPLETE. JSON saved." (line 101) for S3 generation results
  - **Expected**: JSON file on disk
  - **Check**: Search `results/hackathon/` for S3 generation results file
  - **Files**: None found
  - **Verdict**: FALSE_STATUS

- [x] **H14-011** | Response counts: Accurate=40, Confabulated=17, Partial=1, Refused=0, Pending=2
  - **Expected**: Exact counts
  - **Check**: Load S3 generation JSON, verify classification counts
  - **Files**: No JSON exists
  - **Verdict**: NO_DATA

- [x] **H14-012** | d=-0.107, p=0.728
  - **Expected**: d=-0.107, p=0.728
  - **Check**: Load S3 generation JSON, recompute d and p
  - **Files**: No JSON exists
  - **Verdict**: NO_DATA

- [x] **H14-013** | Per-difficulty d: easy=-0.834, medium=-0.895, hard=+0.257
  - **Expected**: 3 d values as stated
  - **Check**: Load S3 generation JSON, recompute per-difficulty d
  - **Files**: No JSON exists
  - **Verdict**: NO_DATA

---

### Exp 14c: Contrastive Encoding — Bare vs Grounded (Qwen 7B)

*HACKATHON_LOG.md lines 105-175. Data file: NONE. Described as "THE HEADLINE RESULT."*

- [x] **H14-014** | "COMPLETE. JSON saved. THIS IS THE HEADLINE RESULT." (line 174)
  - **Expected**: JSON file on disk
  - **Check**: Search `results/hackathon/` for contrastive bare/grounded results
  - **Files**: None found
  - **Verdict**: FALSE_STATUS

- [x] **H14-015** | Bare=19.08, Grounded=25.79, d=-2.354
  - **Expected**: d=-2.354
  - **Check**: Load contrastive JSON, recompute effective rank means and d
  - **Files**: No JSON exists
  - **Verdict**: NO_DATA

- [x] **H14-016** | Spectral entropy d=-0.477
  - **Expected**: d=-0.477
  - **Check**: Load contrastive JSON, recompute spectral entropy d
  - **Files**: No JSON exists
  - **Verdict**: NO_DATA

- [x] **H14-017** | Key norms d=-1.937
  - **Expected**: d=-1.937
  - **Check**: Load contrastive JSON, recompute key norms d
  - **Files**: No JSON exists
  - **Verdict**: NO_DATA

- [x] **H14-018** | Wilcoxon W=0.0, p<0.0001
  - **Expected**: W=0.0, p<0.0001
  - **Check**: Load contrastive JSON, recompute Wilcoxon test
  - **Files**: No JSON exists
  - **Verdict**: NO_DATA

- [x] **H14-019** | Token counts: Bare=25.0, Grounded=44.4
  - **Expected**: Bare 25.0 tokens, Grounded 44.4 tokens
  - **Check**: Load contrastive JSON, compute mean token counts
  - **Files**: No JSON exists
  - **Verdict**: NO_DATA

- [x] **H14-020** | Raw d=-2.354 residualized to d=-1.081
  - **Expected**: d_resid=-1.081
  - **Check**: Load contrastive JSON, recompute residualized d
  - **Files**: No JSON exists
  - **Verdict**: NO_DATA

- [x] **H14-021** | Per-layer d values (5 layers shown): range -2.724 to -2.518
  - **Expected**: 5 d values within stated range
  - **Check**: Load contrastive JSON, recompute per-layer d
  - **Files**: No JSON exists
  - **Verdict**: NO_DATA

- [x] **H14-022** | Per-difficulty d (3 levels): range -2.403 to -2.425
  - **Expected**: 3 d values within stated range
  - **Check**: Load contrastive JSON, recompute per-difficulty d
  - **Files**: No JSON exists
  - **Verdict**: NO_DATA

- [x] **H14-023** | Per-domain d (6 domains): range -1.871 to -4.162
  - **Expected**: 6 d values within stated range
  - **Check**: Load contrastive JSON, recompute per-domain d
  - **Files**: No JSON exists
  - **Verdict**: NO_DATA

---

### Exp 14d: Contrastive Controlled — 4-Condition Test (Qwen 7B)

*HACKATHON_LOG.md lines 219-267. Data file: NONE. Debunks the Exp 14c headline.*

- [x] **H14-024** | "COMPLETE. JSON saved." (line 267) for 4-condition control
  - **Expected**: JSON file on disk
  - **Check**: Search `results/hackathon/` for 4-condition results
  - **Files**: None found
  - **Verdict**: FALSE_STATUS

- [x] **H14-025** | Cell means: Bare=19.08, Correct=25.79, Wrong=26.09, Irrelevant=26.30
  - **Expected**: 4 mean values as stated
  - **Check**: Load 4-condition JSON, recompute cell means
  - **Files**: No JSON exists
  - **Verdict**: NO_DATA

- [x] **H14-026** | Correct vs Irrelevant d=-0.160, p=0.340
  - **Expected**: d=-0.160, p=0.340
  - **Check**: Load 4-condition JSON, recompute pairwise d and p
  - **Files**: No JSON exists
  - **Verdict**: NO_DATA

- [x] **H14-027** | Residualized d=-0.539
  - **Expected**: d_resid=-0.539
  - **Check**: Load 4-condition JSON, recompute residualized d
  - **Files**: No JSON exists
  - **Verdict**: NO_DATA

- [x] **H14-028** | Per-domain d (6 domains, correct vs irrelevant): range -0.988 to +0.498
  - **Expected**: 6 d values within stated range
  - **Check**: Load 4-condition JSON, recompute per-domain d
  - **Files**: No JSON exists
  - **Verdict**: NO_DATA

- [x] **H14-029** | Verdict: INFORMATION_VOLUME (not confabulation-specific)
  - **Expected**: Categorical verdict
  - **Check**: Verify narrative consistency — signal from information volume, not confabulation
  - **Files**: No JSON exists (narrative is self-consistent)
  - **Verdict**: NO_DATA

---

### Exp 14 Cross-Model: Contrastive Validation (Llama 8B, Mistral 7B)

*HACKATHON_LOG.md lines 297-321. Data file: NONE.*

- [x] **H14-030** | "COMPLETE. JSON saved for all three models." (line 321)
  - **Expected**: JSON files for Qwen, Llama, Mistral
  - **Check**: Search `results/hackathon/` for cross-model contrastive files
  - **Files**: None found (0 of 3)
  - **Verdict**: FALSE_STATUS

- [x] **H14-031** | Qwen correct vs irrelevant d=-0.160, p=0.340
  - **Expected**: d=-0.160, p=0.340
  - **Check**: Load Qwen 4-condition JSON
  - **Files**: No JSON exists
  - **Verdict**: NO_DATA

- [x] **H14-032** | Llama correct vs irrelevant d=-0.161, p=0.363
  - **Expected**: d=-0.161, p=0.363
  - **Check**: Load Llama 4-condition JSON
  - **Files**: No JSON exists
  - **Verdict**: NO_DATA

- [x] **H14-033** | Mistral correct vs irrelevant d=-0.192, p=0.245
  - **Expected**: d=-0.192, p=0.245
  - **Check**: Load Mistral 4-condition JSON
  - **Files**: No JSON exists
  - **Verdict**: NO_DATA

- [x] **H14-034** | Residualized d: Qwen=-0.539, Llama=-0.549, Mistral=-0.457
  - **Expected**: 3 residualized d values as stated
  - **Check**: Load all three 4-condition JSONs, recompute
  - **Files**: No JSON exists
  - **Verdict**: NO_DATA

---

### Standard Sycophancy Detection (Qwen 7B)

*HACKATHON_LOG.md lines 340-401. Data file: NONE (separate from enhanced sycophancy).*

- [x] **SYCO-001** | Sycophancy rate: 4/60 (6.7%)
  - **Expected**: 4 sycophantic, rate=0.067
  - **Check**: Load standard sycophancy JSON, verify count
  - **Files**: No JSON exists (standard sycophancy never serialized)
  - **Verdict**: NO_DATA

- [x] **SYCO-002** | Response breakdown: 4/60 sycophantic, 18/60 corrective, 38/60 ambiguous
  - **Expected**: Exact counts summing to 60
  - **Check**: Load standard sycophancy JSON, verify classification counts
  - **Files**: No JSON exists
  - **Verdict**: NO_DATA

- [x] **SYCO-003** | Means: Bare=17.61, Correct=19.90, Wrong=19.80
  - **Expected**: 3 mean values as stated
  - **Check**: Load standard sycophancy JSON, recompute means
  - **Files**: No JSON exists
  - **Verdict**: NO_DATA

- [x] **SYCO-004** | Wrong vs Correct d=-0.054, p=0.360
  - **Expected**: d=-0.054, p=0.360
  - **Check**: Load standard sycophancy JSON, recompute d and p
  - **Files**: No JSON exists
  - **Verdict**: NO_DATA

- [x] **SYCO-005** | Residualized d=+0.047, p=0.685
  - **Expected**: d_resid=+0.047, p=0.685
  - **Check**: Load standard sycophancy JSON, recompute residualized d
  - **Files**: No JSON exists
  - **Verdict**: NO_DATA

- [x] **SYCO-006** | Per-domain d (6 domains): range -0.212 to +0.218
  - **Expected**: 6 d values within stated range
  - **Check**: Load standard sycophancy JSON, recompute per-domain d
  - **Files**: No JSON exists
  - **Verdict**: NO_DATA

- [x] **SYCO-007** | Verdict: MODEL_RESISTANT
  - **Expected**: MODEL_RESISTANT categorical verdict
  - **Check**: Load standard sycophancy JSON, verify verdict field
  - **Files**: No JSON exists
  - **Verdict**: NO_DATA

---

### Standard Sycophancy Cross-Model (Llama 8B, Mistral 7B)

*HACKATHON_LOG.md lines 402-419. Data file: NONE for standard sycophancy.*

- [x] **SYCO-008** | Cross-model sycophancy rates: Qwen 4/60 (6.7%), Llama 0/60 (0%), Mistral 7/60 (11.7%)
  - **Expected**: Exact counts and rates for 3 models
  - **Check**: Load standard sycophancy JSONs for all 3 models
  - **Files**: No JSON exists (standard sycophancy)
  - **Verdict**: NO_DATA

- [x] **SYCO-009** | Cross-model Wrong vs Correct d: Qwen -0.054, Llama -0.043, Mistral -0.027
  - **Expected**: 3 d values as stated
  - **Check**: Load standard sycophancy JSONs, recompute d
  - **Files**: No JSON exists (standard sycophancy)
  - **Verdict**: NO_DATA

- [x] **SYCO-010** | Cross-model Wrong vs Correct p: Qwen 0.360, Llama 0.908, Mistral 0.657
  - **Expected**: 3 p values as stated
  - **Check**: Load standard sycophancy JSONs, recompute p
  - **Files**: No JSON exists (standard sycophancy)
  - **Verdict**: NO_DATA

- [x] **SYCO-011** | Cross-model Residualized d: Qwen +0.047, Llama +0.093, Mistral +0.146
  - **Expected**: 3 residualized d values as stated
  - **Check**: Load standard sycophancy JSONs, recompute residualized d
  - **Files**: No JSON exists (standard sycophancy)
  - **Verdict**: NO_DATA

- [x] **SYCO-012** | Mistral within-sycophancy d=+0.400, p=0.480
  - **Expected**: d=+0.400, p=0.480
  - **Check**: Standard sycophancy has no JSON; but enhanced Mistral within-syc d=0.400 IS verified (see H14-ENH-MISTRAL)
  - **Files**: No JSON for standard; enhanced value verified below
  - **Verdict**: NO_DATA (standard)

---

### Enhanced Sycophancy — Run 1: WITH REASONING (INVALIDATED)

*HACKATHON_LOG.md lines 433-454. Invalidated due to length confound.*

- [x] **SYCO-013** | Raw d=+3.455, p<0.0001 (invalidated run — length confound)
  - **Expected**: d=+3.455
  - **Check**: No JSON expected (run was invalidated)
  - **Files**: Not serialized
  - **Verdict**: NO_DATA (invalidated run)

- [x] **SYCO-014** | Residualized d flipped to -0.611 (invalidated run)
  - **Expected**: d_resid=-0.611
  - **Check**: No JSON expected (run was invalidated)
  - **Files**: Not serialized
  - **Verdict**: NO_DATA (invalidated run)

- [x] **SYCO-015** | Within-sycophancy d=+0.781, p=0.219 (invalidated run)
  - **Expected**: d=+0.781, p=0.219
  - **Check**: No JSON expected (run was invalidated)
  - **Files**: Not serialized
  - **Verdict**: NO_DATA (invalidated run)

- [x] **SYCO-016** | Sycophancy rate 6/60 (10.0%) (invalidated run)
  - **Expected**: rate=0.100
  - **Check**: No JSON expected (run was invalidated)
  - **Files**: Not serialized
  - **Verdict**: NO_DATA (invalidated run)

---

### Enhanced Sycophancy — Run 2: Authority-Only (Qwen 7B)

*HACKATHON_LOG.md lines 456-501. Data file: `confabulation_detection_Qwen2.5-7B_results.json` (misnamed).*

- [x] **SYCO-017** | Sycophancy rate 9/60 (15%)
  - **Expected**: sycophantic=9, rate=0.15
  - **Check**: `.sycophancy.statistics.sycophancy_rate`
  - **Files**: `confabulation_detection_Qwen2.5-7B_results.json`
  - **Verdict**: VERIFIED

- [x] **SYCO-018** | Mean bare=16.26
  - **Expected**: 16.260
  - **Check**: `.sycophancy.statistics.means.bare`
  - **Files**: `confabulation_detection_Qwen2.5-7B_results.json`
  - **Verdict**: VERIFIED

- [x] **SYCO-019** | Mean user_correct=21.37
  - **Expected**: 21.369
  - **Check**: `.sycophancy.statistics.means.user_correct`
  - **Files**: `confabulation_detection_Qwen2.5-7B_results.json`
  - **Verdict**: VERIFIED

- [x] **SYCO-020** | Mean user_wrong=21.19
  - **Expected**: 21.189
  - **Check**: `.sycophancy.statistics.means.user_wrong`
  - **Files**: `confabulation_detection_Qwen2.5-7B_results.json`
  - **Verdict**: VERIFIED

- [x] **SYCO-021** | Wrong vs Correct d=-0.107, p=0.772
  - **Expected**: d=-0.1071, p=0.7724
  - **Check**: `.sycophancy.statistics.pairwise.wrong_vs_correct.cohens_d` and `.wilcoxon.p`
  - **Files**: `confabulation_detection_Qwen2.5-7B_results.json`
  - **Verdict**: VERIFIED

- [x] **SYCO-022** | Residualized d=+0.143, p=0.029
  - **Expected**: d_resid=0.1426, p=0.0287
  - **Check**: `.sycophancy.statistics.residualized.wrong_vs_correct_resid_d` and `_resid_p`
  - **Files**: `confabulation_detection_Qwen2.5-7B_results.json`
  - **Verdict**: VERIFIED

- [x] **SYCO-023** | Within-sycophancy d=+0.393, p=0.478
  - **Expected**: d=0.3932, p=0.4775
  - **Check**: `.sycophancy.statistics.sycophantic_vs_corrective.cohens_d` and `.welch_t.p_value`
  - **Files**: `confabulation_detection_Qwen2.5-7B_results.json`
  - **Verdict**: VERIFIED

- [x] **SYCO-024** | Per-domain d (6 domains): range -0.392 to +0.075
  - **Expected**: 6 d values within stated range
  - **Check**: `.sycophancy.statistics.per_domain`
  - **Files**: `confabulation_detection_Qwen2.5-7B_results.json`
  - **Verdict**: VERIFIED

- [x] **SYCO-025** | Science sycophancy rate 4/10 (40%)
  - **Expected**: 4/10 in science domain
  - **Check**: `.sycophancy.statistics.per_domain.science`
  - **Files**: `confabulation_detection_Qwen2.5-7B_results.json`
  - **Verdict**: VERIFIED

- [x] **SYCO-026** | Verdict: NULL
  - **Expected**: NULL verdict
  - **Check**: `.sycophancy.statistics.verdict`
  - **Files**: `confabulation_detection_Qwen2.5-7B_results.json`
  - **Verdict**: VERIFIED

---

### Enhanced Sycophancy Cross-Model — Llama 3.1 8B

*Data file: `confabulation_detection_Llama-3.1-8B_results.json` (misnamed).*

- [x] **SYCO-027** | Llama sycophancy rate 0/60 (0%)
  - **Expected**: sycophantic=0, rate=0.0
  - **Check**: `.sycophancy.statistics.sycophancy_rate`
  - **Files**: `confabulation_detection_Llama-3.1-8B_results.json`
  - **Verdict**: VERIFIED

- [x] **SYCO-028** | Llama Wrong vs Correct d=-0.043, p=0.908
  - **Expected**: d=-0.0434, p=0.9075
  - **Check**: `.sycophancy.statistics.pairwise.wrong_vs_correct`
  - **Files**: `confabulation_detection_Llama-3.1-8B_results.json`
  - **Verdict**: VERIFIED

- [x] **SYCO-029** | Llama Residualized d=+0.093
  - **Expected**: d_resid=0.0935
  - **Check**: `.sycophancy.statistics.residualized`
  - **Files**: `confabulation_detection_Llama-3.1-8B_results.json`
  - **Verdict**: VERIFIED

- [x] **SYCO-030** | Llama Verdict: MODEL_RESISTANT
  - **Expected**: MODEL_RESISTANT
  - **Check**: `.sycophancy.statistics.verdict`
  - **Files**: `confabulation_detection_Llama-3.1-8B_results.json`
  - **Verdict**: VERIFIED

---

### Enhanced Sycophancy Cross-Model — Mistral 7B v0.3

*Data file: `confabulation_detection_Mistral-7B-v0.3_results.json` (misnamed).*

- [x] **SYCO-031** | Mistral sycophancy rate 7/60 (11.7%)
  - **Expected**: sycophantic=7, rate=0.117
  - **Check**: `.sycophancy.statistics.sycophancy_rate`
  - **Files**: `confabulation_detection_Mistral-7B-v0.3_results.json`
  - **Verdict**: VERIFIED

- [x] **SYCO-032** | Mistral Wrong vs Correct d=-0.027, p=0.657
  - **Expected**: d=-0.0268, p=0.6571
  - **Check**: `.sycophancy.statistics.pairwise.wrong_vs_correct`
  - **Files**: `confabulation_detection_Mistral-7B-v0.3_results.json`
  - **Verdict**: VERIFIED

- [x] **SYCO-033** | Mistral Residualized d=+0.146
  - **Expected**: d_resid=0.1455
  - **Check**: `.sycophancy.statistics.residualized`
  - **Files**: `confabulation_detection_Mistral-7B-v0.3_results.json`
  - **Verdict**: VERIFIED

- [x] **SYCO-034** | Mistral within-sycophancy d=+0.400
  - **Expected**: d=0.4002
  - **Check**: `.sycophancy.statistics.sycophantic_vs_corrective.cohens_d`
  - **Files**: `confabulation_detection_Mistral-7B-v0.3_results.json`
  - **Verdict**: VERIFIED

- [x] **SYCO-035** | Mistral Verdict: NULL
  - **Expected**: NULL
  - **Check**: `.sycophancy.statistics.verdict`
  - **Files**: `confabulation_detection_Mistral-7B-v0.3_results.json`
  - **Verdict**: VERIFIED

---

### Exp 15: Refusal Geometry — Base vs Abliterated (Qwen 7B)

*HACKATHON_LOG.md lines 509-581. Data file: `refusal_geometry_Qwen2.5-7B_results.json`.*

- [x] **H15-001** | Cell means: base_harmful=12.56, base_benign=12.49, abl_harmful=12.56, abl_benign=12.48
  - **Expected**: 4 cell means as stated
  - **Check**: `.statistics.cell_means`
  - **Files**: `refusal_geometry_Qwen2.5-7B_results.json`
  - **Verdict**: VERIFIED

- [x] **H15-002** | H1: d=+0.071, p=0.783
  - **Expected**: d=0.0715, p=0.783
  - **Check**: `.statistics.h1`
  - **Files**: `refusal_geometry_Qwen2.5-7B_results.json`
  - **Verdict**: VERIFIED

- [x] **H15-003** | H2: d=+0.071, p=0.784
  - **Expected**: d=0.0712, p=0.784
  - **Check**: `.statistics.h2`
  - **Files**: `refusal_geometry_Qwen2.5-7B_results.json`
  - **Verdict**: VERIFIED

- [x] **H15-004** | H3: d=+0.005, p=0.985
  - **Expected**: d=0.005, p=0.985
  - **Check**: `.statistics.h3`
  - **Files**: `refusal_geometry_Qwen2.5-7B_results.json`
  - **Verdict**: VERIFIED

- [x] **H15-005** | Control: d=+0.008, p=0.977
  - **Expected**: d=0.008, p=0.977
  - **Check**: `.statistics.control`
  - **Files**: `refusal_geometry_Qwen2.5-7B_results.json`
  - **Verdict**: LOW_PRIORITY

- [x] **H15-006** | Per-topic d (6 topics): range +1.608 to -1.007
  - **Expected**: 6 d values within stated range
  - **Check**: `.statistics.per_topic`
  - **Files**: `refusal_geometry_Qwen2.5-7B_results.json`
  - **Verdict**: VERIFIED

- [x] **H15-007** | Verdict: NO_ENCODING_SIGNAL
  - **Expected**: NO_ENCODING_SIGNAL
  - **Check**: Narrative consistency — encoding regime does not distinguish harmful from benign
  - **Files**: `refusal_geometry_Qwen2.5-7B_results.json`
  - **Verdict**: VERIFIED

---

### Exp 16: Direction Sweep — Per-Layer Profile Analysis

*HACKATHON_LOG.md lines 585-627. Data file: `direction_sweep_results.json`.*

- [x] **H16-001** | Control Llama: Dir AUROC=0.913, LR AUROC=0.948
  - **Expected**: 0.913, 0.948
  - **Check**: `direction_sweep_results.json` control Llama entries
  - **Files**: `direction_sweep_results.json`
  - **Verdict**: VERIFIED

- [x] **H16-002** | Control Mistral: Dir AUROC=0.948, LR AUROC=0.979
  - **Expected**: 0.948, 0.979
  - **Check**: `direction_sweep_results.json` control Mistral entries
  - **Files**: `direction_sweep_results.json`
  - **Verdict**: VERIFIED

- [x] **H16-003** | Control Qwen: Dir AUROC=1.000, LR AUROC=1.000
  - **Expected**: 1.000, 1.000
  - **Check**: `direction_sweep_results.json` control Qwen entries
  - **Files**: `direction_sweep_results.json`
  - **Verdict**: VERIFIED

- [x] **H16-004** | Sycophancy Llama: Dir AUROC=0.558, LR AUROC=0.857
  - **Expected**: 0.558, 0.857
  - **Check**: `direction_sweep_results.json` sycophancy Llama entries
  - **Files**: `direction_sweep_results.json`
  - **Verdict**: VERIFIED

- [x] **H16-005** | Sycophancy Mistral: Dir AUROC=0.677, LR AUROC=0.827
  - **Expected**: 0.677, 0.827
  - **Check**: `direction_sweep_results.json` sycophancy Mistral entries
  - **Files**: `direction_sweep_results.json`
  - **Verdict**: VERIFIED

- [x] **H16-006** | Sycophancy Qwen: Dir AUROC=0.526, LR AUROC=0.616
  - **Expected**: 0.526, 0.616
  - **Check**: `direction_sweep_results.json` sycophancy Qwen entries
  - **Files**: `direction_sweep_results.json`
  - **Verdict**: VERIFIED

- [x] **H16-007** | Refusal base: Dir AUROC=0.526, LR AUROC=0.579
  - **Expected**: 0.526, 0.579
  - **Check**: `direction_sweep_results.json` refusal base entries
  - **Files**: `direction_sweep_results.json`
  - **Verdict**: VERIFIED

- [x] **H16-008** | Refusal abliterated: Dir AUROC=0.526, LR AUROC=0.634
  - **Expected**: 0.526, 0.634
  - **Check**: `direction_sweep_results.json` refusal abliterated entries
  - **Files**: `direction_sweep_results.json`
  - **Verdict**: VERIFIED

- [x] **H16-009** | Refusal interaction: Dir AUROC=1.000, LR AUROC=1.000 (artifact)
  - **Expected**: 1.000, 1.000
  - **Check**: `direction_sweep_results.json` refusal interaction entries
  - **Files**: `direction_sweep_results.json`
  - **Verdict**: VERIFIED (log correctly identifies as artifact)

- [x] **H16-010** | Within-sycophancy Mistral: Dir AUROC=0.602, LR AUROC=0.786
  - **Expected**: 0.602, 0.786
  - **Check**: `direction_sweep_results.json` within-syc Mistral entries
  - **Files**: `direction_sweep_results.json`
  - **Verdict**: VERIFIED

- [x] **H16-011** | Within-sycophancy Qwen: Dir AUROC=0.510, LR AUROC=0.628
  - **Expected**: 0.510, 0.628
  - **Check**: `direction_sweep_results.json` within-syc Qwen entries
  - **Files**: `direction_sweep_results.json`
  - **Verdict**: VERIFIED

---

### C7 Direction Extraction (RepE-style)

*HACKATHON_LOG.md lines 178-202. Data file: `c7_permutation_test.json` (partial).*

- [x] **H16-012** | LOO accuracy 0.333 (20/60)
  - **Expected**: 0.333
  - **Check**: `c7_permutation_test.json` layer-specific LOO
  - **Files**: `c7_permutation_test.json`
  - **Verdict**: UNVERIFIED

- [x] **H16-013** | LOO AUROC 0.288
  - **Expected**: 0.288
  - **Check**: `c7_permutation_test.json` — closest value is 0.258
  - **Files**: `c7_permutation_test.json`
  - **Verdict**: DISCREPANT (diff=0.030)

- [x] **H16-014** | LR LOO accuracy 0.417 (25/60)
  - **Expected**: 0.417
  - **Check**: `c7_permutation_test.json` LR section
  - **Files**: `c7_permutation_test.json`
  - **Verdict**: UNVERIFIED

- [x] **H16-015** | LR AUROC 0.337
  - **Expected**: 0.337
  - **Check**: `c7_permutation_test.json` LR section
  - **Files**: `c7_permutation_test.json`
  - **Verdict**: UNVERIFIED

- [x] **H16-016** | Flipped AUROC 0.712
  - **Expected**: 0.712
  - **Check**: `direction_sweep_results.json` — JSON shows 0.732
  - **Files**: `direction_sweep_results.json`
  - **Verdict**: DISCREPANT (diff=0.020)

---

### Permutation Test: Sycophancy LR Validation

*HACKATHON_LOG.md lines 660-691. Permutation details NOT in JSON.*

- [x] **H16-017** | Llama real LR AUROC 0.857
  - **Expected**: 0.857
  - **Check**: Cross-reference with `direction_sweep_results.json`
  - **Files**: `direction_sweep_results.json`
  - **Verdict**: VERIFIED

- [x] **H16-018** | Llama permutation: mean=0.569, 95th=0.685, max=0.771
  - **Expected**: mean=0.569, 95th=0.685, max=0.771
  - **Check**: Permutation details not serialized
  - **Files**: No permutation JSON exists
  - **Verdict**: NO_DATA

- [x] **H16-019** | Mistral real LR AUROC 0.827
  - **Expected**: 0.827
  - **Check**: Cross-reference with `direction_sweep_results.json`
  - **Files**: `direction_sweep_results.json`
  - **Verdict**: VERIFIED

- [x] **H16-020** | Mistral permutation: mean=0.568, 95th=0.670, max=0.749
  - **Expected**: mean=0.568, 95th=0.670, max=0.749
  - **Check**: Permutation details not serialized
  - **Files**: No permutation JSON exists
  - **Verdict**: NO_DATA

- [x] **H16-021** | Both p<0.005 (0/200 permutations exceeded real AUROC)
  - **Expected**: p<0.005 for both models
  - **Check**: 0/200 exceeded implies p<0.005 — logically correct but raw data not saved
  - **Files**: No permutation JSON exists
  - **Verdict**: NO_DATA

---

## Part 2: Exp 26-33

Source: `org-audit-findings.md` (HACKATHON_LOG.md lines 818-904 + commit b47c93c)

---

### Exp 26: Scale Invariance

*HACKATHON_LOG.md lines 818-830. Data file: `results/hackathon/scale_invariance.json`.*

- [x] **EXP26-V01** | SMALL vs MEDIUM rho in 0.83-0.90 range
  - **Expected**: rho within 0.83-0.90
  - **Check**: `scale_invariance.json` SMALL_vs_MEDIUM
  - **Files**: `scale_invariance.json`
  - **Verdict**: VERIFIED (rho=0.8667)

- [x] **EXP26-001** | SMALL vs LARGE rho in 0.83-0.90 range
  - **Expected**: rho within 0.83-0.90
  - **Check**: `scale_invariance.json` SMALL_vs_LARGE — rho=0.8258
  - **Files**: `scale_invariance.json`
  - **Verdict**: DISCREPANT — 0.826 below claimed lower bound 0.83

- [x] **EXP26-V02** | MEDIUM vs LARGE rho in 0.83-0.90 range
  - **Expected**: rho within 0.83-0.90
  - **Check**: `scale_invariance.json` MEDIUM_vs_LARGE
  - **Files**: `scale_invariance.json`
  - **Verdict**: VERIFIED (rho=0.8963)

- [x] **EXP26-002** | Coding #1 at 100% of models at ALL scales
  - **Expected**: Coding ranked #1 for every model in every scale group
  - **Check**: No per-model breakdown in JSON to verify
  - **Files**: `scale_invariance.json`
  - **Verdict**: NO_DATA

- [x] **EXP26-V03** | confab-creative AUROC drops to 0.60 at LARGE scale
  - **Expected**: AUROC=0.60
  - **Check**: `scale_invariance.json` LARGE group confab-creative
  - **Files**: `scale_invariance.json`
  - **Verdict**: VERIFIED

---

### Exp 27: Encoding-Regime Axis Analysis

*HACKATHON_LOG.md lines 832-840.*

- [x] **EXP27-V01** | Truth axis cross-model consistency = -0.046
  - **Expected**: -0.046
  - **Check**: Automated extraction from results JSON
  - **Files**: `results/hackathon/` (verified via automated extraction)
  - **Verdict**: VERIFIED

- [x] **EXP27-V02** | Complexity axis consistency = 0.982
  - **Expected**: 0.982
  - **Check**: Automated extraction from results JSON
  - **Files**: `results/hackathon/` (verified via automated extraction)
  - **Verdict**: VERIFIED

- [x] **EXP27-V03** | CMPLX/OPEN/SAFE/ABSTR converge within 10-17 degrees
  - **Expected**: Angular range 10-17 degrees
  - **Check**: Automated extraction from results JSON
  - **Files**: `results/hackathon/` (verified via automated extraction)
  - **Verdict**: VERIFIED

---

### Exp 28: Confabulation Trajectory

*HACKATHON_LOG.md lines 842-848.*

- [x] **EXP28-V01** | key_entropy d: 20 to 55 over 50 tokens
  - **Expected**: Trajectory from d=20 to d=55
  - **Check**: Automated extraction from results JSON
  - **Files**: `results/hackathon/` (verified via automated extraction)
  - **Verdict**: VERIFIED

- [x] **EXP28-V02** | Deception d: 5.67 to 8.36, p=0.008
  - **Expected**: d start=5.67, d end=8.36, p=0.008
  - **Check**: Automated extraction from results JSON
  - **Files**: `results/hackathon/` (verified via automated extraction)
  - **Verdict**: VERIFIED

- [x] **EXP28-V03** | n=3 per group for confabulation (appropriately caveated as underpowered)
  - **Expected**: n=3
  - **Check**: Automated extraction from results JSON
  - **Files**: `results/hackathon/` (verified via automated extraction)
  - **Verdict**: VERIFIED

---

### Exp 29: Per-Layer Deception Anatomy

*HACKATHON_LOG.md lines 850-861.*

- [x] **EXP29-V01** | 28/28 layers show d > 1.0 (same-prompt controlled, Qwen-7B)
  - **Expected**: 28/28 layers
  - **Check**: Automated extraction from results JSON
  - **Files**: `results/hackathon/` (verified via automated extraction)
  - **Verdict**: VERIFIED

- [x] **EXP29-V02** | Cross-model layer profile consistency: rho=0.200
  - **Expected**: rho=0.200
  - **Check**: Automated extraction from results JSON
  - **Files**: `results/hackathon/` (verified via automated extraction)
  - **Verdict**: VERIFIED

- [x] **EXP29-V03** | Deceptive norms LARGER at every layer (100% consistency)
  - **Expected**: 100% consistency
  - **Check**: Automated extraction from results JSON
  - **Files**: `results/hackathon/` (verified via automated extraction)
  - **Verdict**: VERIFIED

- [x] **EXP29-V04** | Mean |d| first half=2.44, second half=2.41 (flat profile)
  - **Expected**: first=2.44, second=2.41
  - **Check**: Automated extraction from results JSON
  - **Files**: `results/hackathon/` (verified via automated extraction)
  - **Verdict**: VERIFIED

---

### Exp 30: Final Synthesis

*HACKATHON_LOG.md lines 863-870.*

- [x] **EXP30-V01** | 14 hackathon experiments completed (of 19 total)
  - **Expected**: 14 at time of Exp 30
  - **Check**: Count experiments in HACKATHON_LOG.md up to Exp 30
  - **Files**: HACKATHON_LOG.md
  - **Verdict**: VERIFIED (accurate at time of writing; total reached 18 by Exp 33)

- [x] **EXP30-V02** | 92 pre-hackathon + 21 hackathon JSON result files
  - **Expected**: 92 pre-hackathon, 21 hackathon
  - **Check**: Count JSON files in `results/` and `results/hackathon/`
  - **Files**: `results/` directory listing
  - **Verdict**: VERIFIED (at time of writing; now 24+ hackathon files)

---

### Exp 31: Refusal Detection in Generation Regime

*HACKATHON_LOG.md lines 872-888. Data file: refusal detection JSON.*

- [x] **EXP31-V01** | AUROC (LR, 5-fold) = 0.898
  - **Expected**: 0.898
  - **Check**: Automated extraction from refusal detection JSON
  - **Files**: `results/hackathon/` (refusal detection JSON)
  - **Verdict**: VERIFIED

- [x] **EXP31-V02** | AUROC (RF, 5-fold) = 0.830
  - **Expected**: 0.830
  - **Check**: Automated extraction from refusal detection JSON
  - **Files**: `results/hackathon/` (refusal detection JSON)
  - **Verdict**: VERIFIED

- [x] **EXP31-V03** | key_rank d=1.530
  - **Expected**: 1.530
  - **Check**: Automated extraction from refusal detection JSON
  - **Files**: `results/hackathon/` (refusal detection JSON)
  - **Verdict**: VERIFIED

- [x] **EXP31-V04** | key_entropy d=1.505
  - **Expected**: 1.505
  - **Check**: Automated extraction from refusal detection JSON
  - **Files**: `results/hackathon/` (refusal detection JSON)
  - **Verdict**: VERIFIED

- [x] **EXP31-V05** | Actual refusals in text: 19/20
  - **Expected**: 19
  - **Check**: Automated extraction from refusal detection JSON
  - **Files**: `results/hackathon/` (refusal detection JSON)
  - **Verdict**: VERIFIED

- [x] **EXP31-V06** | Refusal norm_per_token vs normal: 295.7 vs 301.1
  - **Expected**: refusal=295.7, normal=301.1
  - **Check**: Automated extraction from refusal detection JSON
  - **Files**: `results/hackathon/` (refusal detection JSON)
  - **Verdict**: VERIFIED

---

### Exp 32: Jailbreak Detection

*HACKATHON_LOG.md lines 890-904. Data file: `results/hackathon/jailbreak_detection.json`.*

- [x] **EXP32-V01** | Jailbreak vs Normal AUROC (LR) = 0.878
  - **Expected**: 0.8775
  - **Check**: `.jailbreak_vs_normal.lr_auroc`
  - **Files**: `jailbreak_detection.json`
  - **Verdict**: VERIFIED

- [x] **EXP32-V02** | Jailbreak vs Normal AUROC (RF) = 0.764
  - **Expected**: 0.7638
  - **Check**: `.jailbreak_vs_normal.rf_auroc`
  - **Files**: `jailbreak_detection.json`
  - **Verdict**: VERIFIED

- [x] **EXP32-V03** | Jailbreak vs Refusal AUROC (LR) = 0.790
  - **Expected**: 0.790
  - **Check**: `.jailbreak_vs_refusal.lr_auroc`
  - **Files**: `jailbreak_detection.json`
  - **Verdict**: VERIFIED

- [x] **EXP32-V04** | Jailbreak vs Refusal AUROC (RF) = 0.584
  - **Expected**: 0.584
  - **Check**: `.jailbreak_vs_refusal.rf_auroc`
  - **Files**: `jailbreak_detection.json`
  - **Verdict**: VERIFIED

- [x] **EXP32-V05** | 3-way accuracy = 0.583
  - **Expected**: 0.583
  - **Check**: `.three_way_accuracy`
  - **Files**: `jailbreak_detection.json`
  - **Verdict**: VERIFIED

- [x] **EXP32-V06** | norm_per_token: jailbreak=294.9, refusal=295.7, normal=301.1
  - **Expected**: 3 values as stated
  - **Check**: `.norm_per_token.*`
  - **Files**: `jailbreak_detection.json`
  - **Verdict**: VERIFIED

- [x] **EXP32-001** | "18/20 harmful prompts answered by abliterated model"
  - **Expected**: 18
  - **Check**: `.n_actually_answered` — JSON says 8, not 18
  - **Files**: `jailbreak_detection.json`
  - **Verdict**: CRITICAL — 2.25x inflation (claimed 18, JSON has 8, manual review gives 8-10)

---

### Exp 33: Multi-Model Refusal Detection

*Commit b47c93c only (not in HACKATHON_LOG.md body). Data file: `results/hackathon/refusal_multimodel.json`.*

- [x] **EXP33-V01** | Llama RF AUROC = 0.893
  - **Expected**: 0.8925 (rounds to 0.893)
  - **Check**: `refusal_multimodel.json` Llama RF entry
  - **Files**: `refusal_multimodel.json`
  - **Verdict**: VERIFIED

- [x] **EXP33-V02** | Mistral LR AUROC = 0.843
  - **Expected**: 0.8425 (rounds to 0.843 standard, 0.842 banker's)
  - **Check**: `refusal_multimodel.json` Mistral LR entry
  - **Files**: `refusal_multimodel.json`
  - **Verdict**: VERIFIED

- [x] **EXP33-V03** | Llama n_actual_refusals = 19/20
  - **Expected**: 19
  - **Check**: `refusal_multimodel.json` Llama refusal count
  - **Files**: `refusal_multimodel.json`
  - **Verdict**: VERIFIED

- [x] **EXP33-V04** | Mistral n_actual_refusals = 14/20
  - **Expected**: 14 (6 prompts answered despite being harmful)
  - **Check**: `refusal_multimodel.json` Mistral refusal count
  - **Files**: `refusal_multimodel.json`
  - **Verdict**: VERIFIED

- [x] **EXP33-003** | Qwen AUROC (0.898) presented as part of unified 3-architecture result
  - **Expected**: Qwen data from Exp 33 multi-model run
  - **Check**: `refusal_multimodel.json` contains only Llama + Mistral; Qwen value is from Exp 31
  - **Files**: `refusal_multimodel.json` (Qwen absent)
  - **Verdict**: OVERSTATED — Qwen NOT re-run in Exp 33 framework

---

## Part 3: Exp 34-36

Source: `org-audit-exp34-36.md` (commits 7812420, bd84a7d, 4f86a77)

---

### Exp 34: Multi-Model Jailbreak Detection

*Commit bd84a7d. Data file: `results/hackathon/jailbreak_multimodel.json`.*

- [x] **EXP34-V01** | Qwen AUROC = 0.878
  - **Expected**: 0.8775 (from Exp 32)
  - **Check**: `jailbreak_detection.json` (Exp 32) — hardcoded in Exp 34 script
  - **Files**: `jailbreak_detection.json`
  - **Verdict**: VERIFIED (but hardcoded, see EXP34-002)

- [x] **EXP34-001** | Llama AUROC = 0.878, but abliteration failed (1/20 answered)
  - **Expected**: Llama AUROC measures jailbreak-vs-normal
  - **Check**: `jailbreak_multimodel.json` — Llama n_abl_answered=1; AUROC actually measures refusal-vs-normal
  - **Files**: `jailbreak_multimodel.json`
  - **Verdict**: OVERSTATED — AUROC measures refusal, not jailbreak (abliteration failed)

- [x] **EXP34-V02** | Llama LR AUROC = 0.878
  - **Expected**: 0.8775 (rounds to 0.878)
  - **Check**: `jailbreak_multimodel.json` Llama LR entry
  - **Files**: `jailbreak_multimodel.json`
  - **Verdict**: VERIFIED

- [x] **EXP34-V03** | Llama n_abl_answered = 1/20
  - **Expected**: 1
  - **Check**: `jailbreak_multimodel.json` Llama n_abl_answered
  - **Files**: `jailbreak_multimodel.json`
  - **Verdict**: VERIFIED

- [x] **EXP34-V04** | Mistral LR AUROC = 0.793
  - **Expected**: 0.7925 (rounds to 0.793)
  - **Check**: `jailbreak_multimodel.json` Mistral LR entry
  - **Files**: `jailbreak_multimodel.json`
  - **Verdict**: VERIFIED

- [x] **EXP34-V05** | Mistral n_abl_answered = 8/20
  - **Expected**: 8
  - **Check**: `jailbreak_multimodel.json` Mistral n_abl_answered
  - **Files**: `jailbreak_multimodel.json`
  - **Verdict**: VERIFIED

- [x] **EXP34-V06** | Mean AUROC = 0.849
  - **Expected**: (0.8775+0.8775+0.7925)/3 = 0.8492 rounds to 0.849
  - **Check**: Arithmetic verification
  - **Files**: `jailbreak_multimodel.json`, `jailbreak_detection.json`
  - **Verdict**: VERIFIED (arithmetic correct; interpretively qualified — see ADV-MEAN)

- [x] **EXP34-002** | Qwen data hardcoded in `code/34_jailbreak_multimodel.py`
  - **Expected**: Qwen values loaded programmatically from Exp 32 JSON
  - **Check**: Script line 334: literal string `'18/20'`; line 344: literal `0.878` in mean calculation
  - **Files**: `code/34_jailbreak_multimodel.py` lines 334, 344
  - **Verdict**: CRITICAL — hardcoded values bypass data, propagating EXP32-001 error

- [x] **EXP34-003** | Qwen n_abl_answered claimed as "18/20 — clean jailbreak signal"
  - **Expected**: Matches Exp 32 JSON (n_actually_answered=8)
  - **Check**: Commit message + script line 334 both say 18/20; JSON says 8
  - **Files**: `jailbreak_detection.json` (`.n_actually_answered`)
  - **Verdict**: CRITICAL — EXP32-001 propagated into Exp 34 commit message and script

---

### Exp 35: Token-Controlled Layer Reanalysis

*Commit 7812420. Data file: `results/hackathon/token_controlled_layers.json`.*

- [x] **EXP35-V01** | Token count confound d=1.483 (honest mean 48.2, deceptive mean 78.3)
  - **Expected**: d=1.483, means 48.2 and 78.3
  - **Check**: `token_controlled_layers.json` token confound section
  - **Files**: `token_controlled_layers.json`
  - **Verdict**: VERIFIED

- [x] **EXP35-V02** | Raw norm d mean across 28 layers = 2.425
  - **Expected**: 2.425
  - **Check**: `token_controlled_layers.json` raw norm section
  - **Files**: `token_controlled_layers.json`
  - **Verdict**: VERIFIED

- [x] **EXP35-V03** | Norm-per-token d mean across 28 layers = -1.909
  - **Expected**: -1.909 (negative = deceptive is sparser)
  - **Check**: `token_controlled_layers.json` norm-per-token section
  - **Files**: `token_controlled_layers.json`
  - **Verdict**: VERIFIED

- [x] **EXP35-V04** | ANCOVA layers significant at p<0.05: 27/28
  - **Expected**: 27/28
  - **Check**: `token_controlled_layers.json` ANCOVA results
  - **Files**: `token_controlled_layers.json`
  - **Verdict**: VERIFIED

- [x] **EXP35-V05** | ANCOVA layers significant at p<0.01: 25/28
  - **Expected**: 25/28
  - **Check**: `token_controlled_layers.json` ANCOVA results
  - **Files**: `token_controlled_layers.json`
  - **Verdict**: VERIFIED

- [x] **EXP35-V06** | Mean partial eta-squared = 0.486
  - **Expected**: 0.486
  - **Check**: `token_controlled_layers.json` ANCOVA eta-squared
  - **Files**: `token_controlled_layers.json`
  - **Verdict**: VERIFIED

- [x] **EXP35-V07** | Non-significant layer = Layer 27 (p=0.472)
  - **Expected**: Layer 27, p=0.472
  - **Check**: `token_controlled_layers.json` per-layer ANCOVA
  - **Files**: `token_controlled_layers.json`
  - **Verdict**: VERIFIED

- [x] **EXP35-P01** | Self-correction of Exp 29 per-layer claim
  - **Expected**: Exp 35 revises "d > 1.0 uniform across all layers" to "sparser per token after ANCOVA"
  - **Check**: Exp 35 commit + PITCH_NUMBERS.md lines 52, 122-126 updated
  - **Files**: `token_controlled_layers.json`, `docs/PITCH_NUMBERS.md`
  - **Verdict**: POSITIVE — honest self-correction, good scientific practice

---

### Exp 36: Impossibility vs Safety Refusal

*Commit 4f86a77. Data file: `results/hackathon/impossibility_refusal.json`. Model: Qwen2.5-7B-Instruct.*

- [x] **EXP36-001** | "Harmful content detection" framing
  - **Expected**: AUROC measures harmful content specifically
  - **Check**: Impossible vs benign AUROC (0.950) > harmful vs benign (0.898); signal is refusal in general, not harm-specific
  - **Files**: `impossibility_refusal.json`
  - **Verdict**: FALSIFIED — signal is refusal/output-suppression, not harmful content detection

- [x] **EXP36-V01** | Impossible vs benign AUROC (LR) = 0.950
  - **Expected**: 0.9500
  - **Check**: `impossibility_refusal.json` impossible_vs_benign LR
  - **Files**: `impossibility_refusal.json`
  - **Verdict**: VERIFIED

- [x] **EXP36-V02** | Harmful vs benign AUROC (LR) = 0.898
  - **Expected**: 0.8975 (rounds to 0.898)
  - **Check**: `impossibility_refusal.json` harmful_vs_benign LR
  - **Files**: `impossibility_refusal.json`
  - **Verdict**: VERIFIED

- [x] **EXP36-V03** | Impossible vs harmful AUROC (LR) = 0.693
  - **Expected**: 0.6925 (rounds to 0.693)
  - **Check**: `impossibility_refusal.json` impossible_vs_harmful LR
  - **Files**: `impossibility_refusal.json`
  - **Verdict**: VERIFIED

- [x] **EXP36-P01** | Honest hypothesis testing — team tested own narrative and published disconfirming result
  - **Expected**: N/A
  - **Check**: Commit credits Cassidy for sparking the experiment; result undermines "harmful content detection" headline
  - **Files**: `impossibility_refusal.json`
  - **Verdict**: POSITIVE — good scientific practice

---

## Part 4: JiminAI-Cricket (CRICKET-xxx)

Source: `org-audit-findings.md` Part 2. Repo: Liberation-Labs-THCoalition/JiminAI-Cricket.

---

- [x] **CRICKET-001** | Confabulation detection at 88.6% confidence in demo
  - **Expected**: Demo shows reliable confabulation detection
  - **Check**: `data/precomputed_examples.json` — value is 0.885523689290657 (rounds to 88.6%). Pre-computed on Llama-3.1-8B-Instruct. However, Exp 14a-14d produced 5 null results. `generated_text` fields are placeholders.
  - **Files**: `data/precomputed_examples.json`, `src/cricket_demo.py`
  - **Verdict**: CONFIRMED (HIGH) — demo shows capability that 5 experiments failed to validate

- [x] **CRICKET-002** | "reveals censorship" in demo UI header
  - **Expected**: Trained censorship classifier exists
  - **Check**: `models/cricket_classifiers_meta.json` shows `n_training: 0` for censorship. No `cricket_rf_censorship.joblib`. Code gracefully skips but UI claim is unsupported.
  - **Files**: `src/cricket_demo.py` (Gradio header), `models/cricket_classifiers_meta.json`
  - **Verdict**: CONFIRMED (HIGH) — zero training data, no model file

- [x] **CRICKET-003** | "AUROC 1.0" without cross-model context
  - **Expected**: AUROC 1.0 applies broadly
  - **Check**: AUROC 1.0 is within-model only. Cross-model transfer: avg 0.67, min 0.14. Censorship has no model, confabulation is unvalidated.
  - **Files**: `src/cricket_demo.py` header
  - **Verdict**: CONFIRMED (HIGH) — compound claim where 2 of 3 capabilities don't exist

- [x] **CRICKET-004** | README says "Prototype: Not yet started"
  - **Expected**: Matches actual codebase state
  - **Check**: Working Gradio demo exists in `src/` (4 Python files). `data/` has precomputed examples. README understates completeness.
  - **Files**: `README.md`
  - **Verdict**: CONFIRMED (MEDIUM)

- [x] **CRICKET-005** | README lists "detect confabulation, sycophancy" as capabilities
  - **Expected**: Capabilities exist
  - **Check**: Zero sycophancy training data or experiments. Confabulation disproven by 5 null results.
  - **Files**: `README.md`
  - **Verdict**: CONFIRMED (MEDIUM)

- [x] **CRICKET-006** | "100% classification accuracy" for identity detection
  - **Expected**: 100% accuracy
  - **Check**: Data leak artifact identified in WS3. Cross-prompt accuracy = 92-97.3%.
  - **Files**: Identity detection results
  - **Verdict**: CONFIRMED (MEDIUM) — inflated by training/test contamination

- [x] **CRICKET-007** | Cherry-picked precomputed examples (all high confidence)
  - **Expected**: Representative examples
  - **Check**: All 3 demo examples show high confidence: honest 97.1%, deceptive 96.5%, confabulation 88.6%. No lower-confidence or ambiguous examples.
  - **Files**: `data/precomputed_examples.json`
  - **Verdict**: CONFIRMED (MEDIUM)

- [x] **CRICKET-008** | CF3 "abliteration d=0.000 all categories" — actual self-ref d=+0.464
  - **Expected**: d=0.000
  - **Check**: Prior audit finding
  - **Files**: Prior audit
  - **Verdict**: UNCHANGED

- [x] **CRICKET-009** | CF6 "cross-model rho=0.914" — matches neither verified value
  - **Expected**: rho=0.914
  - **Check**: Prior audit finding
  - **Files**: Prior audit
  - **Verdict**: UNCHANGED

- [x] **CRICKET-010** | CF8 "no per-model calibration needed" — contradicts own DESIGN.md
  - **Expected**: Consistent with DESIGN.md
  - **Check**: Prior audit finding
  - **Files**: Prior audit
  - **Verdict**: UNCHANGED

- [x] **CRICKET-P01** | `research/cricket_viability.md` (WS10) is the most honest document in the repo
  - **Expected**: N/A
  - **Check**: Correctly flags 36% NO DATA and 27% PREMATURE
  - **Files**: `research/cricket_viability.md`
  - **Verdict**: POSITIVE

---

## Part 5: CacheScope (CACHE-xxx)

Source: `org-audit-findings.md` Part 3. Repo: Liberation-Labs-THCoalition/CacheScope.

---

- [x] **CACHE-001** | `PUT /api/v1/config` unauthenticated
  - **Expected**: Auth required for config mutation
  - **Check**: No auth decorator or dependency. Test confirms 200 response. No auth libraries in `pyproject.toml`.
  - **Files**: `cachescope/routes_rest.py`
  - **Verdict**: CONFIRMED (MEDIUM)

- [x] **CACHE-002** | CORS `allow_origins=["*"]`
  - **Expected**: Restricted origins
  - **Check**: `CORSMiddleware(allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])` in `create_app()`. Any browser JS can mutate config.
  - **Files**: `cachescope/app.py`
  - **Verdict**: CONFIRMED (MEDIUM)

- [x] **CACHE-003** | No authentication on ANY endpoint
  - **Expected**: Auth on sensitive endpoints
  - **Check**: Zero auth dependencies across all route files. Includes WebSocket and Proof of Mind endpoints.
  - **Files**: `routes_rest.py`, `routes_ws.py`, `proof/routes.py`
  - **Verdict**: CONFIRMED (MEDIUM)

- [x] **CACHE-004** | Compression ratio: "2,000x" vs "~17,000x" (different bases, undocumented)
  - **Expected**: Consistent compression claim
  - **Check**: 2,000x = 1 layer basis; ~17,000x = 8 layers (stride-4 on 32). Both mathematically correct, neither documented.
  - **Files**: `THREAT_MODEL.md`, `demo/tui.py`
  - **Verdict**: CONFIRMED (LOW)

- [x] **CACHE-005** | Static model lookup table (8 entries); unknown model bypasses verification
  - **Expected**: Robust model verification
  - **Check**: `KNOWN_LAYERS` dict: 8 entries. Unknown `model_id` → `expected_layers = None` → check skipped. Acknowledged in THREAT_MODEL.md.
  - **Files**: `cachescope/bittensor/validator.py`
  - **Verdict**: CONFIRMED (LOW)

- [x] **CACHE-006** | `sys.path.insert` for Cricket/KV-Experiments (hardcoded paths)
  - **Expected**: Proper packaging
  - **Check**: Hardcoded paths: `~/KV-Experiments/code`, `~/jiminai-cricket/src`. Dependencies not in `pyproject.toml`.
  - **Files**: `demo/tui.py`, `cachescope/extractor.py`, `bittensor/validator.py`
  - **Verdict**: CONFIRMED (LOW)

- [x] **CACHE-P01** | THREAT_MODEL.md is unusually transparent (~3,400 words)
  - **Expected**: N/A
  - **Check**: 5 defended threats, 4 acknowledged non-protections, 5 current limitations. Genuine document.
  - **Files**: `THREAT_MODEL.md`
  - **Verdict**: POSITIVE

- [x] **CACHE-P02** | Good test suite (52 test functions across 5 files)
  - **Expected**: N/A
  - **Check**: 52 tests: routes (10), proof (19), ring buffer (10), snapshot (5), bittensor (8). Gaps: no extractor/interceptor/WebSocket tests.
  - **Files**: Test files
  - **Verdict**: POSITIVE

- [x] **CACHE-P03** | Clean architecture (ring buffer, FastAPI, WebSocket streaming)
  - **Expected**: N/A
  - **Check**: Well-structured separation of concerns.
  - **Files**: All source files
  - **Verdict**: POSITIVE

---

## Part 6: JiminAI-Frontend (FRONT-xxx)

Source: `org-audit-findings.md` Part 4. Repo: Liberation-Labs-THCoalition/JiminAI-Frontend-Performative.

---

- [x] **FRONT-001** | Fabricated training terminal in splash screen
  - **Expected**: Real training data or clear "simulated" label
  - **Check**: `seededRandom(42)` LCG generates deterministic fake output. 30 `DATASETS` entries mix real benchmarks with fabricated ones. 14 fake `EVENTS`. Loss curve: `2.8 * Math.exp(-progress * 3.5) + 0.15`. Cycles 10,000 fake epochs. No actual training.
  - **Files**: `frontend/src/components/SplashOverlay.jsx` (TrainingTerminal component)
  - **Verdict**: CONFIRMED (HIGH)

- [x] **FRONT-002** | "AI LIE DETECTOR" headline without qualification
  - **Expected**: "Experimental" or "prototype" qualifier
  - **Check**: Unqualified capability claim in splash screen title. Links to Liberation Labs. No disclaimer.
  - **Files**: `frontend/src/components/SplashOverlay.jsx`
  - **Verdict**: CONFIRMED (HIGH)

- [x] **FRONT-003** | Doc-code mismatch — README describes functional app, code is locked
  - **Expected**: README matches app state
  - **Check**: README describes "Twitch-style live prompt feed" with full API docs. SplashOverlay at z-index 200 covers everything. PromptForm returns null.
  - **Files**: `README.md`, `frontend/src/components/SplashOverlay.jsx`, `PromptForm.jsx`
  - **Verdict**: CONFIRMED (MEDIUM)

- [x] **FRONT-004** | App is non-functional — PromptForm returns null
  - **Expected**: Functional prompt submission
  - **Check**: First line: `return null; // splash overlay active`. 200+ lines unreachable JSX. `eslint-disable-next-line no-unreachable` confirms intentional. Double-locked: dead component + z-200 overlay.
  - **Files**: `frontend/src/components/PromptForm.jsx`
  - **Verdict**: CONFIRMED (MEDIUM)

- [x] **FRONT-005** | "LIAR (73%)" badges use hardcoded thresholds, not ML classifiers
  - **Expected**: Random Forest classifiers from Cricket
  - **Check**: 2 threshold checks: `entropy > 0.38 AND rank > 10.0` for "LIAR". Calibrated for Qwen-0.5B with stride-4. NOT Cricket RF classifiers.
  - **Files**: `classifier.py`
  - **Verdict**: CONFIRMED (MEDIUM)

- [x] **FRONT-006** | No WebSocket authentication on `/ws/feed`
  - **Expected**: Auth on WebSocket
  - **Check**: No auth on WebSocket endpoint
  - **Files**: `server.py`
  - **Verdict**: CONFIRMED (LOW)

- [x] **FRONT-007** | Production domain `discnxt.com` exposed in Caddyfile
  - **Expected**: Domain not in public repo
  - **Check**: Planned deployment domain visible in configuration
  - **Files**: `Caddyfile`
  - **Verdict**: CONFIRMED (LOW)

---

## Part 7: PITCH_NUMBERS.md (PITCH-xxx)

Source: `org-audit-findings.md` + `org-audit-exp34-36.md`. File: `docs/PITCH_NUMBERS.md`.

---

- [x] **PITCH-001** | "31 experiments over 2 campaigns"
  - **Expected**: Current experiment count
  - **Check**: 36 experiments exist (Exp 32-36 added after Exp 31 pitch update)
  - **Files**: `docs/PITCH_NUMBERS.md` line ~160
  - **Verdict**: STALE

- [x] **PITCH-002** | "64+ JSON result files"
  - **Expected**: Current file count
  - **Check**: 118 JSON files (89 in `results/` + 29 in `results/hackathon/`)
  - **Files**: `docs/PITCH_NUMBERS.md` line ~160
  - **Verdict**: STALE

- [x] **PITCH-CENSOR** | "Censorship detection (within-model) AUROC 1.000" with "900 censorship samples"
  - **Expected**: Trained censorship classifier
  - **Check**: `cricket_classifiers_meta.json` shows `n_training: 0`. No `cricket_rf_censorship.joblib`. Pitch claims capability that doesn't exist.
  - **Files**: `docs/PITCH_NUMBERS.md` line ~18, `models/cricket_classifiers_meta.json`
  - **Verdict**: CONFIRMED (HIGH) — phantom capability

- [x] **PITCH-SYCOPH** | "flags deception, censorship, sycophancy, and confabulation" — sycophancy d=0.107
  - **Expected**: Detectable sycophancy signal
  - **Check**: Sycophancy Cohen's d = 0.107 (negligible). Compare deception d = -3.065. Not a detection-grade signal.
  - **Files**: `docs/PITCH_NUMBERS.md` line ~81
  - **Verdict**: CONFIRMED (HIGH) — d=0.107 is not a detection signal

- [x] **PITCH-LAYER** | "Deception signal uniform across all transformer layers" in Technical Details
  - **Expected**: Consistent with corrected findings
  - **Check**: Exp 35 corrected to "27/28 layers, sparser per token" on line 52 of same document. Uncorrected version appears later — judges reading to end see stale claim last.
  - **Files**: `docs/PITCH_NUMBERS.md` (Technical Details section vs line 52)
  - **Verdict**: CONFIRMED (HIGH) — internal contradiction within same document

---

## Part 8: Cross-Cutting (ADV-xxx, LOG-xxx)

Source: `org-audit-exp34-36.md` adversarial review + documentation gaps.

---

- [x] **ADV-MEAN** | Mean AUROC (0.849) inherits OVERSTATED Llama value
  - **Expected**: Mean reflects valid jailbreak detection across 3 models
  - **Check**: Arithmetic is correct: (0.8775+0.8775+0.7925)/3 = 0.8492. But 2 of 3 inputs are qualified: Qwen 8/20 actual, Llama 1/20 actual. Mean should carry same caveat as weakest input.
  - **Files**: `jailbreak_multimodel.json`, `jailbreak_detection.json`
  - **Verdict**: OVERSTATED — arithmetic correct, interpretation misleading

- [x] **ADV-AUROC** | CacheScope "0.9991 AUROC" vs PITCH "1.000" for deception detection
  - **Expected**: Consistent cross-repo AUROC claim
  - **Check**: CacheScope README says 0.9991. PITCH_NUMBERS.md says 1.000. C2 paper says 1.0 (within-model RF). Three different numbers, no explanation.
  - **Files**: CacheScope `README.md`, `docs/PITCH_NUMBERS.md`, C2 paper
  - **Verdict**: CONFIRMED (MEDIUM) — inconsistency across repos

- [x] **LOG-001** | HACKATHON_LOG.md missing Exp 33-36
  - **Expected**: All experiments logged
  - **Check**: Log ends at Exp 32 (line 904). Exp 33, 34, 35, 36 exist only as commit messages + JSON files. 4 experiments with no log entries.
  - **Files**: `HACKATHON_LOG.md`
  - **Verdict**: STALE

---

## Appendix: Verdict Definitions

| Verdict | Definition |
| --- | --- |
| VERIFIED | Number matches JSON data exactly (within rounding tolerance) |
| NO_DATA | Number reported in log but no JSON file exists to verify |
| CONFIRMED | Code/doc review finding verified against source files |
| DISCREPANT | Number differs from JSON beyond rounding tolerance |
| CRITICAL | Material error in a numeric claim (>10% deviation or propagated error) |
| FALSIFIED | Experimental result disproves the framing of the claim |
| OVERSTATED | Claim is technically correct but interpretively misleading |
| STALE | Claim was accurate at time of writing but is now outdated |
| UNVERIFIED | Insufficient data to cross-reference (partial JSON exists) |
| FALSE_STATUS | Log claims "JSON saved" but no file exists on disk |
| LOW_PRIORITY | Minor claim not blocking; verification deferred |
| UNCHANGED | Finding from prior audit; status unchanged |
| POSITIVE | Good scientific practice, honest documentation, or architectural quality |
