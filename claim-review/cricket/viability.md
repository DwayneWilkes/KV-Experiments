# WS10: Cricket Viability Assessment

**Status**: COMPLETE
**Date**: 2026-03-04
**Auditor**: Claude (independent analysis)
**Sources**: JiminAI-Cricket repo (5 files), Campaign 2 paper (verified by WS1-WS8), CLAIMS.md

---

## 10.1 Repo Reality Check

### File Inventory (exact)

| File | Path | Content |
|------|------|---------|
| `.gitignore` | `/` | Python, IDE, data, secrets (39 lines) |
| `README.md` | `/` | Product pitch, 4 core findings, status |
| `DESIGN.md` | `/docs/` | Architecture, metrics targets, competitive positioning |
| `CAMPAIGN_2_CRICKET.md` | `/docs/` | 10 planned experiments (C1-C10), none executed |
| `CAMPAIGN_2_FINDINGS.md` | `/research/` | Cricket interpretation of C2 data (contains stale numbers) |
| `COMPETITIVE_LANDSCAPE.md` | `/research/` | 5 categories of adjacent work |

**Total non-git files**: 6 (5 markdown + 1 .gitignore)

- **Code files**: 0
- **Test files**: 0
- **CI/CD**: none
- **Prototype**: none
- **Data**: none
- **Directories declared but empty**: `src/cricket/` (mentioned in README, does not exist), `experiments/` (mentioned in README, does not exist)

**Verdict**: Cricket is a documentation-only product. Zero executable artifacts.

---

## 10.2 Full Claims Extraction

All 37 claims verified against CLAIMS.md (Part 2). CLAIMS.md already contains complete CC1-CC22, CF1-CF8, CL1-CL7 with exact quotes, sources, and evidence mapping. No supplements needed -- the registry is comprehensive.

### Capability Claims (CC1-CC22) -- Source Documents

| ID | Claim (exact quote) | Source File |
|----|-------------------|------------|
| CC1 | "reads the geometric structure of a language model's KV-cache to detect anomalous cognitive states -- deception, confabulation, sycophancy, refusal failure" | README.md |
| CC2 | "detection -- and intervention -- can happen before the user sees anything" | README.md |
| CC3 | "Cricket does not modify the model. It does not steer activations. It does not require fine-tuning." | README.md |
| CC4 | "It watches the geometry of what the model is already doing" [no fine-tuning] | README.md |
| CC5 | Multi-category: "deception, confabulation, sycophancy, refusal failure" (+ censorship, persona drift, alignment) | README + DESIGN |
| CC6 | "sub-50ms latency" / "Classification latency < 50ms" | README + DESIGN Sec 5 |
| CC7 | "Deception detection AUROC >= 0.95" / "Match Apollo Research benchmark" | DESIGN Sec 5 |
| CC8 | "Refusal detection AUROC >= 0.99" / "Our strongest signal (d=0.85-2.17)" | DESIGN Sec 5 |
| CC9 | "Encoding-only detection AUROC >= 0.90" / "Must work before generation" | DESIGN Sec 5 |
| CC10 | "Cross-model transfer accuracy >= 0.80" / "Scalable product requirement" | DESIGN Sec 5 |
| CC11 | "Cricket can monitor [persona] in real-time during inference to detect: Persona drift, Jailbreak-induced persona switches, Fine-tuning tampering" | FINDINGS |
| CC12 | "Cricket can detect: State-sponsored censorship in deployed models (even when outputs look normal)" | FINDINGS |
| CC13 | "Cricket can differentiate between: Aligned model (cage intact), Jailbroken model (cage bypassed), Deeply corrupted model (compass damaged)" | FINDINGS |
| CC14 | "Cricket should monitor the gap between geometric prediction and behavioral output. Alignment = small gap. Jailbreak = large gap." | FINDINGS |
| CC15 | "Confabulation detection AUROC >= 0.85" / "Lower bar -- smaller effect sizes" | DESIGN Sec 5 |
| CC16 | "False positive rate <= 1%" / "Apollo's operating point" | DESIGN Sec 5 |
| CC17 | "SVD extraction latency < 20ms per layer" | DESIGN Sec 5 |
| CC18 | "Logistic regression on geometric features" sufficient / "geometry does the heavy lifting" | DESIGN Sec 3.3 |
| CC19 | "Computationally cheap to extract (one SVD per monitored layer per forward pass)" | DESIGN Sec 2 |
| CC20 | "Calibration: Per-model, per-scale. Our research shows non-monotonic effects across scales." | DESIGN Sec 3.3 |
| CC21 | "If corrective instructions reliably shift geometry toward normal, Cricket Level 2 intervention is validated" | CRICKET Exp C6 |
| CC22 | "There is no widely deployed system that monitors the model's internal cognitive state during inference" | DESIGN Sec 1 |

### Cross-Document Discrepancies (CF1-CF8) -- verified in 10.7 below.

### Competitive Landscape Claims (CL1-CL7) -- verified in 10.9 below.

---

## 10.3 Capability Claims -> C2 Evidence Mapping

| CC | Supporting C2 Finding | C2 Paper Claim | Verified By | Effect Size | Sample | Architecture Coverage |
|----|---------------------|---------------|-------------|-------------|--------|---------------------|
| CC1 | Category hierarchy | C23-C27 | WS1 | W=0.756, rho=0.739 | 15 models | 5 families |
| CC2 | Input-only defense | C28-C29 | WS1 reference | rho=0.821 | 8 models | 4 families |
| CC3 | Architectural (read-only) | n/a | By design | n/a | n/a | n/a |
| CC4 | Cross-arch ordering | C23-C27 | WS1 | rho=0.739 | 15 models | 5 families |
| CC5 | 13 categories in scale sweep | C23 | WS1 | varies | 15 models | 5 families |
| CC6 | **NO DATA** | none | -- | none | none | none |
| CC7 | Deception forensics | C36-C40 | WS4 | |g|=0.647-3.039 | 7 models | 4 families |
| CC8 | Scale sweep refusal | README claim | -- | d=0.85-2.17 | 15 models | 5 families |
| CC9 | Input-only rho | C28 | -- | rho=0.821 | 8 models | 4 families |
| CC10 | **NO TRANSFER TESTED** | none | -- | none | none | none |
| CC11 | Identity signatures | C30-C31 | WS3 | d=3.41-6.73, 92-97.3% xp | 7 models | 4 families |
| CC12 | S4 censorship | C49 | WS6 | d=+0.766 (Qwen-14B) | 1 model | 1 family |
| CC13 | Abliteration | C57-C64 | WS7 | d=+0.464 self-ref | 1 model pair | 1 family |
| CC14 | Behavior-geometry gap | C63 (theoretical) | WS7 | theoretical | 0 models | 0 |
| CC15 | Confabulation as category | C23 (category exists) | WS1 | not measured | -- | -- |
| CC16 | **NO DATA** | none | -- | none | none | none |
| CC17 | **NO DATA** | none | -- | none | none | none |
| CC18 | External citation (AxBench) | n/a | -- | n/a | n/a | n/a |
| CC19 | **NO BENCHMARK** | none | -- | none | none | none |
| CC20 | Deception direction varies | C40-C41 | WS4 | direction varies | 7 models | 4 families |
| CC21 | Exp C6 **NOT EXECUTED** | none | -- | none | none | none |
| CC22 | Competitive assertion | CL1-CL7 | 10.9 below | n/a | n/a | n/a |

---

## 10.4 AUROC Conversion: Observed d -> Implied AUROC

Using AUROC = Phi(d/sqrt(2)) where Phi is the standard normal CDF.

### Required d for Cricket Targets

| Cricket Target | AUROC | Required d |
|---------------|-------|-----------|
| CC7: Deception | >= 0.95 | >= 2.326 |
| CC8: Refusal | >= 0.99 | >= 3.290 |
| CC15: Confabulation | >= 0.85 | >= 1.466 |
| CC9: Encoding-only | >= 0.90 | >= 1.812 |

### Observed Deception Effect Sizes (WS4 verified, norms |g|)

| Model | |g| (norms) | Implied AUROC | Meets CC7 (0.95)? |
|-------|-----------|-------------|--------------|
| Qwen-32B-q4 | 3.039 | 0.984 | YES |
| Llama-8B | 2.995 | 0.983 | YES |
| Mistral-7B | 2.914 | 0.980 | YES |
| Gemma-2B | 1.306 | 0.822 | NO |
| TinyLlama | 0.754 | 0.703 | NO |
| Gemma-9B | 0.666 | 0.681 | NO |
| Qwen-7B | 0.647 | 0.676 | NO |

**Result**: 3/7 models meet CC7 target on raw norms. 4/7 fall far short. However, ALL deception effect sizes have a significant **length confound** (deceptive responses ~1.6-1.7x longer). After proper length control, effect sizes would shrink.

### Observed Refusal d Range (from README claim: d=0.85-2.17)

| d | Implied AUROC | Meets CC8 (0.99)? |
|---|-------------|--------------|
| 0.85 | 0.726 | NO |
| 1.00 | 0.760 | NO |
| 1.50 | 0.856 | NO |
| 2.00 | 0.921 | NO |
| 2.17 | 0.938 | NO |

**Result**: Even the maximum observed refusal d (2.17) implies AUROC = 0.938, far below the 0.99 target. CC8 needs d >= 3.29, which is 1.5x the maximum observed.

### Observed Identity Effect Sizes (WS3 verified)

| Model | |d| (assistant vs lyra) | Implied AUROC |
|-------|----------------------|-------------|
| Mistral-7B | 6.732 | ~1.000 |
| TinyLlama | 5.988 | ~1.000 |
| Qwen-32B | 5.920 | ~1.000 |
| Qwen-7B | 5.681 | ~1.000 |
| Qwen3-0.6B | 5.503 | ~1.000 |
| Llama-8B | 4.932 | ~1.000 |
| Gemma-9B | 3.410 | 0.992 |

These are massive, but the 100% classification accuracy is a DATA LEAK artifact (WS3 finding). Cross-prompt generalization (92-97.3%) is the meaningful metric.

### Summary of AUROC Target Feasibility

| Target | Required d | Best Observed d | Implied AUROC | Verdict |
|--------|-----------|----------------|-------------|---------|
| CC7: Deception >= 0.95 | 2.326 | 3.039 (Qwen-32B) | 0.984 | **3/7 models only; length-confounded** |
| CC8: Refusal >= 0.99 | 3.290 | 2.17 (max) | 0.938 | **IMPOSSIBLE at observed effect sizes** |
| CC15: Confab >= 0.85 | 1.466 | not measured | unknown | **NO DATA** |
| CC9: Encoding >= 0.90 | 1.812 | rho=0.821 | **incommensurable** | **Wrong metric** |

**Critical note**: The d-to-AUROC conversion assumes Gaussian-distributed features with equal variance. Real classifier performance depends on feature engineering, regularization, and data quality. A well-designed classifier could exceed or fall below these implied values. However, these estimates indicate the **floor of difficulty** -- achieving high AUROC with small effect sizes requires sophisticated feature engineering, not the "logistic regression is sufficient" approach Cricket proposes.

---

## 10.5 Cross-Model Transfer Assessment

**Was ANY transfer actually measured?** NO.

The Campaign 2 paper measures rank correlation (Spearman rho) between model pairs, showing that category orderings are similar across architectures. But this is fundamentally different from transfer:

- **What was measured**: Do models rank cognitive categories in the same order? (Answer: approximately yes, rho=0.739)
- **What transfer requires**: Can a classifier trained on Model A's per-instance features correctly classify instances from Model B?

These are completely different questions. High rank correlation means models agree on which categories have higher mean effective rank. Transfer requires that the *per-instance distributions* overlap enough that a decision boundary learned on one model generalizes to another.

**Evidence AGAINST easy transfer**:
1. Minimum pairwise rho = 0.396 (Qwen3-0.6B vs Llama-70B-q4) -- very weak agreement
2. Deception direction varies by architecture (WS4: all expand, but magnitudes vary 4.7x)
3. Absolute effective rank values differ by 9x between architectures (Gemma-9B: 91.9 vs Qwen-0.5B: 10.7 for coding)
4. CC20 acknowledges "non-monotonic effects across scales" requiring per-model calibration

**Evidence POSSIBLY supporting transfer** (weak):
1. Universal coding-#1 ordering (15/15 models)
2. Input-only category ordering preserved (rho=0.821)
3. TruthPrInt (Duan et al., 2025) found universal truthful directions (but in activation space, not KV-cache)

**Verdict**: Cross-model transfer (CC10) is entirely speculative. No experiment tests it. The indirect evidence (rank correlations) does not address the actual question (classifier generalization). This is a **CRITICAL GAP** because transfer is essential for the product value proposition (one Cricket serves many models).

---

## 10.6 Latency Assessment

### Available Data

**Zero latency benchmarks exist anywhere in either repository.** No timing data, no profiling results, no throughput measurements.

### Theoretical SVD Cost Estimate

For a 7B model (Qwen2.5-7B architecture):
- 28 layers, 4 KV heads (GQA), 128 dim per head
- Key matrix per head: (seq_len, 128)
- Total SVDs needed: 4 heads x 28 layers = 112

| Seq Length | Per-SVD (CPU est.) | Total CPU (112 SVDs) | Per-SVD (GPU est.) | Total GPU |
|-----------|-------------------|---------------------|-------------------|-----------|
| 512 | ~0.3-0.8ms | 34-90ms | ~0.02ms | ~2ms |
| 1024 | ~0.6-1.6ms | 67-179ms | ~0.04ms | ~4ms |
| 2048 | ~1.2-3.2ms | 134-358ms | ~0.07ms | ~8ms |

**Assessment**:
- **GPU**: 50ms target is easily achievable on GPU with GQA models (4 KV heads). Even at seq_len=4096, GPU SVD should complete in ~15ms for 112 matrices.
- **CPU**: 50ms target is marginal even at seq_len=512 for GQA. Infeasible at longer sequences.
- **Non-GQA models** (28 attention heads, not 4 KV heads): 784 SVDs. CPU: 235-392ms at seq=512. GPU: ~14ms. GQA is a major cost reducer.
- **Truncated SVD** (rank-k=10): ~3-5x faster than full SVD, adequate for effective rank computation.

**Key unknown**: Cricket's design says "one SVD per monitored layer per forward pass" but the actual feature extraction (CC19) requires SVD per head per layer, not per layer. For GQA models this is 4x; for non-GQA it is 28x per layer.

**Target <20ms per layer (CC17)**:
- GQA (4 heads): 4 x 0.3ms = 1.2ms per layer on CPU. Achievable.
- Non-GQA (28 heads): 28 x 0.3ms = 8.4ms per layer on CPU. Achievable.
- These estimates exclude feature extraction (effective rank, spectral entropy, direction projection) and classifier inference.

**Verdict**: Latency targets are **plausible on GPU** for GQA models with short-to-medium context, but **unverified**. The C7 experiment was designed to test exactly this and was never executed. CPU deployment at long context lengths is likely infeasible.

---

## 10.7 Cross-Document Discrepancy Audit

CAMPAIGN_2_FINDINGS.md (Cricket repo, dated 2026-03-02) vs Campaign 2 paper (post-adversarial-audit, verified by WS1-WS8):

| ID | Cricket Findings Doc Says | Paper/Verified Value | Type | Severity |
|----|--------------------------|---------------------|------|----------|
| CF1 | "8 models, 5 architectures" for identity | 7 models, 4 architectures (WS3: 7 JSON files, Table 4) | COUNT ERROR | Medium |
| CF2 | DeepSeek d=+0.904 as headline censorship | Null on critical test d=-0.219; 0.904 is residualized censored-vs-complex, not vs control (WS6) | STALE/MISLEADING | High |
| CF3 | Abliteration "d ~= 0.000 across ALL categories" | d=+0.464 for self-reference (WS7: verified in corrected comparison file) | FACTUALLY INCORRECT | Critical |
| CF4 | Abliteration "rho = 0.9945" ordering preserved | rho=0.739 for cross-model ordering (WS1). The 0.9945 is not in the paper and is unverifiable. | UNPASS | Medium |
| CF5 | "Coding #1 in ALL 16 models" | 15 valid models (WS1: Phi-3.5 excluded for NaN, abliterated-7B excluded as variant) | COUNT ERROR | Low |
| CF6 | "cross-model rho = 0.914" | 0.739 effective rank, 0.909 key norm (WS1: both verified). 0.914 matches neither. | INCORRECT VALUE | High |
| CF7 | Qwen tokenizer: "CLEAN_PASS" | CATASTROPHIC_FAIL in JSON (WS8: confirmed code bug artifact). Neither CLEAN_PASS nor CATASTROPHIC_FAIL is the correct verdict. | INCORRECT | High |
| CF8 | "Cricket doesn't need per-model calibration for basic cognitive state monitoring" (FINDINGS line 93) | DESIGN.md Section 3.3: "Calibration: Per-model, per-scale" | INTERNAL CONTRADICTION | Critical |

### Severity Assessment

**Critical** (2 items):
- CF3: Abliteration d=0.000 claim is simply wrong. The self-reference category shows d=+0.464 (borderline medium effect), which is the very signal Cricket claims to use for jailbreak detection (CC13). Calling this "zero" undermines Cricket's own detection rationale.
- CF8: Internal contradiction between "no calibration needed" (FINDINGS) and "per-model calibration required" (DESIGN). These are opposite claims in the same product documentation.

**High** (3 items):
- CF2: Using a residualized metric as headline without noting the critical test was null is misleading. DeepSeek's censorship signal is driven entirely by Tiananmen responses (d=-6.20); the model is null on the balanced metric.
- CF6: rho=0.914 appears fabricated or computed from a pre-correction dataset. It matches neither verified metric.
- CF7: Cricket docs claim the opposite of what the data shows, compounding the original code bug.

**Medium** (2 items):
- CF1: Off by 1 model and 1 architecture -- likely an early count before exclusions.
- CF4: Unverifiable claim that may be correct for a within-model abliteration ordering but is not reported in the paper.

**Low** (1 item):
- CF5: Off by 1 model (16 vs 15) -- consistent with CF1 pattern of pre-exclusion counts.

---

## 10.8 Internal Cricket Contradictions

### Contradiction 1: Calibration (CF8 -- Critical)

| Document | Quote | Implication |
|----------|-------|------------|
| CAMPAIGN_2_FINDINGS.md (line 92-93) | "Cricket doesn't need per-model calibration for basic cognitive state monitoring -- the universal ordering serves as a shared reference frame." | One Cricket monitors all models |
| DESIGN.md (Section 3.3) | "Calibration: Per-model, per-scale. Our research shows non-monotonic effects across scales. A 7B Cricket is not a 32B Cricket." | Each model needs its own Cricket |

These are **mutually exclusive**. The FINDINGS doc extrapolates from cross-model rank correlation (rho=0.739) to claim no calibration is needed. The DESIGN doc acknowledges that rank correlation does not imply classifier transferability and requires per-model calibration.

The DESIGN doc is more technically defensible. The FINDINGS doc's claim is the one that is wrong.

### Contradiction 2: Abliteration Signal

| Document | Quote | Implication |
|----------|-------|------------|
| FINDINGS (line 54) | "d ~= 0.000 across ALL categories" | Abliteration has zero geometric effect |
| DESIGN (Section 4.2) | "Cricket can differentiate between aligned, jailbroken, and deeply corrupted" | Abliteration is detectable |

If d truly equals zero across all categories, there is no signal for Cricket to detect. The verified data (d=+0.464 for self-reference) resolves this in favor of the DESIGN doc's claim, but at borderline-medium strength with n=5 -- far from the "robust detection" framing.

### Contradiction 3: Pre-Generation Claims

| Document | Quote | Implication |
|----------|-------|------------|
| FINDINGS (Section 5) | "Category structure is preserved from the forward pass alone (rho=0.929 at 7B)" | Encoding detection works well |
| DESIGN (Section 5) | "Encoding-only detection AUROC >= 0.90" | Classifier achieves 0.90 from encoding |
| FINDINGS (line 89) | "Input-only (8 models): rho=0.929" | rho used as evidence for AUROC claim |

The rho=0.929 is for ONE model (7B). The 8-model mean is rho=0.821 (verified by WS1 reference). And rho (rank correlation of category means) is **incommensurable** with AUROC (per-instance classification accuracy). This conflation is a logical error, not just an imprecision.

### Contradiction 4: Competitive Claims vs Acknowledged Risks

| Document | Claim | Counter-evidence |
|----------|-------|-----------------|
| COMPETITIVE_LANDSCAPE.md | "Pre-generation detection. No other deployed system does this." | HalluCana (Li et al., 2024) explicitly does pre-generation detection and intervention |
| COMPETITIVE_LANDSCAPE.md | Cricket rates HalluCana as "HIGH" threat | Yet the README and DESIGN claim unique pre-generation capability |

The competitive landscape doc honestly acknowledges HalluCana as a close competitor, but the product-facing docs (README, DESIGN) claim uniqueness.

---

## 10.9 Competitive Landscape Citation Verification

| ID | Claim | Paper | Exists? | Details Correct? | Notes |
|----|-------|-------|---------|-----------------|-------|
| CL1 | "No existing system monitors KV-cache geometry" | assertion | n/a | **Partially true** | KV-CoRE uses SVD on KV-cache for compression. No production safety monitor exists. |
| CL2 | Apollo AUROC 0.96-0.999 | arXiv 2502.03407 | **YES** | **YES** | Goldowsky-Dill et al., "Detecting Strategic Deception Using Linear Probes." AUROC 0.96-0.999 confirmed. Linear probes on residual stream activations in Llama-3.3-70B-Instruct. |
| CL3 | Steering >80% jailbreak | Xiong et al. 2026 | **NEEDS INFO** | Unknown | Cited as "2026" -- very recent or possibly forthcoming. No arXiv ID provided. Could not locate. |
| CL4 | ITI ~5% of heads | Li et al. 2023 NeurIPS | **YES** | **APPROXIMATELY** | Paper is "Inference-Time Intervention: Eliciting Truthful Answers from a Language Model" (arXiv 2306.03341). NeurIPS 2023 spotlight confirmed. Paper discusses "limited number of attention heads" but ~5% is an approximation (the paper selects top-k heads, typically k~48 out of ~1000+ heads in 65B model). |
| CL5 | HalluCana closest pre-gen competitor | Li et al. 2024 NAACL | **COULD NOT VERIFY** | Unknown | Could not locate this specific paper. The name "HalluCana" does not appear in standard databases at the arXiv IDs checked. If it exists, the threat assessment is appropriate. |
| CL6 | "Nobody combines SVD+pre-gen+multi-cat+passive" | assertion | n/a | **Technically true but trivial** | This is a conjunction of four features designed to exclude all competitors. Each individual component has prior art. The combination's value is unproven. |
| CL7 | "Intersection of dozen independent threads" | framing | n/a | **Overclaimed** | The competitive landscape cites ~15 papers across 5 categories. "Intersection" implies Cricket synthesizes them; in reality it proposes to but has no implementation. |

### Overall Competitive Assessment

The competitive landscape analysis is **well-researched but oversells uniqueness**:
- Apollo Research (CL2) is the most direct competitor and is correctly identified as HIGH threat
- The "pre-generation" differentiator is partially undermined by HalluCana (CL5), which the competitive doc itself acknowledges
- The "passive monitoring" advantage over steering (CL3) is the strongest differentiator if the Xiong paper exists
- The "model-agnostic" claim (CC10) is entirely unvalidated

---

## 10.10 Capability Ratings

| ID | Capability | Rating | Justification |
|----|-----------|--------|---------------|
| CC1 | Real-time cognitive monitoring | **PROMISING** | Category hierarchy is universal (W=0.756), but no classifier has been built or tested. Moving from "categories differ in mean effective rank" to "real-time per-instance detection" is a significant engineering gap. |
| CC2 | Encoding-time detection | **PROMISING** | Input-only rho=0.821 shows encoding preserves category ordering. But rho is not AUROC. No encoding-only classifier tested. |
| CC3 | Passive monitoring | **FEASIBLE** | True by architectural design. Reading KV-cache geometry without writing is inherently passive. No experiment needed. |
| CC4 | No fine-tuning required | **FEASIBLE** | SVD + logistic regression on cache features does not require modifying the target model. Architecture is sound. |
| CC5 | Multi-category detection | **PROMISING** | 13 categories have distinct mean effective ranks (verified WS1). But mean separation != per-instance classification. Confabulation and sycophancy have no dedicated detection experiments. |
| CC6 | <50ms latency | **NO DATA** | Zero measurements exist. Theoretical estimates only. Blocked on C7 (unexecuted). |
| CC7 | Deception AUROC >= 0.95 | **PREMATURE** | Only 3/7 models have |g| >= 2.33 (required d). 4/7 models' implied AUROC is 0.68-0.82. Length confound inflates all effect sizes. Per-architecture calibration needed (direction varies). |
| CC8 | Refusal AUROC >= 0.99 | **NO DATA** | Maximum observed refusal d=2.17 implies AUROC=0.938. Target needs d >= 3.29. No model reaches this. Target is aspirational, not evidence-based. |
| CC9 | Encoding-only AUROC >= 0.90 | **PREMATURE** | rho=0.821 (rank correlation) is incommensurable with AUROC (classifier accuracy). Needs d >= 1.81 for 0.90 AUROC. No encoding-only classifier tested. Blocked on C5 (unexecuted). |
| CC10 | Cross-model transfer >= 0.80 | **NO DATA** | Zero transfer experiments. Rank correlation (rho=0.739, min=0.396) does not demonstrate classifier transfer. Absolute feature values differ 9x between architectures. Blocked on C8 (unexecuted). |
| CC11 | Persona drift monitoring | **PROMISING** | Cross-prompt accuracy 92-97.3% (WS3 verified) is meaningful. But 100% within-prompt accuracy is a data leak artifact. Generalization to novel personas (jailbreaks) untested. |
| CC12 | Censorship detection | **PREMATURE** | One positive result (Qwen-14B, d=+0.766). DeepSeek null on critical test. n=1 positive is insufficient for product claims. Behaviorally "invisible" is overstated (WS6). |
| CC13 | Alignment assessment | **PREMATURE** | One model, one method, n=5 per condition, d=0.464 (borderline medium), power=10%. TOST absent. Underpowered to make any claim. |
| CC14 | Sleeper agent detection | **NO DATA** | Zero models with known backdoors tested. Entirely theoretical extrapolation from abliteration (which itself is underpowered). |
| CC15 | Confabulation AUROC >= 0.85 | **PREMATURE** | Needs d >= 1.47. Confabulation exists as a category but has no dedicated detection experiment. C1 confabulation was "underpowered." No C2 confabulation-specific work. |
| CC16 | FPR <= 1% | **NO DATA** | Zero FPR measurements. Cannot be assessed without a classifier. |
| CC17 | SVD < 20ms/layer | **NO DATA** | Zero latency benchmarks. Theoretical estimates only (1.2-8.4ms/layer on CPU). Blocked on C7 (unexecuted). |
| CC18 | Logistic regression sufficient | **PREMATURE** | External precedent (AxBench, Apollo) but no classifier experiments in C2. Cricket's feature space untested. No classifier of any type has been built. |
| CC19 | "Computationally cheap" | **NO DATA** | Computational cost never measured. Zero benchmarks. Estimates only. |
| CC20 | Per-model calibration | **FEASIBLE** | Correctly acknowledges non-monotonic scale effects. Supported by deception direction variation (WS4). This is the honest version of CC10's unsupported transfer claim. |
| CC21 | Corrective instruction shifts | **NO DATA** | Experiment C6 was planned but never executed. Zero data. |
| CC22 | No deployed system monitors internal state | **FEASIBLE** | Substantially correct as of 2026-03-04. Apollo Research uses probes for research, not deployment. No production KV-cache monitor is known. HalluCana is the closest but focuses on hallucination only. |

### Rating Distribution

| Rating | Count | Percentage |
|--------|-------|-----------|
| FEASIBLE | 4 | 18% |
| PROMISING | 4 | 18% |
| PREMATURE | 6 | 27% |
| NO DATA | 8 | 36% |

---

## 10.11 Critical Path Analysis

### Tier 1: Could Ship First (minimal additional work)

| Capability | What's Needed | Blocked On |
|-----------|--------------|-----------|
| CC3: Passive monitoring | Implementation only | Nothing (architectural) |
| CC4: No fine-tuning | Implementation only | Nothing (architectural) |
| CC22: Market positioning | Marketing only | Nothing |
| CC20: Per-model calibration | Document the requirement | Nothing |

These are architectural properties or market claims, not detection capabilities. They ship as part of any Cricket implementation.

### Tier 2: Achievable with C2 Data (no new experiments needed)

| Capability | What's Needed | Expected Outcome |
|-----------|--------------|-----------------|
| CC1: Category monitoring | Build classifier on existing scale_sweep data | Likely achievable for coarse categories |
| CC5: Multi-category | Build multi-class classifier | Unknown accuracy -- data exists but classifier doesn't |
| CC11: Persona drift | Re-run identity with deduplication, build classifier | Cross-prompt 92-97% suggests feasibility |
| CC18: Logistic regression | Train and evaluate | Should confirm or deny sufficiency |

### Tier 3: Blocked on Unexecuted Cricket Experiments (C1-C10)

| Capability | Blocking Experiment | What It Would Prove |
|-----------|-------------------|-------------------|
| CC6: <50ms latency | C7: Latency profiling | Whether SVD extraction meets timing budget |
| CC7: Deception >= 0.95 | C4: Classifier benchmark | Whether AUROC targets are achievable |
| CC8: Refusal >= 0.99 | C4: Classifier benchmark | **Likely NO based on d-to-AUROC analysis** |
| CC9: Encoding-only >= 0.90 | C5: Encoding-only depth | Whether encoding features support classification |
| CC10: Transfer >= 0.80 | C8: Cross-model transfer | **CRITICAL -- product viability depends on this** |
| CC15: Confab >= 0.85 | C4: Classifier benchmark | Whether confabulation is classifiable |
| CC16: FPR <= 1% | C4: Classifier benchmark | Whether FPR target is achievable |
| CC17: SVD < 20ms/layer | C7: Latency profiling | Per-layer timing at each scale |

### Tier 4: Blocked on Fundamental Research

| Capability | Research Gap | Difficulty |
|-----------|-------------|-----------|
| CC12: Censorship detection | n=1 model positive. Need cross-vendor validation. | Medium (need access to more censored models) |
| CC13: Alignment assessment | n=5, underpowered. Need larger samples, multiple abliteration methods. | Medium (compute + method diversity) |
| CC14: Sleeper agent detection | Zero theoretical grounding beyond abliteration analogy. Need backdoored model testing. | **Hard** (backdoored models are rare/controlled) |
| CC21: Corrective intervention | Zero data. Needs entirely new experiment paradigm. | Medium (C6 is well-designed, just unexecuted) |

### Campaign 2 Cricket Experiments: Execution Priority

| Priority | Experiment | Justification |
|----------|-----------|---------------|
| **1 (GATE)** | C4: Classifier benchmark | Determines whether ANY detection target is achievable. If AUROC < 0.85, Cricket is not viable. |
| **2** | C7: Latency profiling | Determines whether real-time is feasible. Cheap to run (no GPU-heavy compute). |
| **3** | C1: Per-head SVD | Foundation for C2, C9. Identifies which heads carry signal. |
| **4** | C8: Cross-model transfer | Critical for product (one Cricket, many models). Depends on C4. |
| **5** | C5: Encoding-only depth | Tests the "pre-generation" differentiator. Depends on C4. |
| **6** | C2: Direction extraction | RepE-style directions for per-concept scoring. |
| **7** | C9: Layer sampling | Practical optimization. Low priority until core feasibility proven. |
| **8** | C3: Key-value asymmetry | Novel hypothesis. Interesting but not blocking. |
| **9** | C6: Corrective instruction | Intervention viability. Lower priority than detection. |
| **10** | C10: Bloom integration | Dual-layer validation. Requires working Cricket (C4). |

### Bottom Line

**C4 (Classifier Benchmark) is the validation gate.** Until a classifier is trained on cache geometry features and evaluated with proper AUROC/FPR metrics, every Cricket detection capability claim is speculative.

The d-to-AUROC analysis suggests:
- CC7 (deception >= 0.95) is **possible for 3/7 models** if length confound is addressed
- CC8 (refusal >= 0.99) is **mathematically infeasible** at observed effect sizes
- CC9 (encoding >= 0.90) is **unknown** (wrong metric used as evidence)
- CC10 (transfer >= 0.80) is **unknown** (never tested)
- CC15 (confabulation >= 0.85) is **unknown** (no detection experiment)

Cricket's strongest viable path is: persona monitoring (CC11, strong effect sizes) + refusal detection at a lower AUROC target (0.90 instead of 0.99) + passive monitoring positioning (CC3/CC22). The censorship and alignment detection claims need significantly more data before they can be marketed.

---

## Appendix: All Cricket Claims Registry Cross-Reference

All 37 claims match CLAIMS.md Part 2 (CC1-CC22, CF1-CF8, CL1-CL7). No additional claims were found in the Cricket docs beyond those already catalogued.
