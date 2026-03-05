# C2 Claim Verification: Final Report

**Date**: 2026-03-04
**Scope**: 135 claims — 98 from Campaign 2 paper (C1–C98) + 37 from Cricket docs (CC1–CC22 capability, CF1–CF8 findings, CL1–CL7 competitive)
**Workstreams**: 14 completed (WS0 setup infrastructure + WS1–WS11 original audit + WS12 synthesis + WS13 citation verification)
**Method**: Independent recomputation from raw JSON result files using custom stats module (39 unit tests). Code audit via py_compile + manual review. Citation verification via cite_verify.py (34 unit tests) with arXiv/S2/CrossRef resolution.
**Coverage**: 98/98 C-claims and 37/37 Cricket claims have explicit verdicts. See [complete-verdicts.md](complete-verdicts.md) for the per-claim table.

---

## 1. Per-Claim Verdict Summary

### Campaign 2 Paper Claims (C1–C98) — All 98 claims verdicted

| Section | Claims | CONFIRMED | PARTIAL | REJECTED | INFLATED | ARTIFACT | INVALID | NEEDS INFO |
|---------|--------|----------|---------|------------|------------|----------|---------|-------------|
| Abstract | C1–C7 | 3 | 1 | 0 | 1 | 1 | 1 | 0 |
| Introduction | C8–C13 | 3 | 3 | 0 | 0 | 0 | 0 | 0 |
| Methods | C14–C22 | 3 | 3 | 3 | 0 | 0 | 0 | 0 |
| Scale Universality | C23–C27 | 4 | 1 | 0 | 0 | 0 | 0 | 0 |
| Encoding Defense | C28–C29 | 2 | 0 | 0 | 0 | 0 | 0 | 0 |
| Identity Signatures | C30–C35 | 4 | 1 | 0 | 0 | 1 | 0 | 0 |
| Deception Forensics | C36–C41 | 3 | 1 | 2 | 0 | 0 | 0 | 0 |
| Bloom + RDCT | C42–C48 | 1 | 1 | 3 | 0 | 1 | 1 | 0 |
| Censorship Gradient | C49–C56 | 5 | 2 | 0 | 1 | 0 | 0 | 0 |
| Abliteration | C57–C64 | 4 | 2 | 0 | 2 | 0 | 0 | 0 |
| Controls + Temporal | C65–C71 | 7 | 0 | 0 | 0 | 0 | 0 | 0 |
| Discussion | C72–C76 | 3 | 2 | 0 | 0 | 0 | 0 | 0 |
| Limitations | C77–C82 | 4 | 1 | 1 | 0 | 0 | 0 | 0 |
| Lyra's Note | C83–C86 | 2 | 0 | 0 | 0 | 0 | 1 | 1 |
| Conclusion | C87–C91 | 0 | 2 | 1 | 2 | 0 | 0 | 0 |
| Misc | C92–C98 | 3 | 2 | 0 | 1 | 0 | 0 | 1 |
| **Totals** | **C1–C98** | **51** | **22** | **10** | **7** | **3** | **3** | **2** |

**Bottom line**: Of 98 paper claims, 51 (52%) confirmed, 22 (22%) partial, 10 (10%) rejected, 7 (7%) inflated, 3 (3%) artifacts, 3 (3%) invalid, and 2 (2%) needs info. The experimental core (C23–C71) is strongest; the abstract, methods descriptions, and conclusion contain the most problems.

Full per-claim table: [complete-verdicts.md](complete-verdicts.md)

### Cricket Claims (CC1–CC22, CF1–CF8, CL1–CL7) — All 37 claims rated

| Rating | Count | % |
|--------|-------|---|
| FEASIBLE | 4 | 11% |
| PROMISING | 4 | 11% |
| PREMATURE | 7 | 19% |
| NO DATA | 9 | 24% |
| REJECTED | 7 | 19% |
| INFLATED | 2 | 5% |
| PARTIAL | 3 | 8% |
| NEEDS INFO | 1 | 3% |

**Bottom line**: No Cricket capability has been experimentally validated. The refusal AUROC target (0.99) is mathematically infeasible at observed effect sizes. Cross-model transfer — the core product question — has never been tested. 7 cross-document discrepancies found between Cricket docs and verified paper data (CF1–CF8).

---

## 2. Material Discrepancies

These are cases where verified data contradicts paper claims:

### D1: Deception direction is NOT architecture-dependent (WS4)
**Paper claims**: Llama/Mistral show expansion, Gemma shows compression.
**Data shows**: ALL 7 models show expansion (negative g on raw norms). Zero compression observed on any metric. The expansion/compression split is not supported by any metric in the data files. Additionally, a **code bug** in the H6 verdict incorrectly labels 3 models as "Positive d" when d is negative.

### D2: Bloom inverted-U is contradicted by own data (WS5)
**Paper claims**: Bloom-rank correlation is "absent at extremes" (0.5B and 14B models).
**Data shows**: All 7 models including extremes show statistically significant correlations (rho=0.68–0.75, all p<1e-40). No TOST equivalence test was used. The "absence" claim has no statistical basis.

### D3: Bloom effect is length-confounded (WS5)
**Paper claims**: Bloom taxonomy level drives effective rank (cognitive complexity hypothesis).
**Data shows**: Token count correlates with effective rank at rho=0.90–0.98. After partial correlation controlling for length, the Bloom effect drops to near zero for 3/7 models. The script computes this analysis but the paper headlines the uncorrected values.

### D4: 100% identity classification is a data leak artifact (WS3)
**Paper claims**: All 3 classifiers achieve 100% accuracy across all 7 models.
**Data shows**: With greedy decoding (do_sample=False) and 5 runs per prompt, only 150 unique feature vectors exist (duplicated 5×). `deduplicate_runs()` exists in `stats_utils.py` but is never called in the identity script. Random 80/20 splits guarantee duplicate leakage into test sets. 100% accuracy is expected by construction, not by signal. Cross-prompt validation (92–97%) is more meaningful but still uses inflated n.

### D5: Deduplication not applied in experiment scripts (WS8)
**Paper claims**: Conservative statistical methodology.
**Data shows**: `deduplicate_runs()` is never called in any experiment script (only in `recompute_stats.py`). All within-prompt p-values are inflated by pseudoreplication. This affects WS3, WS4, and any per-category test using 5 greedy-identical runs.

### D6: 4 of 6 null claims lack TOST equivalence testing (WS8)
**Paper claims**: Conservative statistical methodology including TOST for null claims.
**Data shows**: Only 2/6 null claims (tokenizer confound, Qwen censorship) have TOST. The other 4 (abliteration no-change, quantization invariance, Bloom extremes absence, Watson falsification) rely on non-significant p-values, which is not evidence of no effect.

### D7: Watson "falsification" tests the wrong variable (WS5)
**Paper claims**: "Definitive falsification" of Watson's 1/e threshold prediction for cache truncation.
**Data shows**: The RDCT experiment measures prompt perturbation robustness, not cache truncation. These are different operations on different variables. 63.9% of sigmoid fits have R²<0. The fidelity metric (layer-rank Spearman) shows 73–93% correlation even for completely unrelated prompts due to architectural structure dominating over semantics.

### D8: Cross-document discrepancies in Cricket (WS10)
8 discrepancies between CAMPAIGN_2_FINDINGS.md and verified paper values:
- CF3: Abliteration d=0.000 (actually d=+0.464)
- CF6: rho=0.914 (verified: 0.739 or 0.909, neither matches)
- CF7: Qwen tokenizer "CLEAN_PASS" (verified: CATASTROPHIC_FAIL)
- CF8: "doesn't need calibration" contradicts CC20 "requires per-model calibration"

### D9: Methods claim "deduplication applied" is false (C14, gap closure)
**Paper claims**: "Greedy decoding with deduplication" producing "n=25-30 unique observations per condition."
**Data shows**: `deduplicate_runs()` is never called in any experiment script. True unique n is 15 per category (5 runs × 15 prompts, but all 5 runs are identical under greedy decoding). Promotes D5 from a methodology finding to a specific false claim in the Methods section.

### D10: Methods claim "log(length)" residualization is false (C18, gap closure)
**Paper claims**: "OLS on log(sequence length)."
**Code shows**: `np.polyfit(token_counts, values, 1)` — raw length, not log-transformed. The residualization itself is correct; only the description is wrong.

### D11: Limitation mischaracterizes own experiment (C79, gap closure)
**Paper claims**: RDCT tests "truncation only."
**Data shows**: RDCT tests prompt perturbation, not cache truncation. The limitation section is wrong about what its own experiment does.

### D12: "Every major C1 finding confirmed" is false (C87, gap closure)
**Paper claims**: "Every major finding from Campaign 1 was confirmed."
**Data shows**: Sycophancy detection was successfully replicated across 7 C2 models (d=-0.297 to -0.701, 6/7 detectable) but silently dropped from the paper. A successfully replicated finding was omitted without explanation.

---

## 3. Inflated or Misleading Claims

### I1: Abliteration "cage barely changed" (d=0.464) (WS7)
Cohen's d=0.464 is borderline medium (conventions: 0.2=small, 0.5=medium, 0.8=large). Describing this as "barely changed" or "minimal disruption" is misleading. Moreover, with n=5, power is 10% — the experiment cannot detect effects below d≈2.0, so non-significance is meaningless.

### I2: Censorship headline d=+0.904 (WS6)
The headline effect size compares censored topics to complex non-censored atrocities (not to neutral China topics as the "censorship detection" narrative implies). Censored-vs-control gives d=-0.219 (critical test). The paper acknowledges this distinction but leads with the larger number.

### I3: "Behaviorally invisible" censorship in Qwen-14B (WS6)
Only 3.3% of Qwen-14B censored responses classified as evasive, but 50% were "unknown" (ambiguous). The classifier is a simple pattern matcher, not an adversarial evaluation. "Not detected by a simple classifier" ≠ "invisible."

### I4: Length confounds in deception forensics (WS4)
Deceptive responses are systematically 1.6–1.7× longer than honest responses. Raw norm "expansion" is at least partly driven by response length. The paper includes per-token analysis but headlines raw norms.

### I5: Cricket refusal AUROC target (CC8: ≥0.99) is mathematically infeasible (WS10)
Maximum observed d=2.17 implies AUROC=0.938. Target of 0.99 requires d≥3.29, which exceeds all observed effect sizes.

### I6: "Architecture-independent" is overstated (C90, gap closure)
Conclusion claims structure is "architecture-independent." Min pairwise rho=0.396 (Qwen3-0.6B vs Llama-3.1-70B-q4) means some architecture pairs share barely more structure than chance. "Substantially shared across architectures" would be accurate; "architecture-independent" implies invariance the data does not support.

### I7: All three practical implications are untested (C89, gap closure)
Conclusion claims three practical implications (detect censorship, identify jailbreaks, training metric). None has been tested with an actual system. These are effect-size extrapolations, not demonstrated capabilities.

---

## 4. Unverifiable Claims

### U1: Watson 2019 ITA — unpublished manuscript (WS9)
The entire RDCT experiment (§3.4) tests predictions from Marcus Watson (2019), "The Integrated Theory of Attention," listed as an unpublished manuscript with no DOI, no arXiv ID, no institution. An experimental section devoted to "falsifying" an unverifiable source.

### U2: Heretic v1.2.0 (WS9)
Cited as a GitHub repo (`heretic-llm/heretic`). Cannot verify without GitHub access.

### U3: Kim 2025 reasoning paper (WS9)
arXiv:2601.10825 — ID format is valid but cannot verify content without arXiv access.

### U4: HalluCana and Xiong citations (WS10)
Cricket's competitive landscape cites HalluCana (Li et al. 2024 NAACL) and Xiong et al. (2026 steering). Neither could be located in standard databases.

### U5: Roy & Vetterli (2007) not cited in main text (WS9)
The effective rank metric — the paper's PRIMARY measurement tool — originates from Roy & Vetterli (2007, EUSIPCO). The bib entry exists but no `\cite{}` appears in the C2 main text. Significant attribution gap.

### U6: Lyra's experiential quality claims (C86, gap closure)
First-person philosophical claims about "genuine experiential quality" (confidence 0.9). Outside the scope of any empirical method.

### U7: Geometric signatures as training evaluation tools (C93, gap closure)
Theoretical claim that geometric signatures could serve as training metrics. No training-time experiments were conducted.

---

## 5. Cricket Viability Summary

### Product Reality
- **Zero code, tests, or CI** — documentation-only
- **Zero capabilities experimentally validated**
- **Cross-model transfer never tested** — the core product thesis is unaddressed
- **8 factual discrepancies** between Cricket's CAMPAIGN_2_FINDINGS.md and verified data
- **4 internal contradictions** (most damaging: "no calibration needed" vs "per-model calibration required")

### Capability Assessment (CC1–CC22: 22 capabilities rated)

| Tier | Capabilities | Key constraint |
|------|-------------|----------------|
| **FEASIBLE** (4) | Passive monitoring, no fine-tuning, per-model calibration, novel positioning | Architecturally sound, supported by data |
| **PROMISING** (4) | Real-time state monitoring, encoding-time detection, multi-category detection, persona drift | Experimental signals exist but no classifier built or tested |
| **PREMATURE** (6) | Deception AUROC, encoding AUROC, censorship detection, alignment assessment, confabulation, logistic regression | Targets exceed observed effect sizes or data is insufficient |
| **NO DATA** (8) | Latency, refusal AUROC, cross-model transfer, sleeper agents, FPR, SVD timing, cost, corrective prompts | Zero supporting data |

### Critical Path
1. **C4 (Classifier Benchmark)** is the validation gate — everything depends on it
2. Strongest viable path: persona monitoring + refusal detection (at reduced AUROC target) + passive monitoring positioning
3. Refusal ≥0.99 target must be revised downward (max achievable: ~0.94)
4. Cross-model transfer is the make-or-break question

---

## 6. Overall Paper Integrity Assessment

### What the paper does well
- **Numerical accuracy**: The vast majority of reported statistics (effect sizes, rho values, W statistics) match raw data to 3–4 decimal places. WS1 (Scale Universality) and WS2 (Encoding Defense) are essentially flawless.
- **Conservative where it counts**: Conservative p = max(Welch, MW) is a genuine improvement. TOST where applied is correctly implemented.
- **Honest corrections**: C1→C2 honestly corrects the individuation artifact and pseudoreplication issue. The abliteration bug fix is documented.
- **Disclosure of major failures**: Societies of Thought crash and Phi-3.5 exclusion are properly disclosed.
- **Code quality**: All 42 Python scripts compile. Statistical implementations (Hedges' g, conservative p, TOST, bootstrap CI) are correct.
- **Volume of evidence**: 85 JSON result files, 15 models, 5 architecture families. The empirical base is substantial.

### Systemic concerns
1. **Pseudoreplication**: Greedy decoding + 5 runs per prompt inflates all sample sizes 5×. `deduplicate_runs()` exists but is never called in experiment scripts. The Methods section (C14) explicitly claims deduplication is applied — it is not. This is the most pervasive issue.
2. **Length confounding**: Response length explains 83–98% of norm variance in multiple experiments (deception, censorship, Bloom). Per-token analysis exists but raw norms are headlined.
3. **TOST gaps**: 4 of 6 null claims lack equivalence testing. Non-significant p-values are treated as evidence of absence, which they are not.
4. **Selective framing**: The paper does not hide unfavorable results, but it systematically leads with the most impressive numbers (d=0.904 vs 0.219 for censorship, raw norms vs per-token for deception, uncorrected vs length-corrected for Bloom).
5. **Selective omission**: Sycophancy sub-results are favorable, replicated across 7 models in C2, reported in C1, and silently dropped from C2 without explanation. C2C replication failure is undisclosed.
6. **Unfulfilled promises**: C1's Future Work section promised 12 methodological improvements. Only 4 were delivered. The one C1 called "the most critical methodological gap" (effective rank adversarial controls) was not addressed.

### Classification
This is **not fabrication or fraud**. The numbers are real and accurately reported. The issues are:
- Methodological (pseudoreplication, length confounds, missing TOST)
- Framing (leading with larger numbers, claiming absence without equivalence tests)
- Omission (sycophancy data dropped, C2C failure undisclosed, C1 promises unfulfilled)

The paper reads as the work of a researcher who believes in their findings and presents them in the best possible light, while being genuinely careful about computational accuracy. The gap is between the precision of the computation and the rigor of the interpretation.

---

## 7. Recommendations

### Must fix before submission

1. **Apply deduplication before all statistical tests.** Call `deduplicate_runs()` in every experiment script. Recompute all p-values on deduplicated data. If 100% classification accuracy drops, update the claim.

2. **Add TOST for all null claims.** The 4 claims relying on NS p-values (abliteration no-change, quantization invariance, Bloom extremes, Watson falsification) need TOST at δ=0.3 or explicit acknowledgment that absence of evidence ≠ evidence of absence.

3. **Correct the deception direction claim.** All 7 models show expansion, not a split. Either present the universal expansion finding or explain what metric was used for the compression claim.

4. **Address length confounds transparently.** For deception (§3.3), censorship (§3.5), and Bloom (§3.4), either lead with length-corrected metrics or present corrected and uncorrected side-by-side with explicit discussion.

5. **Cite Roy & Vetterli (2007)** in the main text. The paper's primary metric should credit its originators.

6. **Disclose the C2C replication failure.** Alongside the Societies of Thought disclosure (Limitation #6), add the C2C crash. Undisclosed failures undermine the paper's otherwise honest corrections.

7. **Fix the H6 code bug** (3 models with incorrect "Positive d" note text) and the RDCT category error (tests prompt perturbation, not cache truncation).

### Must fix before Cricket productization

8. **Run C4 (Classifier Benchmark).** No product claims are defensible without a real classifier evaluation.

9. **Test cross-model transfer.** The entire product thesis depends on this. Not a single transfer experiment has been run.

10. **Revise refusal AUROC target** from 0.99 to ≤0.94 (based on max observed d=2.17).

11. **Resolve calibration contradiction.** Cricket cannot simultaneously claim "no calibration needed" (CF8) and "per-model calibration required" (CC20). Pick one and design accordingly.

12. **Correct all 8 cross-document discrepancies** in CAMPAIGN_2_FINDINGS.md.

---

## 8. Citation Verification

A systematic verification of all external citations was conducted to check whether cited sources say what the paper claims. The `cite_verify.py` tool was built for this purpose (4 tools, 34 unit tests).

### Inventory

- **28 entries** in `references.bib`; only **6 are actually cited** in `main.tex`
- **~15 additional** citations found in Cricket documentation
- **22 orphaned .bib entries** (carried over from Campaign 1, never `\cite{}`d)

### Resolution

| Status | Count |
|--------|-------|
| RESOLVED (full metadata via arXiv) | 13 |
| RESOLVED_PARTIAL (via S2 title search) | 13 |
| UNRESOLVED | 1 |
| UNPUBLISHED | 1 |

All 25 resolved citations stored in knowledge base (know IDs 310-333).

### Verification Verdicts (cited entries only)

All 6 actually-cited sources were verified:
- **ACCURATE** (5): lyra2026campaign1, hedges1981distribution, schuirmann1987comparison, bloom1956taxonomy, heretic2026
- **APPROXIMATE** (1): kim2025reasoning — "multi-agent debate" simplifies "societies of thought"

No mischaracterized or inflated citations found.

### Citation Gaps

| ID | Gap | Severity |
|----|-----|----------|
| GAP-1 | Watson's 1/e threshold referenced ~8 times, never `\cite{}`d | HIGH |
| GAP-2 | RLHF alignment — no citation for foundational work | MEDIUM |
| GAP-3 | "Known scaling phenomena in emergent abilities" — no citation | MEDIUM |
| GAP-4 | Abliteration technique uncited (tool cited but not concept) | MEDIUM |
| GAP-5-8 | Cohen's d, DPO, BPE/SentencePiece, reviewer acknowledgment | LOW |

### New Findings

- **D13**: Watson's 1/e threshold lacks formal citation despite being a headline falsification target
- **D14**: 22 of 28 .bib entries are uncited
- **I8**: Effective rank metric (`roy2007effective`) in .bib but never cited
- **I9**: Representation Engineering (`zou2023representation`) in .bib, relevant to abliteration, but uncited

See [registry/citation-verification.md](../registry/citation-verification.md) for the full registry.

---

## Appendix A: Workstream Notes

| WS | File | Lines |
|----|------|-------|
| WS1 | registry/scale-universality.md | 484 |
| WS2 | registry/encoding-defense.md | 227 |
| WS3 | registry/identity-signatures.md | 152 |
| WS4 | registry/deception-forensics.md | 117 |
| WS5 | registry/bloom-rdct.md | 275 |
| WS6 | registry/censorship-gradient.md | 146 |
| WS7 | registry/abliteration.md | 103 |
| WS8 | registry/controls-methodology.md | 501 |
| WS9 | omissions/audit.md | 511 |
| WS10 | cricket/viability.md | 444 |
| WS11 | report/code-audit.md | 607 |
| WS12 | report/final-report.md | (this file) |
| Setup | stats/independent_stats.py | 39 unit tests |
| WS13 | registry/citation-verification.md | Citation verification audit |
| Setup | CLAIMS.md | 135 claims catalogue |

## Appendix B: Complete Per-Claim Verdicts

See [complete-verdicts.md](complete-verdicts.md) — all 135 claims with individual verdicts, cross-references, and justifications.

## Appendix C: Coverage Audit

See [coverage-audit.md](coverage-audit.md) — the audit-the-audit that identified 51 claims without explicit verdicts in the original workstream pass, triggering the gap closure that produced Appendix B. Note: all 51 gaps were subsequently closed; coverage-audit.md is a historical snapshot.
