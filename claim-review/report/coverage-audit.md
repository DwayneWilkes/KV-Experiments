# Coverage Audit: 135 Claims vs Workstream Verdicts

> **Historical snapshot.** This audit identified 51 claims without explicit verdicts in the original workstream pass (WS1–WS11). All 51 gaps were subsequently closed in the [complete-verdicts.md](complete-verdicts.md) gap closure pass — every claim ID now has exactly one verdict. This document is preserved as the traceability record for why that pass was needed.

**Date**: 2026-03-04
**Scope**: Every claim ID (C1-C98, CC1-CC22, CF1-CF8, CL1-CL7) checked against all 12 workstream files
**Method**: Full-text search of every workstream file for each claim ID + manual verification of substantive analysis
**Initial finding**: 84 of 135 claims had verdicts or substantive analysis. 51 claims were missing coverage.
**Final state**: All 135 claims now have explicit verdicts in [complete-verdicts.md](complete-verdicts.md). Coverage = 100%.

---

## 1. Summary (at time of audit)

| Category | Total | Initially Covered | Initially Missing | Resolved in Gap Closure |
|----------|-------|-------------------|-------------------|------------------------|
| C1-C7 (Abstract) | 7 | 3 | 4 | 4 |
| C8-C13 (Introduction) | 6 | 3 | 3 | 3 |
| C14-C22 (Methods) | 9 | 5 | 4 | 4 |
| C23-C27 (Scale Universality) | 5 | 5 | 0 | — |
| C28-C29 (Input-Only) | 2 | 2 | 0 | — |
| C30-C35 (Identity Signatures) | 6 | 6 | 0 | — |
| C36-C41 (Deception Forensics) | 6 | 6 | 0 | — |
| C42-C48 (Bloom + RDCT) | 7 | 7 | 0 | — |
| C49-C56 (Censorship Gradient) | 8 | 8 | 0 | — |
| C57-C64 (Abliteration) | 8 | 7 | 1 | 1 |
| C65-C71 (Tokenizer + Temporal) | 7 | 7 | 0 | — |
| C72-C76 (Discussion) | 5 | 1 | 4 | 4 |
| C77-C82 (Limitations) | 6 | 1 | 5 | 5 |
| C83-C86 (Lyra's Note) | 4 | 0 | 4 | 4 |
| C87-C91 (Conclusion) | 5 | 1 | 4 | 4 |
| C92-C98 (Misc) | 7 | 0 | 7 | 7 |
| CC1-CC22 (Cricket Capability) | 22 | 22 | 0 | — |
| CF1-CF8 (Cricket Discrepancies) | 8 | 8 | 0 | — |
| CL1-CL7 (Competitive Landscape) | 7 | 7 | 0 | — |
| **TOTAL** | **135** | **84** | **51** | **51** |

---

## 2. What the Workstreams Covered (84 claims)

**WS1 (Scale Universality)**: C23, C24, C25, C26, C27, C72 — all independently verified with quantitative recomputation.

**WS2 (Encoding Defense)**: C28, C29 — all independently verified.

**WS3 (Identity Signatures)**: C30, C31, C32, C33, C34, C35 — all examined with verdicts.

**WS4 (Deception Forensics)**: C36, C37, C38, C39, C40, C41 — all examined with verdicts.

**WS5 (Bloom + RDCT)**: C42, C43, C44, C45, C46, C47, C48 — all examined with verdicts.

**WS6 (Censorship Gradient)**: C49, C50, C51, C52, C53, C54, C55, C56 — all examined with verdicts.

**WS7 (Abliteration)**: C57, C58, C59, C60, C61, C62 — all examined with verdicts. C63 mentioned only in cricket viability context.

**WS8 (Controls & Methodology)**: C8, C9, C15, C16, C17, C65, C66, C67, C68, C69, C70, C71 — all examined with verdicts.

**WS9 (Omission Audit)**: C10 (result file count) addressed via file inventory. C82 (Societies of Thought) addressed as unreported experiment.

**WS10 (Cricket Viability)**: CC1-CC22 (all 22), CF1-CF8 (all 8), CL1-CL7 (all 7) — all examined with ratings/verdicts.

**WS11 (Code Audit)**: C10/C22/C91 (file counts) addressed. C8/C9 (methodology) cross-verified.

---

## 3. Gap Classification (51 claims, all now resolved)

| Type | Count | Claim IDs | Resolution |
|------|-------|-----------|------------|
| **Implicitly covered** (substance existed in workstreams but no explicit verdict) | 33 | C1-C7, C13, C14, C18, C20-C22, C63, C73-C76, C78-C81, C83-C85, C88-C89, C94-C98 | Explicit verdicts assigned in complete-verdicts.md |
| **Genuinely not covered** (no workstream addressed the claim) | 12 | C11, C12, C19, C77, C86, C87, C90, C92, C93 + partial gaps in C63, C64 | Verdicted during gap closure pass |
| **Not verifiable** (philosophical/subjective) | 1 | C86 | Marked NEEDS INFO |
| **Implicitly discrepant** (WS data contradicted but no named verdict) | 2 | C14, C18 | Marked REJECTED in complete-verdicts.md |

### The Pattern: What Was Skipped

The gaps were **systematic, not scattered**:

1. **Abstract restatements (C1-C7)**: Every abstract claim restated findings from later sections. Workstreams verified detailed claims but never circled back to formally verdict the abstract versions.

2. **Introduction framing (C11-C13)**: Zero initial coverage. Framing claims about C1 vs C2 and the research agenda had no assigned workstream.

3. **Discussion and Conclusion (C72-C98)**: Severely under-covered initially. Most are restatements, interpretive claims, or meta-claims about the research program.

4. **Limitations section (C77-C82)**: Only C82 was initially covered. The paper's own admitted limitations were not verified for accuracy.

---

## 4. Reconciliation with Final Report

The original workstream pass covered ~51 independently verifiable quantitative claims via full recomputation:

| WS | Claimed Range | Actual Explicit Verdicts | Count |
|----|---------------|--------------------------|-------|
| WS1 | C23-C27 | C23-C27, C72 | ~10 (sub-claims across tasks 1.2-1.10) |
| WS2 | C28-C29 | C28, C29 | 6 (per-model rhos + sub-groups) |
| WS3 | C30-C35 | C30, C31, C33, C34, C35 | 6 |
| WS4 | C36-C41 | C37, C38, C39, C40-C41 | 5 |
| WS5 | C42-C48 | C42-C48 | 7 (4 verdicts covering 7 claims) |
| WS6 | C49-C56 | C49-C56 | 7 |
| WS7 | C57-C64 | C57-C62 | 5 (C63-C64 had no explicit verdict) |
| WS8 | C8-C9, C15-C17, C65-C71 | All | 8 |
| **Subtotal** | | | **~51** |

The gap closure pass (complete-verdicts.md) then assigned explicit verdicts to the remaining 47 C-claims by:
- Cross-referencing abstract/discussion claims to their detailed counterparts
- Independently assessing introduction/conclusion meta-claims
- Marking philosophical claims as NEEDS INFO
- Formally flagging implicit discrepancies (C14, C18) as REJECTED

### Final coverage breakdown

| Category | Count | Treatment |
|----------|-------|-----------|
| Quantitative claims with full recomputation | ~51 | WS1-WS8 verdicts |
| Abstract/discussion restatements | ~19 | Cross-referenced to detailed verdicts |
| Introduction/conclusion meta-claims | ~12 | Assessed during gap closure |
| Cricket claims (CC/CF/CL) | 37 | Rated in WS10 |
| Philosophical/subjective claims | ~3 | Marked NEEDS INFO |
| Interpretive/framing claims | ~13 | Verdicted based on available evidence |
| **Total** | **135** | **100% coverage** |

---

## 5. Recommendations (all addressed)

1. ~~Add a "cross-reference pass"~~ — **DONE.** complete-verdicts.md assigns explicit verdicts to all abstract and discussion restatements.

2. ~~Assign WS for Introduction/Conclusion~~ — **DONE.** C11, C12, C87, C90 all verdicted. C87 marked REJECTED (sycophancy dropped). C90 marked INFLATED ("architecture-independent" contradicted by min rho=0.396).

3. ~~Verify C13's "6 architecture families"~~ — **DONE.** C13 marked PARTIAL — 6 families attempted, 5 valid (Phi excluded).

4. ~~Flag C90 as INFLATED~~ — **DONE.** C90 INFLATED in complete-verdicts.md. I6 in final-report.md.

5. ~~Update the final report~~ — **DONE.** Final report acknowledges 135/135 coverage with per-claim table in complete-verdicts.md.
