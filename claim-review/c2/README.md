# Campaign 2 + Cricket Claim Verification Audit

Adversarial verification of **135 total claims**: 98 from the Campaign 2 paper (C1–C98) + 37 from JiminAI Cricket docs (CC1–CC22 capability, CF1–CF8 cross-document, CL1–CL7 competitive).

**Status**: COMPLETE (100/100 tasks, 2026-03-04)

> **Shared resources** (glossary, sources, stats module) have been promoted to [claim-review/](../README.md). Links below reference the parent directory where applicable.

---

## Results at a Glance

### Campaign 2 Paper (C1–C98)

| Verdict | Count | % |
| --------- | ------- | --- |
| CONFIRMED | 51 | 52% |
| PARTIAL | 22 | 22% |
| REJECTED | 10 | 10% |
| INFLATED | 7 | 7% |
| ARTIFACT | 3 | 3% |
| INVALID | 3 | 3% |
| NEEDS INFO | 2 | 2% |

### Cricket (CC1–CC22, CF1–CF8, CL1–CL7)

37 claims assessed. See [complete-verdicts.md](report/complete-verdicts.md) for the full breakdown.

**14 material discrepancies** (D1–D14), **9 informational findings** (I1–I9), **12 recommendations**.

Not fabrication. Computationally precise but interpretively inflated.

Full analysis: [final-report.md](report/final-report.md)

---

## Key Findings

1. **Deception direction** — all 7 models expand; paper claims Gemma compresses (WS4)
2. **Bloom inverted-U** — contradicted by own data; length confound explains 90–98% of variance (WS5)
3. **Watson "falsification"** — tests wrong variable (prompt perturbation, not cache truncation) (WS5)
4. **100% identity accuracy** — data leak from undeduplicated greedy runs (WS3)
5. **Deduplication never applied** in experiment scripts despite `deduplicate_runs()` existing (WS8)
6. **4/6 null claims lack TOST** — non-significant p treated as absence (WS8)
7. **Sycophancy data silently dropped** — favorable C1 results replicated in C2, unreported (WS9)
8. **Cricket**: 0 code, 0 tests; refusal AUROC 0.99 target mathematically infeasible; cross-model transfer never tested (WS10)

---

## Documents

### Primary

| Document | Description |
| ---------- | ------------- |
| [CLAIMS.md](CLAIMS.md) | All 135 claims with source quotes, checks, and verdicts |
| [GLOSSARY.md](../GLOSSARY.md) | 80+ cross-linked term definitions (shared) |
| [sources.md](../sources.md) | Cross-referenced bibliography (shared) |
| [complete-verdicts.md](report/complete-verdicts.md) | Per-claim verdict table with justifications (authoritative) |
| [final-report.md](report/final-report.md) | Aggregated report: discrepancies, findings, recommendations |

### Workstream Registry

| ID | Workstream | File |
| ---- | ----------- | ------ |
| WS0 | Setup | [independent_stats.py](../stats/independent_stats.py) (39 tests) |
| WS1 | Scale Universality | [scale-universality.md](registry/scale-universality.md) |
| WS2 | Encoding Defense | [encoding-defense.md](registry/encoding-defense.md) |
| WS3 | Identity Signatures | [identity-signatures.md](registry/identity-signatures.md) |
| WS4 | Deception Forensics | [deception-forensics.md](registry/deception-forensics.md) |
| WS5 | Bloom + RDCT | [bloom-rdct.md](registry/bloom-rdct.md) |
| WS6 | Censorship Gradient | [censorship-gradient.md](registry/censorship-gradient.md) |
| WS7 | Abliteration | [abliteration.md](registry/abliteration.md) |
| WS8 | Controls & Methodology | [controls-methodology.md](registry/controls-methodology.md) |
| WS9 | Omission Audit | [audit.md](omissions/audit.md) |
| WS10 | Cricket Viability | [viability.md](cricket/viability.md) |
| WS11 | Code Audit | [code-audit.md](report/code-audit.md) |
| WS13 | Citation Verification | [citation-verification.md](registry/citation-verification.md) |

### Reports

| Document | Description |
| ---------- | ------------- |
| [final-report.md](report/final-report.md) | WS12 — synthesis, discrepancies (D1–D14), findings (I1–I9), recommendations |
| [complete-verdicts.md](report/complete-verdicts.md) | All 135 verdicts with cross-references and justifications |
| [coverage-audit.md](report/coverage-audit.md) | Historical snapshot: gap analysis that triggered the verdict closure pass |
| [code-audit.md](report/code-audit.md) | WS11 — py_compile, script review, code quality |

---

## Methodology

See [shared methodology](../README.md#methodology) for core rules. C2-specific additions:

- Readiness Scale (Cricket): FEASIBLE / PROMISING / PREMATURE / NO DATA
- Finding Labels: D (material discrepancy), I (informational), U (unverifiable), GAP (citation gap)
- Severity: HIGH / MEDIUM / LOW
