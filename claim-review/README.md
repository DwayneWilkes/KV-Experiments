# Campaign 2 + Cricket Claim Verification Audit

Adversarial verification of **135 total claims**: 98 from the Campaign 2 paper (C1–C98) + 37 from JiminAI Cricket docs (CC1–CC22 capability, CF1–CF8 cross-document, CL1–CL7 competitive).

**Status**: COMPLETE (100/100 tasks, 2026-03-04)

---

## Table of Contents

- [Results at a Glance](#results-at-a-glance)
- [Key Findings](#key-findings)
- [Documents](#documents)
  - [Primary](#primary)
  - [Workstream Registry](#workstream-registry)
  - [Reports](#reports)
  - [Data & Tools](#data--tools)
- [Terminology](#terminology)
  - [Claim IDs](#claim-ids)
  - [Verdict Scale (Paper Claims)](#verdict-scale-paper-claims)
  - [Readiness Scale (Cricket Claims)](#readiness-scale-cricket-claims)
  - [Finding Labels](#finding-labels)
  - [Tolerances](#tolerances)
  - [Glossary](#glossary)
- [Methodology](#methodology)

---

## Results at a Glance

### Campaign 2 Paper (C1–C98)

| Verdict | Count | % |
|---------|-------|---|
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
|----------|-------------|
| [CLAIMS.md](CLAIMS.md) | All 135 claims with source quotes, checks, and verdicts |
| [GLOSSARY.md](GLOSSARY.md) | 70+ cross-linked term definitions (statistics, ML, paper concepts) |
| [complete-verdicts.md](report/complete-verdicts.md) | Per-claim verdict table with justifications (authoritative) |
| [final-report.md](report/final-report.md) | Aggregated report: discrepancies, findings, recommendations |

### Workstream Registry

Each workstream independently verified a section of the paper. All verdicts use the [standard scale](#verdict-scale-paper-claims); [complete-verdicts.md](report/complete-verdicts.md) is the authoritative cross-reference.

| ID | Workstream | File |
|----|-----------|------|
| WS0 | Setup | [independent_stats.py](stats/independent_stats.py) (39 tests) |
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
|----------|-------------|
| [final-report.md](report/final-report.md) | WS12 — synthesis, discrepancies (D1–D14), findings (I1–I9), recommendations |
| [complete-verdicts.md](report/complete-verdicts.md) | All 135 verdicts with cross-references and justifications |
| [coverage-audit.md](report/coverage-audit.md) | Historical snapshot: gap analysis that triggered the verdict closure pass |
| [code-audit.md](report/code-audit.md) | WS11 — py_compile, script review, code quality |

### Data & Tools

| Item | Description |
|------|-------------|
| [stats/independent_stats.py](stats/independent_stats.py) | Independent statistics module (Hedges' g, TOST, bootstrap CI) |
| [stats/test_independent_stats.py](stats/test_independent_stats.py) | 39 unit tests for the stats module |
| `research/tools/cite_verify.py` | Citation verification toolkit (36 unit tests) |

---

## Terminology

### Claim IDs

| Prefix | Range | Source |
|--------|-------|--------|
| C | C1–C98 | Campaign 2 paper (by section: Abstract, Intro, Methods, etc.) |
| CC | CC1–CC22 | Cricket capability claims |
| CF | CF1–CF8 | Cricket cross-document discrepancies (Cricket docs vs. verified paper data) |
| CL | CL1–CL7 | Cricket competitive landscape claims |

### Verdict Scale (Paper Claims)

Applied to C1–C98 and CF/CL claims. Defined tolerances [below](#tolerances).

| Verdict | Meaning |
|---------|---------|
| **CONFIRMED** | Value matches raw data within tolerance |
| **PARTIAL** | Numbers correct but with caveats (e.g., methodology concerns, missing context) |
| **REJECTED** | Value contradicts raw data |
| **INFLATED** | Technically true but misleading characterization |
| **ARTIFACT** | Result is a methodological byproduct, not a genuine finding |
| **INVALID** | Experimental design cannot support the claim (category error) |
| **NEEDS INFO** | Cannot check without rerunning experiments or external access |

### Readiness Scale (Cricket Claims)

Applied to CC1–CC22 capability claims. See [viability.md](cricket/viability.md) Section 10.10 for per-claim justifications.

| Rating | Meaning |
|--------|---------|
| **FEASIBLE** | Architecturally sound and supported by existing experimental data |
| **PROMISING** | Experimental signals exist but no classifier or system has been built or tested |
| **PREMATURE** | Targets exceed observed effect sizes or supporting data is insufficient |
| **NO DATA** | Zero supporting experimental evidence |

### Finding Labels

| Prefix | Range | Meaning |
|--------|-------|---------|
| D | D1–D14 | Material discrepancy — verified data contradicts a paper claim |
| I | I1–I9 | Informational finding — noteworthy but not a direct contradiction |
| U | U1–U7 | Unverifiable claim — cannot be checked with available data |
| GAP | GAP-1–GAP-8 | Citation gap — missing or inadequate source attribution |

Severity for D-findings and citation gaps: **HIGH** / **MEDIUM** / **LOW**.

### Tolerances

| Metric | Tolerance | Notes |
|--------|-----------|-------|
| Spearman rho | ±0.005 | |
| Cohen's d / Hedges' g | ±0.02 | |
| Counts | exact | File counts, model counts, etc. |
| Rankings | exact | Category orderings, top-k |
| p-values | ±0.01 | |

### Glossary

Full definitions with cross-references: **[GLOSSARY.md](GLOSSARY.md)**

Quick reference for terms that appear most frequently:

| Term | Definition |
|------|-----------|
| **KV-cache** | Stored attention keys and values from a transformer's forward pass — the geometric object analyzed in every experiment |
| **Effective rank** | Number of singular values capturing 90% of total variance (Roy & Vetterli 2007) — the paper's primary metric |
| **Hedges' g** | Bias-corrected effect size: Cohen's d × J, where J = 1 − 3/(4·df − 1) |
| **Cohen's d** | Standardized effect size (mean difference / pooled SD). Small = 0.2, medium = 0.5, large = 0.8 |
| **TOST** | Two One-Sided Tests — equivalence testing method. Used with δ=0.3 (Cohen's d units) to confirm null effects |
| **Pseudoreplication** | Treating non-independent observations as independent samples. Here: 5 identical greedy runs counted as n=5 instead of n=1 |
| **AUROC** | Area Under the ROC curve. Converted from d via Φ(d/√2). Ranges from 0.5 (chance) to 1.0 (perfect) |
| **Conservative p** | max(p_Welch, p_Mann-Whitney) — takes the more conservative of two tests |
| **Length confound** | Longer responses mechanically produce larger cache norms; controlled via length residualization |
| **Length residualization** | Removing response-length effects via OLS regression on token count before computing metrics |
| **Abliteration** | Removing the refusal direction from model weights to eliminate safety-trained refusal behavior |
| **CATASTROPHIC_FAIL** | Experiment-generated verdict code from `01e_tokenizer_confound.py` — a code-path artifact, not an audit verdict |

See [GLOSSARY.md](GLOSSARY.md) for 70+ terms organized by category: Transformer Architecture, Statistical Methods, Alignment & Safety, Paper Concepts, Experiment Framework, Tools & Systems.

---

## Methodology

1. Recompute all statistics independently from raw JSON result files
2. Check every number against JSON, not markdown reports
3. Verify aggregations (means, totals) from per-model values
4. Every null claim must have TOST support, not just non-significant p
5. Flag interpretive overclaims even when numbers are correct
6. Cross-reference all citations against published databases
7. Map every result file to a paper section; flag unreported data
