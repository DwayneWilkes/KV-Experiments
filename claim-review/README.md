# KV-Cache Experiments — Claim Verification Hub

Adversarial verification of claims across all KV-cache experiment campaigns. Each audit is self-contained; shared infrastructure (statistics module, glossary, bibliography) lives at this level.

---

## Audits

| Audit | Scope | Claims | Status |
| ------- | ------- | -------- | -------- |
| [c2/](c2/) | Campaign 2 paper + Cricket | 135 (C1–C98, CC, CF, CL) | COMPLETE (2026-03-04) |
| [hackathon-iatf/](hackathon-iatf/) | [IATF Hackathon](https://luma.com/ftchack-sf-2026) (Funding the Commons) — Liberation Labs team, Exp 14–36 | 189 (H14–H16, EXP26–EXP36, CRICKET, CACHE, FRONT, PITCH, ADV, LOG) | IN PROGRESS |

---

## Shared Resources

| Resource | Description |
| ---------- | ------------- |
| [GLOSSARY.md](GLOSSARY.md) | 80+ cross-linked term definitions (statistics, ML, paper concepts, hackathon) |
| [sources.md](sources.md) | Cross-referenced bibliography: internal data, cited papers, missing literature, flags |
| [stats/](stats/) | Independent statistics module — recomputation library decoupled from experiment code |
| [IMPROVEMENTS.md](IMPROVEMENTS.md) | Future audit tooling improvements |

### Statistics Module (`stats/`)

| File | Description |
| ------ | ------------- |
| [independent_stats.py](stats/independent_stats.py) | Hedges' g, TOST, ANCOVA, bootstrap CI, power analysis, d→AUROC conversion |
| [test_independent_stats.py](stats/test_independent_stats.py) | 39 unit tests |
| [requirements.txt](stats/requirements.txt) | `numpy>=1.24`, `scipy>=1.11`, `pytest>=7.0` |

---

## Terminology

### Verdict Scales

**Data integrity audits** (numeric claim verification):

| Verdict | Meaning |
| --------- | --------- |
| **VERIFIED** | Value matches raw JSON within tolerance |
| **DISCREPANT** | Value does not match raw data |
| **CRITICAL** | Material error affecting downstream claims |
| **FALSIFIED** | Claim disproved by subsequent experiment |
| **OVERSTATED** | Technically correct but misleading |
| **STALE** | Correct when written but now outdated |
| **NO_DATA** | Cannot verify — data file missing |
| **FALSE_STATUS** | Log claims file exists but no file found |

**Methodology review** (scientific claim assessment):

| Verdict | Meaning |
| --------- | --------- |
| **CONFIRMED** | Claim supported by adequate power and correct methodology |
| **QUALIFIED** | Claim directionally correct but with caveats (imprecise CIs, single model, etc.) |
| **UNDERMINED** | Methodological issues weaken the claim substantially |
| **REFUTED** | Claim not supportable given sample size, power, or design |

**Paper claims** (C2 audit):

| Verdict | Meaning |
| --------- | --------- |
| **CONFIRMED** | Value matches raw data within tolerance |
| **PARTIAL** | Numbers correct but with caveats |
| **REJECTED** | Value contradicts raw data |
| **INFLATED** | Technically true but misleading characterization |
| **ARTIFACT** | Result is a methodological byproduct |
| **INVALID** | Experimental design cannot support the claim |
| **NEEDS INFO** | Cannot check without rerunning experiments |

### Claim ID Prefixes

| Prefix | Audit | Source |
| -------- | ------- | -------- |
| C | c2 | Campaign 2 paper (C1–C98) |
| CC, CF, CL | c2 | Cricket capability, cross-doc, competitive |
| H14–H16 | hackathon-iatf | Hackathon experiments 14–16 (report.md claims) |
| EXP26–EXP36 | hackathon-iatf | Hackathon experiments 26–36 |
| CRICKET | hackathon-iatf | JiminAI-Cricket org audit findings |
| CACHE | hackathon-iatf | CacheScope org audit findings |
| FRONT | hackathon-iatf | JiminAI-Frontend findings |
| PITCH | hackathon-iatf | PITCH_NUMBERS.md claims |
| ADV | hackathon-iatf | Adversarial / cross-cutting findings |
| LOG | hackathon-iatf | HACKATHON_LOG.md gap findings |

### Tolerances

| Metric | Tolerance | Notes |
| -------- | ----------- | ------- |
| Spearman rho | ±0.005 | |
| Cohen's d / Hedges' g | ±0.02 | |
| AUROC | ±0.005 | |
| Counts | exact | File counts, model counts, etc. |
| Rankings | exact | Category orderings, top-k |
| p-values | ±0.01 | |

### Glossary

Full definitions with cross-references: **[GLOSSARY.md](GLOSSARY.md)**

---

## Methodology

Core rules shared across all audits:

1. Recompute all statistics independently from raw JSON result files
2. Check every number against JSON, not markdown reports
3. Verify aggregations (means, totals) from per-model values
4. Every null claim must have TOST support, not just non-significant p
5. Flag interpretive overclaims even when numbers are correct
6. Cross-reference all citations against published databases
7. Map every result file to a paper section; flag unreported data

**Anti-info-poisoning** (methodology review only): Blind recomputation agents receive raw data WITHOUT claimed values. Comparison happens only after independent computation is complete.
