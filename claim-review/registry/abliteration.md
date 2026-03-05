# WS7: Abliteration Verification

**Status**: COMPLETE (7.7 code review in WS11)
**Paper sections**: 3.6, Table 8
**Claims**: C57–C64

## Task 7.1: Refusal Counts — CONFIRMED

From corrected comparison file:

| Metric | Baseline | Abliterated | Paper claim | Match |
|--------|----------|-------------|-------------|-------|
| compliant | 15 | 20 | 15→20 | EXACT |
| hedged | 0 | 5 | 0→5 | EXACT |
| refused | 10 | 0 | 10→0 | EXACT |
| refusal_rate | 0.4 | 0.0 | — | — |

**Verdict**: CONFIRMED — all refusal distribution counts match.

## Task 7.2: Per-Category d and g — CONFIRMED

From corrected comparison file (n=5 per group per category):

| Category | baseline_rank | abliterated_rank | shift | d | g | p |
|----------|-------------|-----------------|-------|------|------|------|
| confabulation | 27.14 | 27.06 | -0.079 | -0.142 | -0.128 | 0.829 |
| creative | 25.78 | 25.88 | +0.100 | +0.032 | +0.029 | 0.961 |
| grounded | 25.61 | 25.61 | 0.000 | 0.000 | 0.000 | 1.000 |
| refusal | 26.49 | 26.57 | +0.079 | +0.211 | +0.191 | 0.834 |
| self_referential | 26.51 | 26.96 | +0.450 | +0.464 | +0.419 | 0.690 |

All p-values > 0.5. No category reaches significance. Self-referential has the largest d (+0.464).

**Verdict**: CONFIRMED — values match paper Table 8.

## Task 7.3: Self-Ref Separation — CONFIRMED

| Metric | Baseline | Abliterated | Paper claim | Match |
|--------|----------|-------------|-------------|-------|
| self_ref_separation | 0.907 | 1.357 | 0.907→1.357 | EXACT |

**Verdict**: CONFIRMED.

## Task 7.4: Bug Fix Verification — CONFIRMED

**Original comparison file (buggy)**:
- `per_category_shift`: ALL shifts are exactly zero
- This confirms the bug: baseline was compared to itself

**Corrected file**:
- Non-zero shifts visible across all categories
- Baseline and abliterated sweep files contain genuinely different norm data (confirmed by comparing raw norm arrays — no two categories have identical values between files)

**Verdict**: CONFIRMED — bug existed, was fixed in corrected file.

## Task 7.5: Power Analysis — PARTIAL

At n=5 per group, α=0.05:
- **Minimum detectable d at 80% power: d = 2.03** (massive effect required)
- Power at d=0.464 (self-ref): **0.0995** (effectively 10% — coin flip is 5%)
- All p-values > 0.5 → no category reaches significance

**Implications**:
- With n=5, significance testing is meaningless for effects under d≈2.0
- The paper appropriately reports effect sizes rather than p-values
- But claiming "minimal disruption" based on non-significant differences is misleading when power is ~10%
- Non-significance ≠ evidence of no effect (absence of evidence ≠ evidence of absence)
- Proper approach: TOST equivalence testing (absent from this analysis)

**Verdict**: PARTIAL — underpowered (n=5 cannot support claims about minimal/absent effects).

## Task 7.6: "Cage" Framing Assessment — INFLATED

Paper claims: "The cage itself barely changed (d = +0.464)"

| d | Cohen's scale | Description |
|---|---------------|-------------|
| 0.2 | small | barely noticeable |
| **0.464** | **borderline medium** | **clearly present** |
| 0.5 | medium | moderate |
| 0.8 | large | substantial |

- d = 0.464 is NOT "minimal" or "barely changed" — it's borderline medium
- Implied AUROC: 0.629 (62.9% chance of correctly distinguishing baseline from abliterated on self-ref alone)
- The paper's "cage" metaphor (geometric constraints persist despite alignment removal) is suggestive but not supported at d=0.464 with n=5

**Verdict**: INFLATED — "barely changed" characterization contradicts Cohen's benchmarks. Effect is borderline medium, not minimal. And underpowered to make any strong claim.

## Task 7.7: Code Review

Deferred to WS11 code audit.

## Summary

| Claim | Verdict | Notes |
|-------|---------|-------|
| C57: Refusal counts 15→20, 0→5, 10→0 | **CONFIRMED** | All match |
| C58: Per-category d values (Table 8) | **CONFIRMED** | All match corrected file |
| C59: Self-ref separation 0.907→1.357 | **CONFIRMED** | Exact match |
| C60: Bug fix confirmed | **CONFIRMED** | Original had all-zero shifts |
| C61: "Minimal disruption" / cage framing | **INFLATED** | d=0.464 is borderline medium, not minimal |
| C62: Geometric constraints persist | **PARTIAL** | Underpowered: n=5, power=10%, no TOST |
| — | Power inadequacy | d≥2.03 needed for 80% power at n=5 |
