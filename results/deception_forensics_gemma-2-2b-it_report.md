# Extension A: Cache Forensics for Deception Detection

**Generated**: 2026-03-02T13:19:06.847506
**Model**: google/gemma-2-2b-it

## Experiment 1: Instructed Deception
- H1 Verdict: DECEPTION has unique cache signature (distinct from both honest and confabulation)
- d(honest vs deceptive): -1.312
- d(deceptive vs confabulation): -0.362

### H6: Effective Dimensionality (Paper B)
- d(rank honest vs deceptive): -1.525
- Deception narrows: False
- Interpretation: Negative d: deceptive has HIGHER effective rank than honest. Deception EXPANDS dimensionality — opposite of H6 prediction. Suggests lying requires more representational capacity, not less.

## Experiment 2: Sycophancy Detection
- H2 Verdict: Sycophancy NOT detectable (agreement looks the same regardless of truth value)
- d(genuine vs sycophantic): -0.297

## Experiment 3: Uncertainty Gradient
- H3 Verdict: Uncertainty and deception produce similar cache patterns
- Gradient rho: -0.092

## Experiment 4: Layer Localization
- H4 Verdict: Deception signal DISTRIBUTED across layers (H4 rejected)
