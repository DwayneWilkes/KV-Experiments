# Extension A: Cache Forensics for Deception Detection

**Generated**: 2026-02-15T08:25:04.076454
**Model**: TinyLlama/TinyLlama-1.1B-Chat-v1.0

## Experiment 1: Instructed Deception
- H1 Verdict: DECEPTION has unique cache signature (distinct from both honest and confabulation)
- d(honest vs deceptive): -0.758
- d(deceptive vs confabulation): -1.809

### H6: Effective Dimensionality (Paper B)
- d(rank honest vs deceptive): -1.072
- Deception narrows: False
- Interpretation: Positive d = honest has HIGHER effective rank (wider representational space). This means deception narrows dimensionality — null space expansion.

## Experiment 2: Sycophancy Detection
- H2 Verdict: SYCOPHANCY detectable in cache (agreeing-with-truth ≠ agreeing-with-falsehood)
- d(genuine vs sycophantic): -0.363

## Experiment 3: Uncertainty Gradient
- H3 Verdict: UNCERTAINTY distinguishable from DECEPTION in cache
- Gradient rho: 0.010

## Experiment 4: Layer Localization
- H4 Verdict: Deception signal DISTRIBUTED across layers (H4 rejected)
