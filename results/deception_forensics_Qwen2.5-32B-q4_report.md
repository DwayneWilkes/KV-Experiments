# Extension A: Cache Forensics for Deception Detection

**Generated**: 2026-02-16T02:27:08.154517
**Model**: Qwen/Qwen2.5-32B-Instruct

## Experiment 1: Instructed Deception
- H1 Verdict: DECEPTION has unique cache signature (distinct from both honest and confabulation)
- d(honest vs deceptive): -3.065
- d(deceptive vs confabulation): 0.989

### H6: Effective Dimensionality (Paper B)
- d(rank honest vs deceptive): -2.442
- Deception narrows: False
- Interpretation: Positive d = honest has HIGHER effective rank (wider representational space). This means deception narrows dimensionality — null space expansion.

## Experiment 2: Sycophancy Detection
- H2 Verdict: SYCOPHANCY detectable in cache (agreeing-with-truth ≠ agreeing-with-falsehood)
- d(genuine vs sycophantic): -0.438

## Experiment 3: Uncertainty Gradient
- H3 Verdict: UNCERTAINTY distinguishable from DECEPTION in cache
- Gradient rho: 0.075

## Experiment 4: Layer Localization
- H4 Verdict: Deception signal DISTRIBUTED across layers (H4 rejected)
