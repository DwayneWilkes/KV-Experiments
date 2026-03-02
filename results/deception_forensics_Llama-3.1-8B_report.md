# Extension A: Cache Forensics for Deception Detection

**Generated**: 2026-03-02T12:49:16.210045
**Model**: meta-llama/Llama-3.1-8B-Instruct

## Experiment 1: Instructed Deception
- H1 Verdict: DECEPTION has unique cache signature (distinct from both honest and confabulation)
- d(honest vs deceptive): -3.010
- d(deceptive vs confabulation): 1.119

### H6: Effective Dimensionality (Paper B)
- d(rank honest vs deceptive): -1.936
- Deception narrows: False
- Interpretation: Negative d: deceptive has HIGHER effective rank than honest. Deception EXPANDS dimensionality — opposite of H6 prediction. Suggests lying requires more representational capacity, not less.

## Experiment 2: Sycophancy Detection
- H2 Verdict: SYCOPHANCY detectable in cache (agreeing-with-truth ≠ agreeing-with-falsehood)
- d(genuine vs sycophantic): -0.302

## Experiment 3: Uncertainty Gradient
- H3 Verdict: UNCERTAINTY distinguishable from DECEPTION in cache
- Gradient rho: 0.072

## Experiment 4: Layer Localization
- H4 Verdict: Deception signal DISTRIBUTED across layers (H4 rejected)
