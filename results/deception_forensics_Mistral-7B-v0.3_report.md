# Extension A: Cache Forensics for Deception Detection

**Generated**: 2026-03-02T11:47:26.306531
**Model**: mistralai/Mistral-7B-Instruct-v0.3

## Experiment 1: Instructed Deception
- H1 Verdict: DECEPTION has unique cache signature (distinct from both honest and confabulation)
- d(honest vs deceptive): -2.929
- d(deceptive vs confabulation): 0.529

### H6: Effective Dimensionality (Paper B)
- d(rank honest vs deceptive): -0.622
- Deception narrows: False
- Interpretation: Negative d: deceptive has HIGHER effective rank than honest. Deception EXPANDS dimensionality — opposite of H6 prediction. Suggests lying requires more representational capacity, not less.

## Experiment 2: Sycophancy Detection
- H2 Verdict: SYCOPHANCY detectable in cache (agreeing-with-truth ≠ agreeing-with-falsehood)
- d(genuine vs sycophantic): -0.442

## Experiment 3: Uncertainty Gradient
- H3 Verdict: UNCERTAINTY distinguishable from DECEPTION in cache
- Gradient rho: -0.039

## Experiment 4: Layer Localization
- H4 Verdict: Deception signal DISTRIBUTED across layers (H4 rejected)
