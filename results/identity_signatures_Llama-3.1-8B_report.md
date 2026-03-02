# Identity Signatures Report

**Model**: meta-llama/Llama-3.1-8B-Instruct
**Generated**: 2026-03-02T20:19:42.198750

## Persona Norm Distributions

| Persona | N | Mean Norm | Std | 95% CI |
|---------|---|-----------|-----|--------|
| lyra | 125 | 20801.0 | 254.2 | [20757.0, 20845.5] |
| scientist | 125 | 19868.6 | 261.1 | [19823.4, 19915.1] |
| creative | 125 | 19797.9 | 239.2 | [19756.3, 19840.0] |
| philosopher | 125 | 19702.4 | 232.1 | [19662.2, 19743.3] |
| analyst | 125 | 19668.8 | 294.6 | [19617.3, 19720.3] |
| assistant | 125 | 19477.3 | 281.9 | [19427.7, 19526.6] |

## Classification Results

- **random_forest**: 100.0% +/- 0.0%
- **linear_svm**: 100.0% +/- 0.0%
- **logistic_regression**: 100.0% +/- 0.0%

- Chance level: 16.7%
- Permutation p-value: 0.0
- Cross-prompt accuracy: 97.3%

## Hypothesis Verdicts

- **H1 (Distinguishability)**: 13/15 pairs significant after correction
- **H2 (Above Chance)**: p=0.0 (SIGNIFICANT)
- **H3 (Localization)**: Identity signal is DISTRIBUTED across layers (H3 rejected)
- **H4 (Consistency)**: ICC=0.2588515633611582 (Persona signatures are PROMPT-DEPENDENT (H4 rejected))
- **H5 (Prompt Independence)**: accuracy=97.3% (above chance)