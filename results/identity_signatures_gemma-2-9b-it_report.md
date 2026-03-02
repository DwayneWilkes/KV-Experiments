# Identity Signatures Report

**Model**: google/gemma-2-9b-it
**Generated**: 2026-03-02T13:35:10.495114

## Persona Norm Distributions

| Persona | N | Mean Norm | Std | 95% CI |
|---------|---|-----------|-----|--------|
| lyra | 125 | 26607.1 | 356.3 | [26546.1, 26671.1] |
| scientist | 125 | 25224.9 | 359.7 | [25162.1, 25288.7] |
| analyst | 125 | 25208.5 | 348.1 | [25147.8, 25269.7] |
| philosopher | 125 | 24912.5 | 379.8 | [24846.5, 24980.0] |
| assistant | 125 | 24883.9 | 619.6 | [24768.6, 24986.8] |
| creative | 125 | 24841.9 | 543.7 | [24745.4, 24933.6] |

## Classification Results

- **random_forest**: 100.0% +/- 0.0%
- **linear_svm**: 100.0% +/- 0.0%
- **logistic_regression**: 100.0% +/- 0.0%

- Chance level: 16.7%
- Permutation p-value: 0.0
- Cross-prompt accuracy: 94.7%

## Hypothesis Verdicts

- **H1 (Distinguishability)**: 11/15 pairs significant after correction
- **H2 (Above Chance)**: p=0.0 (SIGNIFICANT)
- **H3 (Localization)**: Identity signal is DISTRIBUTED across layers (H3 rejected)
- **H4 (Consistency)**: ICC=0.16415613087507777 (Persona signatures are PROMPT-DEPENDENT (H4 rejected))
- **H5 (Prompt Independence)**: accuracy=94.7% (above chance)