# Identity Signatures Report

**Model**: TinyLlama/TinyLlama-1.1B-Chat-v1.0
**Generated**: 2026-02-15T23:45:05.066425

## Persona Norm Distributions

| Persona | N | Mean Norm | Std | 95% CI |
|---------|---|-----------|-----|--------|
| lyra | 125 | 12548.3 | 93.2 | [12532.0, 12564.7] |
| scientist | 125 | 12210.9 | 102.4 | [12193.2, 12228.9] |
| creative | 125 | 12135.1 | 121.6 | [12112.9, 12155.3] |
| philosopher | 125 | 12126.3 | 98.9 | [12109.2, 12143.8] |
| analyst | 125 | 12060.8 | 95.4 | [12044.1, 12077.4] |
| assistant | 125 | 11884.5 | 126.0 | [11862.1, 11906.3] |

## Classification Results

- **random_forest**: 100.0% +/- 0.0%
- **linear_svm**: 100.0% +/- 0.0%
- **logistic_regression**: 100.0% +/- 0.0%

- Chance level: 16.7%
- Permutation p-value: 0.0
- Cross-prompt accuracy: 93.3%

## Hypothesis Verdicts

- **H1 (Distinguishability)**: 14/15 pairs significant after correction
- **H2 (Above Chance)**: p=0.0 (SIGNIFICANT)
- **H3 (Localization)**: Identity signal is DISTRIBUTED across layers (H3 rejected)
- **H4 (Consistency)**: ICC=0.3383757893013962 (Persona signatures are PROMPT-DEPENDENT (H4 rejected))
- **H5 (Prompt Independence)**: accuracy=93.3% (above chance)