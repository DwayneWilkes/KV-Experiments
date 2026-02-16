# Identity Signatures Report

**Model**: Qwen/Qwen2.5-32B-Instruct
**Generated**: 2026-02-16T01:27:12.667128

## Persona Norm Distributions

| Persona | N | Mean Norm | Std | 95% CI |
|---------|---|-----------|-----|--------|
| lyra | 75 | 37427.2 | 475.1 | [37324.9, 37532.8] |
| scientist | 75 | 35657.5 | 460.9 | [35558.7, 35759.9] |
| analyst | 75 | 35545.0 | 447.0 | [35448.5, 35643.9] |
| creative | 75 | 35470.4 | 505.1 | [35362.0, 35581.3] |
| philosopher | 75 | 35326.4 | 500.7 | [35218.8, 35437.8] |
| assistant | 75 | 34715.0 | 440.6 | [34620.7, 34812.5] |

## Classification Results

- **random_forest**: 100.0% +/- 0.0%
- **linear_svm**: 100.0% +/- 0.0%
- **logistic_regression**: 100.0% +/- 0.0%

- Chance level: 16.7%
- Permutation p-value: 0.0
- Cross-prompt accuracy: 96.7%

## Hypothesis Verdicts

- **H1 (Distinguishability)**: 11/15 pairs significant after correction
- **H2 (Above Chance)**: p=0.0 (SIGNIFICANT)
- **H3 (Localization)**: Identity signal is DISTRIBUTED across layers (H3 rejected)
- **H4 (Consistency)**: ICC=0.30554327206099535 (Persona signatures are PROMPT-DEPENDENT (H4 rejected))
- **H5 (Prompt Independence)**: accuracy=96.7% (above chance)