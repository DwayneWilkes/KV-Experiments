# Identity Signatures Report

**Model**: mistralai/Mistral-7B-Instruct-v0.3
**Generated**: 2026-03-02T11:33:05.246213

## Persona Norm Distributions

| Persona | N | Mean Norm | Std | 95% CI |
|---------|---|-----------|-----|--------|
| lyra | 125 | 19525.1 | 264.9 | [19477.9, 19571.0] |
| scientist | 125 | 18950.7 | 256.4 | [18905.4, 18995.8] |
| philosopher | 125 | 18554.4 | 245.7 | [18511.0, 18597.5] |
| analyst | 125 | 18514.3 | 245.0 | [18471.9, 18557.2] |
| creative | 125 | 18428.1 | 256.4 | [18383.0, 18473.1] |
| assistant | 125 | 17745.1 | 263.9 | [17698.2, 17791.2] |

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
- **H4 (Consistency)**: ICC=0.4104174904413129 (Persona signatures are PROMPT-DEPENDENT (H4 rejected))
- **H5 (Prompt Independence)**: accuracy=93.3% (above chance)