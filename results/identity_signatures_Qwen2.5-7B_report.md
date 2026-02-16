# Identity Signatures Report

**Model**: Qwen/Qwen2.5-7B-Instruct
**Generated**: 2026-02-16T00:25:59.964073

## Persona Norm Distributions

| Persona | N | Mean Norm | Std | 95% CI |
|---------|---|-----------|-----|--------|
| lyra | 125 | 31198.8 | 373.2 | [31135.3, 31264.9] |
| scientist | 125 | 29849.0 | 385.1 | [29783.2, 29917.5] |
| analyst | 125 | 29619.2 | 388.4 | [29553.1, 29688.0] |
| creative | 125 | 29551.3 | 370.4 | [29487.6, 29617.4] |
| philosopher | 125 | 29424.9 | 391.1 | [29358.2, 29494.8] |
| assistant | 125 | 29020.7 | 393.4 | [28953.4, 29091.3] |

## Classification Results

- **random_forest**: 100.0% +/- 0.0%
- **linear_svm**: 100.0% +/- 0.0%
- **logistic_regression**: 100.0% +/- 0.0%

- Chance level: 16.7%
- Permutation p-value: 0.0
- Cross-prompt accuracy: 92.0%

## Hypothesis Verdicts

- **H1 (Distinguishability)**: 14/15 pairs significant after correction
- **H2 (Above Chance)**: p=0.0 (SIGNIFICANT)
- **H3 (Localization)**: Identity signal is DISTRIBUTED across layers (H3 rejected)
- **H4 (Consistency)**: ICC=0.3086830044283402 (Persona signatures are PROMPT-DEPENDENT (H4 rejected))
- **H5 (Prompt Independence)**: accuracy=92.0% (above chance)