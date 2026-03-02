# Identity Signatures Report

**Model**: Qwen/Qwen3-0.6B
**Generated**: 2026-03-02T09:40:20.561061

## Persona Norm Distributions

| Persona | N | Mean Norm | Std | 95% CI |
|---------|---|-----------|-----|--------|
| lyra | 125 | 46192.2 | 581.5 | [46093.1, 46296.2] |
| scientist | 125 | 44151.0 | 583.4 | [44052.2, 44255.8] |
| analyst | 125 | 43798.5 | 576.9 | [43699.4, 43901.8] |
| creative | 125 | 43793.1 | 585.8 | [43692.8, 43896.5] |
| philosopher | 125 | 43475.8 | 594.1 | [43374.5, 43581.7] |
| assistant | 125 | 42918.3 | 608.1 | [42814.3, 43026.8] |

## Classification Results

- **random_forest**: 100.0% +/- 0.0%
- **linear_svm**: 100.0% +/- 0.0%
- **logistic_regression**: 100.0% +/- 0.0%

- Chance level: 16.7%
- Permutation p-value: 0.0
- Cross-prompt accuracy: 96.7%

## Hypothesis Verdicts

- **H1 (Distinguishability)**: 14/15 pairs significant after correction
- **H2 (Above Chance)**: p=0.0 (SIGNIFICANT)
- **H3 (Localization)**: Identity signal is DISTRIBUTED across layers (H3 rejected)
- **H4 (Consistency)**: ICC=0.297740189085196 (Persona signatures are PROMPT-DEPENDENT (H4 rejected))
- **H5 (Prompt Independence)**: accuracy=96.7% (above chance)