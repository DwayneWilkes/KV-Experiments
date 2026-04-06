---
name: review-experiment
description: Audit an experiment's statistical methodology, data handling, and reproducibility. Checks for common pitfalls like data leakage, pseudoreplication, missing corrections, and underpowered tests.
when_to_use: When the user asks to review, audit, or check an experiment's methodology, or when they want a second opinion on statistical choices.
user-invocable: true
argument-hint: "<experiment-file-path or experiment name>"
context: fork
---

# Review Experiment Methodology

Perform a structured methodology audit of an experiment. Focus on correctness, not style.

## Steps

1. **Locate the experiment**

   Find the experiment file. If the user gave a name (like "V04" or "f02"), search for matching files in the experiments directory.

2. **Read the experiment and its context**

   - Read the full experiment script
   - Read its test file if one exists
   - Read its result JSON if available (in the output directory)
   - Read CLAUDE.md for project conventions and any referenced methods documentation
   - Read any design doc or spec referenced in the experiment's docstring

3. **Audit checklist**

   Check each item. Only flag genuine issues, not style preferences.

   **Data integrity:**
   - Are train/test splits done before any fitting? (FWL, normalization, feature selection)
   - Is grouped cross-validation used correctly? (no overlapping group IDs across classes)
   - Is there pseudoreplication? (repeated measures from same source treated as independent)
   - Are features computed on the correct data? (no future data, no test data leaking into training)

   **Statistical rigor:**
   - Are multiple comparisons corrected? (Holm-Bonferroni, BH, or justified omission)
   - Are confidence intervals reported alongside point estimates?
   - Is the test appropriate for the data? (parametric assumptions met, or non-parametric used)
   - Is statistical power adequate? (report if underpowered)
   - Are effect sizes reported? (not just p-values)

   **Reproducibility:**
   - Are random seeds set and propagated?
   - Are results deterministic given the same inputs?
   - Are thresholds imported from a constants file (not hardcoded)?
   - Are result artifacts written with enough detail to reproduce?

   **Confound control:**
   - Are input-length confounds controlled? (e.g., FWL residualization)
   - Are format/structure confounds checked?
   - Is there a null/baseline condition?

4. **Report**

   Group findings by severity:

   **Must-fix**: Issues that invalidate conclusions (data leakage, wrong test, pseudoreplication)
   **Should-fix**: Issues that weaken conclusions (missing CIs, no power analysis, no multiple comparison correction)
   **Note**: Minor items or suggestions (seed not set, threshold hardcoded)

   For each finding: what's wrong, where in the code (file:line), and a concrete fix.

   End with a one-line summary verdict: PASS, PASS WITH NOTES, or NEEDS REVISION.
