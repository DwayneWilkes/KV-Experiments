---
name: summarize-results
description: Load experiment result files, apply verdict criteria, and generate a cross-experiment comparison table with narrative synthesis. Works with any JSON/YAML result format.
when_to_use: When the user asks to summarize findings, compare results across experiments, generate a verdict table, or synthesize what the experiments show.
user-invocable: true
argument-hint: "<result-dir or experiment IDs to compare>"
context: fork
---

# Summarize Experiment Results

Synthesize findings across multiple experiments into a structured report.

## Steps

1. **Find results**

   - If the user specified a directory, scan it for result files (JSON, YAML)
   - If the user named specific experiments, find their output directories
   - If neither, find the project's results/output directory from CLAUDE.md or by searching for `output/`, `results/`, or similar

2. **Learn the project's verdict system**

   - Read CLAUDE.md for verdict criteria, threshold definitions, and result schemas
   - Read the constants/config file for threshold values
   - Read 1-2 result files to understand the schema (what fields exist, what metrics are reported)
   - If a types/models file defines verdict enums or result dataclasses, read it

3. **Extract metrics from each experiment**

   For each result file:
   - Experiment ID and description
   - Primary metric (AUROC, accuracy, effect size, etc.) with confidence interval
   - P-value and significance after any corrections
   - Power estimate if available
   - The experiment's own verdict if it recorded one
   - Any notable anomalies or edge cases

4. **Cross-experiment analysis**

   - Which experiments agree? Which contradict?
   - Are there shared confounds across experiments?
   - Do results replicate across conditions, models, or datasets?
   - What's the overall weight of evidence?

5. **Generate report**

   Structure:

   **Summary table**: One row per experiment. Columns: ID, description, primary metric + CI, p-value, verdict.

   **Key findings**: 3-5 bullet points capturing the most important conclusions.

   **What survives**: Which claims hold up after all tests?

   **What falls**: Which claims are falsified or fatally weakened?

   **Open questions**: What couldn't be resolved with available data?

   Write the report to a file if the user wants one, otherwise display inline.
