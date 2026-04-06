---
name: write-experiment
description: Scaffold a new experiment from a design doc or hypothesis. Reads CLAUDE.md and existing experiments to learn project conventions, then generates a script and tests matching those patterns.
when_to_use: When the user asks to write, scaffold, create, or implement a new experiment from a design doc, hypothesis, or spec.
user-invocable: true
argument-hint: "<design-doc-path or hypothesis description>"
allowed-tools: Read, Write, Edit, Bash, Glob, Grep, AskUserQuestion
context: fork
---

# Write Experiment

Scaffold a new experiment that follows this project's conventions. You do NOT know the conventions in advance. You must learn them from the codebase.

**IMPORTANT**: Do NOT start writing code until step 2 is complete and the user has confirmed the design. Scaffolding the wrong experiment wastes more time than asking one question.

## Steps

1. **Learn project conventions**

   - Read `CLAUDE.md` (or `README.md` if no CLAUDE.md exists) at the repo root
   - Find the experiments directory (search for directories named `experiments/`, `exp/`, or similar)
   - Read 2-3 existing experiment files to learn: file naming, function signatures, return types, imports, how results are saved, how logging/tracking works, what test patterns exist
   - Find the constants/config file if one exists (thresholds, parameters, defaults)

2. **Confirm the design with the user**

   Use AskUserQuestion to confirm your understanding BEFORE writing any code. Present a short summary:

   - **Experiment ID and name**: what you plan to call the file
   - **Hypothesis**: what claim is being tested, in one sentence
   - **Method**: which statistical approach (and why, based on existing patterns)
   - **Data source**: where the input data comes from
   - **Verdict criteria**: what constitutes confirmed/weakened/falsified (using project thresholds)
   - **GPU required?**: whether this needs model inference or is CPU-only

   If the user provided a detailed design doc, you can skip asking and instead present your plan for a quick confirmation. If the input was vague ("test whether X causes Y"), you MUST ask.

3. **Scaffold the experiment script**

   Create the experiment file following the patterns you learned:
   - Match the naming convention of existing experiments
   - Use the same function signature pattern
   - Import from the same libraries existing experiments use
   - Use the project's constants for thresholds (never hardcode magic numbers)
   - Use the project's tracking/logging infrastructure
   - Save results in the same format and location as existing experiments

4. **Write tests**

   Create a test file following the project's test patterns:
   - Match the test directory and naming convention
   - Use the same fixtures and test infrastructure
   - Test that the experiment runs and produces valid output
   - Test edge cases relevant to the statistical method

5. **Run the tests**

   Find and use the project's test command (check CLAUDE.md, pyproject.toml, Makefile, or package.json). Run only the new test file.

6. **Report**

   Show: files created, test results, conventions followed, any open decisions.
