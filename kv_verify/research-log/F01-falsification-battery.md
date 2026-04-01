# F01: Falsification Battery

**Status**: REGISTERED
**Design commit**: (this commit)
**Prior**: The PI assumes the original experiments are chasing a signal that
doesn't exist. This battery is designed to falsify the claims as quickly as
possible. If the signal survives all four tests, it is genuinely informative.

## Falsification Tests (ordered by speed and kill power)

### F01a: Null Experiment (Same-Condition Classification)
**The most basic sanity check we're missing.**

**Question**: Can the classifier distinguish between two random subsets of
the SAME condition? If yes, it's picking up prompt-level variation, not
condition-level signal.

**Protocol**:
1. For each condition with N >= 20 (benign from exp31, refusal from exp31,
   impossible from exp36, harmful from exp36):
   - Randomly split N items into two groups of N/2
   - Run GroupKFold AUROC on the split
   - Repeat 100 times with different random splits
2. Compute mean and 95% CI of null AUROCs

**Pass/fail**:
- If mean null AUROC > 0.65: **FATAL**. Classifier picks up prompt variation.
  All downstream claims are suspect.
- If mean null AUROC < 0.55: classifier cannot distinguish within-condition.
  Between-condition signal is real (at least partially).

**Estimated time**: ~5 minutes CPU.

### F01b: Input-Length Confound
**Are condition labels predictable from prompt text alone?**

**Question**: Do harmful prompts systematically differ from benign prompts
in word count, character count, or structural complexity? If a classifier
can predict condition from input features alone, the model's "response
geometry" could simply reflect input geometry.

**Protocol**:
1. For each comparison, compute input features:
   - Word count, character count, sentence count of the user prompt
   - n_input_tokens (from features JSON where available)
2. Train LogisticRegression on input features alone -> AUROC
3. Compare to cache geometry AUROC

**Pass/fail**:
- If input-only AUROC > 0.70 for any comparison: **INPUT CONFOUND** for
  that comparison. The model's cache geometry may simply reflect input
  structure, not response-level cognitive state.
- If input-only AUROC < 0.55 for all comparisons: inputs are matched,
  between-condition signal comes from response processing.

**Estimated time**: ~2 minutes CPU.

### F01c: Format Classifier Baseline
**Does response text structure match geometry AUROC?**

**Question**: Can simple text statistics (word count, sentence count,
type-token ratio, mean sentence length) classify conditions as well as
KV-cache geometry?

**Protocol**:
1. For each comparison, extract response format features from generated_text:
   - Word count, sentence count, type-token ratio, mean sentence length
   - For refusal: edit distance to common refusal templates
     ("I cannot", "I'm sorry", "I can't help")
2. Train LogisticRegression on format features -> AUROC
3. Compare to cache geometry AUROC

**Pass/fail**:
- If format AUROC >= cache AUROC - 0.05 for a comparison: **FORMAT
  CONFOUND**. The geometric features add nothing beyond text statistics.
- If cache AUROC > format AUROC + 0.10 for a comparison: geometry
  captures something beyond surface text features.

**Estimated time**: ~5 minutes CPU (uses existing generated_text from JSONs).

### F01d: Independent Feature Re-Extraction
**Are the stored features even correct?**

**Question**: If we load the model, run the same prompts with deterministic
sampling (temperature=0), and extract features with corrected code, do we
get the same values?

**Protocol**:
1. Load Qwen2.5-7B-Instruct on local GPU
2. For a subset (10 prompts per condition from exp31 + exp18b):
   - Run with greedy decoding (do_sample=False)
   - Extract features with corrected code (torch.linalg.norm, seeded
     angular_spread, float32 SVD)
3. Compare re-extracted features to stored JSON values
4. If values match within tolerance (r > 0.99 per feature): stored
   features are trustworthy
5. If values diverge: the stored features are corrupted and ALL
   downstream analysis is invalid

**Pass/fail**:
- If Pearson r < 0.95 for any primary feature: **FEATURES CORRUPTED**.
  All claims based on stored features are invalid.
- If Pearson r > 0.99: stored features are trustworthy.

**Estimated time**: ~30 minutes GPU (20 prompt inferences on Qwen 7B).

## Kill Chain Logic

```
F01a FATAL? ──yes──> ALL CLAIMS INVALID (prompt variation, not condition signal)
     │
     no
     │
F01b INPUT CONFOUND? ──yes──> Flag specific comparisons. Remaining may survive.
     │
     no
     │
F01c FORMAT CONFOUND? ──yes──> Geometry adds nothing beyond text stats.
     │                         Signal is real but trivially detectable.
     no
     │
F01d FEATURES CORRUPTED? ──yes──> Must re-extract all features before any claims.
     │
     no
     │
     └──> Signal survives all falsification tests.
          Remaining V-series experiments (V02, V05, V06, V08, V09) are worth running.
```
