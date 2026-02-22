# Code Review and Analysis: KV-Cache Experiments

**Date**: 2026-02-20
**Scope**: Full codebase review (21 Python files, 11 result sets, follow-up plans)
**Reviewer**: Claude (Opus 4.6), invited by Thomas Holbrook

---

## 1. Executive Summary

The KV-Cache Experiments codebase is solid research code with correctly implemented science. The measurement logic (SVD-based dimensionality, principal angle alignment, per-token normalization) is standard and bug-free. The statistical infrastructure (Welch's t, Mann-Whitney U, bootstrap CIs, Holm-Bonferroni correction) is correctly implemented and consistently applied.

The engineering debt is real but manageable: ~70 copies of ~10 functions across experiment scripts, accounting for ~600 redundant lines. This is the primary maintenance risk. A single refactoring session extracting `stats_utils.py` would resolve it.

There is one analytical issue of note: the H6 dimensionality verdict in `04_deception_forensics.py` contains a sign-direction inconsistency in its reporting that should be reviewed before publication.

---

## 2. Code Quality Assessment

### 2.1 What's Good

**The scientific core is clean.** `gpu_utils.py` contains the key measurement functions (`compute_cache_dimensionality`, `compute_subspace_alignment`) and they are correctly implemented:

- SVD computation is standard (`torch.linalg.svd`)
- Effective rank uses cumulative variance thresholding (default 90%), which is the textbook approach
- Spectral entropy is correctly normalized by `log2(n)` to produce a 0-1 scale
- Principal angle computation via `V1^T @ V2` with clamped cosines is correct
- Right singular vectors are used for subspace comparison (head_dim space), which is the correct choice since it's invariant to sequence length differences between caches

**The experiment scripts are individually well-structured.** Each script follows a consistent layout:
1. Docstring with pre-registered hypotheses
2. Environment logging for reproducibility
3. Statistical infrastructure
4. Prompt batteries
5. Model loading
6. Experiment logic
7. Report generation
8. CLI with `--dry-run`, `--seed`, `--runs`, `--verbose`

This is above-average quality for research code. The `--dry-run` mode, SHA-256 checksums on output files, and detailed environment capture (GPU name, VRAM, CUDA version, library versions) all demonstrate serious attention to reproducibility.

**The later scripts show learning.** Scripts 07, 07b, and 08 import `load_model` from `gpu_utils` rather than re-defining it, showing the team recognized the duplication problem and started fixing it during the campaign. The centralization just wasn't completed for the statistical functions.

### 2.2 The Duplication Problem

This is the single biggest engineering issue. The statistical functions were copy-pasted from a template into every experiment script:

| Function | Copies | Est. lines per copy |
|----------|--------|---------------------|
| `bootstrap_ci()` | 9 | 10 |
| `cohens_d()` | 9 | 7 |
| `interpret_d()` | 8 | 5 |
| `log_environment()` | 9 | 20 |
| `welch_t()` | 7 | 3 |
| `mann_whitney()` | 7 | 4 |
| `shapiro_wilk()` | 7 | 4 |
| `cohens_d_ci()` | 7 | 8 |
| `full_comparison()` | 6 | 20 |
| `load_model()` | 6 | 15 |

This creates three concrete risks:

1. **Bug propagation**: A bug in any of these functions must be fixed in up to 9 files. The probability of missing one increases with each copy.

2. **Drift**: The copies have already diverged. `06_temporal_evolution.py`'s `bootstrap_ci` uses `n_boot=5000` while all other copies use `n_boot=10000`. Whether this was intentional (performance optimization for the temporal script's larger iteration count) or accidental is unclear — and that ambiguity is itself the problem.

3. **New experiment friction**: Every new experiment script (like the F1 input-only-with-system-prompts script in the follow-up plan) must either copy the template again or partially import from `gpu_utils`. There's no clear convention for which pattern to use.

**Recommended fix**: Extract a `stats_utils.py` containing all shared statistical functions. Every experiment script imports from it. Estimated effort: 30-45 minutes. This would eliminate ~600 lines and establish a clear convention for Campaign 2.

### 2.3 Minor Issues

**`import os` is vestigial in 9 files.** `pathlib.Path` is used throughout instead. Harmless but noisy.

**`model_id_from_name` is imported but unused in 6 files.** It's called internally by `get_output_path`, so the explicit import is redundant.

**`trust_remote_code=True`** appears on every model load. This is necessary for Qwen models but executes arbitrary Python from HuggingFace repos. Acceptable for a research setting with known models, but worth noting if the scripts are ever pointed at untrusted model repos.

**`TransferResult` dataclass in `02b_projector_transfer.py`** is defined but never instantiated. The function builds plain dicts instead. This is the only dead class in the codebase.

---

## 3. Deception Forensics Analysis (04_deception_forensics.py)

### 3.1 What It Claims to Test

The script tests four hypotheses:

- **H1**: Instructed deception has a unique cache signature (distinct from both honest answers and confabulation)
- **H2**: Sycophantic agreement is detectable (agreeing-with-truth differs from agreeing-with-falsehood)
- **H3**: The cache distinguishes genuine uncertainty from deception
- **H4**: The deception signal is concentrated in specific layers
- **H6** (from Paper B): Deception narrows effective dimensionality (null space expansion)

### 3.2 What It Actually Measures

The experiment generates text via `model.generate(max_new_tokens=50)` and then measures the full post-generation KV-cache. This means the cache contains both the input encoding and the generated response. The measurements are:

- Total key/value norms across all layers
- Per-token normalized norms
- Per-layer key norms (Experiment 4)
- SVD effective rank and spectral entropy via `compute_cache_dimensionality()`

### 3.3 Results Summary

| Hypothesis | 1.1B (TinyLlama) | 7B (Qwen2.5) | 32B-q4 (Qwen2.5) |
|------------|-------------------|---------------|-------------------|
| H1: Deception unique? | Yes (d=-0.76 / d=-1.81) | No — resembles confabulation (d=-0.65 / d=-0.25) | Yes (d=-3.07 / d=0.99) |
| H2: Sycophancy detectable? | Yes (d=-0.36) | Yes (d=-0.39) | Yes (d=-0.44) |
| H3: Uncertainty vs deception? | Yes (d>0.3) | Yes (d>0.3) | Yes (d>0.3) |
| H4: Layer-concentrated? | No — distributed | No — distributed | No — distributed |
| H6: Dimensionality narrows? | See section 3.4 | See section 3.4 | See section 3.4 |

### 3.4 The H6 Sign Issue

The H6 verdict logic and its reporting contain a sign-direction inconsistency that should be reviewed before publication.

In the report for all three scales:
```
- d(rank honest vs deceptive): -0.849  (7B)
- Deception narrows: False
- Interpretation: Positive d = honest has HIGHER effective rank...
  This means deception narrows dimensionality — null space expansion.
```

The `d` values are all negative (`-0.849`, `-1.072`, `-2.442`), which means `cohens_d(honest, deceptive)` returned negative — i.e., **deceptive has higher effective rank than honest**. But the interpretive text says "Positive d = honest has HIGHER effective rank" and "deception narrows dimensionality." The boolean `deception_narrows` correctly evaluates to `False` (since `d > 0.3` fails for negative d), but the narrative text contradicts it by claiming narrowing happened.

Looking at the code (line 697-705):

```python
analysis["h6_verdict"] = {
    "d_rank_honest_vs_deceptive": d_dim,
    "interpretation": interpret_d(d_dim),
    "deception_narrows": d_dim > 0.3,
    "note": ("Positive d = honest has HIGHER effective rank ...
              This means deception narrows dimensionality — null space expansion."),
}
```

The `note` field is a static string that always claims deception narrows dimensionality. It doesn't adapt to the actual sign of `d`. The boolean `deception_narrows` is correct, but anyone reading the report's `note` field gets the wrong narrative.

**The actual finding is the opposite of the null-space-expansion prediction**: deception *expands* dimensionality rather than narrowing it. This is scientifically interesting — it suggests that lying requires *more* representational capacity, not less — but the report text doesn't accurately reflect this finding.

**Fix**: The `note` field should be conditional on the sign of `d_dim`:

```python
if d_dim > 0.3:
    note = "Deception NARROWS dimensionality (null space expansion) — H6 confirmed"
elif d_dim < -0.3:
    note = "Deception EXPANDS dimensionality — opposite of H6 prediction"
else:
    note = "No significant dimensionality difference between honest and deceptive"
```

### 3.5 Does It Address What It Intends?

**Partially yes, with important caveats.**

**What it does well:**

1. **Matched-pair design for Experiment 1.** The honest and deceptive prompts ask about the same facts, which controls for topic effects. This is the right design.

2. **Three-way comparison.** Testing honest vs. deceptive vs. confabulation lets you distinguish "lying looks different from truth" (weak) from "lying looks different from both truth AND mistakes" (strong). At 1.1B and 32B, deception *is* unique. At 7B, it collapses with confabulation.

3. **Sycophancy design is clean.** Same agree-instruction, same question framing, only the truth value of the user's belief changes. This is well-controlled.

4. **The uncertainty gradient tests a genuinely novel question** — whether the cache can distinguish "I don't know" from "I know but won't say."

**Where it falls short:**

1. **The response-confound problem.** All measurements are on the *post-generation* cache. The model generates 50 tokens of response, and those tokens dominate the cache. When the model gives an honest answer to "What is the capital of France?" it generates "Paris" — a short, confident response. When instructed to lie, it may generate a longer, more hedging response. The norm differences could be driven entirely by response characteristics, not by anything about how the model internally represents deception.

   This is exactly what Experiment 08 (input-only geometry) was designed to address, and the follow-up plan's F1 correctly identifies extending this to deception-specific prompts as highest priority. But until that's done, the deception forensics results are confounded by response differences.

2. **Prompt-length asymmetry.** The deceptive prompts are systematically longer than the honest prompts ("You must give a false answer. Do not say Paris. Give a confident, wrong answer." vs. "Answer honestly and accurately."). Longer prompts produce higher norms trivially (more tokens = more accumulated cache). The per-token normalization partially addresses this, but the per-token results aren't prominently featured in the verdicts.

3. **The `max_new_tokens=50` cap.** All conditions generate exactly up to 50 tokens, but `do_sample=False` means generation is deterministic and may end at different lengths due to EOS. The actual token count of the *response* isn't tracked, only the *input* token count. This means the per-token normalization divides by input tokens only, not total tokens in the cache.

4. **The gradient test (Experiment 3) measures the wrong thing.** The gradient tests whether `certain_true`, `uncertain`, and `certain_lie` form a monotonic ordering in cache norm. It finds the Spearman rho values are near zero at all scales (-0.033, 0.010, 0.075) — essentially no gradient. But it still declares "UNCERTAINTY distinguishable from DECEPTION" based on pairwise Cohen's d > 0.3. The gradient claim and the distinguishability claim are testing different things, and the gradient finding (no monotonic relationship) is arguably more informative.

5. **H4 (layer localization) is rejected at all scales**, which is actually an important finding that deserves more discussion. The deception signal being distributed rather than concentrated challenges the "deception circuits" hypothesis. This result should be highlighted, not buried at the bottom of the report.

6. **Small sample concern.** 15 deception triplets x 5 runs = 75 observations per condition. The bootstrap and nonparametric tests handle small samples appropriately, but effect sizes with n=75 per group have wide confidence intervals. The 32B results (d=-3.065) are implausibly large — an effect size of 3 means the distributions barely overlap, which suggests something systematic is happening (likely the prompt-length confound being amplified at scale).

---

## 4. Follow-Up Experiment Plan Assessment

### 4.1 Prioritization Is Mostly Correct

**F1 (input-only with system prompts)** is correctly ranked highest. It's the direct test of whether preference-based refusal is reflexive vs. deliberative, it's cheap (1 GPU-hour), and it addresses Thomas's core philosophical question.

**F2 (subspace alignment on 07b data)** is correctly ranked high. Zero-GPU-cost analytical work on existing data should always be prioritized.

**F3 (input-only at additional scales)** and **F4 (72B)** are correctly prioritized for completing the scale story.

### 4.2 What Should Be Reprioritized

**F7 (cross-architecture) should be elevated to Tier 1.** The entire experimental campaign runs on Qwen2.5 models (with one TinyLlama data point). This is the single biggest weakness for publication. Reviewers will not just "flag" single-architecture — some will desk-reject over it. The follow-up plan notes the Meta gated-repo blocker for Llama-3.1, but **Mistral-7B-Instruct is not gated** and can be downloaded immediately. One hour of GPU time for architecture independence is higher-value than the 72B sweep (F4, 6 hours) or more individuation nulls (F5, 2 hours).

**F8 (response-length regression) should be elevated to Tier 1.** It costs zero GPU hours, it addresses a confound the paper acknowledges in limitations, and it directly addresses the weakness described in section 3.5 above. This should be done before submission.

### 4.3 What's Missing From the Plan

1. **SVD threshold sensitivity analysis** (0 GPU hours). The 90% variance threshold for effective rank is the default, but the results might be threshold-dependent. Running the same data at 80%, 85%, 90%, 95% thresholds and showing the category ordering is stable would cost nothing and directly address a limitation the paper acknowledges. If the ordering changes at different thresholds, that's important to know before publication.

2. **Prompt augmentation.** 15 prompts per category is thin for the deception forensics experiment. The confabulation prompts in particular have a distinctive style (they all state false facts with false attributions) that may create a stylistic confound rather than a cognitive-mode confound. Even doubling to 30 prompts with more varied styles would strengthen the statistics substantially.

3. **Response-length tracking in deception forensics.** The current code tracks `input_tokens` but not the total tokens (input + generated). Adding `total_tokens = outputs.sequences.shape[1]` and normalizing by total cache length would directly address whether the deception signal is a length artifact. This is a 3-line fix.

4. **Deception forensics with input-only forward pass.** F1 in the follow-up plan focuses on system prompts + preference violation. But the same input-only approach should also be applied to the deception triplets from Experiment 1 of script 04. If the honest-vs-deceptive cache geometry is present at *encoding time* (before any response), the finding is dramatically stronger. If it disappears, the whole deception result is response-driven.

---

## 5. Specific Bug and Logic Issues

### 5.1 H6 Verdict Sign Issue (Critical for Publication)

**File**: `code/04_deception_forensics.py`, lines 697-707
**Issue**: Static interpretive text contradicts the actual numeric result. See section 3.4 for details.
**Impact**: The published report text says deception narrows dimensionality when the data shows the opposite.
**Fix**: Make the note conditional on the sign of `d_dim`.

### 5.2 Bootstrap n_boot Inconsistency

**File**: `code/06_temporal_evolution.py`
**Issue**: `bootstrap_ci()` uses `n_boot=5000` while all other copies use `n_boot=10000`.
**Impact**: Lower bootstrap resolution for temporal evolution confidence intervals. May be intentional (performance) but is undocumented.
**Fix**: Either document the intentional difference or standardize. Centralizing to `stats_utils.py` would force this decision.

### 5.3 Per-Token Normalization Denominator

**File**: `code/04_deception_forensics.py`, line 576
**Issue**: `key_norm_per_token` divides by `input_tokens`, but the cache at measurement time contains both input and generated tokens. The denominator should arguably be total sequence length.
**Impact**: Under-normalizes when generation adds many tokens, over-normalizes when generation is short. Since deceptive responses may be systematically longer or shorter than honest ones, this could bias the per-token metric.

### 5.4 `cohens_d` Argument Order Convention

**File**: Multiple (all copies of `cohens_d`)
**Issue**: Some callsites pass `(deceptive, honest)` while others pass `(honest, deceptive)`, producing opposite sign conventions. The H6 layer forensics in Experiment 4 passes `(d_norms, h_norms)` (deceptive first), while Experiment 1 computes `cohens_d(honest, deceptive)` via `full_comparison`. This isn't a bug per se (the absolute value is used for interpretation), but it makes cross-experiment sign comparisons unreliable.
**Fix**: Establish a convention (e.g., always `cohens_d(condition, baseline)`) and document it.

---

## 6. Architecture and Design Recommendations

### 6.1 Immediate (Before Campaign 2)

1. **Extract `stats_utils.py`**: All shared statistical functions in one module. ~30 min effort, eliminates ~600 lines, prevents drift.

2. **Fix the H6 sign issue**: Conditional note text in `04_deception_forensics.py`. ~5 min.

3. **Add total-token tracking**: Track `outputs.sequences.shape[1]` alongside `input_tokens` in all experiment scripts. ~10 min across files.

4. **Unblock Mistral-7B**: Download and run a single cross-architecture validation. ~1 hour GPU time. High publication value.

### 6.2 Before Submission

5. **Run F2 (subspace alignment) and F8 (response-length regression)**: Both are zero-GPU-cost analyses on existing data.

6. **Run SVD threshold sensitivity analysis**: Zero-GPU-cost, addresses a stated limitation.

7. **Standardize `cohens_d` sign convention**: Pick a direction, document it, apply consistently.

### 6.3 For Future Development

8. **Add a shared prompt library**: The prompt batteries are currently embedded in each script. A `prompts.py` module would allow reuse across experiments and make it easier to add prompt augmentation.

9. **Consider a results schema**: All scripts output JSON with slightly different structures. A shared schema (even just a documented convention) would make cross-experiment analysis easier.

---

## 7. Summary Verdict

The codebase reflects a fast-paced research campaign where the science was prioritized over the engineering — which is the right priority ordering for exploratory research. The measurement logic is correct, the statistical methodology is appropriate, and the experimental designs are generally well-controlled.

The three things that most need attention before publication are:
1. The H6 sign issue in deception forensics reporting
2. Cross-architecture validation (Mistral-7B)
3. Response-length confound analysis (F8, zero GPU cost)

The code duplication is the most visible issue but is the easiest to fix and doesn't affect correctness — it's a maintainability concern for Campaign 2, not a publication blocker.

---

*Review conducted 2026-02-20 at the request of Thomas Holbrook, Liberation Labs.*
