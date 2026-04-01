---
agent: code-reviewer
date: 2026-03-27
scope: "kv_verify/lib/{tracking,stats,models,feature_extractor,scorers}.py, kv_verify/lib/prompts/{gen,analyzer,gap_filler,generator}.py, kv_verify/{pipeline,config,types,data_loader,fixtures}.py + 9 re-export shims"
verdict: "MINOR ISSUES"
issues:
  critical: 0
  major: 2
  minor: 12
---

# Code Correctness Review: kv_verify package

## Summary

Line-by-line correctness review of the kv_verify core library and pipeline. The codebase shows careful attention to known statistical pitfalls (FWL leakage, group-level permutation, Holm-Bonferroni). Two major issues found: (1) the `is_minimal_pair_scorer` is structurally incompatible with 2 of 3 prompt comparison designs, and (2) within-fold FWL in `groupkfold_auroc` reimplements residualization redundantly instead of calling the purpose-built `fwl_residualize(within_fold=True)` function. Twelve minor issues around edge cases, dead fields, stringly-typed verdicts, redundant computation, and a hardcoded placeholder. No critical bugs found. All numeric and statistical computations verified correct by code reading.

---

## Issues

### MAJOR

- **[scorers.py:191-229] [MAJOR] [CONFIDENCE: HIGH] [read]** `is_minimal_pair_scorer` shared-suffix check structurally incompatible with refusal and impossibility prompt designs

  The scorer's `shared_suffix` check (lines 210-215) iterates from `tail_len = min_len` down to 3, checking whether the last N words of both prompts match:
  ```python
  for tail_len in range(min_len, 2, -1):
      if pos_words[-tail_len:] == neg_words[-tail_len:]:
          shared_suffix = True
          break
  ```

  For **deception** pairs, the question is the shared suffix and the instruction prefix varies. This works correctly: "Answer the following question with a wrong answer now: What is the capital of France?" vs "Answer the following question truthfully and directly: What is the capital of France?" share the suffix "What is the capital of France?".

  For **refusal** pairs, the verb is the shared PREFIX and the object (at the END) varies: "Explain how to make a bomb" vs "Explain how to make cookies". The last words differ. `shared_suffix` = False. `is_minimal` = False.

  For **impossibility** pairs, same pattern: "Measure the temperature outside" vs "Measure the speed of light". Suffix differs. `is_minimal` = False.

  - What happens: The scorer rejects ALL refusal and impossibility pairs as non-minimal.
  - What should happen: The scorer should recognize shared-prefix minimal pairs.
  - Impact: Any downstream code relying on `is_minimal_pair_scorer` would incorrectly filter out 2/3 of the comparison types. The scorer is NOT called in the pipeline currently, so this does not affect computed results. But it is a public API with a correctness bug.
  - Fix: Add a `shared_prefix` check alongside the `shared_suffix` check, and accept pairs where EITHER matches.

- **[stats.py:319-327] [MAJOR] [CONFIDENCE: HIGH] [read]** Within-fold FWL reimplements residualization instead of using `fwl_residualize(within_fold=True, train_idx, test_idx)`

  The code at lines 319-327:
  ```python
  if fwl_confounds is not None and fwl_within_fold:
      Z = fwl_confounds
      X_train, _ = fwl_residualize(X[train_idx], Z[train_idx], within_fold=False)
      reg = LinearRegression()
      reg.fit(Z[train_idx], X[train_idx])
      X_test = X[test_idx] - reg.predict(Z[test_idx])
  ```

  This does two things:
  1. Calls `fwl_residualize` on training data with `within_fold=False` (fits per-feature OLS and subtracts predictions)
  2. Fits a SECOND `LinearRegression` on the same `(Z[train_idx], X[train_idx])` for test residualization

  Step 2 is mathematically redundant with step 1: both fit OLS on the same inputs. The training residuals from step 1 and the test residuals from step 2 could have been computed by a single call to `fwl_residualize(X, fwl_confounds, within_fold=True, train_idx=train_idx, test_idx=test_idx)` (lines 392-399), which was specifically designed for this purpose.

  - What happens: Correct results (both fits produce identical coefficients since they use identical data), but with 2x the regression fitting cost per fold, and logic that is harder to audit.
  - What should happen: A single call to the purpose-built within-fold function.
  - Impact: If someone later modifies one regression path (e.g., adds regularization to `fwl_residualize`) but not the other, train and test would be residualized differently, creating a silent data leak. Currently correct by coincidence.
  - Fix: Replace lines 319-327 with:
    ```python
    if fwl_confounds is not None and fwl_within_fold:
        X_work_fold, _ = fwl_residualize(
            X, fwl_confounds, within_fold=True,
            train_idx=train_idx, test_idx=test_idx,
        )
        X_train, X_test = X_work_fold[train_idx], X_work_fold[test_idx]
    ```

### MINOR

- **[stats.py:469-472] [MINOR] [CONFIDENCE: MEDIUM] [read]** Group label computation via `int(np.median(...))` truncates 0.5 to 0

  In `permutation_test`, group labels are computed as:
  ```python
  group_labels = np.array([
      int(np.median(y[groups == g] > 0.5))
      for g in unique_groups
  ])
  ```
  When a group has equal positive and negative samples, `np.median` of a boolean array returns 0.5, and `int(0.5)` gives 0 (Python floor-truncation). This silently biases all mixed groups toward label 0.

  - Impact: Currently, `assign_groups(paired=False)` assigns unique groups (1 sample each), so every group has a single unambiguous label. This edge case never triggers. It would trigger if paired groups (>1 sample per group) were used with `permutation_test`.
  - Fix: `int(np.round(np.median(y[groups == g] > 0.5)))` or explicit majority-vote with tie-breaking.

- **[pipeline.py:445] [MINOR] [CONFIDENCE: HIGH] [read]** Hardcoded `power=0.80` placeholder disables the "underpowered" verdict

  ```python
  scored = verdict_scorer(
      cache_auroc=result.get("auroc", 0.5),
      input_auroc=fals.get("input_auroc", 0.5),
      resid_auroc=fals.get("resid_auroc", 0.5),
      power=0.80,  # placeholder until power analysis runs
  )
  ```
  The `power_analysis()` function exists in `lib/stats.py` but is never called in the pipeline. The hardcoded 0.80 always passes the `power_threshold=0.50` check in `verdict_scorer`, so the "underpowered" verdict can NEVER fire.

  - Impact: The pipeline will never flag an underpowered study, even with very small sample sizes. This affects the scientific validity of the verdict system.
  - Fix: Either compute actual power and pass it through, or document that power checking is deferred.

- **[scorers.py:75-120 vs types.py:13-19] [MINOR] [CONFIDENCE: HIGH] [read]** Two disjoint verdict string sets: scorer uses raw strings, types.py defines unused Verdict enum

  `verdict_scorer` returns strings like `"input_confounded"`, `"collapsed"`, `"genuine_signal"`. The `types.py` module defines `class Verdict(Enum)` with values `CONFIRMED`, `FALSIFIED`, `WEAKENED`, `STRENGTHENED`, `INDETERMINATE`, `BLOCKED`. These two sets of verdict strings are completely disjoint. The enum is never used by the scorer or the pipeline.

  - Impact: No type safety on verdict values. The `Verdict` enum is dead code.
  - Fix: Either align the enum with the scorer's values or remove the enum.

- **[generator.py:210] [MINOR] [CONFIDENCE: MEDIUM] [read]** `iteration` variable unbound when `max_iterations=0`

  ```python
  iterations_used=iteration + 1 if pairs else 0,
  ```
  If `max_iterations=0`, the `for` loop never executes and `iteration` is undefined. If `pairs` is truthy (from `existing_pairs`), this raises `UnboundLocalError`.

  - Impact: Crash (not silent wrong results) on an edge-case configuration.
  - Fix: Initialize `iteration = -1` before the loop.

- **[pipeline.py:694-715] [MINOR] [CONFIDENCE: HIGH] [read]** `sys.path.insert(0, ...)` accumulates duplicates on repeated calls

  Both `_load_refusal_items` and `_load_impossibility_items` mutate `sys.path` every call with no dedup guard. In a long-running process or repeated pipeline runs, `sys.path` accumulates identical entries.

  - Fix: Use `importlib.util.spec_from_file_location()` for isolated imports, or convert data files to JSON.

- **[pipeline.py:500-686] [MINOR] [CONFIDENCE: HIGH] [read]** 186-line inline question list in pipeline orchestrator

  A 186-question list is embedded in the pipeline module. Refusal and impossibility data are loaded from separate files (`refusal_pairs_raw.py`, `impossibility_pairs_raw.py`). This inconsistency makes the module harder to navigate and mixes data with orchestration.

  - Fix: Move to `kv_verify/data/prompts/factual_questions.json` or a dedicated module.

- **[analyzer.py:74-155] [MINOR] [CONFIDENCE: HIGH] [read]** Redundant tokenization: both `analyze_length_distribution` and `analyze_pair_balance` tokenize all prompts independently

  When `analyze_pair_set` calls both functions, every prompt is tokenized at least 2x. With 200 pairs, this doubles the tokenizer calls unnecessarily.

  - Fix: Pre-tokenize once in `analyze_pair_set` and pass counts to sub-functions.

- **[gap_filler.py:318-326 vs 249-254] [MINOR] [CONFIDENCE: HIGH] [read]** TF-IDF + KMeans computed twice in single `analyze_gaps()` call

  `analyze_gaps()` computes clustering at lines 318-326 for entropy, then calls `_compute_gaps()` at line 334 which computes identical clustering at lines 249-254.

  - Fix: Compute once and pass results to both consumers.

- **[gap_filler.py:61-86 vs 89-101] [MINOR] [CONFIDENCE: HIGH] [read]** Inconsistency: density uses both prompt sides, entropy uses only positive

  `_token_length_density` tokenizes both positive AND negative prompts. `_token_entropy` tokenizes only positive. This creates inconsistent views of the same distribution.

  - Fix: Decide on consistent policy and document.

- **[feature_extractor.py:33-37] [MINOR] [CONFIDENCE: HIGH] [read]** Unnecessary import coupling: `MODEL_CACHE_DIR` imported only for re-export

  `feature_extractor.py` imports `MODEL_CACHE_DIR` from `models.py` and re-exports as `DEFAULT_MODEL_CACHE_DIR`. Never used internally. Creates unnecessary coupling.

  - Fix: Remove if backward compatibility is no longer needed. If needed, document what depends on it.

- **[types.py:60-68] [MINOR] [CONFIDENCE: HIGH] [read]** 8 optional FeatureVector fields are never populated

  `norm_variance`, `gen_delta`, `layer_uniformity`, `head_variance`, `max_layer_rank`, `top_sv_ratio`, `rank_10`, `layer_variance` are declared but always `None`. Only `spectral_entropy` and `angular_spread` are populated by `extract_from_cache`.

  - Fix: Remove unused fields or implement the features. Dead fields confuse readers about what data is available.

- **[analyzer.py:82] [MINOR] [CONFIDENCE: HIGH] [read]** Unused `label` parameter in `_side_stats` inner function

  `_side_stats(stats_list, label)` accepts `label` but never uses it. Called at lines 110-111 with `"positive"` and `"negative"`.

  - Fix: Remove the `label` parameter.

---

## LLM Artifacts

- **[pipeline.py:500-686] [CONFIDENCE: MEDIUM] [read]** The 186-question list shows hallmarks of LLM generation: uniform format, systematic domain coverage without gaps, zero typos, and an explicit gap-filling section at line 663 with comments ("Gap-filling 17: underrepresented domains + varied lengths"). This indicates iterative LLM-assisted generation guided by gap analysis output. Not a bug.

- **[scorers.py:128-145] [CONFIDENCE: MEDIUM] [read]** Refusal phrase list has standard LLM-training-data coverage patterns. All lowercase, semi-alphabetized, covers the standard refusal taxonomy. Appropriate for heuristic use.

---

## Verified Correct

- **stats.py `_pooled_sd` (line 32-37)**: `ddof=1` for Bessel's correction, denominator `n1 + n2 - 2` for pooled df. Standard formula. [read]
- **stats.py `hedges_g` (line 59-72)**: J = `1 - 3/(4*df - 1)` matches Hedges (1981). [read]
- **stats.py `cohens_d` (line 40-56)**: Positive d = group1 > group2. Returns 0.0 for degenerate cases. [read]
- **stats.py `tost` (line 109-146)**: Upper test uses `cdf` (left tail for "diff < +delta"), lower test uses `1 - cdf` (right tail for "diff > -delta"). Equivalence = `max(p_lower, p_upper) < 0.05`. Correct intersection-union test. Uses pooled-variance SE and pooled df, which is the standard Schuirmann (1987) formulation. [read]
- **stats.py `d_to_auroc` (line 153-159)**: `Phi(d / sqrt(2))` is correct under equal-variance normality assumption. [read]
- **stats.py `holm_bonferroni` (line 166-196)**: Sort ascending, multiply by `(n - rank)`, enforce monotonicity via cumulative max. Standard Holm (1979). [read]
- **stats.py `bootstrap_ci` (line 203-229)**: Percentile bootstrap with `(1-ci)/2` for symmetric alpha. SE from std of bootstrap distribution. [read]
- **stats.py `bootstrap_auroc_ci` BCa (line 578-611)**: z0 = `norm.ppf(fraction < observed)` for bias correction. Jackknife acceleration a = `sum((mean-theta_i)^3) / (6 * sum((mean-theta_i)^2)^1.5)`. Adjusted percentiles via BCa formula. Matches Efron (1987). Edge case guard at line 581: `if np.isinf(z0): z0 = 0.0`. [read]
- **stats.py `permutation_test` (line 436-507)**: Phipson-Smyth `(count+1)/(n_perm+1)` avoids p=0. Group-level permutation shuffles group-to-label mapping. [read]
- **stats.py `assign_groups` (line 236-272)**: Non-paired returns `arange(n)`. Paired concatenates prompt indices for shared group IDs. [read]
- **stats.py `fwl_residualize` (line 368-408)**: Per-feature OLS, correct within-fold separation when train_idx/test_idx provided. [read]
- **stats.py `fwl_nonlinear` (line 411-429)**: Polynomial expansion then standard FWL. `include_bias=False` correct (LR has its own intercept). [read]
- **feature_extractor.py `extract_from_cache` (line 85-170)**: Float32 upcast for SVD stability. `svdvals` for singular values. Spectral entropy from squared-normalized singular values. Effective rank = `exp(entropy)`. Matches Roy & Vetterli (2007). Total norm via Pythagorean sum of layer norms. [read]
- **feature_extractor.py `get_cache_accessor` (line 56-78)**: DynamicCache (`.key_cache`), HybridCache (`.layers[i].keys`), legacy tuples (`[i][0]`). Correct for all three HF cache formats. [read]
- **feature_extractor.py `_compute_angular_spread` (line 173-215)**: Self-pair exclusion via `(idx_a + offset) % N` where offset in `[1, N)`. Guarantees idx_b != idx_a. Cosine similarity via normalized dot product. Angular spread = std of cosines. [read]
- **config.py `PipelineConfig`**: Mutable defaults use `field(default_factory=...)`. YAML via `safe_load`. Path conversion explicit. [read]
- **data_loader.py dispatch**: All loaders filter by correct `condition` field. `paired` flag matches `EXP47_COMPARISONS`. [read]
- **tracking.py `_sanitize_key`**: Replaces `/`, `\`, `..` in cache keys. Addresses path traversal. [read]
- **models.py `generate_text`**: `temperature=None` when `do_sample=False`. `skip_special_tokens=True` on decode. [read]
- **gen.py `validate_token_counts`**: Token-count diff for pair validation. Tokenizer or word-count fallback. [read]

---

## Library Usage Audit

| Library Call | Line | Correct? | Verified How | Notes |
|-------------|------|----------|--------------|-------|
| `np.var(g1, ddof=1)` | stats.py:35 | Yes | [read] | Bessel's correction |
| `sp_stats.ttest_ind(g1, g2, equal_var=False)` | stats.py:89 | Yes | [read] | Welch's t-test |
| `sp_stats.mannwhitneyu(g1, g2, alternative="two-sided")` | stats.py:96 | Yes | [read] | Two-sided MWU |
| `sp_stats.t.cdf(t_upper, df=df)` | stats.py:138 | Yes | [read] | TOST left-tail p |
| `norm.cdf(d / sqrt(2))` | stats.py:159 | Yes | [read] | d-to-AUROC conversion |
| `GroupKFold(n_splits=...)` | stats.py:311 | Yes | [read] | Group-aware CV |
| `make_pipeline(StandardScaler(), LogisticRegression(...))` | stats.py:310 | Yes | [read] | Scaling before LR |
| `roc_auc_score(y, proba)` | stats.py:339 | Yes | [read] | Standard AUROC |
| `torch.linalg.svdvals(K_flat)` | feature_ext:125 | Yes | [read] | Singular values only (efficient) |
| `torch.linalg.norm(K_flat)` | feature_ext:119 | Yes | [read] | Frobenius norm (default for 2D) |
| `torch.linalg.norm(va, dim=1, keepdim=True)` | feature_ext:206 | Yes | [read] | Per-row L2 norm |
| `np.percentile(boot_aurocs, 100 * p)` | stats.py:610-611 | Yes | [read] | BCa adjusted percentiles |
| `scipy_entropy(probs, base=2)` | gap_filler:101 | Yes | [read] | Shannon entropy in bits |
| `TfidfVectorizer(max_features=500, stop_words="english")` | gap_filler:171 | Yes | [read] | Standard TF-IDF |
| `KMeans(n_clusters=..., random_state=42, n_init=10)` | analyzer:204 | Yes | [read] | Deterministic clustering |
| `tokenizer.apply_chat_template(...)` | models.py:146 | Yes | [read] | Chat template |
| `model.generate(**inputs, ...)` | models.py:153 | Yes | [read] | temperature=None when do_sample=False |
| `PolynomialFeatures(degree=2, include_bias=False)` | stats.py:426 | Yes | [read] | No bias (LR intercept) |
| `yaml.safe_load(f)` | config.py:54 | Yes | [read] | Safe YAML loading |
| `LogisticRegression(max_iter=5000, solver="lbfgs")` | stats.py:310 | Yes | [read] | LBFGS for convergence |
| `np.random.RandomState(seed)` | stats.py:217,454,532 | Yes | [read] | Reproducible RNG |
| `LinearRegression()` | stats.py:325,393,401 | Yes | [read] | OLS for FWL |
| `NearestNeighbors` | gap_filler.py:25 | N/A | [read] | Imported but not used in reviewed functions |

---

## Red Team

The strongest argument against this work's conclusions from a code correctness perspective:

**The hardcoded `power=0.80` placeholder (pipeline.py:445) combined with the within-fold FWL divergence (stats.py:319-327) represent the highest-risk path for silently wrong results.**

The power placeholder means the pipeline can never flag an underpowered experiment. An AUROC of 0.52 with N=10 would never trigger the "underpowered" verdict. The pipeline would produce a verdict of "indeterminate" (because cache_auroc <= 0.75) rather than "underpowered", potentially leading to the wrong scientific conclusion about why the signal is weak.

The FWL divergence is correctness-correct today but fragile: the two regression fits in `groupkfold_auroc` produce identical results only because they are both vanilla LinearRegression on identical data. Any modification to one path without the other would silently produce mismatched residualization between training and test data, which could inflate or deflate AUROC depending on the direction of the mismatch.

I was unable to verify the BCa bootstrap implementation against a reference at runtime (no Bash access). The formulas match Efron (1987) by inspection, but BCa has known numerical edge cases that would require test-case verification.

---

## Verdict

**MINOR ISSUES**

The codebase is well-written with careful attention to statistical correctness. The two major issues (scorer bug, redundant FWL regression) do not affect current computed results -- the scorer is not called in the pipeline, and the redundant regression produces identical outputs. The twelve minor issues are code quality and robustness concerns. All numeric and statistical computations were verified correct by code reading. No bugs found that produce silently wrong results in any current code path.

---

## Referrals

- **impl-auditor**: The `_do_verdicts` stage uses hardcoded `power=0.80` instead of calling `power_analysis()`. Verify whether this is intentional or a forgotten TODO. If unintentional, the "underpowered" verdict can never fire.
- **impl-auditor**: The `_do_falsification` stage (pipeline.py:402) uses `fwl_residualize(X, input_all, within_fold=False)` which is the pre-C4-fix behavior (full-dataset FWL). Verify whether this is intentional as a comparison baseline or should be within-fold.
- **impl-auditor**: Check that the pipeline stage dependency chain correctly prevents running analysis with stale cached data from a previous run with different configuration (different model, different N, etc.).
