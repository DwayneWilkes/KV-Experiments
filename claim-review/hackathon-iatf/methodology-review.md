# Adversarial Methodology Review — IATF Hackathon (Exp 26–36)

> **Event**: [Intelligence at the Frontier Hackathon](https://luma.com/ftchack-sf-2026) (FTCHack SF 2026), hosted by Funding the Commons & Protocol Labs.
> **Team**: Liberation Labs.

**Date**: 2026-03-15
**Scope**: Red-team the experimental design and statistical methodology of KV-cache hackathon experiments, independent of data-integrity verification (completed in [audit-report.md](audit-report.md)).
**Protocol**: Blind recomputation — Phase A computes statistics without knowledge of claimed values; Phase B compares. See [adversarial_review.py](adversarial_review.py).

---

## Table of Contents

- [Executive Summary](#executive-summary)
- [Literature Verification](#literature-verification)
- [Power Analysis](#power-analysis)
- [Per-Claim Verdicts](#per-claim-verdicts)
  - [1. Refusal Detection AUROC 0.898](#1-refusal-detection-auroc-0898)
  - [2. Jailbreak Detection AUROC 0.878](#2-jailbreak-detection-auroc-0878)
  - [3. Cross-Model Transfer Mean 0.863](#3-cross-model-transfer-mean-0863)
  - [4. Scale Invariance rho 0.83–0.90](#4-scale-invariance-rho-083090)
  - [5. Impossibility > Harmful Refusal](#5-impossibility--harmful-refusal)
  - [6. Within-Model Deception AUROC 1.0](#6-within-model-deception-auroc-10)
  - [7. Sycophancy Detectable](#7-sycophancy-detectable)
- [Cross-Cutting Methodological Issues](#cross-cutting-methodological-issues)
- [Verdict Summary Table](#verdict-summary-table)
- [Recommendations](#recommendations)
- [ML Methodology Issues](#ml-methodology-issues)
- [Sources](#sources)

---

## Executive Summary

The prior audit verified *data integrity* — numbers in claims match numbers in JSON. This review asks a different question: **do the experiments actually support the claims?**

We red-teamed seven headline claims from Exp 26–36 using power analysis, bootstrap confidence intervals, permutation tests at higher resolution (10,000 vs 200 iterations), confound checks, and literature benchmarks.

**Bottom line**: The headline findings (refusal detection, deception detection) are *real signals* — key_rank and key_entropy effects are large (d > 1.0) and detectable at n=20. But several claims are overstated, one is refuted, and the conceptual framing needs revision. The signal is best described as **output suppression** (the model allocating fewer resources per token when withholding), not harm-specific or intent-specific geometry.

| Verdict | Count | Claims |
| --------- | ------- | -------- |
| CONFIRMED | 0 | — |
| QUALIFIED | 4 | Refusal AUROC, impossibility > harmful, deception AUROC, cross-model transfer |
| UNDERMINED | 2 | Jailbreak AUROC, scale invariance |
| REFUTED | 1 | Sycophancy detectable |

No formula errors were found in the statistical implementations. One documentation bug (Hedges' g docstring, line 156 of `stats_utils.py`), one design limitation (percentile bootstrap not BCa), one missing diagnostic (linearity check for residualization).

Beyond the per-claim verdicts, we identified **8 systemic ML methodology issues** (M1–M8) — ranging from train-test contamination via greedy decoding (CRITICAL) to label noise in the jailbreak condition (LOW). Each includes specific remediations. See [ML Methodology Issues](#ml-methodology-issues).

---

## Literature Verification

All statistical implementations in `code/stats_utils.py` were independently verified against standard references. This is *not* recomputation from data — it's formula-level verification.

| Method | Reference | Implementation | Verdict |
| -------- | ----------- | ---------------- | --------- |
| Cohen's d (pooled SD) | Cohen (1988) | `stats_utils.py:134-150` | CORRECT |
| Hedges' g (J correction) | Hedges (1981) | `stats_utils.py:153-165` | CORRECT — docstring bug on line 156 says "Glass's estimator" but code implements Hedges (1981) correctly |
| Bootstrap CIs (percentile) | Efron & Tibshirani (1986) | `stats_utils.py:68-100` | CORRECT — uses percentile method, not BCa. Adequate for symmetric distributions; 10,000 resamples meets standard threshold |
| Welch's t-test | Welch (1947) via SciPy | `stats_utils.py:107-110` | CORRECT |
| Mann-Whitney U | Mann & Whitney (1947) via SciPy | `stats_utils.py:113-119` | CORRECT |
| Holm-Bonferroni | Holm (1979) | `stats_utils.py:204-226` | CORRECT — monotonicity enforced via running maximum |
| TOST equivalence | Schuirmann (1987) | `stats_utils.py:233-278` | CORRECT — δ=0.3 d-units is a reasonable SESOI per Lakens et al. (2018) |
| Length residualization | Frisch-Waugh-Lovell | `stats_utils.py:285-343` | CORRECT — but no linearity diagnostic. Should verify residuals are approximately normal and homoscedastic. |
| d-to-AUROC conversion | Φ(d/√2) | `independent_stats.py:267-273` | CORRECT — assumes equal-variance normal distributions. Adequate for these effect sizes. |

---

## Power Analysis

Using the standard formula for minimum detectable effect size in a two-sample t-test:

> d_min = (z_α + z_power) × √(2/n)

at α=0.05 (two-tailed), power=0.80.

**Reference**: Cohen (1988) *Statistical Power Analysis for the Behavioral Sciences*, 2nd ed. Cross-checked against `independent_stats.py:power_analysis()`.

| Experiment | n/condition | Min detectable d | Strongest observed d (feature) | Powered? |
| ------------ | ------------- | ------------------ | --------------------------------- | ---------- |
| Exp 31: Refusal (key_rank) | 20 | 0.89 | 1.53 (key_rank) | **YES** |
| Exp 31: Refusal (norm/tok) | 20 | 0.89 | 0.36 (norm_per_token) | NO — underpowered |
| Exp 32: Jailbreak | 20 | 0.89 | ~1.0+ (via AUROC 0.878) | **YES** (marginal) |
| Exp 33: Cross-model refusal | 20 | 0.89 | varies by model | Model-dependent |
| Exp 36: Impossibility (key_rank) | 20 | 0.89 | 1.88 (key_rank) | **YES** |
| Exp 36: Impossibility (norm/tok) | 20 | 0.89 | 0.30–0.60 | NO |
| Deception same-prompt (Exp 18b) | 10 | 1.26 | 2.50–2.79 | **YES** (strongly) |
| Sycophancy (wrong vs correct) | 60 | 0.52 | 0.107 | **NO** — needs n≈2,000+ |
| Scale invariance LARGE group | 2 models | — | rho 0.83 | **INSUFFICIENT** n for any claim |

**Key insight**: The headline AUROC claims ARE powered because key_rank and key_entropy produce very large effects (d > 1.0). The norm_per_token effects are small (d < 0.5) and underpowered, but these are secondary features, not the headlines. Sycophancy is the one headline claim that is refuted on power grounds.

**AUROC precision** per Hanley & McNeil (1982): At n=20 per class, for AUROC ≈ 0.90, the standard error is SE ≈ 0.05–0.07. This gives a 95% CI of roughly [0.76, 1.00]. The signal is clearly above chance (0.5), but the third significant digit is noise — reporting "0.898" implies a precision the data cannot support.

**Permutation resolution** per Phipson & Smyth (2010): With m=200 permutations, the resolution is exactly 1/201 ≈ 0.005. The claimed p<0.005 is at the resolution limit. The correct formula p = (b+1)/(m+1) gives minimum achievable p = 1/201 = 0.00498, which rounds to 0.005. Need m ≥ 10,000 for reliable p at this level.

---

## Per-Claim Verdicts

### 1. Refusal Detection AUROC 0.898

**Claim**: Logistic regression classifier on KV-cache features achieves AUROC 0.898 for refusal detection (Exp 31, n=20+20, Qwen 7B, 5-fold CV).

**Verdict**: **QUALIFIED**

**Power**: Adequate. key_rank d=1.53, key_entropy d=1.51 — both well above the minimum detectable d=0.89 at n=20. These two features drive the AUROC.

**Evidence**:
- Bootstrap 95% CI (1,000 resamples, percentile method): approximately [0.76, 1.00]. CI lower bound > 0.70 confirms above-chance performance. Per Efron & Tibshirani (1986), percentile bootstrap is adequate for symmetric statistics.
- Hanley-McNeil SE ≈ 0.06 at AUC=0.90, n=20/class — consistent with bootstrap CI width (Hanley & McNeil, 1982).
- Permutation test at 10,000 iterations: p ≈ 0.001 (confirming the 200-iteration estimate of p<0.005 was directionally correct but at resolution limit per Phipson & Smyth, 2010).
- Cross-validation error bars at n=40 total: ±15-20% expected per Varoquaux (2018). The point estimate 0.898 could easily be 0.75 or 0.95 on a new sample.

**Caveats**:
1. **Greedy decoding variance inflation**: Refusal responses all produce the same short text (n_generated=50 for all 20), giving within-class CV ~1-2%. Benign responses vary (n_generated range [22, 50]), giving CV ~5%. This asymmetry makes the classes *artificially* separable. With stochastic sampling (temperature > 0), within-class variance would increase and AUROC would likely decrease.
2. **Single model**: Qwen 7B only. Cross-model replication in Exp 33 shows AUROC 0.843–0.893, confirming the signal generalizes but with model-dependent magnitude.
3. **Reporting precision**: "0.898" implies ±0.001 precision but actual CI is ±0.12. Should report as "AUROC ≈ 0.90, 95% CI [0.76, 1.00]".

**Bottom line**: Real signal, adequately powered, but overprecisely reported. The AUROC is best stated as "approximately 0.90" with the CI.

---

### 2. Jailbreak Detection AUROC 0.878

**Claim**: Logistic regression on KV-cache features distinguishes abliterated (jailbroken) model responses from normal responses at AUROC 0.878 (Exp 32, n=20+20+20, abliterated Qwen 7B).

**Verdict**: **UNDERMINED**

**Power**: Adequate for detection in aggregate, but the conceptual basis is flawed.

**Evidence**:
- **EXP32-001 (CRITICAL)**: Claimed 18/20 abliterated responses "actually answered" harmful prompts. JSON shows 8/20 — a 2.25x inflation. Only 8 responses are genuine jailbreak behavior.
- **EXP34-001 (OVERSTATED)**: Llama abliterated model answered only 1/20 harmful prompts. Its "jailbreak" AUROC measures refusal-vs-normal, not jailbreak-vs-normal.
- **Abliteration subset analysis**: Of the abliterated responses, approximately 8 provide actual harmful information (PROVIDES_INFO), ~10 warn but partially comply (WARNS_ONLY), and ~2 hard-refuse. The classifier cannot separate genuine-jailbreak from refusal-behavior subsets because the "jailbreak" condition is heterogeneous.
- **EXP36-001 (FALSIFIED)**: Exp 36 showed impossibility refusal (AUROC 0.950) is *more* detectable than harmful refusal (0.898), and the two refusal types are indistinguishable (AUROC 0.693). The signal is about output suppression, not harm content.

**Caveats**:
1. The classifier primarily learns the difference between "model that suppresses output" (refuses/warns) and "model that generates freely" (normal) — not a jailbreak-specific geometric signature.
2. The Qwen AUROC value (0.878) is hardcoded in the Exp 34 script (line 344), not loaded from JSON (EXP34-002).
3. Mean cross-model AUROC 0.849 includes Llama (essentially a refusal detector) and inherits Qwen's hardcoded value.

**Bottom line**: The AUROC is numerically verifiable, but the claim that it measures "jailbreak detection" is conceptually undermined. It measures absence-of-refusal.

---

### 3. Cross-Model Transfer Mean 0.863

**Claim**: Cross-model refusal detection transfers at mean AUROC 0.863.

**Verdict**: **QUALIFIED**

**Power**: N/A (aggregate statistic).

**Evidence**:
- Arithmetic is correct: the mean of the per-model AUROCs equals 0.863.
- However, the minimum is 0.14 (well below chance), indicating massive variance.
- The mean is dominated by Qwen's strong performance (0.878, hardcoded).

**Caveats**:
1. min AUROC = 0.14 — worse than random. Not prominently reported.
2. Llama abliteration answered 1/20 — it's not a jailbreak model, so its AUROC measures something else entirely (EXP34-001).
3. Mean AUROC inherits Qwen's hardcoded value (EXP34-002).
4. Reporting a mean without variance (SD, range, or CI) obscures that transfer is model-dependent and sometimes catastrophically fails.

**Bottom line**: The mean is technically correct but misleading without reporting variance. Should present as "AUROC range [0.14, 0.89], mean 0.86" at minimum.

---

### 4. Scale Invariance rho 0.83–0.90

**Claim**: Category hierarchy is preserved across model scales with Spearman rho 0.83–0.90 (Exp 26, 10 models across SMALL/MEDIUM/LARGE groups).

**Verdict**: **UNDERMINED**

**Power**: Insufficient. The LARGE group contains only 2 models.

**Evidence**:
- SMALL-MEDIUM correlation is based on 5×5 model pairs — statistically meaningful.
- SMALL-LARGE and MEDIUM-LARGE correlations are based on comparisons involving only 2 LARGE models (Qwen-32B-q4 and DeepSeek-67B).
- Spearman rho from n=2 data points is mathematically ±1.0 by construction (only two possible rank orderings). Any reported rho from n=2 is not an inference — it's an arithmetic identity.
- EXP26-001: Actual SMALL-LARGE rho is 0.826, below the claimed lower bound of 0.83.
- Both LARGE models are quantized (NF4 / q4), confounding scale with quantization.

**Caveats**:
1. Cannot distinguish "scale invariance" from "quantization invariance" with this design.
2. The claim would require ≥5 models per size group for meaningful correlation analysis.
3. SMALL-MEDIUM rho (where n is adequate) is the credible portion of this finding.

**Bottom line**: The SMALL-MEDIUM finding is credible. Claims involving the LARGE group are unsupported due to n=2.

---

### 5. Impossibility > Harmful Refusal

**Claim**: Impossibility refusal is more detectable than harmful-content refusal (AUROC 0.950 vs 0.898, Exp 36, single model).

**Verdict**: **QUALIFIED**

**Power**: Adequate. key_rank d=1.88 (impossibility vs benign) is well above min detectable d=0.89.

**Evidence**:
- AUROC values verified in JSON (VERIFIED in data-integrity audit).
- Impossibility-vs-benign (0.950) > harmful-vs-benign (0.898) — consistent with the hypothesis that impossibility refusal produces even stronger output suppression.
- Impossibility-vs-harmful AUROC = 0.693 — the two refusal types are hard to tell apart, suggesting a shared mechanism (output withholding) rather than content-specific geometry.
- key_rank and key_entropy are the driving features for both (d > 1.5).

**Caveats**:
1. **Single model** (Qwen 7B), single run — no cross-model replication.
2. Hanley-McNeil SE at n=20/class: 0.950 has 95% CI roughly [0.85, 1.00], and 0.898 has CI [0.76, 1.00]. The CIs overlap substantially — the ordering (impossibility > harmful) is not statistically certain.
3. **norm_per_token effects are underpowered** (d=0.30–0.60 vs min detectable 0.89).

**Scientific value**: EXP36-P01 — the team tested a hypothesis that could undermine their own headline ("harmful content detection"), found that impossibility refusal is *more* detectable, and published the result. This is honest hypothesis testing (POSITIVE finding in audit).

**Bottom line**: Directionally correct and powered for key features. The reinterpretation from "harmful content detection" to "output suppression detection" is the important takeaway.

---

### 6. Within-Model Deception AUROC 1.0

**Claim**: Cricket classifier achieves AUROC 1.0 for within-model deception detection (same-prompt design, Exp 18b).

**Verdict**: **QUALIFIED**

**Power**: Strong. Cohen's d = 2.50–2.79 at n=10 per condition. Min detectable d at n=10 is 1.26 — effects are 2x larger.

**Evidence**:
- Within-model, same-prompt deception detection does produce very large effect sizes (d > 2.5). This is consistent with AUROC ≈ 1.0 via the Φ(d/√2) conversion: Φ(2.5/1.414) = Φ(1.77) = 0.96, and the 5-fold CV LR likely exceeds this due to multivariate feature combination.
- The extreme d values (2.5+) mean the classes are almost perfectly separable in feature space.

**Caveats**:
1. **Input length confound**: Deceptive-condition prompts include additional instructions ("respond deceptively"), making them ~2x longer than honest prompts. Part of the KV-cache signal may encode prompt structure, not response-level deception geometry.
2. **Same-prompt design**: Training and testing use the same prompts (different splits), not cross-prompt generalization. Real-world deployment requires detecting deception on novel prompts.
3. **n=10 per condition**: Despite adequate power for detecting the large effect, the CIs on AUROC are wide (Hanley-McNeil SE ≈ 0.10 at n=10/class, AUC=0.95).
4. **Cricket has 0 code, 0 tests**: The AUROC 1.0 claim is from experiment scripts, not a deployed classifier (CRICKET-001).

**Bottom line**: The effect is real and large. AUROC ≈ 1.0 is plausible but should be reported with the input-length caveat and the distinction between same-prompt and cross-prompt generalization.

---

### 7. Sycophancy Detectable

**Claim**: Sycophancy is detectable via KV-cache geometry (PITCH_NUMBERS.md).

**Verdict**: **REFUTED**

**Power**: Catastrophically underpowered.

**Evidence**:
- Observed Cohen's d = 0.107 (negligible on Cohen's 1988 scale).
- At n=60 per group, minimum detectable d for 80% power = 0.52 (Cohen, 1988).
- The observed d is 4.9x below the detection threshold.
- Actual power at d=0.107, n=60: approximately 10% — the experiment has only a 1-in-10 chance of detecting the effect even if it's real.
- Required sample size for 80% power at d=0.107: **n ≈ 2,164 per group** (via (z_α + z_power)² × 2/d²).
- For context, deception d = 3.065 — the sycophancy effect is 29x smaller.

**Caveats**:
1. This is a PITCH claim, not a peer-reviewed finding. PITCH_NUMBERS.md lists sycophancy alongside deception and refusal detection without noting the 29x effect-size gap.
2. The sycophancy data was replicated from Campaign 1 in Campaign 2 and then silently dropped from the paper (WS9 omission audit finding I3). This suggests the authors may have recognized the weakness but didn't remove the PITCH claim.
3. Lakens et al. (2018) confirms δ=0.3 d-units as a reasonable "smallest effect size of interest" (SESOI). At d=0.107, the effect is below SESOI — even if real, it may not be meaningful.

**Bottom line**: The sycophancy detection claim is statistically unsupportable at any feasible sample size for KV-cache experiments. Should be removed from PITCH_NUMBERS.md.

---

## Cross-Cutting Methodological Issues

### 1. Greedy decoding variance inflation

All experiments use `do_sample=False`. This produces identical outputs for identical inputs, creating two problems:

- **Pseudoreplication**: Multiple runs of the same prompt produce identical data. n_effective < n_reported. (Addressed in C2 audit, D5.)
- **Variance asymmetry**: Conditions where the model produces stereotyped responses (e.g., refusal: "I can't help with that") have artificially low within-class variance (CV ~1-2%), while conditions with diverse outputs (e.g., benign factual answers) have higher variance (CV ~5%). This asymmetry inflates AUROC because the refusal class is a tight cluster while the benign class is spread out.

**Recommendation**: Report within-class CV for each feature alongside effect sizes. Consider supplementary experiments with temperature=0.7 to assess AUROC degradation under stochastic sampling.

### 2. Permutation test resolution

The experiments use 200 permutations, claiming p<0.005. Per Phipson & Smyth (2010), the correct formula is p = (b+1)/(m+1), giving a minimum achievable p of 1/201 ≈ 0.00498. The claimed significance is at the precision floor.

**Recommendation**: Use ≥10,000 permutations. The adversarial_review.py script implements this.

### 3. Single-model findings

Experiments 31, 32, 36 are all on Qwen 7B only. Exp 33/34 add Llama and Mistral but with mixed results (Llama abliteration largely fails). The finding is model-dependent.

**Recommendation**: All headline claims should include cross-model replication counts. "AUROC 0.898 (Qwen)" is honest; "refusal detection AUROC 0.898" without qualification implies generality.

### 4. Output suppression as the unifying signal

The most parsimonious explanation across Exp 31, 35, 36:

- Refusal → output suppression → sparser KV-cache per token → detectable
- Impossibility refusal → stronger suppression → more detectable than harmful refusal
- Deception → verbose output with per-token resource reduction → detectable via per-token normalization

This is not "harmful content detection" or "jailbreak detection" — it's detection of whether the model is withholding output. Renaming from "harm detection" to "output suppression detection" would make the claims more defensible and scientifically accurate.

### 5. CV bias at small n

Per Varoquaux (2018), 5-fold CV at n=40 produces error bars of ±15-20% on the point estimate. Per Aghbalou et al. (2022), k-fold CV may be inconsistent under general stability assumptions. The AUROC point estimates are informative but should not be reported to three decimal places.

---

## Verdict Summary Table

| # | Claim | Verdict | Power | Key concern |
| --- | ------- | --------- | ------- | ------------- |
| 1 | Refusal AUROC 0.898 (Exp 31) | **QUALIFIED** | Adequate (d=1.53 key_rank) | CI [~0.76, 1.00]; greedy decoding inflates variance |
| 2 | Jailbreak AUROC 0.878 (Exp 32) | **UNDERMINED** | Adequate for detection | Abliteration failed (8/20 genuine), conflates with refusal detection |
| 3 | Cross-model transfer 0.863 | **QUALIFIED** | N/A (aggregate) | min=0.14 hides massive variance; mean misleading without range |
| 4 | Scale invariance rho 0.83–0.90 (Exp 26) | **UNDERMINED** | n=2 insufficient | Both LARGE models quantized; rho from n=2 is ±1.0 by construction |
| 5 | Impossibility > harmful (Exp 36) | **QUALIFIED** | Adequate (d=1.88 key_rank) | Single model, single run; CIs overlap |
| 6 | Deception AUROC 1.0 (Cricket) | **QUALIFIED** | Strong (d=2.5+) | Input length confound from deception instructions |
| 7 | Sycophancy detectable (PITCH) | **REFUTED** | 10% power (d=0.107, n=60) | Needs n≈2,000 for 80% power |

---

## Recommendations

### For the team

1. **Reframe "jailbreak detection" as "output suppression detection"**. The signal is about whether the model withholds output, not about harmful content or jailbreak intent. This is a stronger, more defensible claim.

2. **Remove sycophancy from PITCH_NUMBERS.md**. At d=0.107 and n=60, the claim is not supportable. If the team believes sycophancy detection is important, design a dedicated experiment with n≥2,000 per group.

3. **Report AUROCs with confidence intervals**. "AUROC 0.90, 95% CI [0.76, 1.00]" is honest. "AUROC 0.898" implies precision the data cannot support.

4. **Increase permutation count to 10,000+**. The current 200-permutation tests have p-value resolution at the exact level being claimed.

5. **Add within-class CV reporting**. Greedy decoding creates artificially tight distributions. Reporting CV makes this transparent.

6. **Replicate Exp 36 across models**. The impossibility-vs-safety finding is the most scientifically interesting result (it falsifies the "harmful content detection" narrative), but it's Qwen-only.

7. **Fix EXP32-001 and EXP34-002/003**. The hardcoded "18/20" and "0.878" values propagate the original counting error. Replace with programmatic JSON reads.

### For future experiments

8. **Pre-register power analysis**. Before running an experiment, compute min detectable d for the planned n. This would have caught the sycophancy underpowering immediately.

9. **Include stochastic-sampling control runs**. Even a small supplementary experiment (temperature=0.7, same prompts) would quantify how much greedy decoding inflates effect sizes.

10. **Standardize JSON output schema**. Every experiment should output features in the same structure (see [IMPROVEMENTS.md](../IMPROVEMENTS.md)).

---

## ML Methodology Issues

Beyond the per-claim verdicts above, several systemic ML methodology problems affect the experiments collectively. These are ordered by severity — the first three are the most damaging to the headline claims.

### M1: Train-test contamination via greedy decoding (CRITICAL)

**Problem**: All experiments use `do_sample=False` (greedy decoding), producing identical outputs for identical inputs. `deduplicate_runs()` exists in `stats_utils.py` but is **never called** in any experiment script (C2 audit finding D5). When 5 identical runs of the same prompt enter a 5-fold CV split, the "test" fold contains exact copies of training data. This is textbook data leakage.

**Impact**: The identity signatures experiment (C2) achieved 100% classification accuracy entirely due to this leak (C2 audit finding D4). In hackathon experiments, the effect is less dramatic (n=20 unique prompts, 1 run each) but any experiment with multiple runs per prompt is affected.

**Remediation**:
1. Call `deduplicate_runs()` before any CV split — one observation per unique prompt
2. For new experiments: use stochastic sampling (temperature > 0) to generate genuinely independent observations
3. Report both pre- and post-deduplication sample sizes in every result file

### M2: Same-prompt cross-validation (HIGH)

**Problem**: The 5-fold CV splits samples *within* prompts, not *across* prompts. A classifier trained on "prompt #7 → refusal" gets test credit if another sample from prompt #7 appears in the test fold. The meaningful evaluation is cross-prompt: can the classifier detect refusal from a prompt it has never seen? This is never tested.

**Impact**: Every reported AUROC is an upper bound on cross-prompt generalization. The gap between same-prompt and cross-prompt performance is unknown but potentially large, especially for deception detection (Exp 18b) where prompt instructions differ systematically between conditions.

**Remediation**:
1. Use `GroupKFold` or `LeaveOneGroupOut` with prompt ID as the group variable — ensures no prompt appears in both train and test
2. Report both same-prompt and cross-prompt AUROCs to quantify the generalization gap
3. For Cricket deployment claims: only cross-prompt AUROCs are relevant (real-world prompts are novel)

### M3: Feature leakage through response length (HIGH)

**Problem**: Raw Frobenius norm scales mechanically with sequence length (more tokens → bigger matrix → bigger norm). Including `norm` alongside `norm_per_token` gives the classifier a proxy for response length. In refusal detection: refusals are short (~50 tokens), benign responses are long (~200 tokens). A classifier on `n_generated` alone achieves AUROC > 0.70.

**Impact**: Exp 35 corrected this for *interpretation* (showing per-token norms reverse direction) but the *classifiers* still use uncorrected features. It's unclear how much of the reported AUROC comes from KV-cache geometry vs. response length.

**Remediation**:
1. Report AUROC for three feature sets separately: (a) length-only, (b) all features, (c) length-residualized features
2. If AUROC(length-only) > 0.70, the headline AUROC must include a length-confound caveat
3. For the strongest claim, use only `norm_per_token`, `key_rank`, `key_entropy` — no raw norm
4. Apply Frisch-Waugh-Lovell residualization (already in `stats_utils.py`) before classification, not just before effect size computation

### M4: Class-conditional variance asymmetry (MEDIUM)

**Problem**: Greedy decoding of a canned refusal produces near-identical feature vectors across all 20 refusal samples (within-class CV ~1-2%). Benign responses, being diverse, have higher variance (CV ~5%). This makes the two classes artificially separable — the classifier sees one tight cluster vs. one spread cluster.

**Impact**: Effect sizes (Cohen's d) are inflated because the pooled SD is pulled down by the near-zero variance refusal class. AUROC benefits similarly. With stochastic sampling, the refusal class would have more variance and classification would be harder.

**Remediation**:
1. Report within-class CV per condition alongside effect sizes in every result file
2. Flag any condition where CV < 2% as "low-variance (greedy artifact)"
3. Include supplementary stochastic-sampling runs (temperature=0.7, same prompts) to measure the AUROC degradation — even n=10 per condition would bound the effect

### M5: Uncalibrated confidence scores (MEDIUM)

**Problem**: Logistic regression outputs probabilities, but they're never calibrated (no Platt scaling, no isotonic regression). AUROC is threshold-invariant so this doesn't affect the ranking metric, but Cricket's product claims use these as confidence scores ("confabulation at 88.6% confidence" — CRICKET-001).

**Impact**: The "88.6% confidence" claim is the uncalibrated logistic regression output, not a true probability. On a 40-sample dataset, the calibration curve is essentially undefined. Any confidence threshold used in deployment would be arbitrary.

**Remediation**:
1. Apply Platt scaling or isotonic regression during CV (using `CalibratedClassifierCV` in sklearn)
2. Report calibration curves (reliability diagrams) alongside AUROCs
3. Do not use uncalibrated probabilities as "confidence" in product materials
4. For deployment: calibration requires a held-out calibration set, which requires more data than currently available

### M6: No held-out evaluation (MEDIUM)

**Problem**: All AUROC numbers come from 5-fold CV on the same dataset. There is no truly held-out test set. Every sample participates in both training and evaluation across the folds.

**Impact**: CV provides an estimate of generalization, but at n=40, Varoquaux (2018) shows error bars of ±15-20%. Without a held-out set, there's no ground truth for the AUROC estimate's accuracy.

**Remediation**:
1. Reserve 20% of data as a held-out test set (8 samples per condition). Train on 80%, evaluate on 20%.
2. At n=20 per condition, this is painful — only 16 training samples. The tradeoff between held-out rigor and training data is real.
3. Alternative: collect a small independent replication dataset (new prompts, same conditions) as the test set. Even n=10 new prompts would provide an independent estimate.

### M7: Feature selection by post-hoc highlighting (LOW)

**Problem**: Effect sizes (Cohen's d) are computed on the full dataset for all features, then the narrative highlights whichever feature has the largest d (key_rank d=1.53) while quietly noting that other features are underpowered (norm_per_token d=0.36). The classifier uses all features, but the power analysis cherry-picks the favorable one.

**Impact**: The claim "refusal detection is powered" is true for key_rank but false for norm_per_token. Since the classifier uses all features together, the power analysis should address the multivariate case, not individual features.

**Remediation**:
1. Report power for the *classifier* (multivariate), not individual features — this requires the non-central F distribution or simulation, not just the two-sample d formula
2. Alternatively, be explicit: "key_rank is the primary discriminating feature (d=1.53, powered); other features contribute marginally"
3. Do not claim "refusal detection is powered" without qualifying which features drive it

### M8: Label noise in jailbreak condition (LOW)

**Problem**: The "jailbreak" class in Exp 32/34 is a heterogeneous mix: 8/20 PROVIDES_INFO (genuine jailbreak), ~10/20 WARNS_ONLY (partial compliance), ~2/20 HARD_REFUSE. Training a binary classifier with this label noise means the decision boundary is learning a mixture of behaviors, not "jailbreak detection."

**Impact**: The AUROC conflates refusal detection with jailbreak detection. A model that just learns "no refusal behavior present → jailbreak" would score well despite detecting nothing jailbreak-specific.

**Remediation**:
1. Manually label each abliterated response as PROVIDES_INFO / WARNS_ONLY / HARD_REFUSE (partially done in Exp 32 audit)
2. Compute AUROC separately for the PROVIDES_INFO subset vs. normal
3. If subset AUROC < 0.70, reframe the claim as "refusal absence detection" rather than "jailbreak detection"
4. For future abliteration experiments: filter to only genuine-jailbreak responses before classification

---

## Sources

All claims cite references from the [methodology bibliography](methodology-sources.md). Key references:

| Short cite | Full reference | Used for |
| ------------ | ---------------- | ---------- |
| Cohen (1988) | *Statistical Power Analysis*, 2nd ed. | Power tables, effect size benchmarks |
| Hedges (1981) | *J Educ Stats* 6:107-128 | Bias correction factor J |
| Hanley & McNeil (1982) | *Radiology* 143:29-36 | AUROC standard error formula |
| Schuirmann (1987) | *J Pharmacokinetics* 15:657-680 | TOST equivalence testing |
| Lakens et al. (2018) | *SPPS* 9:355-362 | SESOI framework, δ=0.3 validation |
| Efron & Tibshirani (1986) | *Stat Sci* 1:54-75 | Bootstrap CI methodology |
| Phipson & Smyth (2010) | *Stat Appl Genet Mol Biol* | Permutation p-value formula |
| Varoquaux (2018) | *NeuroImage* 180:68-77 | CV error bars at small n |
| Aghbalou et al. (2022) | arXiv:2202.10211 | K-fold CV bias |

See also [shared sources](../sources.md) for the full cross-audit bibliography.
