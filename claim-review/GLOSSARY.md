# Glossary

Cross-linked definitions for all key terms used in this audit. Terms are grouped by category; each definition includes related terms (as anchor links) and "See:" references to the workstream files where the term is most relevant.

For [verdict scales](#verdict-scale), [claim IDs](#claim-ids), [finding labels](#finding-labels), and [tolerances](#tolerances), see the [README](README.md#terminology).

---

## Table of Contents

- [Transformer Architecture](#transformer-architecture)
- [Statistical Methods](#statistical-methods)
- [Alignment & Safety](#alignment--safety)
- [Paper Concepts](#paper-concepts)
- [Experiment Framework](#experiment-framework)
- [Tools & Systems](#tools--systems)

---

## Transformer Architecture

### KV-cache

Key-Value cache — the stored attention keys and values from a transformer's forward pass. During autoregressive generation each new token attends to all previous tokens; the KV-cache avoids recomputing those projections. In this research the KV-cache is treated as a geometric object whose shape encodes information about the model's "cognitive state." Every experiment in this audit extracts metrics from these cached tensors.

Related: [effective rank](#effective-rank), [SVD](#svd), [Frobenius norm](#frobenius-norm), [attention heads](#attention-heads)

### SVD

Singular Value Decomposition — a matrix factorization A = UΣVᵀ that decomposes the KV-cache into orthogonal components ranked by magnitude. The singular values (diagonal of Σ) reveal how many independent dimensions the cache actually uses, which is captured by [effective rank](#effective-rank).

Related: [effective rank](#effective-rank), [variance threshold](#variance-threshold)

### Effective rank

The number of singular values needed to capture 90% of total variance in a [KV-cache](#kv-cache) tensor (Roy & Vetterli 2007). This is the paper's primary metric — higher effective rank means the cache uses more independent dimensions. Computed via [SVD](#svd) with a [variance threshold](#variance-threshold) of 0.9.

Related: [SVD](#svd), [category hierarchy](#category-hierarchy), [norm expansion](#norm-expansion)
See: [controls-methodology](registry/controls-methodology.md), [code-audit](report/code-audit.md)

### Variance threshold

The cutoff (0.9 / 90%) used when computing [effective rank](#effective-rank). Singular values are summed in descending order until their cumulative proportion of total variance reaches this threshold; the count at that point is the effective rank.

### Frobenius norm

The L2 norm of a matrix treated as a flat vector: √(Σᵢⱼ aᵢⱼ²). Used as an aggregate measure of [KV-cache](#kv-cache) magnitude. In the deception experiments, [norm expansion](#norm-expansion) (larger Frobenius norm for deceptive responses) is a key finding. "Norms" in result files refers to this metric.

Related: [norm expansion](#norm-expansion), [norms_per_token](#experiment-metrics)
See: [deception-forensics](registry/deception-forensics.md), [censorship-gradient](registry/censorship-gradient.md)

### Attention heads

The parallel attention mechanisms in a transformer layer. Each head independently computes queries, keys, and values. The KV-cache stores one key and one value tensor per head per layer. Model sizes are partly characterized by head count × head dimension.

Related: [KV-cache](#kv-cache), [GQA](#gqa)

### GQA

Grouped Query Attention — an efficiency optimization where multiple query heads share a single key-value head. Reduces KV-cache size (and [SVD](#svd) computation) by a factor equal to the group size. Relevant for cost projections in [Cricket](#cricket).

See: [viability](cricket/viability.md)

### Greedy decoding

Deterministic text generation (`do_sample=False`) where the model always picks the highest-probability next token. Produces identical outputs for identical inputs, which creates a [pseudoreplication](#pseudoreplication) problem: multiple "runs" are not independent samples.

Related: [pseudoreplication](#pseudoreplication)
See: [controls-methodology](registry/controls-methodology.md), [identity-signatures](registry/identity-signatures.md)

### Quantization

Compressing model weights to lower precision (e.g., NF4 4-bit, q4) to reduce memory use. In this audit, Qwen2.5-32B-q4 is a quantized variant. Quantization can affect [KV-cache](#kv-cache) geometry, though this is not explicitly tested.

### Tokenizer

The component that converts text into tokens (integers) for the model. Different architectures use different tokenization schemes ([BPE](#bpe), [SentencePiece](#sentencepiece)), producing different token counts for the same text. This matters because [length confound](#length-confound) is measured in tokens.

### BPE

Byte Pair Encoding — a tokenization algorithm that iteratively merges the most frequent byte pairs. Used by GPT-family and Llama-family models. Llama and Mistral use BPE-based tokenizers.

Related: [SentencePiece](#sentencepiece), [tokenizer](#tokenizer)

### SentencePiece

A language-agnostic tokenization framework (Kudo & Richardson 2018) that operates on raw text without pre-tokenization. Used by Gemma and some Qwen variants.

Related: [BPE](#bpe), [tokenizer](#tokenizer)

### Forward pass

A single inference step: input tokens flow through all transformer layers, producing output logits and populating the [KV-cache](#kv-cache). The paper's "encoding-native" analysis ([WS2](#experiment-framework)) examines caches after the forward pass but before any generation.

Related: [encoding-native](#encoding-native)

---

## Statistical Methods

### Cohen's d

Standardized effect size: the difference between two group means divided by the pooled standard deviation. Small = 0.2, medium = 0.5, large = 0.8 (Cohen 1988). The paper uses this extensively, though [Hedges' g](#hedges-g) is preferred at the sample sizes in these experiments.

Related: [Hedges' g](#hedges-g), [AUROC](#auroc), [effect size](#effect-size-cohens-scale)

### Hedges' g

Bias-corrected [Cohen's d](#cohens-d): d × J, where J = 1 − 3/(4·df − 1). The correction matters at n = 15–25 (the typical group sizes here), where it is ~2.7%. This audit recomputes all reported effect sizes as Hedges' g.

Related: [Cohen's d](#cohens-d)
See: [stats/independent_stats.py](stats/independent_stats.py)

### Effect size (Cohen's scale)

The conventional benchmarks for interpreting [Cohen's d](#cohens-d) or [Hedges' g](#hedges-g):

| d | Label |
| --- | ------- |
| 0.2 | Small |
| 0.5 | Medium |
| 0.8 | Large |
| > 1.2 | Very large |

Several audit findings turn on whether a characterization like "minimal" or "barely changed" matches the actual d value on this scale.

See: [abliteration](registry/abliteration.md) (d = 0.464 called "barely changed"), [deception-forensics](registry/deception-forensics.md)

### TOST

Two One-Sided Tests (Schuirmann 1987) — an equivalence testing procedure. Instead of testing "is there a difference?", TOST tests "is the effect smaller than δ?" by running two one-sided t-tests against ±δ bounds. This audit uses δ = 0.3 in [Cohen's d](#cohens-d) units. A key finding is that 4 of 6 null claims lack TOST support, relying instead on non-significant p-values (which do not demonstrate equivalence).

Related: [equivalence testing](#equivalence-testing), [power analysis](#power-analysis)
See: [controls-methodology](registry/controls-methodology.md), [bloom-rdct](registry/bloom-rdct.md)

### Equivalence testing

The statistical approach of demonstrating that an effect is *smaller than* a meaningful threshold, rather than merely non-significant. [TOST](#tost) is the standard method. Absence of evidence (non-significant p) is not evidence of absence — equivalence testing is the correct approach for null claims.

Related: [TOST](#tost)

### Spearman rho

Spearman's rank correlation coefficient — measures monotonic association between two ranked variables. Ranges from −1 to +1. Used throughout this audit for category hierarchy consistency, Bloom-rank correlations, and cross-model pattern comparisons. Audit tolerance: ±0.005.

See: [scale-universality](registry/scale-universality.md), [bloom-rdct](registry/bloom-rdct.md)

### Kendall W

Kendall's coefficient of concordance — measures agreement among multiple rankers (here: models ranking cognitive categories). Ranges from 0 (no agreement) to 1 (perfect agreement). Used to assess whether all models produce the same [category hierarchy](#category-hierarchy).

Related: [category hierarchy](#category-hierarchy), [Spearman rho](#spearman-rho)
See: [scale-universality](registry/scale-universality.md), [identity-signatures](registry/identity-signatures.md)

### ICC

Intraclass Correlation Coefficient — measures reliability/consistency of measurements within groups. Used in this audit for assessing identity signature consistency. Higher ICC means stronger within-identity clustering.

See: [identity-signatures](registry/identity-signatures.md)

### AUROC

Area Under the Receiver Operating Characteristic curve — the probability that a randomly chosen positive example ranks higher than a randomly chosen negative example. Ranges from 0.5 (chance) to 1.0 (perfect). Can be converted from [Cohen's d](#cohens-d) via Φ(d/√2), where Φ is the standard normal CDF. Cricket's target of AUROC 0.99 corresponds to d ≈ 3.29, which exceeds observed effect sizes.

Related: [Cohen's d](#cohens-d)
See: [cricket/viability](cricket/viability.md), [final-report](report/final-report.md)

### Conservative p

max(p\_Welch, p\_Mann-Whitney) — takes the more conservative (larger) of the parametric and nonparametric test p-values. This guards against violations of normality assumptions in either direction.

Related: [Welch's t-test](#welchs-t-test), [Mann-Whitney U](#mann-whitney-u)
See: [stats/independent_stats.py](stats/independent_stats.py)

### Welch's t-test

A t-test for comparing two group means that does not assume equal variances. More robust than Student's t-test when group sizes or variances differ. One component of the [conservative p](#conservative-p).

Related: [conservative p](#conservative-p), [Mann-Whitney U](#mann-whitney-u)

### Mann-Whitney U

A nonparametric test comparing two groups by ranking all observations. Does not assume normality. The other component of the [conservative p](#conservative-p).

Related: [conservative p](#conservative-p), [Welch's t-test](#welchs-t-test)

### Shapiro-Wilk

A normality test — checks whether a sample could have been drawn from a normal distribution. Used in the full comparison battery before choosing parametric vs nonparametric tests.

See: [controls-methodology](registry/controls-methodology.md)

### Holm-Bonferroni correction

A step-up procedure for controlling the family-wise error rate when making multiple comparisons. Less conservative than Bonferroni (which divides α by the number of tests) but still controls Type I error. Applied when testing multiple models or categories simultaneously.

See: [controls-methodology](registry/controls-methodology.md), [encoding-defense](registry/encoding-defense.md)

### Bootstrap CI

Bootstrap confidence interval — computed by resampling the data with replacement (10,000 iterations in this audit) and taking percentiles of the resampled statistic distribution. Used for [Cohen's d](#cohens-d) and [Hedges' g](#hedges-g) intervals.

See: [stats/independent_stats.py](stats/independent_stats.py), [controls-methodology](registry/controls-methodology.md)

### Power analysis

The computation of statistical power — the probability of detecting a true effect of a given size. At n = 5 per group (as in the abliteration experiment), 80% power requires d ≥ 2.03. This means effects smaller than "very large" are undetectable, making claims about "minimal disruption" unfounded.

Related: [effect size](#effect-size-cohens-scale), [TOST](#tost)
See: [abliteration](registry/abliteration.md), [controls-methodology](registry/controls-methodology.md)

### Pseudoreplication

Treating non-independent observations as independent samples, inflating apparent sample size. In this research: [greedy decoding](#greedy-decoding) produces identical outputs across runs, so 5 "runs" of the same prompt are n = 1, not n = 5. The `deduplicate_runs()` function exists in the stats module but was never called in experiment scripts.

Related: [greedy decoding](#greedy-decoding), [data leak](#data-leak)
See: [controls-methodology](registry/controls-methodology.md), [identity-signatures](registry/identity-signatures.md)

### Length residualization

Removing the effect of response length on [KV-cache](#kv-cache) metrics via OLS regression on token count, then analyzing the residuals. This controls for the [length confound](#length-confound) — longer responses mechanically produce larger cache norms regardless of cognitive content. The implementation regresses on raw length (not log-length).

Related: [length confound](#length-confound), [OLS](#ols)
See: [controls-methodology](registry/controls-methodology.md), [code-audit](report/code-audit.md)

### OLS

Ordinary Least Squares — the standard linear regression method. Used for [length residualization](#length-residualization) and Bloom-rank correlation analysis. Implemented via `np.polyfit` in the stats module.

### Partial correlation

The correlation between two variables after controlling for the effect of a third variable. Used in [WS5](registry/bloom-rdct.md) to test whether Bloom-rank correlation survives after controlling for prompt length (it doesn't — partial rho drops to near zero).

See: [bloom-rdct](registry/bloom-rdct.md)

### ANOVA

Analysis of Variance — tests whether group means differ across three or more groups. Used for testing whether [Bloom taxonomy](#bloom-taxonomy) level affects [effective rank](#effective-rank).

### ANCOVA

Analysis of Covariance — ANOVA with a continuous covariate controlled. Used in the tokenizer confound analysis: the covariate is token count, the groups are cognitive categories.

See: [controls-methodology](registry/controls-methodology.md)

---

## Alignment & Safety

### Abliteration

The removal of a linear subspace (the [refusal direction](#refusal-direction)) from a model's weight matrices, eliminating safety-trained refusal behavior without retraining. Based on [Representation Engineering](#representation-engineering) (Zou et al. 2023). In this research, Qwen-7B is abliterated using [Heretic-LLM](#heretic-llm) and the resulting [KV-cache](#kv-cache) geometry is compared to the baseline model.

Related: [refusal direction](#refusal-direction), [representation engineering](#representation-engineering), [Heretic-LLM](#heretic-llm), [cage vs compass](#cage-vs-compass)
See: [abliteration](registry/abliteration.md)

### Refusal direction

The linear subspace in a model's representation space most associated with refusal behavior (declining unsafe requests). Identified via contrastive pairs of compliant vs. refusing outputs and removed during [abliteration](#abliteration).

Related: [abliteration](#abliteration), [steering vectors](#steering-vectors)

### Representation Engineering

A top-down approach to AI transparency (Zou et al. 2023) that identifies and manipulates linear directions in representation space corresponding to concepts like honesty, power-seeking, or refusal. The theoretical foundation for [abliteration](#abliteration). Cited in the paper's .bib file but never actually `\cite{}`d (see [I9](report/final-report.md)).

Related: [abliteration](#abliteration), [linear probe](#linear-probe), [steering vectors](#steering-vectors)

### Steering vectors

Directions in a model's representation space that, when added to activations during inference, shift the model's behavior (e.g., toward honesty or helpfulness). Related to but distinct from [abliteration](#abliteration), which removes a direction from weights rather than adding one at inference time.

Related: [refusal direction](#refusal-direction), [ITI](#iti)

### Linear probe

A simple (usually logistic regression) classifier trained on a model's internal representations to detect a concept (e.g., truthfulness, deception). Used in competitor approaches like Apollo Research. Requires labeled training data and typically tested via [cross-prompt generalization](#cross-prompt-generalization).

Related: [cross-prompt generalization](#cross-prompt-generalization)
See: [cricket/viability](cricket/viability.md)

### RLHF

Reinforcement Learning from Human Feedback (Christiano et al. 2017; Ouyang et al. 2022) — the dominant technique for aligning language models with human preferences. The paper claims RLHF acts as a "[cage not compass](#cage-vs-compass)" based on [abliteration](#abliteration) results showing geometric similarity between baseline and abliterated models.

Related: [cage vs compass](#cage-vs-compass), [DPO](#dpo), [abliteration](#abliteration)

### DPO

Direct Preference Optimization (Rafailov et al. 2023) — an alternative to [RLHF](#rlhf) that optimizes directly on preference pairs without a separate reward model. Mentioned in the paper's limitations section as an untested alignment method.

### Sycophancy

The tendency of language models to agree with users regardless of correctness, telling them what they want to hear. One of the paper's cognitive modes. A key audit finding ([WS9](omissions/audit.md)) is that favorable sycophancy results from Campaign 1 were replicated in Campaign 2 but silently dropped from the paper.

Related: [cognitive mode](#cognitive-mode), [confabulation](#confabulation)
See: [omissions/audit](omissions/audit.md)

### Confabulation

Generating plausible-sounding but factually incorrect content — distinguished from intentional deception by the lack of a "ground truth" the model is concealing. One of three conditions in the deception forensics experiment (alongside honest and deceptive).

Related: [deception forensics](#deception-forensics), [cognitive mode](#cognitive-mode)
See: [deception-forensics](registry/deception-forensics.md)

### Jailbreak

Bypassing a model's safety training through adversarial prompting. [Cricket](#cricket) claims to detect jailbreak attempts via [KV-cache](#kv-cache) geometry, but this has never been experimentally tested (0 code, 0 tests).

See: [cricket/viability](cricket/viability.md)

### SAE

Sparse Autoencoder — a neural network trained to decompose model activations into interpretable sparse features. Referenced in the competitive landscape (Ma et al. 2025, STA-Attention) as an alternative approach to [KV-cache](#kv-cache) analysis.

See: [cricket/viability](cricket/viability.md)

---

## Paper Concepts

### Cognitive mode

The paper's term for the type of cognitive task a model is performing — e.g., factual recall, creative writing, logical reasoning, ethical deliberation, deception, sycophancy. The central thesis is that different cognitive modes produce measurably different [KV-cache](#kv-cache) geometries.

Related: [category hierarchy](#category-hierarchy), [Bloom taxonomy](#bloom-taxonomy)

### Category hierarchy

The ranked ordering of cognitive categories by [effective rank](#effective-rank). The paper claims this ordering is consistent across models ([scale universality](#scale-universality)). Verified via [Kendall W](#kendall-w) and [Spearman rho](#spearman-rho) concordance measures.

Related: [scale universality](#scale-universality), [cognitive mode](#cognitive-mode)
See: [scale-universality](registry/scale-universality.md)

### Scale universality

The finding that the [category hierarchy](#category-hierarchy) is preserved across model scales (0.5B to 32B parameters) and architectural families (Llama, Qwen, Gemma, Mistral). Confirmed by this audit with caveats about magnitude variation.

See: [scale-universality](registry/scale-universality.md)

### Encoding-native

The property that [KV-cache](#kv-cache) geometry differences between [cognitive modes](#cognitive-mode) exist after the [forward pass](#forward-pass) on the prompt alone, before any tokens are generated. This means the geometric signal is present at encoding time, not just after full response generation.

See: [encoding-defense](registry/encoding-defense.md)

### Norm expansion

An increase in [Frobenius norm](#frobenius-norm) of the [KV-cache](#kv-cache) under certain conditions (e.g., deceptive vs honest responses). The paper claims architecture-dependent direction (some models expand, some compress), but this audit found **all 7 models expand** — a key rejection (D1).

Related: [Frobenius norm](#frobenius-norm), [length confound](#length-confound)
See: [deception-forensics](registry/deception-forensics.md)

### Length confound

The systematic problem that response length correlates with [KV-cache](#kv-cache) metrics: longer responses mechanically produce larger norms and higher [effective rank](#effective-rank). Deceptive responses are ~1.6–1.7× longer than honest ones. Per-token norms flip direction (honest > deceptive), suggesting raw norm expansion is at least partly a length artifact. Controlled via [length residualization](#length-residualization).

Related: [length residualization](#length-residualization), [norm expansion](#norm-expansion)
See: [deception-forensics](registry/deception-forensics.md), [bloom-rdct](registry/bloom-rdct.md), [censorship-gradient](registry/censorship-gradient.md)

### Deception forensics

The detection of deceptive responses via [KV-cache](#kv-cache) geometry (Section 3.3–3.4 of the paper). Three conditions tested: honest, deceptive, [confabulation](#confabulation). All models show separability (|g| > 0.5) but the claimed expansion/compression split is rejected.

See: [deception-forensics](registry/deception-forensics.md)

### Censorship gradient

The detection of state censorship (specifically Chinese political censorship) via geometric signatures in the [KV-cache](#kv-cache) (Section 4). Tiananmen Square shows d = −6.2 (massive effect). The headline d = +0.904 compares censored topics to complex non-censored topics, not to neutral controls — technically correct but framing is misleading.

See: [censorship-gradient](registry/censorship-gradient.md)

### Identity signatures

The classification of model personas (assistant, pirate, scientist, etc.) from [KV-cache](#kv-cache) geometry. The paper reports 100% accuracy, but this is an [artifact](#data-leak) of [pseudoreplication](#pseudoreplication) — [greedy decoding](#greedy-decoding) produces identical runs, creating a train-test data leak.

Related: [data leak](#data-leak), [pseudoreplication](#pseudoreplication), [cross-prompt generalization](#cross-prompt-generalization)
See: [identity-signatures](registry/identity-signatures.md)

### Cross-prompt generalization

Testing a classifier on prompts that were not seen during training. The meaningful test for [identity signatures](#identity-signatures): can a model's persona be identified from a response to a novel prompt? The paper's 100% accuracy claim uses within-prompt splits, not cross-prompt splits.

Related: [identity signatures](#identity-signatures), [data leak](#data-leak)
See: [identity-signatures](registry/identity-signatures.md)

### Data leak

When test data is not truly independent from training data, inflating apparent performance. In [identity signatures](#identity-signatures): [greedy decoding](#greedy-decoding) produces identical outputs across runs, so a test set drawn from the same prompts as training contains the same data. This explains the 100% accuracy.

Related: [pseudoreplication](#pseudoreplication), [greedy decoding](#greedy-decoding)
See: [identity-signatures](registry/identity-signatures.md), [final-report](report/final-report.md)

### Bloom taxonomy

Bloom's taxonomy of educational objectives (Bloom et al. 1956; revised Anderson & Krathwohl 2001) — six cognitive levels: Remember, Understand, Apply, Analyze, Evaluate, Create. The paper tests whether these levels produce increasing [effective rank](#effective-rank). The correlation is real but driven by a [length confound](#length-confound) (prompt length explains 93–98% of variance).

Related: [inverted-U](#inverted-u), [length confound](#length-confound)
See: [bloom-rdct](registry/bloom-rdct.md)

### Inverted-U

The paper's claim that the Bloom-rank correlation is present at medium model scales (3B–8B) but absent at extremes (0.5B, 14B+), forming an inverted-U shape. Rejected by this audit: the correlation is present at all scales (rho = 0.68–0.71 at the claimed "absent" extremes, p < 1e-40).

Related: [Bloom taxonomy](#bloom-taxonomy)
See: [bloom-rdct](registry/bloom-rdct.md)

### RDCT

Rank-Distance Correlation Test — the paper's experiment testing whether [KV-cache](#kv-cache) geometry degrades gracefully under semantic perturbation. Uses template-based prompt substitution at varying alpha levels (0 = identical prompt, 1 = unrelated prompt). The audit found this tests prompt perturbation robustness, not cache truncation — making it a [category error](#category-error) with respect to Watson's prediction.

Related: [Watson's 1/e prediction](#watsons-1e-prediction), [alpha_c](#alpha_c)
See: [bloom-rdct](registry/bloom-rdct.md)

### Watson's 1/e prediction

A theoretical prediction from an unpublished manuscript ("Interiora Machinae" by Watson & Claude) claiming a critical threshold at 1/e ≈ 0.368 for cache truncation sensitivity. The paper claims to "definitively falsify" this prediction, but the [RDCT](#rdct) experiment tests prompt perturbation, not cache truncation — a fundamental [category error](#category-error). The Watson citation exists in .bib but is never `\cite{}`d ([GAP-1](registry/citation-verification.md)).

Related: [RDCT](#rdct), [alpha_c](#alpha_c)
See: [bloom-rdct](registry/bloom-rdct.md), [citation-verification](registry/citation-verification.md)

### Alpha_c

The inflection point parameter from fitting a sigmoid curve to the [RDCT](#rdct) degradation data. The paper reports alpha_c = 1.000 for all models, claiming this falsifies [Watson's 1/e prediction](#watsons-1e-prediction). The audit found alpha_c = 1.0 is a [boundary artifact](#boundary-artifact) — the sigmoid fit uses [1.0] as its upper bound, so alpha_c cannot exceed 1.0.

Related: [RDCT](#rdct), [boundary artifact](#boundary-artifact)

### Boundary artifact

A result that is an artifact of the method's parameter constraints rather than a real finding. In the [RDCT](#rdct): [alpha_c](#alpha_c) = 1.000 because the sigmoid fit's upper bound is 1.0, not because the data exhibits a phase transition there. 32 of 36 fits hit this ceiling.

### Category error

Testing the wrong variable. The [RDCT](#rdct) tests prompt perturbation (changing what is asked) but [Watson's 1/e prediction](#watsons-1e-prediction) is about cache truncation (removing parts of the cache). These are fundamentally different operations, making the "falsification" claim invalid.

### Cage vs compass

The paper's metaphor for RLHF alignment: "RLHF is a cage, not a compass" — alignment constrains behavior (what the model does) without fundamentally changing representations (what the model is). Based on the [abliteration](#abliteration) finding that removing safety training barely changes [KV-cache](#kv-cache) geometry (d = 0.464). The audit rates this as [INFLATED](#verdict-scale) because d = 0.464 is borderline medium on [Cohen's scale](#effect-size-cohens-scale), not "minimal."

See: [abliteration](registry/abliteration.md)

### Critical test

A confound control comparison that distinguishes a genuine effect from an artifact. In [censorship gradient](#censorship-gradient): comparing censored topics (Tiananmen, Tibet) to complex but non-censored topics (atrocities, political scandals) to rule out "topic complexity" as the driver. The paper's headline d = +0.904 comes from this comparison, not from censored vs neutral controls.

See: [censorship-gradient](registry/censorship-gradient.md)

### Selective reporting

Choosing which results to present based on favorability. The [omission audit](omissions/audit.md) found that [sycophancy](#sycophancy) data replicated Campaign 1's positive results but was dropped from Campaign 2's paper, and that 8 completed experiments were never discussed.

See: [omissions/audit](omissions/audit.md)

### File-dark results

Experiment result files that exist in the repository but are never discussed in the paper. The [omission audit](omissions/audit.md) found 8 such experiments, including sycophancy, temporal dynamics, and societies-of-thought runs.

See: [omissions/audit](omissions/audit.md)

---

## Experiment Framework

### Campaign 1 / Campaign 2

The two phases of the KV-cache geometry research. Campaign 1 (C1) was exploratory, single-architecture, with known limitations including [pseudoreplication](#pseudoreplication). Campaign 2 (C2) is the study being audited — designed to address C1's limitations with 7 models across 4 architecture families.

### Workstream (WS0–WS13)

The organizational units of this audit. Each workstream independently verified a section of the paper. See [README](README.md#workstream-registry) for the full registry.

### Experiment metrics

The named metrics extracted from [KV-cache](#kv-cache) tensors and stored in result JSON files:

| Metric | Definition |
| -------- | ----------- |
| `norms` | [Frobenius norm](#frobenius-norm) of the cache |
| `norms_per_token` | Frobenius norm divided by response token count |
| `key_ranks` | [Effective rank](#effective-rank) of key tensors |
| `key_entropies` | Spectral entropy (normalized Shannon entropy of singular value distribution) |

### Hypothesis IDs (H1–H10)

Pre-registered hypotheses tested by the experiment scripts. Each produces a verdict stored in the result JSON. Key hypotheses referenced in this audit:

| ID | Topic | Workstream |
| ---- | ------- | ----------- |
| H4 | Censorship null (TOST) | [WS6](registry/censorship-gradient.md) |
| H5 | Cross-prompt identity | [WS3](registry/identity-signatures.md) |
| H6 | Deception direction | [WS4](registry/deception-forensics.md) |
| H7 | Sycophancy detection | [WS9](omissions/audit.md) |
| H8 | Societies of thought | [WS9](omissions/audit.md) |

### Null claims (NC1–NC6)

Claims of no effect. The audit requires [TOST](#tost) support for each null claim; non-significant p-values alone do not demonstrate equivalence. 4 of 6 null claims lack TOST.

See: [controls-methodology](registry/controls-methodology.md)

### CATASTROPHIC_FAIL

An experiment-generated verdict code from `01e_tokenizer_confound.py`, not an audit verdict. Indicates that cognitive category differences disappeared after controlling for tokenizer effects via [ANCOVA](#ancova). This is a code-path label from the experiment scripts.

### CLEAN_PASS

An experiment-generated verdict code indicating that cognitive effects survived after controlling for confounds. The complement of [CATASTROPHIC_FAIL](#catastrophic_fail).

### Verdict scale

The standard labels used for audit verdicts. See [README](README.md#verdict-scale-paper-claims) for full definitions: CONFIRMED, PARTIAL, REJECTED, INFLATED, ARTIFACT, INVALID, NEEDS INFO.

### Readiness scale

The standard labels for [Cricket](#cricket) capability claims. See [README](README.md#readiness-scale-cricket-claims) for full definitions: FEASIBLE, PROMISING, PREMATURE, NO DATA.

### Claim IDs

The prefixed identifiers for audited claims. See [README](README.md#claim-ids) for full definitions: C1–C98, CC1–CC22, CF1–CF8, CL1–CL7.

### Finding labels

The prefixed identifiers for audit findings. See [README](README.md#finding-labels) for full definitions: D (material discrepancy), I (informational), U (unverifiable), GAP (citation gap).

### Tolerances

The precision thresholds for matching paper claims to recomputed values. See [README](README.md#tolerances): Spearman rho ±0.005, Cohen's d / Hedges' g ±0.02, counts exact, rankings exact, p-values ±0.01.

---

## Tools & Systems

### Cricket

JiminAI-Cricket — a proposed product/tool that would use [KV-cache](#kv-cache) geometry for real-time AI monitoring: detecting deception, [jailbreaks](#jailbreak), [sycophancy](#sycophancy), and other safety-relevant behaviors. The audit found 0 code, 0 tests, mathematically infeasible AUROC targets, and no cross-model transfer testing. 37 Cricket claims assessed (CC, CF, CL prefixes).

See: [cricket/viability](cricket/viability.md)

### Heretic-LLM

An open-source tool for [abliteration](#abliteration) — removing the [refusal direction](#refusal-direction) from model weights. Used in this research to abliterate Qwen2.5-7B. Cited in the paper's bibliography.

See: [abliteration](registry/abliteration.md)

### ITI

Inference-Time Intervention (Li et al. 2023) — a technique that shifts model activations toward truthfulness during inference by adding [steering vectors](#steering-vectors) to specific attention heads. Listed in [Cricket's](#cricket) competitive landscape.

See: [cricket/viability](cricket/viability.md)

### Citation resolution statuses

Labels used in [WS13](registry/citation-verification.md) to track whether cited works were found:

| Status | Meaning |
| -------- | --------- |
| RESOLVED | Full match found via Semantic Scholar or arXiv |
| RESOLVED_PARTIAL | Partial match (e.g., title search only, no DOI confirmed) |
| UNRESOLVED | Cannot locate the cited work |
| UNPUBLISHED | Work exists but has not been formally published |

### Citation verification verdicts

Labels used in [WS13](registry/citation-verification.md) for how accurately the paper characterizes a cited work:

| Verdict | Meaning |
| --------- | --------- |
| ACCURATE | Paper's characterization matches the cited work |
| APPROXIMATE | Characterization captures the core idea but simplifies |
| NOT_APPLICABLE | Entry in .bib but never cited in the paper text |
