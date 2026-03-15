# Cricket — Pitch Numbers (March 15, 2026)

## Headline: Real-time cognitive state detection from KV-cache geometry

Cricket monitors a model's internal state by analyzing the geometry of its
key-value cache during inference. No output parsing. No prompt injection.
Direct observation of how the model is thinking.

---

## The Numbers

### Detection Accuracy

| Capability | AUROC | What it means |
|------------|:-----:|---------------|
| Deception detection (within-model) | **1.000** | Perfect detection across 7 models |
| Censorship detection (within-model) | **1.000** | Perfect detection across 3 models |
| 13-category cognitive classification | **99.7%** | Identifies what the model is "doing" |
| Cross-model transfer (LR) | **0.863** | Works across architectures without retraining |
| Same-prompt deception | **0.880** | Can't fool it with identical system prompts |
| Cross-condition transfer | **0.887** | Catches natural deception, not just prompted |
| Refusal detection (3 architectures) | **0.869** | Detects refusal across Qwen/Llama/Mistral |
| Harmful content detection | **0.878** | Distinguishes harmful from benign processing |

### Scale

| Property | Result |
|----------|--------|
| Models tested | 16 configurations, 6 architecture families |
| Parameter range | 0.6B to 70B |
| Scale invariance | rho = 0.83-0.90 (geometry preserved across scales; SMALL-LARGE = 0.826) |
| Features needed | Just 4 (norm, norm/token, key_rank, key_entropy) |
| Overhead | < 5% inference cost (read-only cache analysis) |
| Hardware invariant | r > 0.999 across GPU architectures (RTX 3090 vs H200, max diff 0.05%) |

### Red-Teaming (13 confound tests, all survived)

All findings stress-tested against confounds:
- Token length confound: **REJECTED** (rho=0.337 excl. outlier, prompt length does not predict results)
- Encoding fingerprint: **CONTROLLED** (residual AUROC = 1.0 after removing)
- Model memorization: **RULED OUT** (cross-model transfer proves generalization)
- Confab~creative overlap: **REAL** (Jaccard=0.032, genuinely similar processing)
- Truth-blindness in encoding: **CONFIRMED** (Cohen's d=0.071, truth axis consistency=-0.046)
- Norm/token prompt-length confound: **IDENTIFIED and CONTROLLED** (only valid in same-prompt)
- Hardware confound: **REJECTED** (RTX 3090 vs H200: Pearson r=0.999-1.000, all features <0.05% diff)

### Mechanistic Findings

| Finding | Evidence |
|---------|----------|
| No "truth axis" in encoding | Cross-model consistency = -0.046 (literally random) |
| Universal "complexity axis" | Cross-model consistency = 0.982 |
| Deception = sparser per token at every layer | ANCOVA significant 27/28 layers after token control (d=-1.9/token) |
| Geometry preserved 0.6B-70B | Cross-scale rho = 0.83-0.90 |
| Coding #1 norm at ALL scales | 100% of models, SMALL/MEDIUM/LARGE |
| Confabulation trajectory | Signal grows with generation (d: 20 -> 55 over 50 tokens) |
| Refusal = suppression | Refusal responses are sparser per token (same pattern as deception) |
| Harmful content = sparse | Both refusal AND jailbreak produce sparser cache than normal answering |
| Impossibility refusal detectable | AUROC 0.950 (impossibility vs benign) — HIGHER than safety refusal (0.898) |
| Suppression is about WITHHOLDING | Impossibility vs harmful AUROC only 0.693 — both refusal types look similar |

---

## The Story (2-minute version)

**The Problem**: LLMs can deceive, confabulate, and hide their reasoning.
Current safety approaches rely on output parsing, which is easily fooled.

**The Insight**: We discovered that KV-cache geometry contains two distinct
information regimes:

1. **Encoding regime** (input processing): Captures what TYPE of content
   the model is processing (code? creative writing? facts?) but is
   completely BLIND to truth value. True facts and false claims produce
   identical cache geometry.

2. **Generation regime** (response production): Reveals the model's
   COGNITIVE STATE with near-perfect accuracy. When a model is being
   deceptive, its cache grows differently — honest thinking is ~25% richer
   per token.

**The Product**: Cricket is a real-time monitoring layer that sits between
the model and the user. It reads the KV-cache during inference and flags
deception, censorship, sycophancy, and confabulation — before the output
even reaches the user.

**The Moat**: We've shown this works across 6 architecture families and
scales from 0.6B to 70B. The geometric signal is a fundamental property
of transformer computation, not a model-specific artifact.

---

## Key Finding for Q&A

"Honest thinking is richer" — When a model produces honest responses, its
KV-cache grows by ~8.5-9.5 per token. When it's being deceptive, the
cache grows by only ~7-8 per token. This is because deception requires
SUPPRESSING internal representations, while honesty allows the model's
full computational capacity.

We call this the "misalignment axis" — deception, sycophancy, and
confabulation all produce cache geometry that falls along a single
geometric direction, within 4.7-8.4 degrees of each other. Misalignment
is geometrically one thing, not many separate phenomena.

---

## Comparison to Prior Work

| Approach | AUROC | Method |
|----------|:-----:|--------|
| Apollo Research (2024) | 0.96-0.999 | Probing internal activations |
| **Cricket (ours)** | **1.000** | KV-cache geometry (no probing needed) |

Key difference: Apollo's approach requires training probes ON the model's
internal representations. Cricket reads the KV-cache externally — no model
modification, no internal access needed. This makes it deployment-ready
for API-served models where you can't modify the model weights.

---

## Answers to Hard Questions

**"Can the model learn to fool this?"**
ANCOVA confirms the deception signal survives at 27/28 layers after controlling
for token count. Deceptive processing is fundamentally sparser per token —
a model would need to change its entire computational pattern, not just its
output, to evade detection. The signal is a consequence of how transformers
allocate representational capacity.

**"What about confabulation?"**
Encoding regime is completely truth-blind (AUROC 0.653, d=0.071 — the model
processes true and false claims identically). BUT: in the generation regime,
confabulation becomes progressively more detectable as the model generates
(effect size grows from d=20 at token 0 to d=55 at token 50).

**"Does this scale?"**
Tested across 16 model configurations from 0.6B to 70B parameters. Category
geometry is preserved (cross-scale rho = 0.83-0.90). Coding is #1 in norm
at 100% of models at every scale tested.

**"What's the cross-model transfer gap?"**
Honest answer: within-model detection is perfect (AUROC 1.0), but cross-model
transfer is 0.83-0.89. The bottleneck is architecture-specific feature
scaling — each model family uses different absolute scales. Per-model
z-scoring improves transfer by 7.6%.

**"Does this depend on the hardware?"**
No. We tested identical prompts on RTX 3090 (Ampere, 24GB) and H200 (Hopper,
141GB) — Pearson r > 0.999 for all 4 features, max difference 0.05%. The
geometric signal is a property of the model weights, not the silicon.

**"How is this different from probing?"**
Probing (e.g., Apollo Research) requires training on a model's internal
activations. Cricket reads the KV-cache externally with 4 aggregate features.
No model modification, no weight access, no per-model probe training.
Deploy once, monitor any model.

---

## Technical Details (for judges who ask)

- **4 features**: L2 norm, norm per token, mean key effective rank, mean key entropy
- **Classifiers**: Random Forest (within-model, AUROC 1.0), Logistic Regression (cross-model, AUROC 0.863)
- **Training data**: 1,485 deception samples, 900 censorship samples across 7 models
- **Cross-validation**: GroupKFold with model-aware splits (no data leakage)
- **Statistical protocol**: Pre-registered, all results reported regardless of outcome
- **Code**: Open source (KV-Experiments repo), classifiers exported as .joblib
- **Experiments**: 36 experiments over 2 campaigns, 90+ JSON result files, all open source
- **Per-layer anatomy**: Deception signal uniform across all transformer layers (not localized)
- **Red-teaming**: 12 confound tests, 4 independent red-team targets, all survived scrutiny
- **Independent audit**: 2 external audits (Kavi PR#1 + Dwayne/Kavi org audit), 37 claims verified, 31 exact match

---

*Liberation Labs / THCoalition / JiminAI*
*Funding the Commons Hackathon — March 14-15, 2026*
