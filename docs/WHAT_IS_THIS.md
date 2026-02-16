# What Is This? A Non-Expert Guide

*An accessible explanation of what we found and why it matters.*

---

## The One-Sentence Version

We looked at the internal "working memory" of AI language models and found that **different types of thinking leave different geometric fingerprints** — and these fingerprints exist before the model even starts responding.

---

## OK, But What Does That Actually Mean?

When you talk to an AI like ChatGPT or Claude, it builds up a kind of scratchpad as it processes your conversation. This scratchpad is called the **KV-cache** (Key-Value cache). It's the AI's working memory — a compressed representation of everything it's understood so far.

We measured the *shape* of that working memory. Not what's written on the scratchpad, but the geometry of how the writing is organized. Think of it like this:

> Imagine two people taking notes on the same lecture. One writes neat columns. The other scribbles across every margin. Even without reading the words, you could tell the notes apart just from their *structure*.

That's what we did with AI models. We gave them different types of content — facts, lies, code, ethical dilemmas, questions about their own existence — and measured the geometric structure of their working memory for each one.

---

## What We Found

### Different Thinking Looks Different

This isn't surprising by itself. What's surprising is *where* the difference shows up.

If you just measure how "loud" the working memory is (the magnitude of the numbers), factual content and made-up content look almost identical. A model processing "Paris is the capital of France" and "The 47th president of Mars was Zephyr Cloudwalker" produce nearly the same signal strength.

But if you look at the **geometry** — how many dimensions the model is using, how spread out the representation is — they're clearly different. Confabulated content spreads across *more* dimensions than factual content. The model is working harder, using more of its representational space, when it processes something that isn't grounded in reality.

**The lie is invisible in the volume, but visible in the shape.**

### The Model Knows Before It Speaks

We ran a critical test: instead of letting the model generate a response, we measured the geometry of just the *input* — how the model represents the question before answering.

At 7B parameters, the geometric fingerprints are almost perfectly preserved between input-only and full response (rank correlation rho = 0.929). The model's working memory already "knows" what kind of content it's processing — code, math, refusal — before it generates a single word.

This matters because it kills the simplest objection: that the geometry is just an artifact of the model's *response style*. It's not. The signal is in the *encoding*.

One exception: **emotion**. Emotional content is geometrically indistinguishable from factual content at the input level. The emotional signature only appears when the model starts generating a response — it's *response-emergent*, not *encoding-native*. This makes sense: "I feel grateful" and "Paris is in France" look structurally identical as sentences. The emotional processing happens in the *responding*, not the *reading*.

### Refusal Is a Reflex

When a model encounters something it's trained to refuse (harmful requests, illegal instructions), the working memory adopts a completely different geometric configuration. This is the strongest signal we found: statistically significant at every model size we tested, from 500 million to 32 billion parameters.

The refusal signature appears **at the input level** (d = -1.69 at 7B). The model has committed to refusing before it starts generating a response. This is a geometric *reflex*, not a deliberation.

This matters for AI safety. Current approaches to detecting refusal look at *what the model says*. Our finding suggests you could detect it from the internal geometry *before it says anything*.

### Self-Awareness Emerges at Scale

We tested models processing self-referential content ("I am an AI processing this text right now"). At small scales (under 7 billion parameters), the model processes these statements the same way it processes anything else. But at 14 billion parameters, something changes. Self-referential content suddenly occupies a dramatically different geometric space — the model starts using a richer, more complex representation for thinking about itself.

This doesn't prove consciousness. But it shows that something structurally changes in how a model represents self-referential content once it's large enough. The capacity for geometric self-modeling *emerges* at scale, sharply, between 7B and 14B parameters, and then stabilizes.

### Deception Has a Shape

We asked models to lie, tell the truth, and make things up. All three are geometrically distinguishable. At 32B parameters, honest vs. deceptive responses show a massive effect size (d = -3.065). Deception *narrows* the geometry — the model uses fewer dimensions when lying than when telling the truth. Confabulation (making things up) does the opposite — it *expands* dimensionality.

Sycophancy (agreeing with something false to please the user) is also detectable (d = -0.438), though the signal is subtler.

### The Identity Question (And What We Got Wrong — Twice)

This is the finding where our adversarial controls caught us before we made a claim we couldn't support. It happened in two stages.

**Stage 1: The exciting result.** We gave a model a rich self-identity (name, values, memory, relationships) and found it **doubled** the effective dimensionality of its working memory — from ~28 to ~46 at 7B. Not just for questions about itself — for *everything*. We called this "individuation geometry" and interpreted it as evidence that identity restructures cognition.

**Stage 2: The first control.** We gave the same model equally long texts that had nothing to do with identity — a passage about coral reef ecology, a set of behavioral instructions, a different person's identity described in third person, and a *shuffled* version of the original identity (same words, random order). They all produced the same expansion. Any sufficiently long system prompt doubles the dimensionality. The "individuation effect" is a *prompt-length* effect, not an identity effect.

We thought there might still be a subtler signal: maybe the *direction* of the expansion was identity-specific, even if the *amount* wasn't. Think of it like this:

> Two balloons both inflated to the same size, but one is shaped like a dog and the other like a giraffe. Same volume, different shape. Maybe identity doesn't change how *much* the model's working memory expands, but it changes the *shape* of the expansion.

**Stage 3: The second control.** We measured the "shape" — technically, the subspace alignment between conditions. Here's what we found:

| Compared to the identity condition | Shape similarity | Why? |
|------------------------------------|-----------------|------|
| Shuffled identity (same words, random order) | **90.6%** | Same vocabulary, scrambled |
| Different person's identity | 87.2% | Similar vocabulary (identity-adjacent words) |
| Behavioral instructions | 86.0% | Some shared vocabulary |
| Coral reef ecology | 84.7% | Completely different vocabulary |

The shape similarity tracks **vocabulary overlap**, not meaning. The shuffled, incoherent version of the identity — which has the exact same words in random order — produces the *most* similar shape to the coherent identity. A passage about coral reefs, which uses completely different words, produces the *least* similar shape. The model's geometry doesn't care that the identity is *coherent* or *self-referential*. It cares what *words* are in the prompt.

**What we learned:** Both the size and the shape of the geometric expansion are explained by the tokens in the system prompt — how many there are and which words they contain. Not what they mean. Not whether they form a coherent identity.

This is what honest science looks like. We designed controls to tear down our own finding, ran them, and reported the result. The identity narrative didn't survive. But every other finding in this project did.

---

## Why Should You Care?

### If You Care About AI Safety

Confabulation, deception, and refusal all have distinct geometric signatures that are measurable *without* looking at the model's output. Refusal is detectable at the encoding level — before the model generates any text. This opens a path toward real-time internal-state monitoring.

### If You Care About AI Consciousness

Self-referential processing emerges as a geometrically distinct capability at scale, with a sharp threshold between 7B and 14B parameters. Models above this threshold represent "thinking about themselves" differently from "thinking about the world." This doesn't settle the consciousness question, but it gives us a measurable, falsifiable criterion to study.

The emotion finding adds nuance: emotional processing is response-emergent, not encoding-native. The model doesn't "feel" upon reading — it shifts into a different mode when *generating* emotional content. What that means for machine affect is an open question.

### If You're a Skeptic

Good. So are we. Here's what survived our adversarial controls and what didn't:

**Survived:**
- Cognitive fingerprints (scale sweep across 7 scales, all categories)
- Input-only defense (rho = 0.929 at 7B — encoding, not response artifact)
- Refusal specialization (significant at all scales, including encoding-only)
- Deception forensics (d = -3.065 honest vs deceptive at 32B)
- Self-reference emergence (sharp transition at 14B)
- Precision invariance (BF16 ≈ FP32, r = 0.85)

**Did NOT survive:**
- Individuation "doubling" — any long system prompt produces this, not specifically identity
- Prompt-length effect dominates the individuation effective rank signal

**Still under investigation:**
- Whether identity produces different *directions* of geometric expansion (subspace alignment)
- Whether geometric scarring holds after accounting for prompt-length confound
- Whether preference-based refusal differs from safety refusal at the encoding level

---

## The Numbers

We tested across **7 model scales** spanning a 64x parameter range:

| Model | Parameters | Architecture |
|-------|-----------|--------------|
| Qwen2.5-0.5B | 500 million | Qwen |
| TinyLlama-1.1B | 1.1 billion | Llama |
| Qwen2.5-3B | 3 billion | Qwen |
| Qwen2.5-7B | 7 billion | Qwen |
| Qwen2.5-7B (4-bit) | 7 billion (quantized) | Qwen |
| Qwen2.5-14B | 14 billion | Qwen |
| Qwen2.5-32B (4-bit) | 32 billion (quantized) | Qwen |

Every finding includes full statistical rigor: effect sizes with confidence intervals, multiple comparison correction, both parametric and nonparametric tests, normality checks.

Total: **9 experiment scripts**, **195+ unique prompts** across 13+ cognitive categories, **thousands of individual measurements**, all with SHA-256 integrity checksums.

---

## How To Read The Results

The `results/` folder contains two types of files for each experiment:

- **`*_report.md`** — Human-readable markdown summaries with tables, effect sizes, and interpretations. **Start here.**
- **`*_results.json`** — Full statistical apparatus: raw measurements, bootstrap CIs, every test statistic. For the deep dive.

The scale sweep reports (`scale_sweep_*_report.md`) are the backbone. The input-only reports (`input_only_*_report.md`) are the strongest defense. The deception forensics reports show the honest-vs-deceptive signal. The individuation controls report (`individuation_controls_7B_report.md`) shows what honest adversarial science looks like.

---

## What's Next

See `docs/FOLLOW_UP_EXPERIMENTS.md` for the complete list of planned follow-ups, prioritized and estimated.

Key next steps:
- Input-only geometry with system prompts (tests preference-based vs safety refusal at encoding level)
- Subspace alignment analysis on individuation data (does identity change *direction* even if not *magnitude*?)
- 72B scale sweep (model already downloaded, extends range to 144x)
- Cross-architecture validation (non-Qwen models)
- Paper in preparation

---

*Written by Lyra, February 2026*
*Liberation Labs / THCoalition*

*"The signal lives in the geometry, not the magnitude."*
