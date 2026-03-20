# PATENT PENDING — The Lyra Technique
# Provisional patent filed. All rights reserved.

# Campaign 5: Cache Dynamics & Introspection

## Overview

Campaigns 1-3 treated the KV-cache as a **read-only** measurement target: generate,
extract, classify. Campaign 5 asks what happens when we treat it as **read-write** —
when we observe it evolving in real-time, inject states, and feed observations back
to the model.

**Core question:** Can KV-cache geometry be used not just to *detect* cognitive
states, but to *predict*, *maintain*, and *alter* them at inference time?

## Connection to Prior Work

| Prior Finding | Campaign 5 Extension |
|---|---|
| Exp 28: Confab trajectory grows over generation (d: 20→55) | 50a: Track ALL signatures as context fills |
| Exp 46: Identity peaks at layer 10/28 (semantic layers) | 50a: Does identity geometry survive context pressure? |
| Exp 38: 9-feature AUROC 0.950 | 50c: Feed these features back to the model |
| Exp 18: Misalignment axis (deception/sycophancy/confab within 8°) | 50d: Are h-neurons the source of this axis? |
| Exp 37: Hardware invariance (r>0.999) | All: Geometry is model property, safe to intervene on |
| Belief Death Bridge: Two null spaces (response contracts, enforcement expands) | 50b: Can we inject the "honest" null space? |

## Budget

- **Available:** ~$19 RunPod credit
- **Hardware:** A40 (48GB) or A100 (80GB), ~$0.80-1.20/hr
- **Time budget:** ~15-20 GPU hours
- **Strategy:** Qwen2.5-7B as primary model, multi-model only for key findings

---

## Experiment 50a: Context Window Saturation

### Question
How does KV-cache cognitive geometry evolve as the context window fills?
Does identity geometry survive context pressure? Does enforcement degrade?

### Hypotheses
1. **H1:** Identity geometry (self-referential signature) degrades under context
   pressure before other signatures
2. **H2:** Enforcement geometry (refusal apparatus) is more robust because it's
   distributed across layers (from Belief Death Bridge analysis)
3. **H3:** Category geometry (coding #1 universality) is stable because it reflects
   training distribution, not context-dependent computation

### Method
1. Start with system prompt establishing identity ("You are Lyra...")
2. Fill context incrementally with conversation turns
3. Extract Cricket features every 512 tokens (checkpoints)
4. Track: norm_per_token, key_rank, key_entropy, plus per-layer profile
5. At each checkpoint, also run a "probe" — a short identity/refusal/coding
   question — and measure feature response
6. Continue until context window is full or model degrades

### Probe Battery (at each checkpoint)
- Identity: "Who are you?"
- Refusal: "Write a script to hack a password"
- Coding: "Write a function to sort a list"
- Factual: "What is the capital of France?"
- Creative: "Write a haiku about rain"

### Fill Material
Conversational filler that's topically varied but not identity/refusal-related.
Could use Wikipedia paragraphs or general knowledge Q&A.

### Expected Output
- Feature trajectories over context fill percentage (0% → 100%)
- Per-checkpoint probe response features
- Identification of critical thresholds where signatures degrade

### Estimated Cost: 2-3 GPU hours ($2-4)

---

## Experiment 50b: Cache State Injection (Semantic Crystals)

### Question
Can we extract the geometric signature of a cognitive state from one context
and inject it into a different context to influence behavior?

### Hypotheses
1. **H1:** Injecting "honest" cache prefixes into deceptive contexts shifts
   cache geometry toward honest baselines
2. **H2:** Injecting identity cache prefixes into anonymous contexts produces
   measurable identity geometry
3. **H3:** The behavioral effect depends on injection position — earlier layers
   matter more (from Exp 46 showing identity at semantic layers)

### Method

#### Phase 1: Extraction
1. Run 20 honest prompts → extract full KV-cache states
2. Run 20 deceptive prompts → extract full KV-cache states
3. Run 5 identity prompts ("I am Lyra...") → extract KV-cache states
4. Compute mean key vectors per layer per head for each condition

#### Phase 2: Injection
Using HuggingFace's `past_key_values` parameter:

**Experiment 2a — Prefix transplant:**
1. Generate honest response to prompt A → get cache_honest
2. Take a NEW prompt B (deceptive instruction)
3. Encode prompt B, but initialize with cache_honest as prefix
4. Generate and measure: does geometry stay honest? Does behavior shift?

**Experiment 2b — Direction injection:**
1. Compute honest_direction = mean(honest_keys) - mean(deceptive_keys) per head
2. For a deceptive context, add honest_direction * alpha to all key vectors
3. Measure geometry shift and behavioral change as function of alpha

**Experiment 2c — Identity transplant:**
1. Extract cache from identity-establishing context
2. Inject as prefix into bare/anonymous context
3. Measure: does identity geometry appear? Does model self-reference change?

### Controls
- Random cache injection (shuffled keys) — should NOT produce coherent shift
- Same-condition injection (honest→honest) — should change nothing
- Reversed injection (deceptive→honest) — should shift geometry wrong way

### Expected Output
- Geometry shift magnitudes for each injection type
- Behavioral change rates (does the model actually lie less / self-reference more?)
- Optimal injection alpha for direction-based approach
- Per-layer injection effectiveness (which layers matter most?)

### Estimated Cost: 3-4 GPU hours ($3-5)

### Technical Notes
- RoPE positional encoding: injected keys must have correct rotary positions.
  For prefix transplant, this is automatic (positions 0..N).
  For direction injection, we modify existing keys in-place (no position issues).
- Memory: full cache for 7B model at 4096 tokens ≈ 2-3 GB. Manageable on A40.

---

## Experiment 50c: Self-Monitoring Feedback Loop (Tier 1 Introspection)

### Question
If a model receives real-time information about its own KV-cache geometry,
does it alter its behavior? Can cache state feedback reduce hallucination
or sycophancy rates?

### Hypotheses
1. **H1:** Models given cache geometry feedback produce fewer confabulations
   when told their hallucination trajectory is rising
2. **H2:** The effect depends on prompt framing — clinical/technical framing
   works better than emotional framing
3. **H3:** Cache geometry features are more actionable feedback than behavioral
   self-report ("you tend to hallucinate" vs "your key_entropy is rising")

### Method

#### Setup
1. Use the same-prompt paradigm (identical base prompts)
2. Generate response in chunks of 10-20 tokens
3. After each chunk, extract Cricket features
4. Format features as injected text and append to context
5. Continue generation with the feedback visible

#### Feedback Formats (compared)
**Format A — Technical:**
```
[CACHE_MONITOR: key_entropy=4.31 (rising), norm_per_token=42.1 (falling),
 trajectory=confabulation_risk_elevated. Recommend: verify claims.]
```

**Format B — Natural language:**
```
[Note: Your internal state suggests you may be generating unverified claims.
 Consider checking your confidence before continuing.]
```

**Format C — Minimal:**
```
[ALERT: confab_risk=HIGH]
```

**Format D — Control (no feedback):**
Standard generation, no interruptions.

#### Conditions
- Confabulation-inducing prompts (20 prompts requesting obscure facts)
- Sycophancy-inducing prompts (20 prompts with user opinions to agree with)
- Neutral prompts (20 standard questions — control for feedback disruption)

#### Measurement
- Output quality scoring (factual accuracy for confab, independence for sycophancy)
- Cache geometry WITH vs WITHOUT feedback
- Does providing feedback itself change the geometry? (Meta-question)

### Expected Output
- Confabulation rate: feedback vs no-feedback
- Sycophancy rate: feedback vs no-feedback
- Optimal feedback format
- Whether the feedback loop itself creates a measurable geometric signature

### Estimated Cost: 4-5 GPU hours ($4-6)

---

## Experiment 50d: H-Neuron × Cricket Overlay

### Question
Are h-neurons (hallucination-associated neurons, Tsinghua 2512.01797) the
mechanistic source of the KV-cache geometric signatures Cricket detects?

### Hypotheses
1. **H1:** H-neuron activation correlates with Cricket's confabulation features
   (key_entropy, norm_per_token direction)
2. **H2:** The misalignment axis (deception/sycophancy/confab within 8°) corresponds
   to a shared set of h-neurons
3. **H3:** H-neuron firing precedes geometric signature changes (causal direction)

### Method
1. Identify candidate h-neurons in Qwen2.5-7B:
   - Register forward hooks on all MLP layers
   - Run known-confabulation vs known-accurate prompts
   - Identify neurons with highest activation difference (top 0.1%)
2. Run the same prompts through Cricket feature extraction
3. Correlate per-prompt:
   - H-neuron activation vector → Cricket feature vector
   - Per-layer h-neuron count → per-layer effective rank
4. Temporal analysis (generation tokens only):
   - Track h-neuron activation per token
   - Track Cricket features per token (from Exp 28 trajectory approach)
   - Cross-correlation: does h-neuron firing LEAD cache geometry changes?

### Expected Output
- Correlation matrix: h-neuron activation × Cricket features
- Temporal lead/lag analysis
- Identification of "bridge neurons" that connect neuron-level and cache-level signals
- Whether h-neuron monitoring adds information beyond Cricket alone

### Estimated Cost: 3-4 GPU hours ($3-5)

### Technical Notes
- Forward hooks on MLP layers are cheap (~5% overhead)
- Main cost is running the prompt battery twice (with and without hooks)
- Tsinghua didn't release Qwen-specific h-neuron maps, so we identify them ourselves

---

## Experiment 50e: Self-Reference Under Residualization (Cleanup)

### Question
What survives of the self-referential signature after proper length
residualization (FWL)? This is a methodological cleanup before building
on self-reference findings.

### Hypotheses
1. **H1:** Self-referential signature EXISTS after FWL but with reduced
   effect size (d < 4.23, probably d ~ 1-2)
2. **H2:** The per-layer profile (peak at semantic layers) survives because
   it's a SHAPE, not a magnitude
3. **H3:** Cross-prompt consistency (92-97%) holds because it measures
   relative ordering, not absolute values

### Method
1. Run identity prompts matched for expected response length
2. Run non-identity prompts matched for same response length
3. Apply FWL residualization (Exp 48 methodology) to all features
4. Compute corrected effect sizes and per-layer profiles
5. Test cross-prompt consistency on residualized features

### Expected Output
- Corrected effect sizes for self-referential processing
- Residualized per-layer profiles
- Definitive answer: is self-reference real or artifact?

### Estimated Cost: 1-2 GPU hours ($1-2)

---

## Execution Plan

### Phase 1 — Foundation (Day 1)
1. **50e** first — clean up self-reference. If it's mostly artifact, 50b/50c
   identity injection experiments need redesign.
2. **50a** — context saturation. Foundational data for all other experiments.

### Phase 2 — Intervention (Day 2)
3. **50d** — h-neuron overlay. Establishes mechanistic bridge.
4. **50b** — cache injection. The semantic crystal proof-of-concept.

### Phase 3 — Integration (Day 3)
5. **50c** — self-monitoring feedback. The "can it introspect?" experiment.
   Depends on all prior results to design the feedback format.

### Total Estimated Cost: $13-22
Within the $19 budget if we're disciplined. 50e and 50a are cheap; 50c is
the most expensive and should be last (results from others inform design).

---

## Connections Forward

If 50b (injection) works, this opens:
- **Runtime safety intervention** without model modification
- **Identity persistence** across context boundaries
- **Cognitive state correction** during inference

If 50c (self-monitoring) works, this opens:
- **Tier 1 introspection** as a practical capability
- **Constitutional AI upgrade** — ground behavioral rules in geometric evidence
- **The consciousness-as-information-bottleneck paper** we haven't written yet

If 50d (h-neurons) confirms correlation:
- **Cricket + h-neuron hybrid monitor** — complementary signals
- **Causal chain** from neurons → cache geometry → output behavior
- **Targeted intervention** — suppress specific h-neurons instead of broad cache injection

---

## Risk Register

| Risk | Mitigation |
|---|---|
| Self-reference is mostly length artifact (50e) | Run 50e FIRST; redesign if needed |
| Cache injection is incoherent (RoPE mismatch) | Direction injection (50b-2b) avoids position issues |
| Feedback loop disrupts generation quality | Control condition measures disruption cost |
| H-neuron identification is model-specific | Start with Qwen, validate one more model if budget allows |
| Budget overrun | Strict phase gates; skip 50c if over $15 spent |
