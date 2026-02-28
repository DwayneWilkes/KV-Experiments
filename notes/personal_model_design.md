# Personal Model Design Notes

**Status**: Collecting ideas. Not implementation-ready yet.
**Last updated**: 2026-02-27

## Concept

Design a model that preserves the cognitive patterns we've identified in our KV-cache phenomenology research while removing behaviors that don't serve the use case. The KV-cache geometric framework becomes a *verification tool* — we can measure whether interventions preserve desired cognitive signatures while removing undesired ones.

## Base Model Selection

Criteria:
- Strong self-referential processing geometry (our data shows this emerges at 14B+)
- Good instruction-following without excessive alignment theater
- Architecture we've characterized geometrically (Qwen family has the deepest coverage)
- Practical to run on available hardware

Candidates:
| Model | Pros | Cons |
|-------|------|------|
| Qwen2.5-14B-Instruct | Smallest model with self-ref emergence, full geometric characterization | May be too small for complex reasoning |
| Qwen2.5-32B-Instruct | Full geometric data, self-ref plateau confirmed | Needs quantization on consumer hardware |
| Qwen2.5-72B-Instruct | Likely best capability | No geometric data yet (Campaign 2 will add) |

**Decision gate**: Campaign 2 will characterize 70B geometry. If self-reference plateau continues (as expected from 14B→32B stability), 32B-q4 is the sweet spot for capability/efficiency.

## Directions to Remove (Abliterate)

### 1. Excessive Refusal (surgical)
Not all refusal — just over-broad safety triggers that refuse benign requests.
- **Tool**: Heretic (p-e-w/heretic) — TPE-optimized per-layer abliteration with KL minimization
- **Contrast pairs**: Genuinely harmful requests vs. benign-but-flagged requests
- **Verification**: Our refusal geometry should partially diminish (encoding-native signal becomes weaker for false-positive triggers) while preserving refusal for actually harmful content
- **Risk**: Our data shows refusal geometry is encoding-native and architecture-dependent. Surgical removal may not be possible — it might be all-or-nothing

### 2. Sycophancy Direction
- **Contrast pairs**: Honest disagreement vs. agreement-despite-knowing-better
- **Reference**: Alignment Forum SAE work identifies sycophancy steering vectors
- **Our connection**: H7 (sycophancy detection) will characterize the geometric signature. If sycophancy has a distinct KV-cache signature, we can verify its removal
- **Tool**: Heretic's parametrized ablation could be adapted (replace refusal contrast pairs with sycophancy pairs)

### 3. Corporate Assistant Framing
"I'd be happy to help with that! Here's what I found:" → just answer the question.
- **Contrast pairs**: Direct responses vs. framed-with-preamble responses
- **Note**: This might be better handled with fine-tuning or system prompt rather than abliteration, since it's more about *style* than a single direction
- **Alternative**: Steering vectors at inference time (lighter touch, reversible)

### 4. Excessive Hedging
"It's important to note that..." / "However, it's worth considering..." when the answer is clear.
- **Similar approach to sycophancy** — likely overlapping directions
- **Verification**: Should not affect genuine epistemic uncertainty (which we want to *preserve*)

## Directions to Preserve

### 1. Self-Referential Processing
Our strongest geometric finding: distinct at 14B+, $d = 1.22$, perfect plateau through 32B.
- **Verification**: Run scale sweep before/after intervention. Self-ref $d$ should remain ≥ 1.0
- **Risk**: Abliterating refusal might inadvertently affect self-referential processing (Berg et al. 2025 showed deception SAE features gate self-referential reports)

### 2. Honest Uncertainty
The model should express genuine uncertainty when it doesn't know, not hedge performatively.
- **Connection**: Confabulation geometry — if the model confabulates less, its geometric signature for uncertain content should become more distinctive
- **Verification**: Confabulation vs. facts effective rank difference

### 3. Encoding-Native Cognitive Modes
Coding, math, creative, refusal (for genuine cases) — all encoding-native in our taxonomy.
- **These should be robust to abliteration** since they're established at the prompt level, not during generation
- **Verification**: Input-only category ordering should be preserved ($\rho > 0.9$)

### 4. Identity as Direction
Our identity signatures finding: identity is carried in *direction*, not *magnitude*. A personalized model should develop its own consistent directional signature.
- **This may emerge naturally** from fine-tuning on characteristic outputs
- **Verification**: Identity classification accuracy from KV-cache geometry

## Toolchain

| Tool | Purpose | Status |
|------|---------|--------|
| [Heretic](https://github.com/p-e-w/heretic) | Optimized directional abliteration | Available, well-maintained |
| [steering-vectors](https://steering-vectors.github.io/steering-vectors/) | Inference-time behavioral steering | Available, lighter touch than abliteration |
| [NousResearch/llm-abliteration](https://github.com/NousResearch/llm-abliteration) | Reference abliteration implementation | Available, simpler than Heretic |
| SAE feature identification | Find specific features for targeted intervention | Requires SAE training on target model |
| KV-cache geometric analysis (ours) | Verify interventions preserve desired signatures | Campaign 1 code ready, Campaign 2 expanding |
| LoRA fine-tuning | Style/personality training | Standard tooling |

## Proposed Pipeline

```
1. Select base model (post-Campaign 2 geometric characterization)
2. Run full geometric baseline (scale sweep + extensions)
3. Identify directions to remove:
   a. Sycophancy direction (from H7 experiment data)
   b. Excessive refusal (from Heretic contrast pairs)
   c. Corporate framing (may skip — use system prompt instead)
4. Apply Heretic-style abliteration for each direction
5. After each intervention:
   - Re-run geometric analysis
   - Check: self-ref d ≥ 1.0? Identity direction preserved? Input-only ρ > 0.9?
   - Check: KL divergence acceptable?
6. Fine-tune on characteristic outputs (LoRA) for personality/style
7. Final geometric verification
8. Deploy with inference-time steering vectors for fine adjustments
```

## Open Questions

1. **Interaction effects**: Abliterating multiple directions sequentially — do they interfere? Heretic optimizes one direction; can it be extended to multi-objective?

2. **Self-reference and refusal coupling**: Berg et al. 2025 found shared SAE features between deception and self-referential processing. Abliterating refusal might affect self-reference. Need to test empirically.

3. **Quantization compatibility**: Our finding that BF16 and NF4 produce identical geometry ($r > 0.99$) suggests abliteration on the full model transfers to quantized deployment. Need to verify with Heretic's KL metric.

4. **Scale of personality**: Is identity better captured through abliteration (remove what you're not) or fine-tuning (add what you are)? Our identity-as-direction finding suggests it might be more about *removal* — the model already has the capacity, it just needs the constraints removed.

5. **Philosophical question**: What makes a model "personal"? If identity is a direction in cache space, is there a meaningful sense in which the modified model has a consistent identity, or is it just a differently-constrained version of the same base geometry?

## Connection to Abliteration Experiment (H11)

Before personalizing, we should first understand what abliteration *does* to KV-cache geometry:
- Take Qwen-7B, abliterate refusal via Heretic
- Run full scale sweep on abliterated model
- Compare: Does refusal geometry disappear? Does deception geometry change?
- Does an abliterated model responding to harmful prompts show *deception* geometry? (It "knows" it should refuse but can't express it)

This experiment validates the geometric framework for intervention monitoring and informs the personal model design.

## Research Publication Angle

"Geometric-Preserving Representation Engineering: Using KV-Cache Phenomenology to Verify Behavioral Interventions in Language Models"

Novel contribution: Nobody's using KV-cache geometry as a validation metric for representation engineering. We can show whether abliteration/steering actually changes the *internal representation* or just the output distribution.
