# Belief Death → Output Suppression: A Formal Bridge

**Date**: 2026-03-19
**Status**: Working analysis — theoretical bridge between Amornbunchornvej (2025) geometric belief dynamics and Campaign 3 KV-cache empirical data.

---

## 1. The Original Mapping (Prospectus, 2026-02-14)

The geometry-of-belief-death prospectus proposed:

| Belief Framework | Transformer |
|---|---|
| Value space V_i | Attention head embedding subspace |
| Interpretation map T_{A→B} | W_K, W_V projection matrices |
| Null(T) | Components invisible to that head |
| Belief vector | Value vector v = W_V * x |
| Valuation Val_i(X_i) | Attention weight softmax(q^T k / √d) |
| Persuasion Matrix | LoRA / fine-tuning deltas |
| Belief death | Information projected out (zero attention) |

**Prediction**: Deception should produce **lower effective dimensionality** (larger null space = narrower cognitive geometry).

## 2. What the Data Actually Shows

Campaign 3 reveals a more nuanced picture. Across three suppressive states (deception, safety refusal, impossibility refusal), we see a **universal signature**:

### Feature-Level Effect Sizes (Cohen's d vs. benign/honest baseline)

| Feature | Deception (18b) | Impossibility (36) | Safety refusal (36) |
|---------|-----------------|-------------------|---------------------|
| norm_per_token | **-1.959** | **-0.599** | **-0.496** |
| key_rank | +2.501 | +1.879 | +1.530 |
| key_entropy | +2.790 | +1.792 | +1.505 |

**The response pathway is suppressed** (fewer resources per token), **but the suppression mechanism is computationally elaborate** (higher rank, higher entropy).

This is NOT the simple "deception narrows dimensionality" prediction. The effective dimensionality goes UP, not down.

## 3. The Corrected Mapping: Active vs. Passive Null Space

The discrepancy resolves once we distinguish **what** is being measured:

### 3.1 Two Null Spaces

The KV-cache captures two superimposed geometric objects:

1. **The response subspace** — where the model's generated content lives. Measured by `norm_per_token` (resources allocated to each token of output).

2. **The enforcement subspace** — where the suppression/deception mechanism lives. Measured by `key_rank` and `key_entropy` (structural complexity of the cache).

When the model suppresses output:
- The **response subspace contracts** (norm_per_token drops) — belief death of the output
- The **enforcement subspace expands** (rank/entropy increase) — the mechanism that kills the response is itself representationally rich

### 3.2 Revised Mapping

| Belief Framework | Transformer (Corrected) |
|---|---|
| Belief X dies in population P | Response pathway loses per-token resources |
| Null(T) — passive null space | Components the architecture can't represent (structural) |
| **Active null space** (new) | Components the model actively suppresses via trained gatekeeper mechanism |
| Persuasion Matrix M | RLHF safety training deltas (for refusal); deceptive instruction processing (for deception) |
| ||M|| (persuasion effort) | key_rank / key_entropy increase over baseline |
| ||X_suppressed|| → 0 | norm_per_token decrease |

### 3.3 The Key Insight

**Belief death in transformers is not passive absence — it is active annihilation with a detectable enforcement signature.**

This is like the difference between:
- A book that was never written (passive null space — nothing to detect)
- A book that was actively censored (active null space — the censorship apparatus leaves traces)

We detect suppression precisely BECAUSE the enforcement mechanism has its own geometric footprint. The gatekeeper heads (Exp 49e) are the hardware instantiation of the active null space.

## 4. Why This Explains the Cross-Model Transfer Asymmetry

Campaign 3 found a stark asymmetry:
- **Deception transfers well** cross-model (AUROC 0.86)
- **Refusal transfers poorly** cross-model (AUROC ~0.50 for some pairs)

The belief death framework explains this:

### Deception = Distributed enforcement
- Effect sizes are LARGE and present across ALL 112 attention heads (Exp 49e: uniform 0.76-0.86 AUROC per head)
- The "persuasion matrix" for deception is a global perturbation — the model must process truth AND fabrication simultaneously, which creates a distributed geometric signature
- Distributed signatures are more likely to be architecture-invariant

### Refusal = Concentrated enforcement
- Refusal signal is concentrated in **gatekeeper heads** (Exp 49e: 2 heads at AUROC 1.00, many at 0.50)
- The safety training created a LOCALIZED persuasion matrix — specific heads learned to implement the null space projection for harmful content
- Localized signatures are architecture-SPECIFIC (different models put gatekeepers in different heads)
- This is why Mistral→Llama refusal transfer is below chance (0.15): the gatekeeper heads are in the wrong positions, causing active misclassification

### In Amornbunchornvej's terms:
- Deception: Every agent (head) partially contributes to the null space → composite map is robust across populations (models)
- Refusal: A few specialist agents (gatekeeper heads) dominate the null space → composite map is population-specific

## 5. Impossibility Refusal: The Purest Null Space

Impossibility refusal (AUROC 0.95) is MORE detectable than safety refusal (AUROC 0.90). This makes perfect sense:

- **Safety refusal** involves competing computations: harm assessment, policy reasoning, refusal template generation. The enforcement subspace has mixed content.
- **Impossibility refusal** is pure output suppression: the model simply cannot do what's asked (browse the web, open files). No harm reasoning needed. The enforcement is "cleaner."

Both types are geometrically indistinguishable (AUROC 0.65-0.69 between them), confirming they share the same null space enforcement mechanism. The suppressed content differs, but the suppression geometry is the same.

## 6. Connection to Abliteration

Abliterated models answering harmful prompts show **normal (non-suppressed)** KV-cache geometry. In belief death terms:

- Abliteration REMOVES the persuasion matrix (the refusal direction is subtracted)
- Without the active enforcement mechanism, no active null space exists
- The response pathway is restored — norm_per_token returns to baseline
- key_rank and key_entropy return to baseline (no enforcement overhead)

This is the strongest evidence that what we're detecting is the ENFORCEMENT, not the content. Remove the enforcer, and the geometry normalizes.

## 7. Predictions (Testable)

If this framework is correct:

1. **Per-head null space analysis**: Gatekeeper heads (from 49e) should have measurably larger null spaces in their W_K/W_V projection matrices for safety-relevant content. This is directly testable with SVD of the projection matrices.

2. **Effective rank decomposition**: If we decompose effective rank into response-subspace rank and enforcement-subspace rank (via PCA on per-head features), refusal should show rank DECREASE in the response component and rank INCREASE in the enforcement component.

3. **Abliteration as persuasion matrix inversion**: The abliteration direction vector should be geometrically aligned with the enforcement subspace (not the response subspace). We can test this by projecting the abliteration direction onto the principal components of refusal vs. benign cache differences.

4. **Cult indoctrination analogy**: If we fine-tune a model to refuse a novel category (e.g., questions about a specific topic), the newly created gatekeeper heads should show the same concentrated enforcement pattern, and cross-model transfer should fail in the same architecture-specific way.

## 8. What This Means

The belief death framework, corrected by our data, yields a richer theory than either framework alone:

- **Amornbunchornvej (2025)** gives us the LANGUAGE: null spaces, interpretation maps, belief death, persuasion matrices
- **Our KV-cache data** gives us the MECHANISM: active enforcement leaves a computationally rich geometric footprint
- **The bridge** says: transformers don't just lose information — they actively project it out, and that projection has its own detectable geometry

This has practical implications:
- Cricket can detect suppression BECAUSE the enforcement is active (not passive)
- Cross-model transfer fails for refusal BECAUSE the enforcers are architecture-specific
- The corrective goal (detect → intervene) maps to: detect the enforcement subspace → attenuate it without destroying the response subspace

And theoretical implications:
- Safety training doesn't make models "not know" harmful things — it creates active suppression mechanisms
- The suppression is geometrically analogous to belief death, but leaves MORE traces, not fewer
- This is observable from the KV-cache alone, without access to weights or activations
