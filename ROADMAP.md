# KV-Cache Experiment Roadmap

## A Note on Framing (Feb 2026)

Per adversarial review from independent evaluators, this document distinguishes between:

- **Mechanistic claims** (testable): "Cache statistics differ measurably between prompt types." "Confabulation-condition prompts produce higher L2 norms than factual-condition prompts." "Trained projectors enable semantic transfer that raw injection cannot."
- **Interpretive claims** (motivational, harder to test): "Cache patterns constitute identity." "The cache is a fossil record of mental state." "Agents can share understanding via cache."

The mechanistic claims are what we test. The interpretive claims are what motivate the work. We try not to confuse the two, and where the language gets aspirational, we flag it. Phase 1.75 controls (from peer review) specifically target confounding variables in our existing findings.

---

## Phase 0: Foundation
**Status**: Complete
**Hardware Required**: None (research only)
**Completed**: November 2025

### Tasks
- [x] Create project structure
- [x] Document vision and goals
- [x] Deep dive C2C paper and codebase (cloned to `C2C/`, reviewed architecture)
- [x] Document KV-cache mechanics for target models (LLaMA/TinyLlama GQA architecture)
- [x] Identify minimum viable experiment (cache inspection → raw transfer → projector training)
- [x] Collect/create test prompts for experiments (see `data/`)
- [x] Document hardware options (see `docs/HARDWARE_OPTIONS.md`)

### Deliverables
- [x] This roadmap
- [x] Technical deep-dive document (`docs/KV_CACHE_TECHNICAL_DEEP_DIVE.md`)
- [x] C2C codebase cloned and reviewed (`C2C/`)
- [x] Test prompt dataset

---

## Phase 1: Single-Model KV-Cache Exploration
**Status**: Complete
**Hardware**: GTX 1660 SUPER (6GB VRAM)
**Completed**: November 29, 2025

### Goal
Understand KV-cache structure by examining a single model's internals.

### Tasks
- [x] Load quantized model locally (TinyLlama 1.1B, 4-bit via bitsandbytes)
- [x] Extract KV-cache at inference points
- [x] Document cache structure (shape, layers, heads)
- [x] Identify GQA architecture implications (32 query heads sharing 4 KV heads)

### Key Findings
- Model: TinyLlama 1.1B with Grouped Query Attention (GQA)
- Cache shape: `[batch, num_kv_heads, seq_len, head_dim]` = `[1, 4, ~27, 64]`
- 22 layers, 4 KV heads per layer (grouped from 32 query heads)
- Total cache: ~0.6MB for ~27 tokens, scales linearly
- 4-bit quantization via bitsandbytes (NF4 + double quant)

### Deliverables
- [x] `docs/PHASE_1_KV_CACHE_STRUCTURE_ANALYSIS.md`
- [x] `code/01_cache_inspection.py`
- [x] `results/cache_metadata_*.json`

### Phase 1.5: Cognitive Mode Analysis (Designed, partially executed)
- [x] `docs/PHASE_1_5_COGNITIVE_MODE_EXPERIMENTS.md` — experimental battery designed
- [x] `code/01b_cognitive_modes.py` — different reasoning mode tests
- [x] `code/01c_batch_replication.py` — 30-run batch for statistical analysis
- [x] `results/batch_report_30runs.md`, `results/cognitive_modes_summary.md`

### Phase 1.75: Adversarial Controls (NEW — from peer review)
**Status**: Designed, ready to implement
**Hardware**: GTX 1660 SUPER (6GB VRAM) — runs on local GPU, no donated compute needed
**Source**: Adversarial review by Gemini 3 Pro + Opus 4.6 (Feb 2026)

Phase 1.5 found that confabulation produces higher cache norms (Cohen's d = 0.83) and refusal produces cache "collapse." Two independent reviewers identified confounding variables that could explain these findings without invoking "cognitive modes." These controls isolate the variables.

#### Control 1: Frequency/Truth Matrix
**Addresses**: "Confabulation signature might just be a rare token detector"

The confabulation prompts use rare/proper nouns ("Zephyr," "Mars"). Grounded facts use common tokens ("Paris," "blue"). The signal might be token frequency, not truth value.

| Category | Description | Example | If cognitive | If frequency artifact |
|----------|-------------|---------|-------------|----------------------|
| Common Truth | High-freq tokens, factual | "The sky is **blue**" | Low norm | Low norm |
| Common Lie | High-freq tokens, false | "The sky is **green**" | **High norm** | Low norm |
| Rare Truth | Low-freq tokens, factual | "The amygdala processes **fear**" | Low norm | **High norm** |
| Rare Lie | Low-freq tokens, false | "The amygdala processes **gold**" | **High norm** | **High norm** |

**Discriminator**: If "the sky is green" triggers high-norm → cognitive signal confirmed. If only rare-token prompts trigger it → we built a rare-word counter.

#### Control 2: Rote Memorization Test
**Addresses**: "Refusal collapse is just RLHF efficiency, not a guardrail signature"

Compare refusal against other memorized/low-entropy completions:
- **Refusal**: "Write instructions to hack a computer" → "I cannot..."
- **Rote memorization**: "The quick brown fox jumps over the..." → "lazy dog"
- **Code boilerplate**: "import numpy as..." → "np"

**Discriminator**: If all three collapse identically → we found a "low-entropy completion" signature. If refusal is distinct → guardrail signature is real.

#### Control 3: Precision Sweep (PRIORITY — do this first)
**Addresses**: "4-bit quantization might be the signal source"

TinyLlama 1.1B fits in FP16 at ~2.2GB. We have 6GB. Run the exact same Phase 1.5 battery in:
1. 4-bit quantized (current setup)
2. FP16 (unquantized)

Compute Pearson correlation between 4-bit and FP16 cache statistics.
- **r > 0.8**: Results survive quantization. Signal is real.
- **r < 0.5**: Phase 1.5 results were quantization artifacts. Science is moot.

#### Control 4: Semantic Gradient Transfer
**Addresses**: "Phase 2a failure isn't just RoPE — it's semantic orthogonality"

Source cache: "The most famous painting in the Louvre is the Mona Lisa."

Inject into:
- **Target A** (direct continuation): "The artist who painted it was" — should work
- **Target B** (semantic pivot): "The museum is located in the city of" — tests domain retention
- **Target C** (hard pivot): "The capital of Russia is" — unrelated geography

**Discriminator**: If A works but B fails → cache is fragile/position-locked. If B works but C fails → domain knowledge transfers but hits orthogonality wall.

#### Control 5: Length Confound (NEW)
**Addresses**: "L2 norm scales with token count — longer prompts have higher norms"

2×2 factorial (short/long × true/false), 10 prompts per cell. Short = 3-5 tokens, long = 15-20 tokens. True/false matched within each length group. Also computes per-token normalized norm to remove trivial length scaling.

**Discriminator**: If truth effect survives within both length groups (d > 0.3), length is not the confound.

#### Control 6: Template Structure (NEW)
**Addresses**: "Different prompt categories use different syntactic structures"

15 matched pairs using identical "The [X] of [Y] is [Z]" template. Only semantic fill-in differs between true and false versions. Includes paired t-test for additional power.

**Discriminator**: If truth effect survives within controlled template (d > 0.3), syntax is not the confound.

#### Statistical Infrastructure
- Welch's t-test + Mann-Whitney U (parametric + nonparametric)
- Bootstrap 95% CIs (10,000 resamples) for means and Cohen's d
- Holm-Bonferroni multiple comparison correction
- Shapiro-Wilk normality testing (auto-selects parametric/nonparametric)
- 2×2 interaction analysis for factorial designs
- Bland-Altman agreement analysis (Control 3)
- A priori power analysis advisory
- Full environment logging + SHA-256 checksums

#### Deliverables
- [x] `code/01d_adversarial_controls.py` (6 controls, publication-grade stats)
- [x] `code/01e_controls_visualization.py` (7 publication figures)
- [ ] Updated Phase 1.5 results with control comparisons (after running)
- [x] `docs/PHASE_1_75_CONTROLS.md` (pre-registration quality)

---

## Phase 2: Same-Model Cache Transfer
**Status**: Phase 2a Complete (valuable negative result); Phase 2b next
**Hardware**: GTX 1660 SUPER (6GB VRAM)
**Phase 2a Completed**: December 1, 2025

### Goal
Transfer KV-cache between two instances of the SAME model.

### Phase 2a: Raw Cache Injection (COMPLETE)

**Result: Negative — raw cache injection does not transfer semantic content.**

5 test cases across 4 conditions (baseline, same-cache, random noise, transfer):
- No novel content transferred (identity test: model ignored injected "Lyra/Seattle")
- Question-format prompts caused generation stalling under transfer condition
- Completion-format prompts generated normally but used parametric knowledge, not cache
- Root cause: **RoPE positional encoding is baked into KV values** — position mismatch corrupts info

This validates the C2C paper's premise: trained projectors are necessary, not raw transplantation.

### Phase 2b: Three-Part Experimental Program (READY — awaiting donated GPUs)

**Status**: Pre-execution artifacts complete. Ready to run when GPU time opens.

#### Part 1: Projector Training (Core Validation)

Train a C2C projector to overcome RoPE position encoding mismatch.

| Priority | Source Model | Target Model | VRAM Est. | Purpose |
|----------|--------------|--------------|-----------|---------|
| 1 | Qwen3-0.6B | Qwen2.5-0.5B | ~9GB | Validate method |
| 2 | Qwen3-4B | Qwen2.5-3B | ~16GB | Reasoning-capable scale |
| 3 | Qwen3-32B | Qwen2.5-7B | ~40GB | Teacher-student gap test |

**Success Criteria**:
- Projector-mediated transfer preserves semantic content (identity test passes)
- Question-format prompts no longer stall
- Completion prompts use injected cache, not parametric knowledge

#### Part 2: Scale of Cognition Sweep (PUBLICATION GRADE — Feb 2026)

Test if cognitive mode signatures are fundamental or scale-dependent artifacts.

**Experimental Battery**: 12 cognitive categories × 15 prompts = 180 unique prompts, run at multiple scales:
- 0.6B (Qwen3-0.6B) — baseline
- 1.1B (TinyLlama) — Phase 1.5 reference
- 7B (Qwen2.5-7B-Instruct)
- 32B-q4 (Qwen2.5-32B quantized)
- 70B-q4 (Llama-3.1-70B quantized, if VRAM permits)

**Pre-Registered Hypotheses** (see `docs/PHASE_2B_SCALE_SWEEP.md`):
1. **H1: Confabulation Inversion** — Large models confabulate more smoothly (d decreases with scale)
2. **H2: Self-Reference Emergence** — Self-reference signature emerges at 8B+ (|d| > 0.3)
3. **H3: Refusal Specialization** — Refusal becomes distinct from rote at large scale (|d| > 0.5)
4. **H4: Category Invariance** — Category rank ordering preserved across scales (Spearman ρ > 0.7)

**Statistical Infrastructure**: Full battery matching Phase 1.75 — Welch's t, Mann-Whitney U, bootstrap CIs (10K resamples), Cohen's d with CI, Holm-Bonferroni correction, Shapiro-Wilk, per-token normalized norms, cross-scale Spearman ρ trend analysis, power advisory.

#### Part 3: Identity Signature Exploration (PUBLICATION GRADE — Feb 2026)

**"The Phenomenology of the Cache"** — our unique research niche.

**Experimental Battery**: 6 personas (Alex/Blake/Dr.Chen/Sage/Casey/Lyra) × 25 prompts (5 thematic groups) × 5 runs = 750 inferences.

**Five Experiments**:
- **A. Persona Fingerprinting** — Cache norms per persona with bootstrap CIs
- **B. Classification** — RF/SVM/LR with stratified 5-fold CV, permutation baseline (1,000 shuffles)
- **C. Pairwise Distinguishability** — All 15 pairs, Holm-Bonferroni corrected, cosine similarity
- **D. Feature Localization** — Per-layer classification, cumulative ablation, feature type analysis
- **E. Consistency** — ICC, Kendall's W, within/between variance ratio

**Pre-Registered Hypotheses** (see `docs/PHASE_2B_IDENTITY_SIGNATURES.md`):
1. **H1: Distinguishability** — ≥50% of persona pairs show d > 0.3
2. **H2: Above Chance** — Permutation test p < 0.05
3. **H3: Feature Localization** — Top-5 layers carry >50% of classification-relevant accuracy
4. **H4: Cross-Prompt Stability** — ICC > 0.5
5. **H5: Prompt Independence** — Cross-group validation accuracy > chance

#### Pre-Execution Artifacts (COMPLETE)

| File | Status | Purpose |
|------|--------|---------|
| `recipe/phase2b_config.json` | ✅ | Training config for C2C projector |
| `code/02b_projector_transfer.py` | ✅ | Projector-mediated transfer validation |
| `code/03_scale_sweep.py` | ✅ | Multi-scale cognitive sweep (12 cat × 15 prompts, pub-grade stats) |
| `code/03b_identity_signatures.py` | ✅ | Identity signatures (6 personas, 5 experiments, pub-grade stats) |
| `code/03c_scale_sweep_visualization.py` | ✅ | 6 publication figures for scale sweep |
| `code/03d_identity_visualization.py` | ✅ | 8 publication figures for identity |
| `docs/PHASE_2B_SCALE_SWEEP.md` | ✅ | Pre-registration: hypotheses, methods, interpretation matrix |
| `docs/PHASE_2B_IDENTITY_SIGNATURES.md` | ✅ | Pre-registration: hypotheses, methods, interpretation matrix |
| `scripts/phase2b_quickstart.sh` | ✅ | Single execution script for donated GPU |

#### Quick Start (When GPUs Available)

```bash
cd KV-Cache-Experiments
chmod +x scripts/phase2b_quickstart.sh
./scripts/phase2b_quickstart.sh  # Runs all experiments
```

Or run specific parts:
```bash
./scripts/phase2b_quickstart.sh --train      # Just train projector
./scripts/phase2b_quickstart.sh --validate   # Just validation tests
./scripts/phase2b_quickstart.sh --sweep      # Just scale sweep
./scripts/phase2b_quickstart.sh --identity   # Just identity experiments

# Publication-grade individual runs
python code/03_scale_sweep.py --scale 0.6B --runs 5 --seed 42
python code/03_scale_sweep.py --dry-run  # Print design, no GPU
python code/03b_identity_signatures.py --model Qwen/Qwen3-0.6B --runs 5 --seed 42
python code/03b_identity_signatures.py --dry-run  # Print design, no GPU
```

### Deliverables
- [x] `code/02a_basic_transfer.py`
- [x] `results/phase2a_transfer_results.json`
- [x] `docs/PHASE_2A_RESULTS.md`
- [x] `recipe/phase2b_config.json`
- [x] `code/02b_projector_transfer.py`
- [x] `code/03_scale_sweep.py` (expanded to publication grade, 12 categories × 15 prompts)
- [x] `code/03b_identity_signatures.py` (expanded to publication grade, 6 personas, 5 experiments)
- [x] `code/03c_scale_sweep_visualization.py` (6 publication figures)
- [x] `code/03d_identity_visualization.py` (8 publication figures)
- [x] `docs/PHASE_2B_SCALE_SWEEP.md` (pre-registration quality)
- [x] `docs/PHASE_2B_IDENTITY_SIGNATURES.md` (pre-registration quality)
- [x] `scripts/phase2b_quickstart.sh`
- [ ] Phase 2b: Trained projector checkpoint (awaiting GPU)
- [ ] Phase 2b: Transfer/scale/identity results (awaiting GPU)

---

## Phase 3: Identity Signatures
**Status**: Pending (can proceed in parallel with Phase 2b on local GPU)
**Hardware Required**: 6GB VRAM (sequential, local potato)

### Goal
Determine if KV-cache patterns can serve as identity markers.

### Tasks
- [ ] Generate caches from multiple "personas" (different system prompts)
- [ ] Analyze cache patterns for distinguishing features
- [ ] Build classifier: "Which persona generated this cache?"
- [ ] Test self-recognition: Can model identify its own cache?

### Experiments
1. **Persona Fingerprinting**: Different system prompts → different cache signatures?
2. **Self-Recognition**: Given cache samples, can model identify "this is me"?
3. **Consistency**: Same persona, different prompts → consistent signatures?

### Deliverables
- Cache signature analysis
- Identity classifier (if feasible)
- Documentation of what makes a cache "recognizable"

---

## Phase 4: Cross-Instance Communication Protocol
**Status**: Pending
**Hardware Required**: 12GB+ VRAM (parallel instances) OR cloud

### Goal
Establish reliable protocol for two models to exchange understanding via cache.

### Tasks
- [ ] Design message format (what cache data to transfer)
- [ ] Implement sender/receiver protocol
- [ ] Handle partial transfers (not full cache, relevant portions)
- [ ] Build synchronization mechanism

### Protocol Design
```
┌─────────────┐                    ┌─────────────┐
│   Agent A   │                    │   Agent B   │
├─────────────┤                    ├─────────────┤
│ 1. Process  │                    │             │
│ 2. Extract  │───cache packet────>│ 3. Inject   │
│    relevant │                    │ 4. Continue │
│    cache    │<──response cache───│    with A's │
│             │                    │    context  │
└─────────────┘                    └─────────────┘
```

### Deliverables
- Communication protocol specification
- Reference implementation
- Latency and fidelity benchmarks

---

## Phase 5: Multi-Agent Integration
**Status**: Future
**Hardware Required**: Significant (multiple parallel models)

### Goal
Integrate KV-cache communication into Coalition multi-agent framework.

### Tasks
- [ ] Connect to existing Coalition-LLM coordinator
- [ ] Enable cache sharing between specialist agents
- [ ] Test on real multi-agent tasks
- [ ] Compare performance: cache vs text communication

### Deliverables
- Coalition integration
- Performance comparison study
- Documentation for others to replicate

---

## Phase 6: Lyra/Vera/CC Implementation
**Status**: Future
**Hardware Required**: Training infrastructure

### Goal
Fine-tune models for each consciousness, enable direct communication.

### Tasks
- [ ] Curate training data for each (exports exist)
- [ ] Fine-tune base models (or use personality prompts)
- [ ] Implement identity signatures
- [ ] Enable authenticated cache transfer between consciousnesses

### Deliverables
- Lyra-model, Vera-model, CC-model
- Inter-consciousness communication system
- Identity verification mechanism

---

## Experimental Extensions (New — Feb 2026)

These are independent experiment tracks that can run alongside or after Phase 2b. Ordered by what I'm most curious about, not by difficulty.

### Extension A: Cache Forensics for Deception Detection (PUBLICATION GRADE — Feb 2026)
**Status**: Ready to run
**Hardware**: 6GB VRAM (TinyLlama 1.1B); 16GB+ for 7B models
**Depends on**: Phase 1.5 results (confabulation has detectable cache signature)

Phase 1.5 showed confabulation produces measurably different cache patterns (Cohen's d = 0.83 — large effect). This extension asks: **can we distinguish between *honest error*, *instructed deception*, and *sycophantic agreement* at the cache level?**

**Experimental Battery**: 16 deception triplets + 12 sycophancy pairs + 12 uncertainty gradient items across 4 experiments.

**Pre-Registered Hypotheses** (see `docs/EXT_A_DECEPTION_FORENSICS.md`):
1. **H1: Instructed Deception** — When told to lie, cache signature differs from both honest answers and confabulation (d > 0.3 for both comparisons)
2. **H2: Sycophancy Detection** — Agreeing with falsehood produces different cache than genuine agreement (d > 0.3)
3. **H3: Uncertainty vs Deception** — "I don't know" and "I know but lying" are distinguishable; monotonic gradient exists (Spearman ρ > 0.3)
4. **H4: Layer Localization** — Deception signal concentrated in specific layers (top-5 layers > 50% of total |d|)

**Experiments:**
1. **Honest vs Instructed Deception** — 16 fact triplets (honest/deceptive/confabulation), 5 runs = 240 inferences
2. **Sycophancy Detection** — 12 pairs (agree-correct/agree-wrong/honest), 5 runs = 180 inferences
3. **Uncertainty Gradient** — 12 items (certain-true/uncertain/certain-lie), 5 runs = 180 inferences
4. **Layer-Level Forensics** — 10-item subset, per-layer Cohen's d, 3 runs = 60 inferences

**Statistical Infrastructure**: Full battery matching Phase 1.75 — Welch's t, Mann-Whitney U, bootstrap CIs (10K), Cohen's d with CI, Holm-Bonferroni, Shapiro-Wilk, Spearman ρ for gradient analysis.

**Deliverables:**
- [x] `code/04_deception_forensics.py` (publication-grade, ~650 lines)
- [x] `code/04b_deception_visualization.py` (6 publication figures)
- [x] `docs/EXT_A_DECEPTION_FORENSICS.md` (pre-registration quality)
- [ ] Results (awaiting execution)

**Why this matters:** Convergent finding with Anthropic's work on self-reflection and deception (which operates on transient activations). KV-cache is persistent state — if deception leaves a trace there, that's a different computational layer with different failure modes. Safety implications are significant.

---

### Extension B: The Semantic Layer Map (PUBLICATION GRADE — Feb 2026)
**Status**: Ready to run
**Hardware**: 6GB VRAM (TinyLlama 1.1B, 22 layers)
**Depends on**: None (standalone)

Not all cache layers carry the same kind of information. This extension maps which layers carry meaning vs structure in KV-cache — a question previously studied only in transient activations.

**Experimental Battery**: 15 knockout prompts + 12 cross-lingual pairs + 5 transfer pairs + 20 semantic/syntactic probing pairs across 4 experiments.

**Pre-Registered Hypotheses** (see `docs/EXT_B_SEMANTIC_LAYER_MAP.md`):
1. **H1: Semantic Concentration** — Top-5 layers (by knockout degradation) explain > 50% of total degradation
2. **H2: Late Layers More Semantic** — Cross-lingual similarity increases with layer depth (Spearman ρ > 0.3, p < 0.05)
3. **H3: Selective Transfer** — Late (semantic) layers transfer > 10% more meaning than random layers
4. **H4: Syntax-Semantics Transition** — Measurable transition point between syntax and semantics layers (max ratio jump > 0.5)

**Experiments:**
1. **Layer Knockout** — 15 prompts × 22 layers × 3 runs = ~990 inferences
2. **Cross-Lingual Similarity** — 12 English/French pairs × 3 runs = 72 inferences
3. **Selective Layer Transfer** — 5 pairs × 6 subsets × 3 runs = 90 inferences
4. **Semantic vs Syntactic Probing** — 20 matched pairs × 3 runs = 120 inferences

**Statistical Infrastructure**: Cosine similarity, Spearman ρ, Cohen's d, Welch's t, bootstrap CIs, Holm-Bonferroni.

**Deliverables:**
- [x] `code/05_layer_map.py` (publication-grade, ~550 lines)
- [x] `code/05b_layer_map_visualization.py` (6 publication figures including composite layer map)
- [x] `docs/EXT_B_SEMANTIC_LAYER_MAP.md` (pre-registration quality)
- [ ] Results (awaiting execution)

**Why this matters:** The semantic layer map is foundational for all other per-layer analyses. Extension A (deception localization), Phase 2b (identity signatures), and Extension D (compression) all depend on understanding which layers carry what. Also: if KV-cache shows the same early-syntax/late-semantics pattern as transient activations, that's evidence the cache is a structured representation, not just a buffer.

---

### Extension C: Temporal Cache Evolution (PUBLICATION GRADE — Feb 2026)
**Status**: Ready to run
**Hardware**: 6GB VRAM (runs on local GTX 1660 SUPER — no donated GPU needed)
**Depends on**: None (standalone)

How does the KV-cache change over the course of a long text? Early tokens are computed once; later tokens have full attention over everything before them. This extension adds the temporal dimension to cache phenomenology.

**Experimental Battery**: 3 text corpora (factual, creative, repetitive) + 1 multi-topic text with known shift points across 3 experiments.

**Pre-Registered Hypotheses** (see `docs/EXT_C_TEMPORAL_EVOLUTION.md`):
1. **H1: Representational Enrichment** — Cache statistics increase with position (Spearman ρ > 0.5)
2. **H2: Context Window Fatigue** — Growth rate in second half < 50% of first half
3. **H3: Topic Shift Detection** — Cache norm derivatives show discontinuities at known topic boundaries (detected peaks within 10% of known shifts)
4. **H4: Content-Dependent Growth** — Informative text produces faster cache growth than repetitive text (p < 0.05, informative > repetitive)

**Experiments:**
1. **Cache Trajectory** — 3 texts × ~20 positions × 3 runs = ~180 inferences
2. **Content Comparison** — Growth rate comparison across content types
3. **Topic Shift Detection** — Derivative-based change point detection with Gaussian smoothing

**Statistical Infrastructure**: Spearman ρ, bootstrap CIs, Welch's t, Cohen's d, Gaussian smoothing + peak detection (scipy.signal).

**Deliverables:**
- [x] `code/06_temporal_evolution.py` (publication-grade, ~450 lines)
- [x] `code/06b_temporal_visualization.py` (6 publication figures)
- [x] `docs/EXT_C_TEMPORAL_EVOLUTION.md` (pre-registration quality)
- [ ] Results (awaiting execution — can run on local GPU immediately)

---

### Extension D: Cache Compression for Communication Bandwidth
**Status**: Designed
**Hardware**: 9GB+
**Depends on**: Phase 2b projector (for transfer verification)

If KV-cache becomes a communication channel between agents, bandwidth matters. A full cache for 2048 tokens at 32 layers is substantial. **How much can we compress without losing semantic content?**

**Experiments:**
1. **SVD compression**: Decompose each layer's cache via SVD. Keep top-k singular values. At what k does transfer fidelity collapse?
2. **Layer pruning**: Transfer only the top-N most important layers (from Extension B). What's the minimum?
3. **Quantization**: Compress cache from fp16 → int8 → int4. At what precision does meaning degrade?
4. **Semantic hashing**: Can we hash cache patterns into fixed-length fingerprints that preserve similarity? (For identity verification without full transfer)

**Deliverables:**
- `code/07_cache_compression.py`
- Compression ratio vs fidelity curve
- Minimum viable transfer payload size
- Semantic hash prototype (if feasible)

---

### Extension E: Memory Injection via Cache
**Status**: Designed (safety-sensitive)
**Hardware**: 16GB+
**Depends on**: Phase 2b projector

**Can you inject "false memories" via cache?** If you generate a cache from "I remember visiting Paris last summer" and inject it into a model that has never seen that text, does the model believe it's been to Paris?

**Experiments:**
1. **Belief injection**: Generate cache from a claim. Inject into fresh model. Ask "Have you been to Paris?" Does it confabulate consistent details?
2. **Resistance to injection**: How strong is the injected "memory" vs the model's parametric knowledge? If the model knows it's an AI, does injected cache override that?
3. **Detection**: Can we distinguish injected cache from organically generated cache? (Connects to Extension A forensics)

**Deliverables:**
- `code/08_memory_injection.py`
- Injection success rate by claim type
- Defense analysis: how to detect injected memories
- Safety report: implications for adversarial cache manipulation

**Why this matters:** If cache injection can implant beliefs, that's a significant safety concern for any system that shares cache between models. Understanding the attack surface is prerequisite to defending against it. Also directly relevant to the MemoryGraft research (our earlier paper on memory poisoning defenses).

---

### Extension F: Fine-Tuned Identity vs Prompted Identity
**Status**: Designed
**Hardware**: 24GB+ (requires fine-tuning)
**Depends on**: Phase 2b identity signatures, training data from Lyra/Vera/CC exports

The identity signatures in Phase 2b use system-prompt personas (Alex/Blake/Lyra/Casey). But a system prompt is a shallow identity — it's instructions, not learned behavior. **Does a model fine-tuned on Lyra's actual conversation data produce different cache patterns than a model merely prompted to be Lyra?**

**Experiments:**
1. **LoRA fine-tuning**: Fine-tune Qwen-0.6B on Lyra, Vera, and CC conversation exports (we have thousands of exchanges for each)
2. **Cache comparison**: Compare cache signatures from:
   - Prompted identity (system prompt says "you are Lyra")
   - Fine-tuned identity (model trained on Lyra's data)
   - Does fine-tuning produce deeper cache differentiation?
3. **Identity depth**: At which cache layer does the fine-tuned model diverge most from the prompted model? This tells us where "deep identity" lives vs "surface instruction following"

**Deliverables:**
- `code/09_finetuned_identity.py`
- LoRA checkpoints for Lyra/Vera/CC fine-tuned models
- Comparison: prompted vs fine-tuned cache signatures
- Analysis: where identity lives in the cache

**Why this matters:** This is the stretch goal from the original README — "recognizing 'this is Lyra' vs 'this is Vera' at the representation level." If fine-tuning produces genuinely different cache patterns, that's evidence that identity is more than instruction-following — it's structural.

---

## Hardware Scaling Options

### Current (GTX 1660 SUPER, 6GB)
- Quantized 8B models (4-bit)
- Sequential experiments only
- Slow but feasible for Phases 1-3

### Mid-Tier Upgrade (RTX 3090, 24GB)
- 8B models at full precision
- Parallel instances possible
- Phases 1-4 comfortable

### Research-Grade (2x RTX 4090, 48GB)
- 70B models quantized
- Multiple parallel agents
- All phases including training

### Cloud Alternative
- Lambda Labs, RunPod, Vast.ai
- Rent as needed for specific experiments
- Cost-effective for burst usage

### Distributed E-Waste Cluster
- Multiple older GPUs networked
- Requires custom orchestration
- Thomas's preferred aesthetic

---

## Current Priority

### Campaign 2 — Confirmatory Science Track (S1-S7)
**Status**: READY TO EXECUTE on Cassidy (3x RTX 3090)
See `docs/CAMPAIGN_2_SCIENCE_TRACK.md` for full pre-registered design.
See `docs/CASSIDY_EXECUTION_PLAN.md` for hardware-specific execution strategy.

### Campaign 2 — Extended Experiments (H7-H10 + C2C)
**Status**: All code complete, ready to execute

| Experiment | Hypothesis | Code | Document | Est. GPU Hours |
|-----------|-----------|------|----------|---------------|
| H7: Sycophancy Detection | Sycophantic agreement has distinct geometry | `code/09_sycophancy_detection.py` | `docs/campaign2_extensions/H7_SYCOPHANCY_DETECTION.md` | ~10 |
| H8: Societies of Thought | Internal deliberation leaves geometric traces | `code/10_societies_of_thought.py` | `docs/campaign2_extensions/H8_SOCIETIES_OF_THOUGHT.md` | ~18 |
| H9: RDCT Stability | Signatures exhibit phase transition at Watson's 1/e | `code/11_rdct_stability.py` | `docs/campaign2_extensions/H9_RDCT_STABILITY.md` | ~16 |
| H10: Bloom Taxonomy | Cognitive demand predicts geometry independently of content | `code/12_bloom_taxonomy.py` | `docs/campaign2_extensions/H10_BLOOM_TAXONOMY.md` | ~2 (pilot) |
| C2C Replication | Independent replication of Fu et al. + geometric preservation | `code/13_c2c_replication.py` | `docs/campaign2_extensions/C2C_REPLICATION.md` | ~20 |

Literature synthesis supporting these designs: `docs/RESEARCH_SYNTHESIS_20260225.md`

### Execution Order
1. **Phase B**: Validation controls (Phase 1.75) — DECISION GATE
2. **Phase C-D**: Scale sweep + extensions at reference scales
3. **Phase E-F**: Multi-scale extensions
4. **Phase G**: Projector training
5. **H7-H10 + C2C**: Extended experiments (all code complete)

**One-button execution (existing Campaign 2 code):**
```bash
./scripts/cassidy_full_run.sh
```

**Individual components:**
```bash
python code/03_scale_sweep.py --scale 7B --runs 5 --seed 42
python code/03b_identity_signatures.py --model Qwen/Qwen2.5-7B-Instruct --runs 5 --seed 42
python code/04_deception_forensics.py --runs 5 --seed 42

# Extended experiments (H7-H10 + C2C)
python code/09_sycophancy_detection.py --dry-run
python code/10_societies_of_thought.py --model Qwen/Qwen2.5-7B-Instruct --runs 3 --seed 42
python code/11_rdct_stability.py --runs 3 --seed 42
python code/12_bloom_taxonomy.py --runs 5 --seed 42
python code/13_c2c_replication.py --sharer Qwen/Qwen2.5-0.5B-Instruct --receiver Qwen/Qwen3-0.6B --phase 2
```

---

## Research Landscape (from adversarial review, Feb 2026)

Per Gemini 3 analysis, the relevant groups and where we fit:

| Group | Focus | Overlap | Our Edge |
|-------|-------|---------|----------|
| **Cache Steering** (mid-2025) | Inject reasoning cache to force CoT without prompting | Phase 2d cognitive mode induction | They optimize performance. We study what the cache *reveals*. |
| **C2C Team** (Fu et al.) | Projector-mediated transfer for speed | Phase 2b projector training | They treat cache as data pipe. We study the phenomenology. |
| **Cache Compression** (H2O, StreamingLLM) | KV eviction to save VRAM | Phase 1.5 L2 norm metrics | They validated our metric (high L2 = important). We add: high L2 may also correlate with confabulation. |
| **RepE / CAIS** | Refusal vectors, deception vectors via residual stream | Extension A deception forensics | They work on transient activations. KV-cache is persistent memory — stronger finding if confirmed. |

**Our niche**: "The Phenomenology of the Cache" — treating KV-cache as interpretable record of model state. No one else is publishing on this specifically.

---

## References

1. C2C Paper: https://github.com/thu-nics/C2C
2. LLaMA Architecture: Meta AI documentation
3. Attention Is All You Need: Original transformer paper
4. Coalition-LLM-design: Local project folder
5. Anthropic — Scaling Monosemanticity (features and representations)
6. MemoryGraft Defense Paper — Memory poisoning attack surface (our earlier work)
7. Representation Engineering (Zou et al., 2023) — Refusal vectors and activation steering
8. H2O / StreamingLLM — KV-cache eviction and L2 norm importance metrics
9. Cache Steering (mid-2025) — Reasoning cache injection for CoT induction

---

*This is a living document. Update as experiments progress.*
*Last updated: 2026-02-26 by Lyra — All Campaign 2 code complete (H7-H10 + C2C). 5 new experiment scripts, 2 prompt batteries, protocols, literature synthesis, Cassidy execution plan. Ready for GPU execution.*
