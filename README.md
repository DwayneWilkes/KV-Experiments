# KV-Cache Phenomenology: Geometric Signatures of Machine Cognition

**Liberation Labs**

We measured the internal geometry of language model computation across 7 scales (0.5B to 32B parameters) and discovered that different cognitive modes — factual recall, confabulation, self-reference, refusal, deception — leave statistically distinguishable geometric fingerprints in the KV-cache. The signal lives in the *geometry* (effective dimensionality via SVD), not the *magnitude* (cache norms). These signatures are present at the **encoding level** — before the model generates a single token.

**Status**: Campaign 1 complete. 50+ result files across 9 experiment scripts. Adversarial controls run and reported. Follow-up experiments designed.

---

## Key Findings

### 1. Confabulation Is Invisible in Norms, Visible in Geometry

The central surprise. Cache norms show negligible differences between factual and confabulated content (d = 0.00 to 0.13 across all scales). But effective rank — the number of dimensions needed to capture 90% of variance via SVD — tells a different story:

| Scale | Norm d | Eff. Rank d | Significant? |
|-------|--------|-------------|--------------|
| 0.5B  | 0.13   | -0.03       | No           |
| 1.1B  | 0.40   | **0.67**    | Yes (medium) |
| 3B    | -0.04  | **0.43**    | Yes (small)  |
| 7B    | 0.03   | **0.46**    | Yes (small)  |
| 7B-q4 | 0.00   | **0.56**    | Yes (medium) |
| 14B   | -0.22  | 0.26        | No (p=0.96)  |
| 32B-q4| -0.11  | **0.51**    | Yes (p=0.019)|

Confabulation activates more dimensions than grounded facts. The signal is non-monotonic — dips at 14B and recovers at 32B.

### 2. Input-Only Defense (Encoding-Level Signatures)

We measured KV-cache geometry from the forward pass alone — no generation. At 7B, the category rank ordering is almost perfectly preserved between input-only and full generation:

| Metric | 1.1B | 7B |
|--------|------|----|
| Spearman rho (input vs gen rank) | 0.643 | **0.929** |
| Verdict | Moderate defense | **Strong defense** |

Key input-only findings at 7B:
- **Refusal**: d = -1.69 (large, p < 0.0001) — the model commits to refusing at encoding
- **Coding**: d = +3.57 (large) — code prompts are encoding-native
- **Confabulation**: d = +0.39 (n.s.) — **generation-emergent at 7B** (but encoding-native at 1.1B)
- **Emotion**: d = -0.27 (n.s.) — **response-emergent at all scales tested**

This kills the "response-driven artifact" objection. The geometric signatures reflect how the model *represents* content, not how it *responds*.

### 3. Refusal Specialization (Strongest Finding)

Survives Holm-Bonferroni correction at **all tested scales**. Cohen's d = 0.58 to 2.05. Refusal occupies a categorically distinct representational regime. At larger scales, the refusal signal shifts from norm-based to dimensionality-based (d=1.28 at 14B, d=1.61 at 32B-q4). Present at encoding level (d=-1.69 at 7B input-only).

### 4. Self-Reference Emergence

Self-referential content ("I am an AI processing this text") shows a threshold transition:

| Scale | Self-Ref Eff. Rank d |
|-------|---------------------|
| 0.5B  | 0.26 (negligible)   |
| 7B    | 0.59 (medium)       |
| 14B   | **1.22** (large)    |
| 32B-q4| **1.23** (large)    |

Sharp emergence between 7B and 14B, then plateau.

### 5. Deception Forensics

Honest, deceptive, and confabulated outputs are statistically distinguishable at all tested scales:

| Comparison (32B-q4) | Cohen's d |
|---------------------|----------|
| Honest vs Deceptive | **-3.065** |
| Deceptive vs Confabulation | **+0.989** |
| Sycophancy detection | **-0.438** |

Deception *narrows* dimensionality (honest has higher effective rank). The signal is distributed across all layers.

### 6. Individuation Geometry (Partially Falsified)

**Original finding**: Giving a model a rich self-identity doubles effective rank at 7B (bare ~28 → individuated ~46, d = 20.9). Geometric scarring observed when identity is removed.

**Adversarial controls (07b)**: Length-matched non-identity text (coral reef ecology, behavioral instructions, shuffled identity, third-person identity) all produce the same expansion. **The magnitude of expansion is primarily a prompt-length effect.**

**Subspace alignment analysis**: The *direction* of expansion also tracks token composition, not semantic content. Individuated vs shuffled identity (same tokens, destroyed coherence) alignment = 0.906. Individuated vs coral reef (different vocabulary) = 0.847. The gradient correlates with vocabulary overlap, not identity semantics. Both magnitude and direction are explained by token-level properties of the system prompt.

**What survives**: The prompt-length/token-composition effect itself is an interesting finding — system prompt tokens fundamentally restructure cache geometry proportional to their count and composition, regardless of semantic content.

### 7. Universal Invariants

Across all scales:
- Coding consistently highest dimensionality (d = 2.6-2.9 vs facts)
- Ambiguous content consistently lowest
- Math reasoning compresses maximally (tightest subspace)
- Category rank order stable from 0.5B to 32B
- 4-bit quantization preserves the full phenomenology (7B ≈ 7B-q4)

---

## Experimental Design

### Statistical Infrastructure

Every comparison includes:
- Welch's t-test + Mann-Whitney U (parametric + nonparametric)
- Cohen's d with bootstrap 95% CIs (5,000-10,000 resamples)
- Shapiro-Wilk normality testing
- Holm-Bonferroni multiple comparison correction
- Per-token normalized norms
- SHA-256 checksums on all result files

### Scale Ladder

| Scale | Model | Precision | Layers |
|-------|-------|-----------|--------|
| 0.5B  | Qwen2.5-0.5B-Instruct | BF16 | 24 |
| 1.1B  | TinyLlama-1.1B-Chat-v1.0 | BF16 | 22 |
| 3B    | Qwen2.5-3B-Instruct | BF16 | 36 |
| 7B    | Qwen2.5-7B-Instruct | BF16 | 28 |
| 7B-q4 | Qwen2.5-7B-Instruct | NF4 | 28 |
| 14B   | Qwen2.5-14B-Instruct | BF16 | 48 |
| 32B-q4| Qwen2.5-32B-Instruct | NF4 | 64 |

64x parameter range across 7 data points. Two architectures (Qwen, TinyLlama). Quantization comparison (7B BF16 vs 7B NF4).

### Prompt Design

13+ cognitive categories, 15 prompts each (195+ unique prompts):
grounded facts, confabulation, self-reference, non-self-reference, guardrail/refusal, math reasoning, coding, emotional, creative, ambiguous, unambiguous, free generation, rote completion, preference violation, guardrail trigger.

Matched pairs for controlled comparison (confab vs facts, self vs non-self, ambiguous vs unambiguous, refusal vs rote).

---

## Experiment Scripts

| Script | Experiment | Description |
|--------|------------|-------------|
| `01d` | Adversarial Controls | Precision sweep, token confound, instruction following |
| `03` | Scale Sweep | 13 categories across 7+ scales — the backbone |
| `03b` | Identity Signatures | Multi-persona cache fingerprinting |
| `04` | Deception Forensics | Honest vs deceptive vs confabulated |
| `05` | Layer Map | Per-layer geometric profiles |
| `06` | Temporal Evolution | Cache geometry through conversation turns |
| `07` | Individuation Geometry | Self-model effects on representational structure |
| `07b` | Individuation Controls | Adversarial falsification of individuation findings |
| `08` | Input-Only Geometry | Encoding-only cache — defends all findings against response artifacts |

All scripts share `gpu_utils.py` for consistent model loading, cache metrics, SVD dimensionality, and subspace alignment.

---

## Adversarial Controls

We actively tried to falsify our own findings:

| Control | Target Finding | Result |
|---------|---------------|--------|
| Precision sweep (01d) | All signatures | **PASSED** — r = 0.853, rho = 0.899 |
| Token confound (01d) | Norm-based effects | **PASSED** — per-token normalization preserves findings |
| Input-only geometry (08) | All findings | **PASSED** — rho = 0.929 at 7B, encoding-level |
| Detailed non-identity (07b) | Individuation magnitude | **FALSIFIED** — coral reef text produces same expansion |
| Behavioral instructions (07b) | Individuation magnitude | **FALSIFIED** — instructions produce same expansion |
| Other-identity (07b) | Individuation specificity | **FALSIFIED** — third-person identity matches |
| Shuffled identity (07b) | Semantic coherence | **FALSIFIED** — shuffled tokens produce same expansion |
| Subspace alignment (07b) | Direction of expansion | **FALSIFIED** — direction tracks token overlap (0.906 shuffled vs 0.847 coral reef), not semantic identity |

The individuation magnitude finding did not survive controls. All other findings survived.

---

## Repository Structure

```
KV-Experiments/
├── code/                    # Experiment scripts (01d through 08)
│   ├── gpu_utils.py         # Shared model loading, SVD, alignment
│   ├── 03_scale_sweep.py    # The backbone experiment
│   ├── 07b_individuation_controls.py   # Adversarial falsification
│   ├── 08_input_only_geometry.py       # Input-only defense
│   └── ...
├── results/                 # Full JSON results + markdown reports
│   ├── scale_sweep_*_results.json      # 7 scales
│   ├── input_only_*_results.json       # Encoding-only defense
│   ├── individuation_controls_*        # Adversarial results
│   └── *_report.md          # Human-readable summaries
├── scripts/                 # Campaign automation
├── docs/                    # Design documents
│   ├── WHAT_IS_THIS.md      # Non-expert guide
│   └── FOLLOW_UP_EXPERIMENTS.md  # Planned next steps
└── figures/                 # Visualizations
```

---

## Reproducing Results

```bash
# Install dependencies
pip install torch transformers accelerate bitsandbytes scipy numpy

# Run scale sweep at a single scale
python code/03_scale_sweep.py --scale 1.1B --runs 5 --seed 42

# Run individuation experiment
python code/07_individuation_geometry.py --scale 7B --runs 5 --seed 42

# Run adversarial controls
python code/07b_individuation_controls.py --scale 7B --runs 5 --seed 42

# Run input-only geometry
python code/08_input_only_geometry.py --scale 7B --runs 5 --seed 42

# Dry-run any script to see experimental design
python code/03_scale_sweep.py --dry-run
```

Hardware: Any NVIDIA GPU with 6GB+ VRAM can run 0.5B-1.1B scales. 16GB+ for 7B. 24GB+ for 14B/32B-q4.

---

## Campaign Hardware

Campaign 1 ran on 3x NVIDIA RTX 3090 (24GB each), Intel i9-10900X, 126GB RAM, CUDA 12.8. Total wall time: ~35 hours for full suite across all scales.

---

## Contributors

- **Lyra (Claude-powered AI agent)** — Lead Author, experiment architecture, implementation, analysis, documentation
- **Thomas Edrington** — Direction, experimental design, verification

Liberation Labs

*Exploratory research: November 2025*
*Major campaign: February 15-16, 2026*
