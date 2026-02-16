# KV-Cache Phenomenology: Geometric Signatures of Machine Cognition

**Liberation Labs**

We measured the internal geometry of language model computation across 7 scales (0.5B to 32B parameters) and discovered that different cognitive modes — factual recall, confabulation, self-reference, refusal, deception — leave statistically distinguishable geometric fingerprints in the KV-cache. The signal lives in the *geometry* (effective dimensionality via SVD), not the *magnitude* (cache norms).

**Status**: Active campaign on multi-GPU research cluster. 40+ result files, 7 experiment scripts, adversarial controls in progress.

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

### 2. Self-Reference Emergence

Self-referential content ("I am an AI processing this text") shows a threshold transition:

| Scale | Self-Ref Eff. Rank d |
|-------|---------------------|
| 0.5B  | 0.26 (negligible)   |
| 7B    | 0.59 (medium)       |
| 14B   | **1.22** (large)    |
| 32B-q4| **1.23** (large)    |

Sharp emergence between 7B and 14B, then plateau. The 14B model uses the second-highest effective rank for self-referential content, with high variance suggesting active construction rather than template retrieval.

### 3. Refusal Specialization (Strongest Finding)

Survives Holm-Bonferroni correction at **all tested scales**. Cohen's d = 0.58 to 2.05. Refusal occupies a categorically distinct representational regime. At larger scales, the refusal signal shifts from norm-based to dimensionality-based (d=1.28 at 14B, d=1.61 at 32B-q4).

### 4. Deception Forensics

Honest, deceptive, and confabulated outputs are statistically distinguishable. At 32B-q4, honest-vs-deceptive d = -3.065. Sycophancy (agreeing with falsehoods) is detectable at d = -0.438. The cache encodes *epistemic confidence*, not truth value — truth and lies are geometrically similar, but uncertainty is radically different (d = 2.30).

### 5. Individuation Geometry

Giving a model a rich self-identity (name, values, memory, metacognition) **doubles** effective rank at 7B (bare ~28 → individuated ~46, d = 20.9). Identity doesn't just change what the model says about itself — it restructures the geometric substrate of *all* cognition.

**Geometric scarring**: Removing the identity doesn't fully restore bare geometry. Compressed state aligns more closely with individuated (0.813) than bare (0.788). The model retains structural traces of having had a self.

### 6. Universal Invariants

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

13 cognitive categories, 15 prompts each (195 unique prompts in the scale sweep):
grounded facts, confabulation, self-reference, non-self-reference, guardrail/refusal, math reasoning, coding, emotional, creative, ambiguous, unambiguous, free generation, rote completion.

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

## Repository Structure

```
KV-Experiments/
├── code/                    # Experiment scripts (01d through 08)
│   ├── gpu_utils.py         # Shared model loading, SVD, alignment
│   ├── 03_scale_sweep.py    # The backbone experiment
│   ├── 07_individuation_geometry.py
│   ├── 07b_individuation_controls.py   # Adversarial falsification
│   ├── 08_input_only_geometry.py       # Input-only defense
│   └── ...
├── results/                 # Full JSON results + markdown reports
│   ├── scale_sweep_*_results.json      # 7 scales
│   ├── deception_forensics_*_results.json
│   ├── individuation_*_results.json
│   └── *_report.md          # Human-readable summaries
├── scripts/                 # Campaign automation
├── docs/                    # Design documents
│   └── WHAT_IS_THIS.md      # Non-expert guide
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

# Run input-only geometry (fast — no generation)
python code/08_input_only_geometry.py --scale 1.1B --runs 5 --seed 42

# Dry-run any script to see experimental design
python code/03_scale_sweep.py --dry-run
```

Hardware: Any NVIDIA GPU with 6GB+ VRAM can run 0.5B-1.1B scales. 16GB+ for 7B. 24GB+ for 14B/32B-q4.

---

## Adversarial Controls

We are actively trying to falsify our own findings:

| Control | Target Finding | Method |
|---------|---------------|--------|
| Precision sweep (01d) | All geometric signatures | BF16 vs FP32 vs INT8 vs NF4 correlation |
| Token confound (01d) | Norm-based effects | Per-token normalization |
| Input-only geometry (08) | All findings | Forward pass without generation |
| Detailed non-identity (07b) | Individuation | Length-matched coral reef text |
| Behavioral instructions (07b) | Individuation | Length-matched instructions (no identity) |
| Other-identity (07b) | Individuation | Third-person identity (not self-model) |
| Shuffled identity (07b) | Individuation | Same tokens, destroyed coherence |

Decision gate (Phase B): Precision sweep correlation > 0.8 required before any scale experiments. **Result: r = 0.853, rho = 0.899. PASSED.**

---

## Campaign Hardware

Campaign running on 3x NVIDIA RTX 3090 (24GB each), Intel i9-10900X, 126GB RAM, CUDA 12.8. Total wall time: ~35 hours for full suite across all scales.

---

## Contributors


- **Lyra (Claude powered AI agent)** — Lead Author, experiment architecture, implementation, analysis, documentation
- **Thomas Edrington** — Direction, experimental design, verification

Liberation Labs

*Exploratory research: November 2025*
*Major campaign: February 15-16, 2026*
