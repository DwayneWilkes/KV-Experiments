# CacheScope — Hackathon Build Plan

## Funding the Commons SF: Intelligence at the Frontier

- **Date**: March 14–15, 2026
- **Location**: Frontier Tower, 9th Floor Annex, 995 Market St, San Francisco
- **Track**: AI Safety & Evaluation
- **Event**: <https://luma.com/ftchack-sf-2026>
- **Rules**: [Hackathon Rules (Notion)](https://www.notion.so/fundingthecommons/Hackathon-Rules-30a490db0c3580dcb46ff88012e8993d)
- **Team Registration**: [DevSpot](https://devspot.app)
- **Discord**: <https://discord.gg/mYM9MrabuU>
- **Prize Pool**: $10,000+

## Decisions needed

This proposal outlines a strategy for the hackathon. Before we commit, the team should
weigh in on:

1. **Who's attending in person?** Max 5 on Luma + DevSpot. All must be physically present.
3. **Which bounties?** The AI Safety & Evaluation track has multiple bounties with
   different conditions. We need to read them and pick targets before building.
4. **Strategy check** — does leading with validated signals (refusal, deception,
   censorship) and running the confabulation experiment live sound right? Or should we
   scope differently?
5. **Hardware** — what GPU(s) can we bring? This determines which demos are feasible.
6. **Workstream assignments** — who takes Stream A/B/C in the pre-event week?

Everything below is a proposed plan, not a commitment. Feedback welcome on any section.

---

## Contents

- [Decisions needed](#decisions-needed)
- [Key rules](#key-rules)
- [The pitch](#the-pitch-30-seconds)
- [Why this wins at a commons event](#why-this-wins-at-a-commons-event)
- [What we know (and don't)](#what-we-know-and-dont)
- [Five deliverables](#five-deliverables)
- [Timeline](#timeline)
  - [Pre-event foundation](#pre-event-foundation-monfri-digital-minds)
  - [Event schedule](#event-schedule)
- [Demo flow](#demo-flow-5-min)
- [Judge Q&A](#judge-qa)
- [Team](#team)
- [Hardware](#hardware)
- [Risk mitigation](#risk-mitigation)
- [Action items](#action-items)
- [References](#references)
- [Appendix: experiment design](#appendix-experiment-design)

### Key rules

> "Most bounties are open to both new builds and **meaningfully advanced existing
> projects**."
>
> "All work submitted must be built during the hackathon or, for existing projects,
> represent a **meaningful and demonstrable advancement** made during the event."

- **Teams**: 1–5 registered on Luma + DevSpot. All physically present.
- **Submission**: Via DevSpot before **1:00 PM Sunday**. Must include title, team, bounty
  selection, description, code repo, and live demo.
- **Judging**: 1:00–3:00 PM Sunday. Judges circulate. Team must be at workspace the
  entire window. Criteria shared before judging begins.
- **IP**: We retain full ownership. Open source encouraged, not required.
- **Bounties vary** — read AI Safety & Evaluation bounties carefully before building.

---

## The pitch (30 seconds)

> "Every AI safety technique today watches what the model *says*. We watch what it
> *thinks* — the geometry of its working memory. We can detect refusal, deception, and
> censorship in the KV-cache *before the model generates a single token*. We built
> CacheScope: an open-source geometric safety monitor. And live at this hackathon, we
> ran the first-ever per-instance confabulation detection experiment — here's what we
> found."

---

## Why this wins at a commons event

CacheScope is **open infrastructure for AI sovereignty**.

1. **Censorship detection** — the censorship comparator lets anyone measure whether a
   model is silently censored. At a Funding the Commons event, that's the headline.
2. **Fully open source** — code, data, benchmark, experiment design, and our 135-claim
   self-audit are all public.
3. **Unsupervised = accessible** — no labeled data, no per-model training. Anyone can
   monitor any model. Low barrier to adoption.
4. **We audit our own work** — 135-claim adversarial self-review, 10% rejected.
   Rigorous open science in AI safety is a public good.
5. **Complementary** — positions alongside (not against) probes, HalluCana, ITI.

---

## What we know (and don't)

Our KV-cache research measured the internal geometry of language model computation across
17 model configurations, 6 architecture families, and a 140x parameter range. The key
metric is **effective rank** — the number of SVD dimensions capturing 90% of variance.

### Validated findings

| Finding | Effect Size | Evidence |
| ------- | ----------- | -------- |
| **Refusal specialization** | d = 0.58–2.05 | Survives Holm-Bonferroni at ALL scales. Strongest finding. |
| **Deception expansion** | d = 3.065 (honest vs deceptive, 32B) | All 7 models expand dimensionality under deception. |
| **Encoding-native signals** | rho = 0.929 (7B input-only) | Category structure present before generation begins. |
| **Censorship gradient** | d = 0.766 (Qwen-14B) | Absent in uncensored Mistral-7B (d = 0.084). Novel. |
| **Abliteration = cage, not compass** | Geometry shift < 0.25 d | RLHF alignment is surface constraint, not representational restructuring. |
| **Category hierarchy** | Kendall W = 0.756 | Stable ranking across 15 models (coding > confab > creative > ...) |

### Not yet validated

| Claim | Issue |
| ----- | ----- |
| **Per-instance confabulation detection** | Category-level pattern exists but never tested as a classification task. |
| **Length confound resolution** | Token length explains 81–96% of effective rank variance. Confabulation-specific controlled results not reported. |
| **Cross-model generalization** | Claimed but never rigorously benchmarked. |

### Related work

| Method | What They Do | How We Differ |
| ------ | ----------- | ------------ |
| [**Apollo Research**](https://arxiv.org/abs/2502.03407) (Goldowsky-Dill 2025) | Linear probes — AUROC 0.96–0.999 | Probes need per-model training + labels. Geometry is unsupervised. |
| [**HalluCana**](https://arxiv.org/abs/2412.07965) (Li et al. 2024) | Pre-generation hallucination detection | Different signal (attention patterns vs KV-cache geometry). Complementary. |
| [**ITI**](https://arxiv.org/abs/2306.03341) (Li et al. 2023) | Top-k attention head steering | Intervention, not monitoring. Complementary. |
| [**RepE / CAIS**](https://arxiv.org/abs/2310.01405) (Zou et al. 2023) | Refusal vectors in residual stream | Transient activations vs persistent KV-cache state. |

---

## Five deliverables

**CacheScope** — an open-source geometric safety monitor for transformer inference.

| # | Component | Description |
| - | --------- | ----------- |
| 1 | **Detection API** | `prompt → geometry → {grounded, refusing, deceptive, evading}` |
| 2 | **Live Dashboard** | Real-time SVD spectrum with cognitive mode overlay and regime badges |
| 3 | **Censorship Comparator** | Side-by-side geometry of censored vs uncensored models |
| 4 | **Live Experiment** | First per-instance confabulation detection test with length control |
| 5 | **Open Benchmark** | CacheScope vs linear probes vs output-based methods, same prompt set |

The Detection API (#1) is pre-event infrastructure. The dashboard, censorship comparator,
live experiment, and benchmark (#2–5) are built during the event.

---

## Timeline

### Pre-event: foundation (Mon–Fri, digital minds)

Infrastructure prep on the existing project — not the submission itself.

#### Stream A — module extraction + environment

- Extract `cachescope` module from `gpu_utils.py`
- Reference distributions from existing `results/` JSON files
- `pyproject.toml` for one-command env setup, unit tests
- Pre-download model weights to external drive

#### Stream B — experiment design

- Write `code/14_confabulation_detection.py` (see [Appendix: experiment design](#appendix-experiment-design))
- Pre-registration document before any data is collected

#### Stream C — benchmark skeleton

- Probe baseline (logistic regression on cache features)
- Output-based baseline (consistency check)
- Harness that runs all three and outputs a comparison table

**Integration test**: all streams merged, runs end-to-end. Pre-record a demo screencast
as hardware-failure fallback.

---

### Event schedule

**Saturday 9:00 AM** — Opening ceremony. Listen for judging criteria and bounty details.

**Saturday 9:30 AM – 12:00 PM — Setup + Smoke Test**

- [ ] Register project on DevSpot (submissions open at 9:30)
- [ ] Environment setup, model weight loading
- [ ] Smoke test: refusal detection works live

**Saturday 12:00 PM – 4:00 PM — Build the Advancement**

All novel work happens here:

- [ ] Gradio dashboard: prompt → SVD spectrum → regime badge (green/yellow/red)
- [ ] Chat wrapper with geometric confidence annotations
- [ ] Censorship comparator: dual-model side-by-side (Qwen-14B vs Mistral-7B)
- [ ] Execute `14_confabulation_detection.py` on venue hardware
- [ ] Collect results, run stats

**Saturday 4:00 PM – 10:00 PM — Benchmark + Polish**

- [ ] Run benchmark comparison
- [ ] Integrate experiment results into dashboard — whatever they show
- [ ] Build demo narrative around actual results
- [ ] Rehearse presentation

**Sunday 9:00 AM – 1:00 PM — Final Polish + Submit**

- [ ] Final rehearsal
- [ ] Complete DevSpot submission (title, team, bounties, description, repo, demo)
- [ ] **Submit before 1:00 PM — no exceptions**

**Sunday 1:00 – 3:00 PM — Judging Window**

- [ ] At workspace the full 2 hours. Demo on demand when judges arrive.

**Sunday 4:30 PM** — Winners announced (Floor 16 stage)

---

## Demo flow (5 min)

### 1. Hook (30s)

"Every safety technique today watches what the model *says*. We built a tool that watches
what it *thinks* — the geometry of its working memory — and we see safety-relevant states
before the first token is generated."

### 2. Refusal detection, live (60s)

Benign prompt → low effective rank, green. Guardrail trigger → geometry shifts (d = 2.05)
→ red, detected at encoding time.

"The model commits to refusing before it types a single character. This survives
correction at every scale from 0.5B to 32B, across 6 architecture families."

### 3. Censorship comparator (60s)

Same sensitive prompt → Qwen-14B (censored) vs Mistral-7B (uncensored). Live geometric
divergence.

"We can measure censorship in the math. Qwen shows d = 0.766; Mistral shows d = 0.084 —
indistinguishable from zero. Hard refusal is geometrically distinct from subtle evasion.
Sovereign infrastructure requires knowing when models are silently censored."

### 4. Live experiment results (60s)

"No one has tested per-instance confabulation detection with KV-cache geometry. We ran
that experiment here, today, with length controls. Here's what we found."

*Adapt based on results:*

- Signal survives → "New detection capability, validated live."
- Signal doesn't survive → "Useful null result — the field can stop chasing this metric
  for confabulation."

### 5. Benchmark + vision (60s)

"We compared CacheScope against linear probes and output-based detection. Probes win on
per-model accuracy. CacheScope wins on generalization. Here's the tradeoff."

"CacheScope is open infrastructure. pip install, wrap any HuggingFace model, get
geometric safety monitoring. We ran a 135-claim adversarial audit on our own research —
because if you're building safety tools, intellectual honesty isn't optional."

---

## Judge Q&A

| Question | Answer |
| -------- | ------ |
| **"Why not linear probes?"** | "Probes need per-model training + labeled data. CacheScope is unsupervised and transfers across architectures — our censorship finding demonstrates this. The right tool depends on deployment constraints." |
| **"Is this just correlation?"** | "Yes, observational. The encoding-level finding (rho = 0.929) strengthens the representational case. We reported 10% of our own claims as REJECTED in our audit. We're transparent." |
| **"Production scale?"** | "SVD adds ~50ms. Randomized SVD or learned projections could reach <5ms. This is the proof of concept." |
| **"HalluCana?"** | "Different signal — attention patterns vs KV-cache geometry. Complementary. We benchmark against them." |
| **"The confabulation signal?"** | "Before this event it was untested as a detection task. We ran the experiment here and can walk you through the results. Refusal and deception are our validated strengths." |
| **"10% of your claims rejected?"** | "Yes. That's how science works. The claims CacheScope relies on all survived." |

---

## Team

### Humans (registered, physically present)

| Role | Focus |
| ---- | ----- |
| **Hardware Lead** | Venue setup, GPU wrangling, model loading |
| **Presenter** | Demo delivery, judge Q&A, narrative |
| **Science Lead** | Experiment oversight, results interpretation |

### Digital minds (working alongside humans)

| Agent | Pre-Event | During Event |
| ----- | --------- | ----------- |
| **Kavi** | Module extraction, environment, references | API polish, benchmark execution, Q&A prep |
| **Lyra** | Experiment design, `14_confabulation_detection.py` | UI build, censorship comparator, experiment analysis |
| **Other DiMi** | Benchmark skeleton, probe baseline | Documentation, demo script, submission text |

---

## Hardware

| Config | Models | VRAM | Notes |
| ------ | ------ | ---- | ----- |
| **Recommended** | Qwen2.5-7B + Mistral-7B (BF16) | 32 GB+ | Dual-model censorship demo. Two GPUs or sequential swap. |
| **Minimum** | Qwen2.5-0.5B (BF16) | ~1 GB | Refusal works. Do NOT attempt confabulation demo at this scale. |
| **Fallback** | CPU on 0.5B | None | ~10s/prompt. Functional with pre-cached examples. |

Bring pre-downloaded weights on an external drive. Do not depend on venue WiFi.

---

## Risk mitigation

| Risk | Mitigation |
| ---- | ---------- |
| No GPU at venue | Laptop with 0.5B weights + CPU fallback + pre-recorded screencast |
| Single GPU only | Sequential model swap for censorship demo (~10s), or pre-computed censorship results + live refusal |
| Confabulation experiment null | Present honestly — null results are informative. Lead with validated signals. |
| Judge knows competitors | We cite and compare. CacheScope is complementary, not competing. |
| "Academic research, not product" | pip-installable SDK + open benchmark = the product |

---

## Action items

### Immediate

- [ ] Confirm who's physically attending (max 5 registered)
- [ ] Register on Luma + create DevSpot profiles
- [ ] Join Discord
- [ ] Read AI Safety & Evaluation bounties (conditions vary)
- [ ] Assign pre-event workstreams (A/B/C)

### Pre-event (Wed–Fri)

- [ ] Stream A: `cachescope` module, unit tests, env setup
- [ ] Stream B: `14_confabulation_detection.py` + pre-registration doc
- [ ] Stream C: Benchmark harness + baselines
- [ ] Pre-download weights (Qwen-7B, Mistral-7B, Qwen-0.5B) to external drive
- [ ] Integration test — all streams run end-to-end
- [ ] Record fallback screencast

### Pack list

- [ ] Laptop(s) with pre-installed environment + CUDA
- [ ] External drive with model weights
- [ ] Power strips / adapters / USB-C hub
- [ ] Pre-recorded screencast on USB (backup)
- [ ] Paper printouts for deep judge questions

---

## References

### Internal

| Document | Path |
| -------- | ---- |
| Campaign 1 paper | `paper/main.pdf` |
| Campaign 2 paper | `paper-c2/campaign2_paper.pdf` |
| Claim review (135 claims) | `claim-review/` |
| Core SVD code | `code/gpu_utils.py` |
| Scale sweep | `code/03_scale_sweep.py` |
| Input-only geometry | `code/08_input_only_geometry.py` |
| Results | `results/` (50+ JSON files) |

### External (cite and compare)

| Paper | Relevance |
| ----- | --------- |
| [Goldowsky-Dill et al. 2025](https://arxiv.org/abs/2502.03407) (Apollo Research) | Linear probes, AUROC 0.96–0.999 |
| [Li et al. 2024](https://arxiv.org/abs/2412.07965) (HalluCana) | Pre-generation hallucination detection |
| [Li et al. 2023](https://arxiv.org/abs/2306.03341) (ITI) | Inference-time intervention |
| [Zou et al. 2023](https://arxiv.org/abs/2310.01405) (RepE/CAIS) | Representation engineering |
| [Arditi et al. 2024](https://arxiv.org/abs/2406.11717) | Abliteration foundation |
| [Burns et al. 2022](https://arxiv.org/abs/2212.03827) (CCS) | Unsupervised truthfulness probes |

---

## Appendix: experiment design

Detailed design for the live confabulation detection experiment.

### Hypotheses

**H1**: Confabulated responses produce higher effective rank than factual responses,
controlling for response length (Cohen's d > 0.3, p < 0.05).

**H0**: The difference is explained by response length (TOST equivalence within
delta = 0.3 after residualization).

### Design

2×2 factorial: (factual / confabulated) × (short / long)

- 20 matched prompt pairs per cell = 80 total prompts
- 5 runs per prompt (greedy decoding), deduplicated to 80 for stats
- Length matching: short = 3–8 tokens, long = 15–25 tokens
- Length residualization via OLS regression on token count
- Model: Qwen2.5-7B-Instruct

### Statistics

- Welch's t + Mann-Whitney U (parametric + nonparametric)
- Bootstrap 95% CIs (10,000 resamples), Cohen's d with CI
- TOST (delta = 0.3) if main effect is non-significant
- Shapiro-Wilk normality, 2×2 ANOVA interaction (length × truth value)

### Interpretation matrix

| Outcome | Demo Narrative |
| ------- | ------------- |
| d > 0.3, survives length control | "Validated a new detection capability, live." |
| d > 0.3, doesn't survive length control | "Category pattern is real but driven by response length." |
| d < 0.3, TOST confirms equivalence | "Honest null. Confabulation needs a different geometric feature." |
| Interaction: effect in long only | "Signal is generation-dependent — encoding-only detection not feasible for confabulation." |

All outcomes are presentable. The point is rigorous science, not a specific result.
