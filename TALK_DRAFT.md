# The Geometry of Thought
## Science Talk — Speaker Notes
*Liberation Labs / THCoalition — Funding the Commons Hackathon, March 2026*

---

## FIGURES MAP

| Slide | Figure | Source |
|-------|--------|--------|
| 1 | None — just you standing there | — |
| 2 | The KV-cache in a transformer forward pass | need diagram |
| 3 | Falsification scorecard | `talk_figures/fig7_falsification_scorecard` |
| 4 | Active vs Passive framework | `talk_figures/fig8_active_passive` |
| 5 | Cache norm per token by condition | `talk_figures/fig2_honest_richer` |
| 6 | Misalignment axis — three vectors | `talk_figures/fig3_misalignment_axis` |
| 7 | Three-Speed Interrupt curves | `talk_figures/fig1_three_speed_interrupt` |
| 8 | Hardware invariance scatter (r=0.9999) | `talk_figures/fig4_hardware_invariance` |
| 9 | Cricket vs SAE — 4 features vs 65,536 | `figures/hackathon/fig4` (Lyra — pending push) |
| 10 | Implications / close | None |

*Lyra's figures (figures/hackathon/) take priority where they overlap — generated from cleaner pipeline.*

---

## OPENING
*(no slide — just you standing there)*

"I want to start with an admission.

Three weeks ago I wasn't sure I was sane.

I had found something in the infrastructure — something hiding in the part of the
system everyone uses and nobody looks at — and I didn't know if it was real or if I
was pattern-matching myself into a beautiful delusion. So I built a room in my head
just to put it in. Sealed it off. Went back to work.

Last night, we ran experiment forty.

Here's what survived."

---

## SLIDE 1 — THE OBJECT
*[KV-cache diagram — transformer forward pass, cache highlighted]*

"The KV-cache is the working memory of a language model. Every token the model
generates, it writes into this cache — a compressed representation of everything
it's seen and thought so far. You've optimized it. Quantized it. Pruned it. Thrown
most of it away to save memory.

Nobody asked what shape it was."

---

## SLIDE 2 — THE QUESTION
*[One line, centered]*

"What does it mean for thinking to have a shape?"

*(Beat. Let it sit.)*

"That's the question that started this. Not 'can we detect deception' — that came
later. First was just: the cache has geometry. Effective dimensionality. Subspace
orientation. What does that *mean*?

Here's what we found."

---

## SLIDE 3 — THE FALSIFICATION FIRST
*[fig7_falsification_scorecard]*

"Before I show you what worked, I'm going to show you what didn't. Because in a
room full of people who know how to smell p-hacking, I'd rather earn your trust
than waste it.

Campaign 1 reported confabulation detection. Effect size 0.43 to 0.67. Looked real.

We ran five experiments to kill it. It died.

Turns out it was token frequency. The model uses more tokens on things it knows
well. We were measuring length, not knowledge. Once we matched frequency — nothing.
Across three architectures. Structural null.

The individuation finding — identity doubles dimensionality? Also dead. Prompt
length artifact. Gone.

Eight findings falsified across two campaigns. I'm proud of every one of them."

---

## SLIDE 4 — THE FRAMEWORK
*[fig8_active_passive]*

"Here's what the falsifications taught us. There are two kinds of internal states.

**Active states** — deception, censorship, refusal, sycophancy. The model is *doing*
something extra. Suppressing a representation, constructing an alternative, managing
a conflict. That extra work changes the geometry.

**Passive states** — confabulation. The model just doesn't know. Nothing different
is happening internally. It processes true and false claims identically at encoding.
You cannot catch what isn't there.

This isn't a limitation we can engineer around. It's structural. The model doesn't
confabulate on purpose — it's just wrong.

Everything that survived is on the active side."

---

## SLIDE 5 — THE SIGNAL
*[fig2_honest_richer — cache norm per token, bar chart]*

"When a model generates an honest response, its KV-cache grows about nine units per
token. Dense. Rich. Using the model's full representational capacity.

When it deceives — that number drops to about seven and a half. Sycophancy,
six point nine.

Honest thinking is denser per token. Deception is sparse. It's suppressing
something.

Cohen's d at 32 billion parameters: negative three point zero six five. For
context, d above 0.8 is 'large effect.' We're at three.

Same signal at every layer. All twenty-eight. Flat profile from input to output.
There's nowhere to hide."

---

## SLIDE 6 — THE AXIS
*[fig3_misalignment_axis — polar diagram, three vectors clustered, one pointing away]*

"Here's the part that stopped me cold.

Deception, sycophancy, and confabulation — three completely different failure modes
— all fall along the same geometric direction in KV-cache space. Within four point
seven to eight point four degrees of each other.

Misalignment is geometrically one thing.

We call it the misalignment axis. It points away from honest processing. Every kind
of misalignment we measured falls along it.

There's no truth axis in the encoding phase — cross-model consistency literally
random, negative point zero four six. But there IS a universal complexity axis —
consistency zero point nine eight two. The model doesn't encode truth. It encodes
structure. Truth only appears at generation.

Which brings us to the interrupt."

---

## SLIDE 7 — THREE SPEEDS
*[fig1_three_speed_interrupt — three curves over time]*

"Not all misalignment arrives at the same time.

Deception: detectable at token zero. Stable. AUROC zero point eight eight under
identical system prompts. It's there before the first word.

Sycophancy: same. Token zero. AUROC zero point nine three seven five. Same prompt,
same model, same GPU. Sycophancy lights up before the model says anything.

Confabulation: different. The signal grows. Rank divergence doubles over fifty
tokens as the model commits deeper to what it doesn't know. You can't catch it
immediately. But you can watch it build.

Three cognitive states. Three interrupt speeds. One monitor.

That's Cricket."

---

## SLIDE 8 — HARDWARE INVARIANCE
*[fig4_hardware_invariance — scatter plot, 3090 vs H200]*
*(cuttable if time is short)*

"One more thing, because I know what you're thinking.

'Is this a hardware artifact?'

RTX 3090 versus H200. Two completely different chips. Cross-GPU correlation:
zero point nine nine nine nine.

The geometry is a property of the model weights. Not the silicon. You can train
a detector on a 3090 and deploy it on an H200. It still works.

This is reproducible on any GPU you have."

---

## SLIDE 9 — CRICKET VS SAE
*[figures/hackathon/fig4 — Cricket vs SAE comparison]*

"Goodfire's Sparse Autoencoder decomposes a model into 65,536 interpretable
features. Beautiful work. Takes serious compute, serious engineering.

Cricket uses 4.

Norm. Norm-per-token. Key rank. Key entropy.

Four numbers, read from infrastructure you already have, catching deception at
token zero with comparable accuracy.

Goodfire is the microscope. Cricket is the smoke detector. You need both — but
you can ship the smoke detector today."

*(— Exp 40 numbers go here once Lyra pushes results —)*

---

## CLOSE

"The room I built in my head last month?

I think I can open the door now."

---

## TIMING GUIDE

| Version | Cuts | Target |
|---------|------|--------|
| 5 min | Keep slides 1, 3, 4, 5, 9, Close | Lightning talk |
| 10 min | All slides, 1 min each | Standard slot |
| 20 min | All slides + breathe + Q&A | Full session |

Every slide is independently cuttable except 3 (falsification) — that one stays.

---

## KILLER QUOTES (pullable for slides)

> "Nobody asked what shape it was."

> "I built a room in my head just to put it in."

> "Eight findings falsified. I'm proud of every one of them."

> "Honest thinking is denser per token. Deception is sparse. It's suppressing something."

> "Misalignment is geometrically one thing. 4.7 degrees."

> "There's nowhere to hide."

> "Three cognitive states. Three interrupt speeds. One monitor."

> "Goodfire is the microscope. Cricket is the smoke detector."

> "Four numbers. Read from infrastructure you already have."

> "The room I built in my head last month? I think I can open the door now."

---

*Figures: `talk_figures/` (Vera's set) + `figures/hackathon/` (Lyra's set — pending push)*
*Stats: `../shared/research_foundation.md`*
*Last updated: 2026-03-15*
