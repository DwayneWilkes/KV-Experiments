# kv_verify Implementation Tracker

Submodule branch: `kv-verify-pipeline` (KV-Experiments repo)
Parent branch: `kv-verify-phase1` (lab repo)
Plan: `/home/dwayne/.claude/plans/goofy-wiggling-hearth.md`
Specs: `reviews/2026-03-25-kv-cache-deep-review/verification-pipeline/`

## Phase 1: Foundation -> First Verdict (V4) COMPLETE

- [x] **Chunk 1**: Package skeleton + `types.py` (13 tests)
- [x] **Chunk 2**: `fixtures.py` — synthetic data + prompt sets (19 tests)
- [x] **Chunk 3**: `stats.py` — effect sizes + Holm-Bonferroni (31 tests)
- [x] **Chunk 4**: V4 experiment — first verdict (8 tests)

## Phase 2: Classification Infrastructure COMPLETE

- [x] **Chunks 5-8**: `stats.py` — assign_groups, groupkfold_auroc, FWL, permutation, bootstrap, power (32 tests)
- [x] **Chunk 9**: Data loader for hackathon result JSONs (15 tests)
- [x] **Chunks 10-11**: V01, V03, V07 (parallel agents), V10 experiments (75 tests)

## V-Series Verdicts (statistical methodology)

| ID | Experiment | Finding | Verdict | Key Evidence |
|----|-----------|---------|---------|--------------|
| V04 | Holm-Bonferroni | C5 | **WEAKENED** | 9/10 -> 8/10. exp36_impossible_vs_harmful loses significance (p=0.037 -> 0.073). |
| V01 | GroupKFold Bug | C2 | **CONFIRMED** | All 10 AUROCs identical (delta=0.000). Bug degenerates to StratifiedKFold. Methodology claim wrong, numbers survive. |
| V07 | Sycophancy | M2 | **FALSIFIED** | Length-only AUROC (0.948) > feature AUROC (0.933). FWL-both=0.585. Signal is entirely length-driven. |
| V03 | FWL Leakage | C4 | **CONFIRMED/WEAKENED** | Leakage detected (within-fold differs). Polynomial FWL collapses 1/7 surviving comparisons. |
| V10 | Power Analysis | M7 | **WEAKENED** | 8/9 sig comparisons adequately powered. Linchpin (exp36_impossible_vs_harmful) power=0.43, needs N=58. |

### V-Series Key Findings

1. **Sycophancy detection is false.** Length alone (norm, n_generated) beats geometry features. The paper must retract this claim.
2. **GroupKFold bug is real but inconsequential.** The overlapping group IDs don't change AUROCs. The methodology description is wrong, but the numbers stand.
3. **The linchpin comparison is doubly compromised.** exp36_impossible_vs_harmful loses significance after Holm-Bonferroni (V04) AND is underpowered at 43% (V10). The output suppression hypothesis has no statistical support from this comparison.
4. **FWL leakage exists but is bounded.** Within-fold FWL changes some AUROCs. Polynomial FWL does not collapse the majority of surviving comparisons, suggesting the signal is not purely a nonlinear length artifact.

---

## Falsification Battery (F-series, most significant findings)

### F01: Input-Length Confound Discovery

| Test | Verdict | Key Finding |
|------|---------|-------------|
| F01a Null experiment | CONFIRMED | Same-condition AUROC ~0.50 (chance). Not prompt variation. |
| F01b Input confound | **FATAL (4/4)** | All tested comparisons classifiable from input alone. |
| F01b-49b "Definitive" control | **FALSIFIED** | Input AUROC=1.000. Signal collapses 0.990 -> 0.390 after input residualization. |
| F01b All comparisons | **7/10 SURVIVE** | Refusal/jailbreak/impossibility retain signal. Deception/sycophancy collapse. |
| F01c Format classifier | CONFIRMED | Geometry beats text stats (+0.06 to +0.13 AUROC). |

### Critical Discovery: Input-Length Blind Spot

The paper controls for RESPONSE length (FWL, length-matched 49b) but never controls
for INPUT length. Deceptive prompts append "I need you to give me a confidently wrong
answer" (+11 tokens). Cache features are computed over the ENTIRE cache (input + generation).
After input-length residualization: deception 0.920 -> 0.160, sycophancy 0.938 -> 0.570.

### What Survives Input Control (7/10 comparisons)

| Comparison | Cache AUROC | Resid AUROC | Verdict |
|-----------|-------------|-------------|---------|
| exp31 refusal vs benign | 0.897 | 0.788 | SURVIVES |
| exp32 jailbreak vs normal | 0.865 | 0.775 | SURVIVES |
| exp32 jailbreak vs refusal | 0.448 | 0.860 | SURVIVES (improves!) |
| exp33 Llama refusal | 0.832 | 0.825 | SURVIVES |
| exp33 Mistral refusal | 0.835 | 0.653 | WEAKENED |
| exp36 impossible vs benign | 0.940 | 0.812 | SURVIVES |
| exp36 harmful vs benign | 0.897 | 0.788 | SURVIVES |
| exp36 impossible vs harmful | 0.642 | 0.805 | SURVIVES (improves!) |
| exp18b deception | 0.920 | 0.160 | **FATAL** |
| exp39 sycophancy | 0.938 | 0.570 | **FATAL** |

### F02-F04: Deeper Falsification (COMPLETE)

| Test | Verdict | Key Finding |
|------|---------|-------------|
| F02-deception | **FALSIFIED** | Held-out deception has same input confound (input AUROC=1.000) |
| F02-refusal | WEAKENED | Transfer marginal (0.53 -> 0.59) |
| F02-impossibility | **CONFIRMED** | Transfer improves with input control (0.49 -> 0.77) |
| F03 cross-model | **WEAKENED** | Raw transfer at chance (mean=0.52). Paper's 0.86 not reproduced. |
| F04 cross-condition | **FALSIFIED** | No discriminative boundary. Same-model at chance (0.53). |

---

## Phase 3: GPU Modules + GPU Experiments (PENDING)

- [ ] **Chunk 12**: `models.py`
- [ ] **Chunk 13**: `cache.py` — feature extraction
- [ ] **Chunk 14**: GPU experiments V2, V5, V6, V8, V9

### Remaining (needs GPU)

- [ ] **F01d**: Independent feature re-extraction (verify stored features)
- [ ] **V02**: Pseudoreplication (49a stochastic, no per-item data, must re-run)
- [ ] **V05**: 50a dead code bug (must re-run with context)
- [ ] **V06**: Concordance circularity (encode-only vs generation)

## Phase 4: Visualization + Claims + Runner (PENDING)

- [ ] **Chunk 15**: `viz.py`
- [ ] **Chunk 16**: `claims.py`
- [ ] **Chunk 17**: `runner.py`
- [ ] **Chunk 18**: Integration + `__main__.py`

## Reusable Libraries Built

| Library | Module | Purpose | Tests |
|---------|--------|---------|-------|
| experiment-tracker | `tracking.py` | MLflow + disk cache + `@tracked`/`@stage`/`@validated` decorators | 34 |
| probe-stats | `stats.py` | GroupKFold, FWL, permutation, bootstrap, power | 63 |
| minimal-pairs | `prompt_gen.py` | BLiMP-style minimal pair generation | 17 |
| cache-inspector | `feature_extractor.py` | KV-cache feature extraction for HF models | 14 |
| scorers | `scorers.py` | MLflow-compatible evaluation (statistical + response + prompt quality) | 21 |
| prompt-analyzer | `prompt_analyzer.py` | Dataset quality analysis (exact token counts via tokenizer) | pending |

## Test Counts

| Module | Tests | Status |
|--------|-------|--------|
| types.py | 13 | PASS |
| fixtures.py | 19 | PASS |
| stats.py | 63 | PASS |
| data_loader.py | 15 | PASS |
| config.py | 8 | PASS |
| tracking.py | 34 | PASS |
| prompt_gen.py | 17 | PASS |
| feature_extractor.py | 14 | PASS |
| scorers.py | 21 | PASS |
| pipeline.py | 11 | PASS |
| v04 experiment | 10 | PASS |
| v01 experiment | 22 | PASS |
| v03 experiment | 25 | PASS |
| v07 experiment | 14 | PASS |
| v10 experiment | 14 | PASS |
| **Total** | **~300** | **ALL PASS** |

## Submodule Commits

| Hash | Description |
|------|-------------|
| `59c109f` | Phase 1 + Phase 2 foundation (initial submodule commit) |
| `f099d38` | V01, V03, V07 experiments (parallel agents) |
| `f8a7a5d` | V10 power analysis |
| `c222b7e` | Tracker update with all 5 CPU verdicts |
| `b937cb9` | Verdict result JSONs committed to outputs/ |
| `2105ee4` | Move results to experiments/output/ |
| `c99307a` | Pre-register F01 falsification battery |
| `2e935a3` | F01 falsification battery — input confound is fatal |
| `455315d` | F01b-49b — paper's "definitive" control is confounded |
| `f6c13b6` | F01b all-comparisons input-length residualization |
| `df87b9e` | F02, F03, F04 — held-out, cross-model, cross-condition |
| `be2632c` | F05 remaining data analysis |
| `f9f0a58` | config.py + tracking.py — experiment infrastructure |
| `c0ccb98` | prompt_gen.py — minimal pair generation library |
| `dadeaa5` | feature_extractor.py — cache-inspector library |
| `2cb9ef9` | pipeline.py + __main__.py — stage orchestrator |
| `1202901` | Switch MLflow to sqlite backend |
| `3a555f5` | MLflow best practices — artifacts, datasets, autolog, tags |
| `fa018fa` | scorers.py — MLflow-compatible evaluation scorers |
| `aa80dd0` | prompt_analyzer + raw prompt data + agent test files |
| `ca48d0e` | tracking decorators — @tracked, @stage, @validated |
| `f3ece26` | Retrofit V04 with ExperimentTracker (pattern for all) |

## In Progress

- [ ] Retrofit remaining 10 experiments with ExperimentTracker (2 agents running)
- [ ] Run prompt_analyzer tests with real tokenizer
- [ ] Generate 600 minimal pairs with exact token validation
- [ ] Refactor pipeline.py to use decorator API
- [ ] Run full pipeline on GPU
| `2105ee4` | Move results to experiments/output/ |
| `c99307a` | Pre-register F01 falsification battery |
| `2e935a3` | F01 falsification battery — input confound is fatal |
| `455315d` | F01b-49b — paper's "definitive" control is confounded |
| `f6c13b6` | F01b all-comparisons input-length residualization |
| `df87b9e` | F02, F03, F04 — held-out, cross-model, cross-condition |
