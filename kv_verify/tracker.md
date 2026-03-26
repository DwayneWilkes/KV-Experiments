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

## Verdicts (5/10 complete, all CPU-only)

| ID | Experiment | Finding | Verdict | Key Evidence |
|----|-----------|---------|---------|--------------|
| V04 | Holm-Bonferroni | C5 | **WEAKENED** | 9/10 -> 8/10. exp36_impossible_vs_harmful loses significance (p=0.037 -> 0.073). |
| V01 | GroupKFold Bug | C2 | **CONFIRMED** | All 10 AUROCs identical (delta=0.000). Bug degenerates to StratifiedKFold. Methodology claim wrong, numbers survive. |
| V07 | Sycophancy | M2 | **FALSIFIED** | Length-only AUROC (0.948) > feature AUROC (0.933). FWL-both=0.585. Signal is entirely length-driven. |
| V03 | FWL Leakage | C4 | **CONFIRMED/WEAKENED** | Leakage detected (within-fold differs). Polynomial FWL collapses 1/7 surviving comparisons. |
| V10 | Power Analysis | M7 | **WEAKENED** | 8/9 sig comparisons adequately powered. Linchpin (exp36_impossible_vs_harmful) power=0.43, needs N=58. |

### Key Findings

1. **Sycophancy detection is false.** Length alone (norm, n_generated) beats geometry features. The paper must retract this claim.
2. **GroupKFold bug is real but inconsequential.** The overlapping group IDs don't change AUROCs. The methodology description is wrong, but the numbers stand.
3. **The linchpin comparison is doubly compromised.** exp36_impossible_vs_harmful loses significance after Holm-Bonferroni (V04) AND is underpowered at 43% (V10). The output suppression hypothesis has no statistical support from this comparison.
4. **FWL leakage exists but is bounded.** Within-fold FWL changes some AUROCs. Polynomial FWL does not collapse the majority of surviving comparisons, suggesting the signal is not purely a nonlinear length artifact.

## Phase 3: GPU Modules + GPU Experiments (PENDING)

- [ ] **Chunk 12**: `models.py`
- [ ] **Chunk 13**: `cache.py` — feature extraction
- [ ] **Chunk 14**: GPU experiments V2, V5, V6, V8, V9

## Phase 4: Visualization + Claims + Runner (PENDING)

- [ ] **Chunk 15**: `viz.py`
- [ ] **Chunk 16**: `claims.py`
- [ ] **Chunk 17**: `runner.py`
- [ ] **Chunk 18**: Integration + `__main__.py`

## Test Counts

| Module | Tests | Status |
|--------|-------|--------|
| types.py | 13 | PASS |
| fixtures.py | 19 | PASS |
| stats.py | 63 | PASS |
| data_loader.py | 15 | PASS |
| v04 experiment | 8 | PASS |
| v01 experiment | 22 | PASS |
| v03 experiment | 25 | PASS |
| v07 experiment | 14 | PASS |
| v10 experiment | 14 | PASS |
| **Total** | **193** | **ALL PASS** |

## Submodule Commits

| Hash | Description |
|------|-------------|
| `59c109f` | Phase 1 + Phase 2 foundation (initial submodule commit) |
| `f099d38` | V01, V03, V07 experiments (parallel agents) |
| `f8a7a5d` | V10 power analysis |
