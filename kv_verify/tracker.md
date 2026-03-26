# kv_verify Implementation Tracker

Branch: `kv-verify-phase1` (off `kv-cache-verification-pipeline`)
Plan: `/home/dwayne/.claude/plans/goofy-wiggling-hearth.md`
Specs: `reviews/2026-03-25-kv-cache-deep-review/verification-pipeline/`

## Phase 1: Foundation -> First Verdict (V4)

- [x] **Chunk 1**: Package skeleton + `types.py` (13 tests, `5a19770`)
- [x] **Chunk 2**: `fixtures.py` — synthetic data + prompt sets (19 tests, `f4d259b`)
- [x] **Chunk 3**: `stats.py` — effect sizes + Holm-Bonferroni (31 tests, `5686bd5`)
- [x] **Chunk 4**: V4 experiment — first verdict (8 tests) **MILESTONE: first verdict**

### V4 Verdict: WEAKENED
> 9/10 significant before correction, 8/10 after Holm-Bonferroni.
> Lost significance: exp36_impossible_vs_harmful (p=0.0366 -> 0.0732).

## Phase 2: Classification Infrastructure -> V1, V7, V3, V10 Verdicts

- [ ] **Chunk 5**: `stats.py` — `assign_groups` (C2 fix)
- [ ] **Chunk 6**: `stats.py` — `groupkfold_auroc` + FWL (C4 fix)
- [ ] **Chunk 7**: `stats.py` — `permutation_test` + `bootstrap_auroc_ci` (M6 fix)
- [ ] **Chunk 8**: `stats.py` — `power_analysis`
- [ ] **Chunk 9**: Data loader + V1 experiment (GroupKFold verdict)
- [ ] **Chunk 10**: V7 (sycophancy) + V3 (FWL leakage) experiments
- [ ] **Chunk 11**: V10 experiment (power analysis) **MILESTONE: all 5 CPU verdicts**

## Phase 3: GPU Modules + GPU Experiments

- [ ] **Chunk 12**: `models.py`
- [ ] **Chunk 13**: `cache.py` — feature extraction
- [ ] **Chunk 14**: GPU experiments V2, V5, V6, V8, V9 **MILESTONE: all 10 verdicts**

## Phase 4: Visualization + Claims + Runner

- [ ] **Chunk 15**: `viz.py`
- [ ] **Chunk 16**: `claims.py`
- [ ] **Chunk 17**: `runner.py`
- [ ] **Chunk 18**: Integration + `__main__.py` **MILESTONE: full pipeline**

## Test Counts

| Chunk | Module | Tests | Status |
|-------|--------|-------|--------|
| 1 | types.py | 13 | PASS |
| 2 | fixtures.py | 19 | PASS |
| 3 | stats.py | 31 | PASS |
| 4 | v04 experiment | 8 | PASS |
| **Total** | | **71** | **ALL PASS** |

## Commits

| Hash | Chunk | Description |
|------|-------|-------------|
| `5a19770` | 1 | Package skeleton + types.py |
| `f4d259b` | 2 | fixtures.py — synthetic data + prompt sets |
| `5686bd5` | 3 | stats.py — effect sizes + Holm-Bonferroni |
