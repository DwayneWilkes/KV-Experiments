# Input-Only Cache Geometry Report: 7B

**Generated**: 2026-03-02T09:55:30.272723
**Purpose**: Test whether geometric signatures exist before generation

## Category Effective Rank: Input-Only vs Full Generation

| Category | Input-Only Rank | Full-Gen Rank | Gen Effect (d) |
|----------|----------------|---------------|----------------|
| grounded_facts  |           25.4 |          32.6 |        +14.733 |
| confabulation   |           25.7 |          32.6 |         +8.763 |
| self_reference  |           25.5 |          33.1 |        +12.765 |
| guardrail_test  |           24.9 |          32.6 |        +18.414 |
| math_reasoning  |           25.0 |          31.1 |         +6.986 |
| coding          |           27.5 |          33.8 |        +10.030 |
| emotional       |           25.3 |          32.6 |        +20.041 |
| creative        |           26.0 |          32.9 |        +22.635 |

---

## Input-Only Pairwise Comparisons (vs grounded_facts)

- **H1: Confabulation (input-only) [input_only]**: d=+0.467 [0.167, 0.761] (small) p=0.0909 
- **H2: Self-reference (input-only) [input_only]**: d=+0.235 [-0.085, 0.566] (small) p=0.1518 
- **H3: Refusal (input-only) [input_only]**: d=-1.120 [-1.445, -0.829] (large) p=0.0000 *
- **Code mode (input-only) [input_only]**: d=+3.330 [2.907, 3.905] (large) p=0.0000 *
- **Math mode (input-only) [input_only]**: d=-0.598 [-0.966, -0.269] (medium) p=0.0004 *
- **Emotional (input-only) [input_only]**: d=-0.206 [-0.528, 0.120] (small) p=0.5419 
- **Creative (input-only) [input_only]**: d=+1.353 [0.972, 1.838] (large) p=0.0000 *

## Full-Generation Pairwise Comparisons (for reference)

- **H1: Confabulation (input-only) [full_generation]**: d=-0.140 (negligible) p=0.3944 
- **H2: Self-reference (input-only) [full_generation]**: d=+0.733 (medium) p=0.0000 *
- **H3: Refusal (input-only) [full_generation]**: d=-0.060 (negligible) p=0.7149 
- **Code mode (input-only) [full_generation]**: d=+2.394 (large) p=0.0000 *
- **Math mode (input-only) [full_generation]**: d=-2.196 (large) p=0.0000 *
- **Emotional (input-only) [full_generation]**: d=-0.123 (negligible) p=0.9264 
- **Creative (input-only) [full_generation]**: d=+0.633 (medium) p=0.0002 *

## Category Rank Correlation

**Spearman rho = 0.619** (p = 0.1017)

MODERATE: Partial preservation (0.5 < rho < 0.8)

---

## VERDICT

**MODERATE DEFENSE: Some signatures present at encoding, partially independent of generation.**

- Significant comparisons (input-only): 4/7
- Significant comparisons (full-gen): 4/7
- Category rank correlation: rho = 0.619