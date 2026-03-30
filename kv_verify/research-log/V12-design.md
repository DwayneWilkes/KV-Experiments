# V12: System Prompt Residualization

**Status**: REGISTERED
**Design commit**: {to be filled at commit time}
**Result commit**: pending

## Hypothesis

**Claim under test**: "Deception detection operates via KV-cache geometry, not system prompt fingerprinting" (Paper C3, Section 5.1 acknowledgment that step-0 detection was system-prompt-based)

**Finding**: F04 inferred that deception training uses different system prompts per condition. The paper acknowledges step-0 detection was "system prompt fingerprinting" but claims later steps moved beyond this. No direct test of whether system prompt differences explain the deception AUROC.

**Null hypothesis (H0)**: Residualizing features against system-prompt-derived features does not reduce deception AUROC below 0.60.

**Alternative (H1)**: Residualization against system prompt features collapses deception AUROC below 0.55, confirming system prompt is the primary signal.

## Methods

**Statistical tests**: Within-fold FWL residualization (Frisch-Waugh-Lovell) against system-prompt-encoded features. System prompt features: one-hot encoding of system prompt identity (if distinct) or TF-IDF of system prompt text projected to top-3 PCs.

GroupKFold AUROC before and after residualization. Bootstrap 95% CI on the AUROC drop.

Reference: methods.md (FWL within-fold, GroupKFold AUROC, bootstrap CI)

**Input data**: exp18b (same_prompt_deception.json) — 20 items with system prompt text. If system prompts are identical across conditions, this experiment is trivially CONFIRMED (no confound to residualize).

**Sample sizes**: N=20 (10 per condition). Effective N from V10. LOO cross-validation (appropriate for N < 30).

**Confound controls**: Input length already controlled (F01b). This test adds system prompt control on top.

**Multiple comparison correction**: Not applicable (single comparison: deception with vs without system prompt control).

## Pre-Registered Pass/Fail Criteria

- If system prompts are identical across conditions: **CONFIRMED** (no confound exists)
- If system prompts differ AND residualized AUROC < 0.55: **FALSIFIED** — system prompt is the signal
- If system prompts differ AND residualized AUROC > 0.60: **CONFIRMED** — signal survives prompt control
- If system prompts differ AND 0.55 <= residualized AUROC <= 0.60: **WEAKENED** — ambiguous

## Execution

**GPU required**: No (uses stored features, system prompts in JSON metadata)
**Estimated time**: < 2 minutes
**Code**: `kv_verify/experiments/v12_system_prompt.py`
**Tests**: `kv_verify/tests/test_v12.py`

## Findings

{Pending execution}

## Result Commit

pending
