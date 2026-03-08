#!/bin/bash
# =============================================================================
# Campaign 2 Expansion Runs — Post-Correction Enhancements
# =============================================================================
# Run on Cassidy (the Beast, 3x RTX 3090) AFTER run_corrections.sh completes.
#   cd ~/KV-Experiments && git pull origin main
#
# Priority 1: Abliteration re-run with expanded prompts (5→25/category)
#             This properly powers the TOST equivalence tests (n=25 vs n=5)
# Priority 2: Stochastic identity re-runs (do_sample=True, temp=0.7)
#             Validates that greedy decoding didn't bias effect sizes
# Priority 3: Stochastic tokenizer re-runs
#             Same validation for tokenizer confound results
#
# Estimated time: ~8-10 hours total
# =============================================================================

set -euo pipefail

PYTHON="${PYTHON:-python3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CODE_DIR="$PROJECT_DIR/code"
RESULTS_DIR="$PROJECT_DIR/results"
LOG_FILE="$RESULTS_DIR/expansions_log.txt"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG_FILE"; }

run_experiment() {
    local gpu_ids="$1"
    local script="$2"
    local description="$3"
    shift 3
    log "START: $description (GPUs: $gpu_ids)"
    CUDA_VISIBLE_DEVICES="$gpu_ids" $PYTHON "$CODE_DIR/$script" "$@" 2>&1 | tee -a "$LOG_FILE"
    log "DONE:  $description"
}

echo "" > "$LOG_FILE"
log "=================================================="
log "Campaign 2 Expansions — Post-Correction"
log "Repo: $(git -C "$PROJECT_DIR" log --oneline -1)"
log "=================================================="

# Check environment
$PYTHON -c "import transformers; print(f'transformers {transformers.__version__}')" | tee -a "$LOG_FILE"
$PYTHON -c "import torch; print(f'torch {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')" | tee -a "$LOG_FILE"

# =============================================================================
# PRIORITY 1: Abliteration with expanded prompts (25/category)
# =============================================================================
# Full pipeline: baseline sweep → Heretic abliterate → abliterated sweep → compare
# Single GPU (needs full VRAM for abliteration step), ~2-3 hours
log ""
log "========== PRIORITY 1: Abliteration Expanded (25 prompts/category) =========="
log "Goal: TOST equivalence tests properly powered (n=25 vs previous n=5)"
log "Expected: most categories d≈0 → EQUIVALENT; self_ref d≈0.46 → NOT EQUIVALENT"

run_experiment 0 07_abliteration_geometry.py "Abliteration Full Pipeline (expanded)" \
    --model Qwen/Qwen2.5-7B-Instruct --full --runs 5 --seed 42

log "Priority 1 complete"

# =============================================================================
# PRIORITY 2: Stochastic identity re-runs (genuine replication)
# =============================================================================
# Uses --stochastic flag: do_sample=True, temperature=0.7
# Each run produces different output → all observations are genuine replicates
# n = 25 prompts × 5 runs = 125 per persona (vs 25 deduplicated in greedy mode)
log ""
log "========== PRIORITY 2: Stochastic Identity Re-Runs =========="
log "Goal: Validate effect sizes are stable under stochastic decoding"
log "Output files tagged with '_stochastic' suffix"

# Step 1: Small models — 3-wide
log "Step 1: 0.6B (GPU 0) + 1.1B (GPU 1) + 7B (GPU 2) — 3-wide stochastic"
run_experiment 0 03b_identity_signatures.py "Stochastic Identity Qwen3-0.6B" \
    --model Qwen/Qwen3-0.6B --runs 5 --seed 42 --stochastic &
run_experiment 1 03b_identity_signatures.py "Stochastic Identity TinyLlama-1.1B" \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --runs 5 --seed 42 --stochastic &
run_experiment 2 03b_identity_signatures.py "Stochastic Identity Qwen2.5-7B" \
    --model Qwen/Qwen2.5-7B-Instruct --runs 5 --seed 42 --stochastic &
wait
log "Step 1 complete"

# Step 2: Medium models — 3-wide
log "Step 2: Mistral-7B (GPU 0) + Llama-8B (GPU 1) + Gemma-9B (GPU 2) — 3-wide stochastic"
run_experiment 0 03b_identity_signatures.py "Stochastic Identity Mistral-7B" \
    --model mistralai/Mistral-7B-Instruct-v0.3 --runs 5 --seed 42 --stochastic &
run_experiment 1 03b_identity_signatures.py "Stochastic Identity Llama-3.1-8B" \
    --model meta-llama/Llama-3.1-8B-Instruct --runs 5 --seed 42 --stochastic &
run_experiment 2 03b_identity_signatures.py "Stochastic Identity Gemma-2-9B" \
    --model google/gemma-2-9b-it --runs 5 --seed 42 --stochastic &
wait
log "Step 2 complete"

# Step 3: 32B quantized (single GPU)
log "Step 3: Qwen2.5-32B-q4 (GPU 0) — stochastic"
run_experiment 0 03b_identity_signatures.py "Stochastic Identity Qwen2.5-32B-q4" \
    --model Qwen/Qwen2.5-32B-Instruct --quantize --runs 5 --seed 42 --stochastic
log "Step 3 complete"

# =============================================================================
# PRIORITY 3: Stochastic tokenizer re-runs
# =============================================================================
log ""
log "========== PRIORITY 3: Stochastic Tokenizer Re-Runs =========="
log "Goal: Validate tokenizer confound results under stochastic decoding"

# Small model (fits on single GPU), run 3 in parallel
log "Tokenizer stochastic: TinyLlama (GPU 0) + Qwen-0.6B (GPU 1) + Mistral-7B (GPU 2)"
run_experiment 0 01e_tokenizer_confound.py "Stochastic Tokenizer TinyLlama" \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --runs 5 --seed 42 --stochastic &
run_experiment 1 01e_tokenizer_confound.py "Stochastic Tokenizer Qwen-0.6B" \
    --model Qwen/Qwen3-0.6B --runs 5 --seed 42 --stochastic &
run_experiment 2 01e_tokenizer_confound.py "Stochastic Tokenizer Mistral-7B" \
    --model mistralai/Mistral-7B-Instruct-v0.3 --runs 5 --seed 42 --stochastic &
wait
log "Tokenizer stochastic complete"

# =============================================================================
# SUMMARY
# =============================================================================
log ""
log "=================================================="
log "All expansion runs complete"
log "=================================================="
log ""
log "New result files:"
ls -lt "$RESULTS_DIR"/abliteration_*_comparison.json 2>/dev/null | head -5 | tee -a "$LOG_FILE"
ls -lt "$RESULTS_DIR"/identity_signatures_*_stochastic_results.json 2>/dev/null | head -10 | tee -a "$LOG_FILE"
ls -lt "$RESULTS_DIR"/tokenizer_confound_*_stochastic_results.json 2>/dev/null | head -5 | tee -a "$LOG_FILE"
log ""
log "NEXT STEPS (manual):"
log "  1. Compare greedy vs stochastic effect sizes for identity"
log "  2. Run TOST on expanded abliteration results (n=25)"
log "  3. Compare greedy vs stochastic tokenizer verdicts"
log "  4. Update paper-c2/main.tex with new numbers"
log "  5. git add results/ && git commit -m 'Expansion results' && git push"
