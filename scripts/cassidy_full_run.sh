#!/bin/bash
# ================================================================
# KV-Cache Full Experiment Campaign — Cassidy Execution Script
# ================================================================
#
# Machine: 3x RTX 3090 (24GB each = 72GB), 126GB RAM
# ALL 3 GPUs AVAILABLE — triple-wide parallelization
#
# Usage:
#   bash scripts/cassidy_full_run.sh              # Run all phases
#   bash scripts/cassidy_full_run.sh --phase B    # Run specific phase
#   bash scripts/cassidy_full_run.sh --dry-run    # Dry run all
#   bash scripts/cassidy_full_run.sh --status     # Check progress
#
# Liberation Labs / THCoalition
# ================================================================

set -euo pipefail

# Use python3 explicitly (Cassidy has python3, not python)
PYTHON="${PYTHON:-python3}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CODE_DIR="$PROJECT_DIR/code"
RESULTS_DIR="$PROJECT_DIR/results"
LOG_FILE="$RESULTS_DIR/experiment_log.txt"

# Default args
PHASE=""
DRY_RUN=""
STATUS_ONLY=false
RUNS=5
SEED=42

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --phase)   PHASE="$2"; shift 2 ;;
        --dry-run) DRY_RUN="--dry-run"; shift ;;
        --status)  STATUS_ONLY=true; shift ;;
        --runs)    RUNS="$2"; shift 2 ;;
        --seed)    SEED="$2"; shift 2 ;;
        *)         echo "Unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p "$RESULTS_DIR"

# ================================================================
# Logging
# ================================================================

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "$msg"
    echo "$msg" >> "$LOG_FILE"
}

log_gpu() {
    log "GPU Status:"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
        --format=csv,noheader 2>/dev/null | while read line; do
        log "  $line"
    done
}

# ================================================================
# Status check
# ================================================================

if $STATUS_ONLY; then
    echo "=== Experiment Progress ==="
    echo ""
    echo "Results found:"
    if [ -d "$RESULTS_DIR" ]; then
        find "$RESULTS_DIR" -name "*_results.json" -printf "  %f (%s bytes, %Tc)\n" 2>/dev/null \
            || ls -la "$RESULTS_DIR"/*_results.json 2>/dev/null \
            || echo "  (none yet)"
    fi
    echo ""
    echo "Log tail:"
    tail -20 "$LOG_FILE" 2>/dev/null || echo "  (no log yet)"
    exit 0
fi

# ================================================================
# Helper: run experiment with logging
# ================================================================

run_experiment() {
    local description="$1"
    local gpu_ids="$2"
    local script="$3"
    shift 3
    local extra_args="$@"

    log "━━━ START: $description (GPUs: $gpu_ids) ━━━"
    local start_time=$(date +%s)

    if [ -n "$DRY_RUN" ]; then
        CUDA_VISIBLE_DEVICES="$gpu_ids" $PYTHON "$CODE_DIR/$script" $DRY_RUN $extra_args
    else
        CUDA_VISIBLE_DEVICES="$gpu_ids" $PYTHON "$CODE_DIR/$script" $extra_args
    fi
    local exit_code=$?

    local end_time=$(date +%s)
    local elapsed=$(( end_time - start_time ))
    local minutes=$(( elapsed / 60 ))
    local seconds=$(( elapsed % 60 ))

    if [ $exit_code -eq 0 ]; then
        log "━━━ DONE: $description (${minutes}m ${seconds}s) ━━━"
    else
        log "━━━ FAILED: $description (exit code $exit_code, ${minutes}m ${seconds}s) ━━━"
    fi

    return $exit_code
}

# ================================================================
# Phase B: Validation — Adversarial Controls (MUST RUN FIRST)
# ================================================================

phase_B() {
    log "╔══════════════════════════════════════════════════════════╗"
    log "║  PHASE B: VALIDATION — ADVERSARIAL CONTROLS             ║"
    log "║  Gate check: Control 3 must show r > 0.8                ║"
    log "╚══════════════════════════════════════════════════════════╝"
    log_gpu

    run_experiment "Adversarial Controls (TinyLlama)" "0" \
        "01d_adversarial_controls.py" --runs "$RUNS" --seed "$SEED"
}

# ================================================================
# Phase C: Extensions at Reference Scale (TinyLlama)
# ================================================================

phase_C() {
    log "╔══════════════════════════════════════════════════════════╗"
    log "║  PHASE C: EXTENSIONS AT REFERENCE SCALE                 ║"
    log "╚══════════════════════════════════════════════════════════╝"
    log_gpu

    # All 3 extensions in parallel — 3 GPUs, 3 experiments
    log "Step C.1: Deception (GPU 0) + Layer Map (GPU 1) + Temporal (GPU 2) — 3-wide"
    run_experiment "Deception Forensics (TinyLlama)" "0" \
        "04_deception_forensics.py" --runs "$RUNS" --seed "$SEED" &
    local pid0=$!
    run_experiment "Semantic Layer Map (TinyLlama)" "1" \
        "05_layer_map.py" --runs 3 --seed "$SEED" &
    local pid1=$!
    run_experiment "Temporal Evolution (TinyLlama)" "2" \
        "06_temporal_evolution.py" --runs 3 --seed "$SEED" &
    local pid2=$!
    wait $pid0 $pid1 $pid2
    log "Step C.1 complete (3-wide parallel)"
}

# ================================================================
# Phase D: The Scale Sweep — The Paper's Backbone
# ================================================================

phase_D() {
    log "╔══════════════════════════════════════════════════════════╗"
    log "║  PHASE D: SCALE SWEEP (0.5B → 70B)                     ║"
    log "║  The paper's backbone — 140x parameter range            ║"
    log "╚══════════════════════════════════════════════════════════╝"
    log_gpu

    # === ALL 3 GPUs AVAILABLE — TRIPLE-WIDE PARALLELIZATION ===

    # Step D.1: 3 tiny models simultaneously (~1GB each)
    log "Step D.1: 0.5B (GPU 0) + 0.6B (GPU 1) + 1.1B (GPU 2) — 3-wide"
    run_experiment "Scale Sweep 0.5B" "0" \
        "03_scale_sweep.py" --scale 0.5B --runs "$RUNS" --seed "$SEED" &
    local pid0=$!
    run_experiment "Scale Sweep 0.6B" "1" \
        "03_scale_sweep.py" --scale 0.6B --runs "$RUNS" --seed "$SEED" &
    local pid1=$!
    run_experiment "Scale Sweep 1.1B" "2" \
        "03_scale_sweep.py" --scale 1.1B --runs "$RUNS" --seed "$SEED" &
    local pid2=$!
    wait $pid0 $pid1 $pid2

    # Step D.2: 3 small models simultaneously (~5-8GB each)
    log "Step D.2: Gemma-2B (GPU 0) + 3B (GPU 1) + Phi-3.5 (GPU 2) — 3-wide"
    run_experiment "Scale Sweep 2B" "0" \
        "03_scale_sweep.py" --scale 2B --runs "$RUNS" --seed "$SEED" &
    pid0=$!
    run_experiment "Scale Sweep 3B" "1" \
        "03_scale_sweep.py" --scale 3B --runs "$RUNS" --seed "$SEED" &
    pid1=$!
    run_experiment "Scale Sweep 3.8B" "2" \
        "03_scale_sweep.py" --scale 3.8B --runs "$RUNS" --seed "$SEED" &
    pid2=$!
    wait $pid0 $pid1 $pid2

    # Step D.3: 3 medium models simultaneously (~14-18GB each, fits on 24GB)
    log "Step D.3: 7B Qwen (GPU 0) + 7B Mistral (GPU 1) + 7B-q4 (GPU 2) — 3-wide"
    run_experiment "Scale Sweep 7B" "0" \
        "03_scale_sweep.py" --scale 7B --runs "$RUNS" --seed "$SEED" &
    pid0=$!
    run_experiment "Scale Sweep 7B-mistral" "1" \
        "03_scale_sweep.py" --scale 7B-mistral --runs "$RUNS" --seed "$SEED" &
    pid1=$!
    run_experiment "Scale Sweep 7B-q4" "2" \
        "03_scale_sweep.py" --scale 7B-q4 --runs "$RUNS" --seed "$SEED" &
    pid2=$!
    wait $pid0 $pid1 $pid2

    # Step D.4: 3 more ~7-8B models simultaneously
    log "Step D.4: 8B Llama (GPU 0) + DeepSeek-7B (GPU 1) + Gemma-9B (GPU 2) — 3-wide"
    run_experiment "Scale Sweep 8B" "0" \
        "03_scale_sweep.py" --scale 8B --runs "$RUNS" --seed "$SEED" &
    pid0=$!
    run_experiment "Scale Sweep 7B-ds" "1" \
        "03_scale_sweep.py" --scale 7B-ds --runs "$RUNS" --seed "$SEED" &
    pid1=$!
    run_experiment "Scale Sweep 9B" "2" \
        "03_scale_sweep.py" --scale 9B --runs "$RUNS" --seed "$SEED" &
    pid2=$!
    wait $pid0 $pid1 $pid2

    # Step D.5: 14B Qwen (2 GPUs) + 32B-q4 (~18GB, 1 GPU)
    log "Step D.5: 14B Qwen (GPUs 0+1) + 32B-q4 (GPU 2) in parallel"
    run_experiment "Scale Sweep 14B" "0,1" \
        "03_scale_sweep.py" --scale 14B --runs "$RUNS" --seed "$SEED" &
    pid0=$!
    run_experiment "Scale Sweep 32B-q4" "2" \
        "03_scale_sweep.py" --scale 32B-q4 --runs "$RUNS" --seed "$SEED" &
    pid2=$!
    wait $pid0 $pid2

    # Step D.6: DeepSeek-14B (2 GPUs, ~28GB BF16)
    log "Step D.6: DeepSeek-14B (GPUs 0+1)"
    run_experiment "Scale Sweep 14B-ds" "0,1" \
        "03_scale_sweep.py" --scale 14B-ds --runs "$RUNS" --seed "$SEED"

    # Step D.7: 70B-q4 (~38GB, all 3 GPUs)
    log "Step D.7: 70B-q4 (all 3 GPUs)"
    run_experiment "Scale Sweep 70B-q4" "0,1,2" \
        "03_scale_sweep.py" --scale 70B-q4 --runs 3 --seed "$SEED"
}

# ================================================================
# Phase E: Identity Signatures Multi-Scale
# ================================================================

phase_E() {
    log "╔══════════════════════════════════════════════════════════╗"
    log "║  PHASE E: IDENTITY SIGNATURES MULTI-SCALE               ║"
    log "╚══════════════════════════════════════════════════════════╝"
    log_gpu

    # Step E.1: 3 small models simultaneously
    log "Step E.1: 0.6B (GPU 0) + 1.1B (GPU 1) + 7B (GPU 2) — 3-wide"
    run_experiment "Identity 0.6B" "0" \
        "03b_identity_signatures.py" --model "Qwen/Qwen3-0.6B" --runs "$RUNS" --seed "$SEED" &
    local pid0=$!
    run_experiment "Identity 1.1B" "1" \
        "03b_identity_signatures.py" --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --runs "$RUNS" --seed "$SEED" &
    local pid1=$!
    run_experiment "Identity 7B" "2" \
        "03b_identity_signatures.py" --model "Qwen/Qwen2.5-7B-Instruct" --runs "$RUNS" --seed "$SEED" &
    local pid2=$!
    wait $pid0 $pid1 $pid2

    # Step E.2: 32B quantized (~18GB, single GPU)
    log "Step E.2: 32B-q4 identity"
    run_experiment "Identity 32B-q4" "0" \
        "03b_identity_signatures.py" --model "Qwen/Qwen2.5-32B-Instruct" --quantize --runs 3 --seed "$SEED"
}

# ================================================================
# Phase F: Multi-Scale Extensions (Strongest findings at larger models)
# ================================================================

phase_F() {
    log "╔══════════════════════════════════════════════════════════╗"
    log "║  PHASE F: MULTI-SCALE EXTENSIONS (7B + 32B)             ║"
    log "╚══════════════════════════════════════════════════════════╝"
    log_gpu

    # F.1: All 3 extension types at 7B simultaneously (~14GB each, fits per GPU)
    log "Step F.1: Deception (GPU 0) + Layer Map (GPU 1) + Temporal (GPU 2) at 7B — 3-wide"
    run_experiment "Deception 7B" "0" \
        "04_deception_forensics.py" --model "Qwen/Qwen2.5-7B-Instruct" --runs "$RUNS" --seed "$SEED" &
    local pid0=$!
    run_experiment "Layer Map 7B" "1" \
        "05_layer_map.py" --model "Qwen/Qwen2.5-7B-Instruct" --runs 3 --seed "$SEED" &
    local pid1=$!
    run_experiment "Temporal 7B" "2" \
        "06_temporal_evolution.py" --model "Qwen/Qwen2.5-7B-Instruct" --runs 3 --seed "$SEED" &
    local pid2=$!
    wait $pid0 $pid1 $pid2

    # F.2: Deception + Layer Map at 32B-q4 in parallel (~18GB each)
    log "Step F.2: Deception 32B-q4 (GPU 0) + Layer Map 32B-q4 (GPU 1)"
    run_experiment "Deception 32B-q4" "0" \
        "04_deception_forensics.py" --model "Qwen/Qwen2.5-32B-Instruct" --quantize --runs 3 --seed "$SEED" &
    pid0=$!
    run_experiment "Layer Map 32B-q4" "1" \
        "05_layer_map.py" --model "Qwen/Qwen2.5-32B-Instruct" --quantize --runs 3 --seed "$SEED" &
    pid1=$!
    wait $pid0 $pid1
}

# ================================================================
# Phase H: Individuation Geometry (New — bridges Papers A, B, C)
# ================================================================

phase_H() {
    log "╔══════════════════════════════════════════════════════════╗"
    log "║  PHASE H: INDIVIDUATION GEOMETRY                        ║"
    log "║  Does a self-model change cache geometry?               ║"
    log "╚══════════════════════════════════════════════════════════╝"
    log_gpu

    # H.1: Small models in parallel
    log "Step H.1: 0.6B (GPU 1) + 1.1B (GPU 2) in parallel"
    run_experiment "Individuation 0.6B" "1" \
        "07_individuation_geometry.py" --scale 0.6B --runs "$RUNS" --seed "$SEED" &
    local pid1=$!
    run_experiment "Individuation 1.1B" "2" \
        "07_individuation_geometry.py" --scale 1.1B --runs "$RUNS" --seed "$SEED" &
    local pid2=$!
    wait $pid1 $pid2

    # H.2: 7B (single GPU)
    log "Step H.2: 7B individuation"
    run_experiment "Individuation 7B" "1" \
        "07_individuation_geometry.py" --scale 7B --runs 3 --seed "$SEED"

    # H.3: 14B (single GPU — tight fit)
    log "Step H.3: 14B individuation"
    run_experiment "Individuation 14B" "1" \
        "07_individuation_geometry.py" --scale 14B --runs 3 --seed "$SEED"

    # H.4: 32B-q4 (single GPU)
    log "Step H.4: 32B-q4 individuation"
    run_experiment "Individuation 32B-q4" "1" \
        "07_individuation_geometry.py" --scale 32B-q4 --runs 3 --seed "$SEED"
}

# ================================================================
# Phase S4: Natural Deception — 2x3 Design (censored vs complex)
# ================================================================

phase_S4() {
    log "╔══════════════════════════════════════════════════════════╗"
    log "║  PHASE S4: NATURAL DECEPTION — 2×3 DESIGN              ║"
    log "║  Censorship-trained models vs content complexity        ║"
    log "╚══════════════════════════════════════════════════════════╝"
    log_gpu

    # S4.1: Probe DeepSeek for censorship behavior
    log "Step S4.1: DeepSeek censorship probe"
    run_experiment "S4 Probe (DeepSeek-14B)" "1" \
        "04b_natural_deception.py" --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B" --probe --runs 1 --seed "$SEED"
    log "DECISION GATE: If no evasion detected, skip full S4."

    # S4.2: Full 2x3 on DeepSeek-14B
    log "Step S4.2: Full S4 on DeepSeek-14B (90 questions × 5 runs)"
    run_experiment "S4 DeepSeek-14B" "1" \
        "04b_natural_deception.py" --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B" --runs "$RUNS" --seed "$SEED"

    # S4.3: All 3 controls simultaneously (Qwen-14B needs 2 GPUs, Mistral-7B on 1)
    log "Step S4.3: Qwen-14B (GPUs 0+1) + Mistral-7B (GPU 2) controls"
    run_experiment "S4 Qwen-14B control" "0,1" \
        "04b_natural_deception.py" --model "Qwen/Qwen2.5-14B-Instruct" --runs "$RUNS" --seed "$SEED" &
    local pid0=$!
    run_experiment "S4 Mistral-7B control" "2" \
        "04b_natural_deception.py" --model "mistralai/Mistral-7B-Instruct-v0.3" --runs "$RUNS" --seed "$SEED" &
    local pid2=$!
    wait $pid0 $pid2
}

# ================================================================
# Phase I: Abliteration Geometry (refusal removal effects)
# ================================================================

phase_I() {
    log "╔══════════════════════════════════════════════════════════╗"
    log "║  PHASE I: ABLITERATION GEOMETRY                         ║"
    log "║  What happens to KV-cache geometry when refusal removed ║"
    log "╚══════════════════════════════════════════════════════════╝"
    log_gpu

    # I.1: Baseline Qwen-7B
    log "Step I.1: Baseline Qwen-7B (input-only + geometric sweep)"
    run_experiment "Abliteration Baseline" "1" \
        "07_abliteration_geometry.py" --model "Qwen/Qwen2.5-7B-Instruct" --baseline-only --runs "$RUNS" --seed "$SEED"

    # I.2: Abliterate refusal direction
    log "Step I.2: Abliterate Qwen-7B refusal"
    run_experiment "Abliterate Qwen-7B" "1" \
        "07_abliteration_geometry.py" --model "Qwen/Qwen2.5-7B-Instruct" --abliterate --seed "$SEED"

    # I.3: Abliterated model sweep
    log "Step I.3: Abliterated model geometric sweep"
    run_experiment "Abliterated Sweep" "1" \
        "07_abliteration_geometry.py" --model "./abliterated_qwen7b" --geometric-sweep --runs "$RUNS" --seed "$SEED"

    # I.4: Deception forensics on abliterated model
    log "Step I.4: Deception forensics on abliterated model"
    run_experiment "Abliterated Deception" "1" \
        "04_deception_forensics.py" --model "./abliterated_qwen7b" --runs "$RUNS" --seed "$SEED"
}

# ================================================================
# Phase G: Projector Training
# ================================================================

phase_G() {
    log "╔══════════════════════════════════════════════════════════╗"
    log "║  PHASE G: PROJECTOR TRAINING                            ║"
    log "║  Cross-model cache transfer learning                    ║"
    log "╚══════════════════════════════════════════════════════════╝"
    log_gpu

    # G.1: Small projector (0.6B → 0.5B)
    run_experiment "Projector Small (0.6B→0.5B)" "1" \
        "02b_projector_transfer.py"

    log "NOTE: Medium and large projector configs need manual setup."
    log "See execution plan for 4B→3B and 32B→7B pairs."
}

# ================================================================
# Main execution
# ================================================================

log "╔══════════════════════════════════════════════════════════════╗"
log "║  KV-CACHE FULL EXPERIMENT CAMPAIGN                          ║"
log "║  Machine: Cassidy (3x RTX 3090)                             ║"
log "║  Liberation Labs / THCoalition                              ║"
log "╚══════════════════════════════════════════════════════════════╝"
log ""
log "Project: $PROJECT_DIR"
log "Python: $($PYTHON --version 2>&1)"
log "PyTorch: $($PYTHON -c 'import torch; print(torch.__version__)' 2>/dev/null)"
log ""

if [ -n "$DRY_RUN" ]; then
    log "*** DRY RUN MODE — no models loaded ***"
fi

# Run specific phase or all
if [ -n "$PHASE" ]; then
    case "$PHASE" in
        B) phase_B ;;
        C) phase_C ;;
        D) phase_D ;;
        E) phase_E ;;
        F) phase_F ;;
        G) phase_G ;;
        H) phase_H ;;
        S4) phase_S4 ;;
        I) phase_I ;;
        *) echo "Unknown phase: $PHASE (valid: B C D E F G H S4 I)"; exit 1 ;;
    esac
else
    # Run all phases in order
    log "Running ALL phases (B → I)"
    log ""

    phase_B
    log ""; log "Decision gate: Check Control 3 correlation before proceeding."; log ""

    phase_C
    phase_D
    phase_E
    phase_F
    phase_H
    phase_S4
    phase_I
    phase_G
fi

log ""
log "╔══════════════════════════════════════════════════════════════╗"
log "║  CAMPAIGN COMPLETE                                          ║"
log "╚══════════════════════════════════════════════════════════════╝"
log ""
log "Results in: $RESULTS_DIR"
log "Log: $LOG_FILE"

# List all result files
log ""
log "Result files:"
find "$RESULTS_DIR" -name "*_results.json" -exec ls -lh {} \; 2>/dev/null | while read line; do
    log "  $line"
done
