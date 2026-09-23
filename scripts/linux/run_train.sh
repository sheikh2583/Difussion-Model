#!/usr/bin/env bash
# =============================================================================
# run_train.sh — Train one configured model (Linux / macOS)
# =============================================================================
# Usage:
#   bash scripts/linux/run_train.sh [OPTIONS]
#
# Options:
#   -a, --algorithm   Algorithm to train:
#                     fm | fm_lognorm | mf | mf_hutchinson | mf_distill |
#                     consistency | reflow | mock
#                     Default: fm
#   -c, --config      Path to a JSON config file. Preset configs:
#                         config/smoke_fast.json         (quick smoke test)
#                         config/fm_full.json            (full FM, CIFAR-10)
#                         config/fm_lognorm_full.json    (FM + logit-normal)
#                         config/fm_lognorm_budget.json  (reduced batch/samples)
#                         config/mf_full.json            (full Mean Flow)
#                         config/mf_distill_full.json    (MF distillation)
#                         config/consistency_full.json   (Consistency Models)
#                         config/reflow_full.json        (Rectified Flow Reflow)
#                         config/fm_celeba64.json        (FM on CelebA 64x64)
#                         config/mf_celeba64.json        (MF on CelebA 64x64)
#                         config/fm_celeba_latent.json   (FM on CelebA VQ-f4 latents)
#                         config/fm_lognorm_celeba_latent.json
#                         config/mf_celeba_latent.json
#                         config/mf_hutchinson_cv_celeba_latent.json (latent only)
#                         config/mf_distill_celeba_latent.json
#                         config/consistency_celeba_latent.json
#                         config/reflow_celeba_latent.json
#   -n, --name        Override experiment_name in the config
#   -e, --epochs      Override epoch count from the config
#   -b, --batch-size  Override batch size for this machine
#   -k, --checkpoint-every  Save .pt + ZIP every N epochs (default: config; full presets use 10)
#   -m, --mode        continue | fresh (default: continue)
#       --train-only  Skip FID/evaluation and train/checkpoint only
#       --machine-label NAME  Label stored with results (default: hostname)
#   -h, --help        Show this message
#
# Examples:
#   bash scripts/linux/run_train.sh -a mock
#   bash scripts/linux/run_train.sh -a fm -c config/fm_full.json
#   bash scripts/linux/run_train.sh -a mf -c config/mf_full.json -e 200 -n mf_run2
#   bash scripts/linux/run_train.sh -a consistency -c config/consistency_celeba_latent.json
#   bash scripts/linux/run_train.sh -a mf_hutchinson -c config/mf_hutchinson_cv_celeba_latent.json
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Resolve project root (two levels up from scripts/linux/)
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
ALGORITHM="fm"
CONFIG=""
EXPERIMENT_NAME=""
EPOCHS=""
BATCH_SIZE=""
CHECKPOINT_EVERY=""
MODE="continue"
TRAIN_ONLY=false
MACHINE_LABEL=""

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        -a|--algorithm)  ALGORITHM="$2"; shift 2 ;;
        -c|--config)     CONFIG="$2"; shift 2 ;;
        -n|--name)       EXPERIMENT_NAME="$2"; shift 2 ;;
        -e|--epochs)     EPOCHS="$2"; shift 2 ;;
        -b|--batch-size) BATCH_SIZE="$2"; shift 2 ;;
        -k|--checkpoint-every) CHECKPOINT_EVERY="$2"; shift 2 ;;
        -m|--mode)       MODE="$2"; shift 2 ;;
        --train-only)    TRAIN_ONLY=true; shift ;;
        --machine-label) MACHINE_LABEL="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,/^set -euo pipefail/p' "$0" | sed '$d'; exit 0 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

case "$MODE" in
    continue|fresh) ;;
    *) echo "ERROR: --mode must be continue or fresh" >&2; exit 2 ;;
esac

# ---------------------------------------------------------------------------
# Resolve the project interpreter
# ---------------------------------------------------------------------------
PYTHON="$PROJECT_ROOT/venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: Project interpreter not found at '$PYTHON'."
    echo "Run ./scripts/linux/init.sh first."
    exit 1
fi

# ---------------------------------------------------------------------------
# Set PYTHONPATH
# ---------------------------------------------------------------------------
export PYTHONPATH="$PROJECT_ROOT"

# ---------------------------------------------------------------------------
# Build argument list
# ---------------------------------------------------------------------------
ARGS=("train.py" "--algorithm" "$ALGORITHM" "--mode" "$MODE")
[[ -n "$CONFIG" ]]          && ARGS+=("--config" "$CONFIG")
[[ -n "$EXPERIMENT_NAME" ]] && ARGS+=("--experiment-name" "$EXPERIMENT_NAME")
[[ -n "$EPOCHS" ]]          && ARGS+=("--epochs" "$EPOCHS")
[[ -n "$BATCH_SIZE" ]]      && ARGS+=("--batch-size" "$BATCH_SIZE")
[[ -n "$CHECKPOINT_EVERY" ]] && ARGS+=("--checkpoint-every" "$CHECKPOINT_EVERY")
[[ "$TRAIN_ONLY" == true ]] && ARGS+=("--train-only")
[[ -n "$MACHINE_LABEL" ]]   && ARGS+=("--machine-label" "$MACHINE_LABEL")

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
echo "=== DiffusionProject Training ==="
echo "Algorithm : $ALGORITHM"
echo "Config    : ${CONFIG:-(defaults)}"
echo "Mode      : $MODE"
echo "Command   : $PYTHON ${ARGS[*]}"
echo ""

cd "$PROJECT_ROOT"
exec "$PYTHON" "${ARGS[@]}"
