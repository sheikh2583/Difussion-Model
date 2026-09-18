#!/usr/bin/env bash
# =============================================================================
# run_evaluate.sh — Evaluate a trained checkpoint (Linux / macOS)
# =============================================================================
# Usage:
#   bash scripts/run_evaluate.sh [OPTIONS]
#
# Required options:
#   -a, --algorithm    Algorithm: mock | fm | fm_lognorm | mf
#   -k, --checkpoint   Path to .pt checkpoint file
#   -c, --config       Path to config.json saved with the checkpoint
#
# Optional options:
#   --make-plots       Generate standard evaluation plots after evaluation
#   -h, --help         Show this message
#
# Examples:
#   bash scripts/run_evaluate.sh \
#       -a fm \
#       -k results/fm_cifar10/checkpoints/run_1/FlowMatchingAlgorithm_epoch100.pt \
#       -c results/fm_cifar10/config.json
#
#   bash scripts/run_evaluate.sh \
#       -a mf \
#       -k results/mf_cifar10/checkpoints/run_1/MeanFlowAlgorithm_epoch100.pt \
#       -c results/mf_cifar10/config.json \
#       --make-plots
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

ALGORITHM=""
CHECKPOINT=""
CONFIG=""
MAKE_PLOTS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -a|--algorithm)   ALGORITHM="$2"; shift 2 ;;
        -k|--checkpoint)  CHECKPOINT="$2"; shift 2 ;;
        -c|--config)      CONFIG="$2"; shift 2 ;;
        --make-plots)     MAKE_PLOTS="--make-plots"; shift ;;
        -h|--help)        sed -n '2,/^# ===/p' "$0"; exit 0 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$ALGORITHM" || -z "$CHECKPOINT" || -z "$CONFIG" ]]; then
    echo "ERROR: --algorithm, --checkpoint, and --config are all required."
    echo "Run with -h for usage."
    exit 1
fi

VENV_ACTIVATE="$PROJECT_ROOT/venv/bin/activate"
if [[ ! -f "$VENV_ACTIVATE" ]]; then
    echo "ERROR: Virtual environment not found at '$VENV_ACTIVATE'."
    exit 1
fi
# shellcheck source=/dev/null
source "$VENV_ACTIVATE"
export PYTHONPATH="$PROJECT_ROOT"

ARGS=("evaluate.py"
      "--algorithm"  "$ALGORITHM"
      "--checkpoint" "$CHECKPOINT"
      "--config"     "$CONFIG")
[[ -n "$MAKE_PLOTS" ]] && ARGS+=("$MAKE_PLOTS")

echo "=== DiffusionProject Evaluation ==="
echo "Algorithm  : $ALGORITHM"
echo "Checkpoint : $CHECKPOINT"
echo "Config     : $CONFIG"
echo "Command    : python ${ARGS[*]}"
echo ""

cd "$PROJECT_ROOT"
python "${ARGS[@]}"
