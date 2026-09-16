#!/usr/bin/env bash
# =============================================================================
# run_train.sh — Train a generative model on CIFAR-10 (Linux / macOS)
# =============================================================================
# Usage:
#   bash scripts/run_train.sh [OPTIONS]
#
# Options:
#   -a, --algorithm   Algorithm to train:
#                     fm | fm_lognorm | mf | mf_distill | consistency | reflow | mock
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
#   -n, --name        Override experiment_name in the config
#   -e, --epochs      Override epoch count from the config
#   -h, --help        Show this message
#
# Examples:
#   bash scripts/run_train.sh -a mock
#   bash scripts/run_train.sh -a fm -c config/fm_full.json
#   bash scripts/run_train.sh -a mf -c config/mf_full.json -e 200 -n mf_run2
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Resolve project root (one level up from scripts/)
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
ALGORITHM="fm"
CONFIG=""
EXPERIMENT_NAME=""
EPOCHS=""

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        -a|--algorithm)  ALGORITHM="$2"; shift 2 ;;
        -c|--config)     CONFIG="$2"; shift 2 ;;
        -n|--name)       EXPERIMENT_NAME="$2"; shift 2 ;;
        -e|--epochs)     EPOCHS="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,/^# ===/p' "$0"; exit 0 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Activate virtual environment
# ---------------------------------------------------------------------------
VENV_ACTIVATE="$PROJECT_ROOT/venv/bin/activate"
if [[ ! -f "$VENV_ACTIVATE" ]]; then
    echo "ERROR: Virtual environment not found at '$VENV_ACTIVATE'."
    echo "Create it with: python -m venv venv && pip install -r requirements.txt"
    exit 1
fi
# shellcheck source=/dev/null
source "$VENV_ACTIVATE"

# ---------------------------------------------------------------------------
# Set PYTHONPATH
# ---------------------------------------------------------------------------
export PYTHONPATH="$PROJECT_ROOT"

# ---------------------------------------------------------------------------
# Build argument list
# ---------------------------------------------------------------------------
ARGS=("train.py" "--algorithm" "$ALGORITHM")
[[ -n "$CONFIG" ]]          && ARGS+=("--config" "$CONFIG")
[[ -n "$EXPERIMENT_NAME" ]] && ARGS+=("--experiment-name" "$EXPERIMENT_NAME")
[[ -n "$EPOCHS" ]]          && ARGS+=("--epochs" "$EPOCHS")

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
echo "=== DiffusionProject Training ==="
echo "Algorithm : $ALGORITHM"
echo "Config    : ${CONFIG:-(defaults)}"
echo "Command   : python ${ARGS[*]}"
echo ""

cd "$PROJECT_ROOT"
python "${ARGS[@]}"
