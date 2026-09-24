#!/usr/bin/env bash
# =============================================================================
# run_inference.sh — Start the read-only thesis UI (Linux / macOS)
# =============================================================================
# Usage:
#   bash scripts/linux/run_inference.sh [OPTIONS]
#
# Options:
#   --host HOST           Bind address. Default: 127.0.0.1
#   --port PORT           TCP port.    Default: 8000
#   --results-dir DIR     Root directory for trained model results.
#                         Default: ./results
#   --self-test           Validate the result catalog and exit; runs no model.
#   --generate-outputs    Generate real cached outputs before starting the UI.
#   --wait-for-gpu        With --generate-outputs, wait for active training.
#   -h, --help            Show this message.
#
# Examples:
#   bash scripts/linux/run_inference.sh
#   bash scripts/linux/run_inference.sh --port 9000 --results-dir /data/models
#   bash scripts/linux/run_inference.sh --self-test
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Defaults
HOST="127.0.0.1"
PORT="8000"
RESULTS_DIR=""
SELF_TEST=""
GENERATE_OUTPUTS=false
WAIT_FOR_GPU=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --host)         HOST="$2"; shift 2 ;;
        --port)         PORT="$2"; shift 2 ;;
        --results-dir)  RESULTS_DIR="$2"; shift 2 ;;
        --self-test)    SELF_TEST="--self-test"; shift ;;
        --generate-outputs) GENERATE_OUTPUTS=true; shift ;;
        --wait-for-gpu) WAIT_FOR_GPU=true; shift ;;
        -h|--help)      sed -n '2,/^# ===/p' "$0"; exit 0 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

PYTHON="$PROJECT_ROOT/venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: Project interpreter not found at '$PYTHON'. Run ./scripts/linux/init.sh first."
    exit 1
fi
export PYTHONPATH="$PROJECT_ROOT"

if [[ "$GENERATE_OUTPUTS" == true ]]; then
    PREPARE=("$PROJECT_ROOT/scripts/linux/generate_inference_outputs.sh")
    [[ "$WAIT_FOR_GPU" == false ]] || PREPARE+=(--wait)
    "${PREPARE[@]}"
fi

ARGS=("web/inference_server.py" "--host" "$HOST" "--port" "$PORT")
[[ -n "$RESULTS_DIR" ]] && ARGS+=("--results-dir" "$RESULTS_DIR")
[[ -n "$SELF_TEST" ]]   && ARGS+=("$SELF_TEST")

echo "=== DiffusionProject Read-Only Thesis UI ==="
[[ -z "$SELF_TEST" ]] && echo "UI available at: http://${HOST}:${PORT}"
[[ -z "$SELF_TEST" ]] && echo "Mode: browser reconstruction only; no checkpoint/GPU execution"
echo "Command: $PYTHON ${ARGS[*]}"
echo ""

cd "$PROJECT_ROOT"
exec "$PYTHON" "${ARGS[@]}"
