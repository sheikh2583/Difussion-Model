#!/usr/bin/env bash
# =============================================================================
# run_inference.sh — Start the local CIFAR-10 inference UI (Linux / macOS)
# =============================================================================
# Usage:
#   bash scripts/run_inference.sh [OPTIONS]
#
# Options:
#   --host HOST           Bind address. Default: 127.0.0.1
#   --port PORT           TCP port.    Default: 8000
#   --results-dir DIR     Root directory for trained model results.
#                         Default: ./results
#   --self-test           Generate one image per available model then exit.
#   -h, --help            Show this message.
#
# Examples:
#   bash scripts/run_inference.sh
#   bash scripts/run_inference.sh --port 9000 --results-dir /data/models
#   bash scripts/run_inference.sh --self-test
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Defaults
HOST="127.0.0.1"
PORT="8000"
RESULTS_DIR=""
SELF_TEST=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --host)         HOST="$2"; shift 2 ;;
        --port)         PORT="$2"; shift 2 ;;
        --results-dir)  RESULTS_DIR="$2"; shift 2 ;;
        --self-test)    SELF_TEST="--self-test"; shift ;;
        -h|--help)      sed -n '2,/^# ===/p' "$0"; exit 0 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

VENV_ACTIVATE="$PROJECT_ROOT/venv/bin/activate"
if [[ ! -f "$VENV_ACTIVATE" ]]; then
    echo "ERROR: Virtual environment not found at '$VENV_ACTIVATE'."
    echo "Create it with: python -m venv venv && pip install -r requirements.txt"
    exit 1
fi
# shellcheck source=/dev/null
source "$VENV_ACTIVATE"
export PYTHONPATH="$PROJECT_ROOT"

ARGS=("web/inference_server.py" "--host" "$HOST" "--port" "$PORT")
[[ -n "$RESULTS_DIR" ]] && ARGS+=("--results-dir" "$RESULTS_DIR")
[[ -n "$SELF_TEST" ]]   && ARGS+=("$SELF_TEST")

echo "=== DiffusionProject Inference Server ==="
[[ -z "$SELF_TEST" ]] && echo "UI available at: http://${HOST}:${PORT}"
echo "Command: python ${ARGS[*]}"
echo ""

cd "$PROJECT_ROOT"
python "${ARGS[@]}"
