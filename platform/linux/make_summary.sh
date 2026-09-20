#!/usr/bin/env bash
# Build the aggregate metrics summary on Linux without starting or stopping training.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="$PROJECT_ROOT/venv/bin/python"
ALLOW_RUNNING=false
DRY_RUN=false
ARGS=()

usage() {
    cat <<'EOF'
Usage: ./platform/linux/make_summary.sh [OPTIONS]

Rebuild results/aggregate from the canonical metrics JSONL files.

Options:
  --results-root PATH       Metrics root (default: results)
  --output-dir PATH         Aggregate output (default: results/aggregate)
  --allow-running           Permit a read-only snapshot while training is active
  --dry-run                 Print the command without writing outputs
  -h, --help                Show this help

By default the script exits if this project is training, preventing a summary
from capturing a metrics file while it is being appended. It never signals,
pauses, or otherwise modifies a training process.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --allow-running) ALLOW_RUNNING=true; shift ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help) usage; exit 0 ;;
        --results-root|--output-dir)
            [[ $# -ge 2 ]] || { echo "ERROR: $1 requires a path." >&2; exit 2; }
            ARGS+=("$1" "$2"); shift 2 ;;
        *) echo "ERROR: unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ -x "$PYTHON" ]] || {
    echo "ERROR: virtual-environment Python not found: $PYTHON" >&2
    echo "Run ./platform/linux/init.sh first." >&2
    exit 1
}

COMMAND=("$PYTHON" "$PROJECT_ROOT/scripts/aggregate_results.py" "${ARGS[@]}")
printf 'Command:'
printf ' %q' "${COMMAND[@]}"
printf '\n'
[[ "$DRY_RUN" == true ]] && exit 0

TRAIN_PATTERN="$PROJECT_ROOT/venv/bin/python train.py"
if [[ "$ALLOW_RUNNING" == false ]] && pgrep -f -- "$TRAIN_PATTERN" >/dev/null; then
    echo "Training is active; aggregate outputs were not changed." >&2
    echo "Run this after training, or add --allow-running for a read-only snapshot." >&2
    exit 3
fi

cd "$PROJECT_ROOT"
exec "${COMMAND[@]}"
