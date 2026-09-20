#!/usr/bin/env bash
# Build the minimal thesis review ZIP on Linux without starting or stopping training.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="$PROJECT_ROOT/venv/bin/python"
ALLOW_RUNNING=false
DRY_RUN=false

usage() {
    cat <<'EOF'
Usage: ./platform/linux/make_minimal_zip.sh [OPTIONS]

Create thesis_review_package_minimal.zip from text configs, documentation,
aggregate tables, and per-run JSONL metrics. Checkpoints and other large binary
artifacts are excluded. Publication is atomic.

Options:
  --allow-running           Permit a read-only snapshot while training is active
  --dry-run                 Print the command without writing the ZIP
  -h, --help                Show this help

By default the script exits while this project is training so the ZIP cannot
capture a metrics file mid-write. It never signals, pauses, or modifies a
training process.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --allow-running) ALLOW_RUNNING=true; shift ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ -x "$PYTHON" ]] || {
    echo "ERROR: virtual-environment Python not found: $PYTHON" >&2
    echo "Run ./platform/linux/init.sh first." >&2
    exit 1
}

COMMAND=("$PYTHON" "$PROJECT_ROOT/scripts/package_review.py")
printf 'Command:'
printf ' %q' "${COMMAND[@]}"
printf '\n'
[[ "$DRY_RUN" == true ]] && exit 0

TRAIN_PATTERN="$PROJECT_ROOT/venv/bin/python train.py"
if [[ "$ALLOW_RUNNING" == false ]] && pgrep -f -- "$TRAIN_PATTERN" >/dev/null; then
    echo "Training is active; the minimal ZIP was not changed." >&2
    echo "Run this after training, or add --allow-running for a read-only snapshot." >&2
    exit 3
fi

cd "$PROJECT_ROOT"
exec "${COMMAND[@]}"
