#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="$PROJECT_ROOT/venv/bin/python"
ALLOW_RUNNING=false
DRY_RUN=false
ARGS=()
for argument in "$@"; do
    case "$argument" in
        --allow-running) ALLOW_RUNNING=true ;;
        --dry-run) DRY_RUN=true ;;
        *) ARGS+=("$argument") ;;
    esac
done
COMMAND=("$PYTHON" "$PROJECT_ROOT/scripts/generate_result_gifs.py" --dataset cifar10 "${ARGS[@]}")
printf 'Command:'; printf ' %q' "${COMMAND[@]}"; printf '\n'
[[ "$DRY_RUN" == false ]] || exit 0
[[ -x "$PYTHON" ]] || { echo "ERROR: project environment not found. Run ./scripts/linux/init.sh first." >&2; exit 1; }
TRAIN_PATTERN="$PROJECT_ROOT/venv/bin/python train.py"
if [[ "$ALLOW_RUNNING" == false ]] && pgrep -f -- "$TRAIN_PATTERN" >/dev/null; then
    echo "Training is active; animations were not changed." >&2
    echo "Run this after training, or add --allow-running for a read-only snapshot." >&2
    exit 3
fi
cd "$PROJECT_ROOT"
exec "${COMMAND[@]}"
