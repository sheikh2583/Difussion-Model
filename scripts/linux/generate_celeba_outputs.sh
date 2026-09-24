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
PIXEL_COMMAND=("$PYTHON" "$PROJECT_ROOT/scripts/generate_result_gifs.py" --dataset celeba "${ARGS[@]}")
LATENT_COMMAND=("$PYTHON" "$PROJECT_ROOT/scripts/generate_result_gifs.py" --dataset celeba_latent "${ARGS[@]}")
SAMPLE_COMMAND=("$PYTHON" "$PROJECT_ROOT/scripts/generate_checkpoint_samples.py" --dataset-family celeba --seeds 0)
for command_name in PIXEL_COMMAND LATENT_COMMAND SAMPLE_COMMAND; do
    declare -n command_ref="$command_name"
    printf 'Command:'; printf ' %q' "${command_ref[@]}"; printf '\n'
done
[[ "$DRY_RUN" == false ]] || exit 0
[[ -x "$PYTHON" ]] || { echo "ERROR: project environment not found. Run ./scripts/linux/init.sh first." >&2; exit 1; }
TRAIN_PATTERN="$PROJECT_ROOT/venv/bin/python train.py"
cd "$PROJECT_ROOT"
if pgrep -f -- "$TRAIN_PATTERN" >/dev/null; then
    if [[ "$ALLOW_RUNNING" == false ]]; then
        echo "Training is active; outputs were not changed." >&2
        echo "Run this after training, or add --allow-running for metric-only snapshots." >&2
        exit 3
    fi
    echo "Training is active; generating read-only metric animations only." >&2
    "${PIXEL_COMMAND[@]}"
    exec "${LATENT_COMMAND[@]}"
fi
"${PIXEL_COMMAND[@]}"
"${LATENT_COMMAND[@]}"
exec "${SAMPLE_COMMAND[@]}"
