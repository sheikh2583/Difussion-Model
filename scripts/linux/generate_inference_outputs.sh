#!/usr/bin/env bash
# Pre-render real checkpoint generations for the browser-only thesis demo.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
PYTHON="$PROJECT_ROOT/venv/bin/python"
EXPERIMENTS="fm_lognorm_celeba"
NFE="1,5,10,20,50"
SEEDS="0,1,2,3"
N_SAMPLES=64
WAIT_FOR_GPU=false
ALL_CELEBA=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --experiments) EXPERIMENTS="$2"; shift 2 ;;
        --nfe) NFE="$2"; shift 2 ;;
        --seeds) SEEDS="$2"; shift 2 ;;
        --n-samples) N_SAMPLES="$2"; shift 2 ;;
        --all-celeba) ALL_CELEBA=true; shift ;;
        --wait) WAIT_FOR_GPU=true; shift ;;
        -h|--help)
            echo "Usage: $0 [--experiments NAMES | --all-celeba] [--nfe VALUES] [--seeds VALUES] [--n-samples N] [--wait]"
            echo "Defaults: --experiments fm_lognorm_celeba --nfe 1,5,10,20,50 --seeds 0,1,2,3 --n-samples 64"
            exit 0 ;;
        *) echo "ERROR: unknown argument: $1" >&2; exit 2 ;;
    esac
done

[[ -x "$PYTHON" ]] || {
    echo "ERROR: project interpreter not found at '$PYTHON'." >&2
    exit 1
}
[[ "$N_SAMPLES" =~ ^[1-9][0-9]*$ ]] || {
    echo "ERROR: --n-samples must be a positive integer." >&2
    exit 2
}

cd "$PROJECT_ROOT"
if [[ "$WAIT_FOR_GPU" == true ]]; then
    echo "Waiting for the project GPU workflow lock to clear..."
    while [[ -e results/.lock ]]; do sleep 15; done
fi

COMMAND=(
    "$PYTHON" scripts/generate_checkpoint_samples.py
    --nfe "$NFE"
    --seeds "$SEEDS"
    --n-samples "$N_SAMPLES"
)
if [[ "$ALL_CELEBA" == true ]]; then
    COMMAND+=(--dataset-family celeba)
else
    COMMAND+=(--experiments "$EXPERIMENTS")
fi

printf 'Command:'; printf ' %q' "${COMMAND[@]}"; printf '\n'
exec "${COMMAND[@]}"
