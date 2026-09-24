#!/usr/bin/env bash
# Stage 2: six-algorithm CelebA 64x64 pixel-space suite.
# Defaults to a non-destructive fresh run; later CLI options override defaults.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$PROJECT_ROOT"

export DIFFUSION_ENTRY_LAUNCHER="scripts/linux/02_train_celeba_pixel.sh"
source scripts/linux/thesis_stage_queue.bash
run_queued_thesis_stage "02-celeba-pixel" \
  ./scripts/linux/train_all_datasets.sh --dataset celeba --mode fresh "$@"
