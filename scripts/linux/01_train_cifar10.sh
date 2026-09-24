#!/usr/bin/env bash
# Stage 1: six-algorithm CIFAR-10 32x32 suite on the current SimpleUNet.
# Defaults to a non-destructive fresh run; later CLI options override defaults.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$PROJECT_ROOT"

export DIFFUSION_ENTRY_LAUNCHER="scripts/linux/01_train_cifar10.sh"
source scripts/linux/thesis_stage_queue.bash
run_queued_thesis_stage "01-cifar10" \
  ./scripts/linux/train_all_datasets.sh \
  --dataset cifar10 --cifar-backbone current --mode fresh "$@"
