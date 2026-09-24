#!/usr/bin/env bash
# Stage 3: seven-algorithm CelebA VQ-f4 latent-space suite.
# Defaults to a non-destructive fresh run; later CLI options override defaults.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$PROJECT_ROOT"

export DIFFUSION_ENTRY_LAUNCHER="scripts/linux/03_train_celeba_latent.sh"
source scripts/linux/thesis_stage_queue.bash
run_queued_thesis_stage "03-celeba-latent" \
  ./scripts/linux/train_celeba_latent.sh --mode fresh "$@"
