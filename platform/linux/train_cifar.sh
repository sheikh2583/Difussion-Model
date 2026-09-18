#!/usr/bin/env bash
set -euo pipefail
PLATFORM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$PLATFORM_DIR")")"
exec "$PROJECT_ROOT/scripts/train_all.sh" \
  --dataset cifar10 \
  --mode continue \
  --checkpoint-every 10 \
  --train-only \
  "$@"

